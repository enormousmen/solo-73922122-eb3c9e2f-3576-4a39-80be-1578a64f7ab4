"""Single-process micro-batch inference engine with fault-frame localization.

Concurrency model
-----------------
* Everything except the blocking model forward pass runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch is ever actually
  inferring at any moment. Failure localization retries run on the SAME
  executor (the original compute channel), serialized with normal batches.
* Capacity accounting: an order occupies a slot while it is queued AND while
  it is in the compute pipeline (batch computing, awaiting a per-frame retry,
  or its retry computing — even if it was cancelled or expired mid-flight).
  The slot is released only when the order leaves the compute pipeline.

Fault-frame localization (bounded retries)
------------------------------------------
When a batch forward pass raises (e.g. one camera frame breaks TorchScript
batch inference), the worker re-infers each still-eligible frame ONE AT A
TIME on the same executor, at most `retry_attempts` times per frame:

* frames that succeed individually complete with their normal scores;
* frames that also fail individually are marked `failed` (with `error`);
* frames cancelled/expired before their retry starts never re-enter the
  model and release their slot when the pipeline passes them;
* a frame whose retry compute is already running keeps its slot until the
  compute ends; a late result is written back ONLY to an order still in
  `processing` whose deadline has not passed.

Retries are bounded: a frame costs at most 1 (batch) + `retry_attempts`
model calls, so even a completely broken model terminates every order with
`failed` in bounded time and cannot stall later orders forever. Submit,
query and cancel stay responsive throughout because all model calls run on
the executor thread.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held
                                until its in-flight/pipeline compute ends)
processing -> expired          (deadline passed mid-flight; result discarded)
processing -> completed        (batch ok, or individual retry ok after batch failure)
processing -> failed           (batch failed AND bounded individual retries exhausted)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"


def _short_error(exc: BaseException, limit: int = 300) -> str:
    """One-line, size-bounded error string for the API.

    torch.jit.Error carries a multi-line TorchScript traceback; the useful
    part (e.g. "builtins.RuntimeError: anomaly: overexposed frame in batch")
    is the last non-empty line.
    """
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    msg = lines[-1] if lines else type(exc).__name__
    if len(msg) > limit:
        msg = msg[: limit - 3] + "..."
    return msg


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    tensor: torch.Tensor
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    error: Optional[str] = None
    batch_id: Optional[str] = None
    attempts: int = 0  # model invocations involving this frame (batch + retries)
    slot_held: bool = False  # occupies compute-pipeline capacity (worker-managed)
    created_at: float = field(default_factory=time.monotonic)


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
        retry_attempts: int = 1,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0
        # Bounded per-frame retries after a batch failure (0 = legacy behavior:
        # a failed batch fails all its frames immediately).
        self.retry_attempts = max(0, retry_attempts)

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        self._orders: dict[str, Order] = {}
        self._inflight = 0  # orders holding a compute-pipeline slot
        self._not_empty = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._worker: Optional[asyncio.Task] = None
        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0
        self.model_calls = 0  # every model invocation (batches + retries)
        self.retries_started = 0  # individual retry invocations after batch failures

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Mark queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        if o.state in (QUEUED, PROCESSING) and o.deadline_at <= time.monotonic():
            if o.state == QUEUED:
                try:
                    self._pending.remove(o)
                except ValueError:
                    pass
            o.state = EXPIRED

    def _hold_slot(self, o: Order) -> None:
        o.slot_held = True
        self._inflight += 1

    def _release_slot(self, o: Order) -> None:
        """Idempotent: an order's compute-pipeline slot is released exactly once."""
        if o.slot_held:
            o.slot_held = False
            self._inflight -= 1

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return len(self._pending) + self._inflight

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        for o in self._orders.values():
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "capacity_used": len(self._pending) + self._inflight,
            "max_capacity": self.max_capacity,
            "queued": len(self._pending),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "model_calls": self.model_calls,
            "retries_started": self.retries_started,
            "retry_attempts": self.retry_attempts,
            "orders": counts,
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(self, tensor: torch.Tensor, deadline_ms: int) -> Optional[Order]:
        """Accept an image, or return None when at capacity (16 by default)."""
        if self.capacity_used() >= self.max_capacity:
            return None
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self._not_empty.set()
        return order

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)
        return o

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            return None
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the compute
            # pipeline passes the order (batch compute end / its retry turn);
            # it never re-enters the model once cancelled.
            o.state = CANCELLED
        return o

    # ------------------------------------------------------------------ #
    # batching worker
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._not_empty.wait()

            # Batch-formation window: the idle model waits at most
            # `batch_wait_s` for more work, up to `batch_size` images.
            batch: list[Order] = []
            window_end = loop.time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                self._purge_expired_pending()
                while self._pending and len(batch) < self.batch_size:
                    batch.append(self._pending.popleft())
                if len(batch) >= self.batch_size:
                    break
                remaining = window_end - loop.time()
                if remaining <= 0:
                    break
                self._not_empty.clear()
                try:
                    await asyncio.wait_for(self._not_empty.wait(), remaining)
                except asyncio.TimeoutError:
                    break

            # Final gate: cancelled/expired orders never enter the model.
            now = time.monotonic()
            batch = [o for o in batch if o.state == QUEUED and o.deadline_at > now]
            if not batch:
                continue

            batch_id = uuid.uuid4().hex[:8]
            for o in batch:
                o.state = PROCESSING
                o.batch_id = batch_id
                o.attempts += 1
                self._hold_slot(o)
            self.batches_started += 1
            self.last_batch_size = len(batch)
            tensors = torch.stack([o.tensor for o in batch])

            try:
                scores = await loop.run_in_executor(self._executor, self._infer, tensors)
            except Exception as exc:  # noqa: BLE001 - model failure must not kill the worker
                log.exception("inference failed for batch %s; localizing faulty frames", batch_id)
                try:
                    await self._localize_failures(batch, batch_id, exc)
                except Exception:  # noqa: BLE001 - never let the worker die
                    log.exception("failure localization crashed for batch %s", batch_id)
                    for o in batch:
                        if o.state == PROCESSING:
                            o.state = FAILED
                            o.error = "internal error during failure localization"
                        self._release_slot(o)
                continue

            now = time.monotonic()
            for o, sc in zip(batch, scores):
                if o.state == PROCESSING:
                    if o.deadline_at <= now:
                        o.state = EXPIRED
                    else:
                        o.state = COMPLETED
                        predicted = max(range(len(sc)), key=lambda i: sc[i])
                        o.result = {"predicted_class": predicted, "scores": sc}
                # Cancelled mid-flight: result is discarded; the late result
                # must NOT revive the order.
                self._release_slot(o)

    async def _localize_failures(
        self, batch: list[Order], batch_id: str, batch_exc: Exception
    ) -> None:
        """Bounded per-frame retries on the SAME executor after a batch failure.

        Each frame still in `processing` is re-inferred alone, at most
        `retry_attempts` times. Total model calls for the batch are bounded
        by 1 + len(batch) * retry_attempts, so a completely broken model
        fails every frame in bounded time instead of stalling later orders.
        """
        loop = asyncio.get_running_loop()
        for o in batch:
            now = time.monotonic()
            if o.state != PROCESSING or o.deadline_at <= now:
                # Cancelled mid-flight, or expired while the batch / earlier
                # retries were computing: terminal already, and it does NOT
                # participate in retries that have not started. Release the
                # slot as the pipeline passes it.
                if o.state == PROCESSING:
                    o.state = EXPIRED
                self._release_slot(o)
                continue

            scores: Optional[list[float]] = None
            last_exc: BaseException = batch_exc
            for _ in range(self.retry_attempts):
                # Re-checked before EVERY attempt: a cancel/expiry that landed
                # while an earlier frame was computing means this retry never
                # starts (the order shows its terminal state already).
                if o.state != PROCESSING or o.deadline_at <= time.monotonic():
                    break
                o.attempts += 1
                self.retries_started += 1
                try:
                    rows = await loop.run_in_executor(
                        self._executor, self._infer, o.tensor.unsqueeze(0)
                    )
                    scores = rows[0]
                    break
                except Exception as exc:  # noqa: BLE001 - per-frame failure is expected here
                    last_exc = exc
                    log.warning("retry failed for order %s (batch %s)", o.id, batch_id)

            now = time.monotonic()
            if o.state == PROCESSING:
                if o.deadline_at <= now:
                    o.state = EXPIRED
                elif scores is not None:
                    o.state = COMPLETED
                    predicted = max(range(len(scores)), key=lambda i: scores[i])
                    o.result = {"predicted_class": predicted, "scores": scores}
                else:
                    o.state = FAILED
                    o.error = (
                        f"inference failed after {o.attempts} attempt(s): "
                        f"{_short_error(last_exc)}"
                    )
            # Else: cancelled/expired while its retry was computing — the slot
            # was held until the compute actually ended; the late result is
            # discarded and must not overwrite the terminal state.
            self._release_slot(o)

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — the only shared state it touches
    # is the model_calls counter (single writer, readers tolerate staleness)
    # ------------------------------------------------------------------ #
    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        self.model_calls += 1
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return [[float(v) for v in row] for row in probs]
