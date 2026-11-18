"""Fault-frame localization tests: bounded per-frame retries after batch failure.

The anomaly demo model (app/model_anomaly.pt) raises RuntimeError whenever a
batch contains an overexposed frame, simulating the camera image that breaks
TorchScript batch inference. The engine must then retry each still-eligible
frame individually on the same compute channel: clean frames complete with
their normal scores, the poison frame is marked `failed`.

A second fixture model always raises, simulating a completely broken model:
orders must still terminate (failed) within a bounded number of model calls.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
import torch
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
ANOMALY_MODEL = Path(__file__).resolve().parent.parent / "app" / "model_anomaly.pt"
POISON = "sample_overexposed.png"
CLEAN3 = ["sample_gradient.png", "sample_circles.png", "sample_stripes.png"]


class _AlwaysFail(torch.nn.Module):
    """TorchScript model whose every forward pass raises (broken model demo)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("model is broken")
        return x


@pytest.fixture(scope="session")
def broken_model_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("models") / "always_fail.pt"
    torch.jit.script(_AlwaysFail()).save(str(path))
    return str(path)


def make_settings(**kw) -> Settings:
    kw.setdefault("model_path", str(ANOMALY_MODEL))
    return Settings(**kw)


class Client:
    """httpx client wrapper with helpers (same shape as tests/test_api.py)."""

    def __init__(self, http: httpx.AsyncClient, app):
        self.http = http
        self.app = app

    @property
    def engine(self):
        return self.app.state.engine

    async def submit(self, name: str, deadline_ms: int = 60_000):
        data = (SAMPLES / name).read_bytes()
        return await self.http.post(
            "/predict",
            files={"file": (name, data, "image/png")},
            data={"deadline_ms": str(deadline_ms)},
        )

    async def get(self, order_id: str):
        return (await self.http.get(f"/orders/{order_id}")).json()

    async def cancel(self, order_id: str):
        return await self.http.post(f"/orders/{order_id}/cancel")

    async def stats(self):
        return (await self.http.get("/stats")).json()

    async def wait_state(self, order_id: str, states, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        body = {}
        while time.monotonic() < deadline:
            body = await self.get(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")

    async def wait_for(self, predicate, timeout: float = 10.0, interval: float = 0.01):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(interval)
        raise AssertionError("condition not met within timeout")


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        app = create_app(make_settings(**settings_kw))
        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        client = Client(http, app)
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------- #
# core: one poison frame fails the batch; mates still get their scores
# --------------------------------------------------------------------- #
async def test_fault_frame_isolated_and_batch_mates_complete(make_client):
    c = await make_client()

    # reference: each clean image inferred alone through the anomaly model
    single_scores = {}
    for name in CLEAN3:
        r = await c.submit(name)
        assert r.status_code == 201, r.text
        body = await c.wait_state(r.json()["order_id"], {"completed"})
        assert body["attempts"] == 1  # no retry needed
        single_scores[name] = body["result"]["scores"]

    # one batch of 4 including the poison frame -> batch forward raises
    names = CLEAN3[:2] + [POISON, CLEAN3[2]]
    responses = await asyncio.gather(*[c.submit(n) for n in names])
    ids = [r.json()["order_id"] for r in responses]
    bodies = await asyncio.gather(
        *[c.wait_state(i, {"completed", "failed"}) for i in ids]
    )

    assert len({b["batch_id"] for b in bodies}) == 1, "expected one shared batch"
    assert c.engine.last_batch_size == 4

    by_name = dict(zip(names, bodies))
    # clean frames: completed via individual retry, scores match single inference
    for name in CLEAN3:
        body = by_name[name]
        assert body["status"] == "completed", body
        assert body["attempts"] == 2  # batch attempt + one individual retry
        assert body["error"] is None
        for a, b in zip(body["result"]["scores"], single_scores[name]):
            assert abs(a - b) <= 1e-5, f"{name}: {a} vs {b}"
    # poison frame: failed on its own too -> terminal `failed` with an error
    poison = by_name[POISON]
    assert poison["status"] == "failed"
    assert poison["result"] is None
    assert poison["attempts"] == 2
    assert "overexposed" in poison["error"]

    # bounded model usage: 3 singles + 1 batch + 4 individual retries
    assert c.engine.retries_started == 4
    assert c.engine.model_calls == 3 + 1 + 4
    # all slots released after the failure pipeline drained
    assert (await c.stats())["capacity_used"] == 0


async def test_poison_frame_fails_even_without_retry(make_client):
    """retry_attempts=0 keeps the legacy behavior: batch failure fails all."""
    c = await make_client(retry_attempts=0)
    responses = await asyncio.gather(*[c.submit("sample_gradient.png"), c.submit(POISON)])
    ids = [r.json()["order_id"] for r in responses]
    bodies = await asyncio.gather(*[c.wait_state(i, {"failed"}) for i in ids])
    for body in bodies:
        assert body["status"] == "failed"
        assert body["attempts"] == 1  # the batch attempt only, no retries
        assert body["error"]
    assert c.engine.retries_started == 0
    assert c.engine.model_calls == 1


# --------------------------------------------------------------------- #
# completely broken model: bounded termination, API stays responsive
# --------------------------------------------------------------------- #
async def test_broken_model_bounded_and_later_orders_not_stuck(make_client, broken_model_path):
    c = await make_client(
        model_path=broken_model_path, retry_attempts=2, inference_delay_ms=250
    )

    responses = await asyncio.gather(*[c.submit(n) for n in CLEAN3])
    ids = [r.json()["order_id"] for r in responses]

    # while retries are running, submit/query/cancel must stay responsive
    await c.wait_for(lambda: c.engine.retries_started > 0)

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    (_, dt1) = await timed(c.http.get("/stats"))
    (r_new, dt2) = await timed(c.submit("sample_noise.png"))
    (r_cancel, dt3) = await timed(c.cancel(r_new.json()["order_id"]))
    (_, dt4) = await timed(c.http.get(f"/orders/{ids[0]}"))
    for dt in (dt1, dt2, dt3, dt4):
        assert dt < 0.5, f"API blocked for {dt:.3f}s during failure localization"
    assert r_new.status_code == 201
    assert r_cancel.json()["status"] == "cancelled"

    # every frame terminates as failed after exactly 1 batch + 2 retry calls
    bodies = await asyncio.gather(*[c.wait_state(i, {"failed"}) for i in ids])
    for body in bodies:
        assert body["attempts"] == 3
        assert "model is broken" in body["error"]
    assert c.engine.model_calls == 1 + 3 * 2  # bounded: 7 calls, no infinite loop

    # a later order also terminates (failed) instead of waiting forever
    r = await c.submit("sample_noise.png")
    body = await c.wait_state(r.json()["order_id"], {"failed"})
    assert body["attempts"] == 3
    assert c.engine.model_calls == 10
    assert (await c.stats())["capacity_used"] == 0


# --------------------------------------------------------------------- #
# cancellation / expiry interaction with the retry pipeline
# --------------------------------------------------------------------- #
async def test_cancel_before_retry_skips_frame(make_client):
    c = await make_client(inference_delay_ms=800)
    names = ["sample_gradient.png", "sample_circles.png", POISON, "sample_stripes.png"]
    responses = await asyncio.gather(*[c.submit(n) for n in names])
    ids = dict(zip(names, [r.json()["order_id"] for r in responses]))

    await asyncio.sleep(0.3)  # batch of 4 is computing now
    r = await c.cancel(ids["sample_circles.png"])
    assert r.json()["status"] == "cancelled"  # terminal immediately

    # while the first individual retry runs, all 4 frames still occupy slots
    # (frames awaiting retry keep their capacity)
    await c.wait_for(lambda: c.engine.retries_started >= 1)
    stats = await c.stats()
    assert stats["capacity_used"] == 4, stats

    gradient = await c.wait_state(ids["sample_gradient.png"], {"completed"})
    stripes = await c.wait_state(ids["sample_stripes.png"], {"completed"})
    poison = await c.wait_state(ids[POISON], {"failed"})
    assert gradient["attempts"] == 2 and stripes["attempts"] == 2
    assert poison["attempts"] == 2

    # the cancelled frame never re-entered the model: exactly one attempt
    # (the original batch) and no individual retry for it
    victim = await c.get(ids["sample_circles.png"])
    assert victim["status"] == "cancelled"
    assert victim["attempts"] == 1
    assert victim["result"] is None
    assert c.engine.retries_started == 3  # gradient, poison, stripes — not the cancelled one


async def test_cancel_during_retry_holds_slot_and_late_result_discarded(make_client):
    c = await make_client(inference_delay_ms=800)
    # sequential submits fix the retry order: poison first, clean second
    poison_id = (await c.submit(POISON)).json()["order_id"]
    clean_id = (await c.submit("sample_gradient.png")).json()["order_id"]

    # wait until the clean frame's individual retry is actually computing
    await c.wait_for(lambda: c.engine.retries_started >= 2, timeout=10.0)
    body = await c.get(clean_id)
    assert body["attempts"] == 2 and body["status"] == "processing"

    r = await c.cancel(clean_id)
    assert r.json()["status"] == "cancelled"  # marked immediately
    assert (await c.get(clean_id))["status"] == "cancelled"

    # its retry compute already started -> slot held until compute ends:
    # only 15 free slots remain although the order is cancelled
    for _ in range(15):
        r = await c.submit("sample_noise.png")
        assert r.status_code == 201, r.text
    r = await c.submit("sample_noise.png")
    assert r.status_code == 429, "in-flight retry of a cancelled order must hold its slot"

    # after the compute ends, the late result must not revive the order
    await asyncio.sleep(1.0)
    body = await c.get(clean_id)
    assert body["status"] == "cancelled"
    assert body["result"] is None
    assert body["attempts"] == 2

    poison = await c.wait_state(poison_id, {"failed"})
    assert "overexposed" in poison["error"]


async def test_frame_expiring_while_awaiting_retry_is_not_retried(make_client):
    c = await make_client(inference_delay_ms=700)
    # poison first so its retry runs while the clean frame's deadline passes
    poison_id = (await c.submit(POISON)).json()["order_id"]
    short_id = (await c.submit("sample_gradient.png", deadline_ms=1100)).json()["order_id"]

    poison = await c.wait_state(poison_id, {"failed"})
    assert poison["attempts"] == 2

    # the clean frame expired while the poison retry was computing:
    # terminal state, and its pending retry never started
    body = await c.get(short_id)
    assert body["status"] == "expired"
    assert body["attempts"] == 1  # batch attempt only
    assert body["result"] is None
    assert c.engine.retries_started == 1  # only the poison frame was retried
    assert (await c.stats())["capacity_used"] == 0
