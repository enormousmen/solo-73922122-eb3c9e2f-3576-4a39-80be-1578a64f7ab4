"""Generate the fixed-weight TorchScript model(s) shipped with the service.

Weights are initialized with a fixed seed, so the produced model.pt is
byte-stable in behavior (same architecture + same seed => same weights).
The script prints a SHA-256 fingerprint of the weights for verification.

Two variants are available:

* ``normal``  (default): plain 10-class CNN classifier.
* ``anomaly``: same architecture and SAME weights, but ``forward`` raises
  ``RuntimeError`` when any frame in the batch is "overexposed" (normalized
  per-frame mean above ``--threshold``). A single poison frame therefore
  fails the whole batch, which is exactly the fault-frame scenario the
  engine's bounded per-frame retry/localization handles. Switch at runtime
  with ``MODEL_PATH=app/model_anomaly.pt``.

Usage:
    python scripts/make_model.py [--out app/model.pt]
    python scripts/make_model.py --variant anomaly --out app/model_anomaly.pt
"""
from __future__ import annotations

import argparse
import hashlib
import io

import torch
from torch import nn

SEED = 20240521
NUM_CLASSES = 10
ANOMALY_THRESHOLD = 0.8  # normalized mean in [-1, 1]; 0.8 ~= raw mean >= 230/255


class TinyClassifier(nn.Module):
    """Small CNN for 64x64 RGB images -> 10-class logits."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AnomalyClassifier(TinyClassifier):
    """Demo classifier that raises on batches containing an overexposed frame.

    The check runs BEFORE the conv layers, so one poison frame fails the
    whole batch — and also fails when retried alone, while clean frames
    retried alone succeed. Weights are identical to TinyClassifier (same
    constructor, same seed), so clean-frame scores match the normal model.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, threshold: float = ANOMALY_THRESHOLD) -> None:
        super().__init__(num_classes)
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.mean(dim=(1, 2, 3))  # per-frame normalized mean
        if bool((means > self.threshold).any()):
            raise RuntimeError("anomaly: overexposed frame in batch")
        # TinyClassifier.forward inlined: TorchScript super() support is limited.
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def weights_fingerprint(model: nn.Module) -> str:
    buf = io.BytesIO()
    for p in model.state_dict().values():
        buf.write(p.detach().numpy().tobytes())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="app/model.pt")
    parser.add_argument("--variant", choices=["normal", "anomaly"], default="normal")
    parser.add_argument("--threshold", type=float, default=ANOMALY_THRESHOLD,
                        help="anomaly variant: normalized per-frame mean above this fails the batch")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    if args.variant == "anomaly":
        model = AnomalyClassifier(threshold=args.threshold)
    else:
        model = TinyClassifier()
    model.eval()

    scripted = torch.jit.script(model)
    scripted.save(args.out)

    fp = weights_fingerprint(model)
    print(f"saved TorchScript model -> {args.out}")
    print(f"variant={args.variant}  seed={SEED}  classes={NUM_CLASSES}  weights_sha256={fp}")

    # sanity: reload and run one forward pass on a clean (dark) frame
    loaded = torch.jit.load(args.out, map_location="cpu")
    loaded.eval()
    with torch.inference_mode():
        out = loaded(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, NUM_CLASSES), out.shape
    print("reload + forward OK, output shape", tuple(out.shape))

    if args.variant == "anomaly":
        # sanity: an overexposed frame must trip the anomaly check
        poison = torch.full((1, 3, 64, 64), 1.0)
        try:
            loaded(poison)
        except Exception as exc:  # noqa: BLE001 - torch.jit.Error on raise
            print("anomaly check OK, overexposed frame raises:", type(exc).__name__)
        else:
            raise SystemExit("anomaly model did NOT raise on an overexposed frame")


if __name__ == "__main__":
    main()
