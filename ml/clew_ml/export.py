"""
python -m clew_ml.export — ship SpeedNet for on-device inference.

Writes to artifacts/:
  speednet.onnx     portable graph (ONNX Runtime Mobile / web)
  speednet.tflite   TensorFlow Lite / LiteRT, if litert-torch is installed
  speednet.weights.json  raw weights + metadata (for a dependency-free JS/Kotlin port)
Input: float32 (1, C, WINDOW) raw features (normalisation is inside the graph).
Output: (1, 3) = [speed m/s, log-variance, stationary logit].
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .features import CHANNELS, WINDOW
from .model import SpeedNet

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def main():
    m = SpeedNet()
    m.load_state_dict(torch.load(os.path.join(ART, "speednet.pt"), weights_only=True))
    m.eval()
    x = torch.randn(1, len(CHANNELS), WINDOW)
    ref = m(x).detach().numpy()

    onnx_path = os.path.join(ART, "speednet.onnx")
    torch.onnx.export(m, x, onnx_path, input_names=["imu"], output_names=["out"], dynamo=False, opset_version=17)
    print(f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1024:.0f} KB)")

    try:
        try:
            import litert_torch as converter  # type: ignore  # current name
        except ImportError:
            import ai_edge_torch as converter  # type: ignore  # older name

        edge = converter.convert(m, (x,))
        tfl = os.path.join(ART, "speednet.tflite")
        edge.export(tfl)
        got = edge(x)
        print(f"wrote {tfl} ({os.path.getsize(tfl) / 1024:.0f} KB), max |Δ| vs PyTorch {np.abs(np.asarray(got) - ref).max():.2e}")
    except (ImportError, AttributeError):
        print("litert-torch not installed — skipped TFLite (pip install litert-torch, then re-run)")

    with open(os.path.join(ART, "speednet.weights.json"), "w") as f:
        json.dump(
            {
                "channels": CHANNELS,
                "window": WINDOW,
                "hz": 10,
                "outputs": ["speed_mps", "log_variance", "stationary_logit"],
                "state_dict": {k: v.tolist() for k, v in m.state_dict().items()},
            },
            f,
        )
    print("wrote speednet.weights.json")


if __name__ == "__main__":
    main()
