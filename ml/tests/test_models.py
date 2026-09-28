"""Unit tests for the model plumbing that doesn't need the dataset."""
import numpy as np
import torch

from clew_ml.drift import FEATURES as DRIFT_FEATURES, OutageTracker, pinball, quantiles
from clew_ml.integrity import FEATURES, FixChecker
from clew_ml.nets import MLP, TCN


def drive_straight(ck: FixChecker, v: float, seconds: int):
    for _ in range(seconds * 10):
        ck.imu_step(0.0, v)


def test_fixchecker_flags_a_jump_but_not_a_consistent_fix():
    ck = FixChecker()
    ck.after_fix(0.0, 0.0, 0.0, 4.0, 10.0, 0.0, 10.0, trusted=True)  # heading north at 10 m/s
    drive_straight(ck, 10.0, 1)
    ok = ck.features(0.0, 10.0, 1.0, 4.0, 10.0, 0.0, 10.0)
    jump = ck.features(60.0, 10.0, 1.0, 4.0, 10.0, 0.0, 10.0)
    assert ok is not None and jump is not None and len(ok) == len(FEATURES)
    i = FEATURES.index("log_innov_m")
    assert ok[i] < np.log1p(1.0)
    assert jump[i] > np.log1p(50.0)


def test_fixchecker_needs_an_anchor():
    assert FixChecker().features(0, 0, 0, 4, 0, 0, None) is None


def test_outage_tracker_features():
    tr = OutageTracker(0.0)
    for _ in range(100):
        tr.step(10.0, 0.05)
    f = tr.features(0.5, 12.0)
    assert len(f) == len(DRIFT_FEATURES)
    assert abs(f[DRIFT_FEATURES.index("log_dist")] - np.log1p(100.0)) < 1e-4
    assert abs(f[DRIFT_FEATURES.index("turn_amount")] - 0.5) < 1e-6
    assert abs(f[DRIFT_FEATURES.index("heading_change")] - 0.5) < 1e-6


def test_quantiles_are_monotone_and_pinball_is_finite():
    o = torch.randn(64, 3)
    q = quantiles(o)
    assert torch.all(q[:, 1] >= q[:, 0]) and torch.all(q[:, 2] >= q[:, 1])
    assert torch.isfinite(pinball(o, torch.rand(64)))


def test_nets_shapes():
    assert TCN(8, 5)(torch.randn(2, 8, 100)).shape == (2, 5)
    assert MLP(9, 1)(torch.randn(3, 9)).shape == (3, 1)
