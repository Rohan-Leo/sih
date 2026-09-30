"""Map matching on a synthetic grid city with a biased dead-reckoned track."""
import numpy as np

from clew_ml.mapmatch import OnlineHMM, RoadGraph


def grid(n=8, block=120.0):
    lines = []
    for i in range(n):
        lines.append(np.array([[0, i * block], [(n - 1) * block, i * block]]))
        lines.append(np.array([[i * block, 0], [i * block, (n - 1) * block]]))
    # split long lines at intersections so the graph is connected
    out = []
    for L in lines:
        pts = np.linspace(L[0], L[1], n)
        out.append(pts)
    return out


def drive(block=120.0):
    """Truth: east 3 blocks, north 2, east 2, south 1 — at 10 m/s, sampled at 1 Hz."""
    legs = [(np.array([1, 0]), 3), (np.array([0, 1]), 2), (np.array([1, 0]), 2), (np.array([0, -1]), 1)]
    p = np.array([0.0, block])
    pts = [p.copy()]
    for d, nb in legs:
        for _ in range(int(nb * block / 10)):
            p = p + d * 10
            pts.append(p.copy())
    return np.array(pts)


def test_hmm_pulls_drifting_dr_back_onto_roads():
    g = RoadGraph(grid())
    truth = drive()
    # dead reckoning with 4 % speed error and a 1.5 °/min heading drift
    steps = np.diff(truth, axis=0)
    ang = 0.0
    dr_open = [truth[0]]  # dead reckoning alone
    est = truth[0].copy()  # dead reckoning with map-matched feedback (as the UKF does)
    matched = [truth[0]]
    hmm = OnlineHMM(g)
    hmm.step(est, sigma=10, travelled=0.0)
    for i, s in enumerate(steps):
        ang += np.radians(1.5) / 60
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        inc = 1.04 * R @ s
        dr_open.append(dr_open[-1] + inc)
        est = est + inc
        c = hmm.step(est, sigma=12, travelled=float(np.linalg.norm(inc)))
        if c is not None:
            est = c.point.copy()  # feedback: position measurement at the matched point
        matched.append(est.copy())
    dr = np.array(dr_open)
    matched = np.array(matched)
    raw_err = np.linalg.norm(dr - truth, axis=1)
    mm_err = np.linalg.norm(matched - truth, axis=1)
    # every matched point lies on a road
    on_road = [min(abs(q[0] % 120) , abs(q[1] % 120)) < 1e-6 or min(120 - q[0] % 120, 120 - q[1] % 120) < 1e-6 for q in matched]
    assert all(on_road)
    assert mm_err.mean() < 0.5 * raw_err.mean(), (mm_err.mean(), raw_err.mean())
    print(f"mean error: dead reckoning {raw_err.mean():.1f} m → map matched {mm_err.mean():.1f} m")


if __name__ == "__main__":
    test_hmm_pulls_drifting_dr_back_onto_roads()
