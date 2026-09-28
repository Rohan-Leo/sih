"""
Unscented Kalman Filter for a road vehicle, phone-IMU driven.

State  x = [E, N, ψ, v, b_ω, b_a]
  E, N   position, metres (local east / north)
  ψ      heading, rad, clockwise from north (compass sense)
  v      forward speed, m/s
  b_ω    gyro yaw-rate bias, rad/s
  b_a    longitudinal accelerometer bias, m/s²

Process (unicycle / non-holonomic: the vehicle moves only along its heading)
  ψ ← ψ + (ω − b_ω)·dt        v ← v + (a_long − b_a)·dt
  E ← E + v·sin ψ·dt          N ← N + v·cos ψ·dt

Measurements (each optional per step)
  GNSS position / speed / course
  NHC centripetal:  a_lat = v·(ω − b_ω)   (no sideways slip ⇒ speed observable in turns)
  ZUPT:             v = 0 while stationary
  Learned speed:    v = v̂ (SpeedNet), with the network's own variance
  Learned bias:     b_ω = −δ̂ (HeadingNet), with the network's own variance
"""
from __future__ import annotations

import numpy as np

N_X = 6
IDX_PSI = 2


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class UKF:
    def __init__(self, alpha: float = 0.5, beta: float = 2.0, kappa: float = 0.0):
        n = N_X
        lam = alpha**2 * (n + kappa) - n
        self.c = n + lam
        self.wm = np.full(2 * n + 1, 1 / (2 * self.c))
        self.wc = self.wm.copy()
        self.wm[0] = lam / self.c
        self.wc[0] = lam / self.c + (1 - alpha**2 + beta)
        self.x = np.zeros(n)
        self.P = np.eye(n)
        # process noise per second (scaled by dt)
        self.q = np.array([0.05, 0.05, 0.02, 1.0, 2e-4, 5e-3]) ** 2

    def init(self, E: float, N: float, psi: float, v: float):
        self.x = np.array([E, N, psi, v, 0.0, 0.0])
        self.P = np.diag([5.0, 5.0, np.radians(10), 1.0, 0.02, 0.2]) ** 2

    def copy(self) -> "UKF":
        u = UKF.__new__(UKF)
        u.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return u

    # ── unscented machinery ──
    def _sigmas(self) -> np.ndarray:
        try:
            S = np.linalg.cholesky(self.c * self.P)
        except np.linalg.LinAlgError:
            self.P = 0.5 * (self.P + self.P.T) + 1e-6 * np.eye(N_X)
            S = np.linalg.cholesky(self.c * self.P + 1e-6 * np.eye(N_X))
        X = np.empty((2 * N_X + 1, N_X))
        X[0] = self.x
        X[1 : N_X + 1] = self.x + S.T
        X[N_X + 1 :] = self.x - S.T
        return X

    def _mean_x(self, X: np.ndarray) -> np.ndarray:
        m = self.wm @ X
        m[IDX_PSI] = np.arctan2(self.wm @ np.sin(X[:, IDX_PSI]), self.wm @ np.cos(X[:, IDX_PSI]))
        return m

    def predict(self, omega: float, a_long: float, dt: float):
        X = self._sigmas()
        psi = X[:, 2] + (omega - X[:, 4]) * dt
        v = np.maximum(0.0, X[:, 3] + (a_long - X[:, 5]) * dt)
        vm = 0.5 * (X[:, 3] + v)
        pm = X[:, 2] + 0.5 * (psi - X[:, 2])
        Y = X.copy()
        Y[:, 0] = X[:, 0] + vm * np.sin(pm) * dt
        Y[:, 1] = X[:, 1] + vm * np.cos(pm) * dt
        Y[:, 2] = wrap(psi)
        Y[:, 3] = v
        self.x = self._mean_x(Y)
        d = Y - self.x
        d[:, 2] = wrap(d[:, 2])
        self.P = (self.wc[:, None] * d).T @ d + np.diag(self.q * dt)

    def update(self, z: np.ndarray, h, R: np.ndarray, angle_rows: tuple[int, ...] = (), gate: float | None = None) -> bool:
        """Generic unscented update. h maps a sigma-point matrix (2n+1, N_X) → (2n+1, m)."""
        X = self._sigmas()
        Z = h(X)
        zm = self.wm @ Z
        for r in angle_rows:
            zm[r] = np.arctan2(self.wm @ np.sin(Z[:, r]), self.wm @ np.cos(Z[:, r]))
        dz = Z - zm
        for r in angle_rows:
            dz[:, r] = wrap(dz[:, r])
        dx = X - self.x
        dx[:, 2] = wrap(dx[:, 2])
        S = (self.wc[:, None] * dz).T @ dz + R
        C = (self.wc[:, None] * dx).T @ dz
        y = z - zm
        for r in angle_rows:
            y[r] = wrap(y[r])
        Si = np.linalg.inv(S)
        if gate is not None and float(y @ Si @ y) > gate:
            return False  # innovation gating: reject inconsistent measurements (e.g. GNSS spoofing/multipath)
        K = C @ Si
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        self.x[3] = max(self.x[3], 0.0)
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    # ── measurement helpers ──
    def gnss(self, E: float, N: float, acc: float, speed: float | None, course_deg: float | None):
        self.update(np.array([E, N]), lambda X: X[:, :2], np.eye(2) * max(acc, 2.0) ** 2, gate=25.0)
        if speed is not None:
            self.update(np.array([speed]), lambda X: X[:, 3:4], np.array([[0.3**2]]))
        if course_deg is not None and speed is not None and speed > 3:
            self.update(np.array([np.radians(course_deg)]), lambda X: X[:, 2:3], np.array([[np.radians(3) ** 2]]), angle_rows=(0,))

    def nhc_centripetal(self, a_lat: float, omega: float, sigma: float = 0.6):
        self.update(np.array([a_lat]), lambda X: (X[:, 3] * (omega - X[:, 4]))[:, None], np.array([[sigma**2]]))

    def zupt(self):
        self.update(np.array([0.0]), lambda X: X[:, 3:4], np.array([[0.05**2]]))

    def gyro_bias_obs(self, b: float, var: float):
        """Learned gyro yaw-rate bias (HeadingNet) as a direct measurement of b_ω."""
        self.update(np.array([b]), lambda X: X[:, 4:5], np.array([[var]]))

    def speed_obs(self, v: float, var: float):
        self.update(np.array([v]), lambda X: X[:, 3:4], np.array([[var]]))
