"""
IO-VNBD loading.

Each drive folder holds a phone log (S-*.csv) and a vehicle log (V-*.csv),
both at 10 Hz. Despite the "synchronised" folder name, the two streams are
offset by several seconds, so every drive is re-aligned by cross-correlating
phone GNSS speed against vehicle speed.

Per drive we keep:
  phone:  accelerometer, gravity, gyroscope (phone frame)
  gnss:   phone-like 1 Hz GNSS, synthesised from the vehicle receiver with
          noise. The phone's own GNSS log is only ~0.1 Hz in most drives, too
          sparse to stand in for a live receiver; it is kept as phone_gnss_*.
  truth:  vehicle GNSS position (ENU metres), vehicle speed from the
          wheel-speed sensors (the training label), vehicle heading
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HZ = 10
DT = 1.0 / HZ
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


@dataclass
class Drive:
    name: str
    # phone sensors, shape (T, 3)
    acc: np.ndarray
    grav: np.ndarray
    gyro: np.ndarray
    # phone GNSS
    gnss_en: np.ndarray  # (T, 2) ENU metres
    gnss_speed: np.ndarray  # m/s
    gnss_acc: np.ndarray  # reported 1-sigma accuracy, m
    gnss_course: np.ndarray  # degrees
    gnss_fresh: np.ndarray  # bool: a new fix arrived at this sample
    # the phone's own GNSS log (kept for reference; ~0.1 Hz in most IO-VNBD drives)
    phone_gnss_en: np.ndarray
    phone_gnss_speed: np.ndarray
    phone_gnss_fresh: np.ndarray
    # ground truth (vehicle)
    truth_en: np.ndarray  # (T, 2)
    truth_speed: np.ndarray  # m/s, wheel-speed derived
    truth_heading: np.ndarray  # degrees
    origin: tuple[float, float]  # lat, lon of ENU origin
    lag: int  # samples the phone GNSS stream was shifted by
    imu_lag: int  # extra shift applied to the phone IMU (logged ~4-5 s late)
    imu_corr: float  # |corr| of aligned phone gyro with vehicle heading rate

    @property
    def T(self) -> int:
        return len(self.acc)


def to_en(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> np.ndarray:
    """Local equirectangular east/north in metres (fine at city scale)."""
    R = 6371008.8
    e = np.radians(lon - lon0) * R * np.cos(np.radians(lat0))
    n = np.radians(lat - lat0) * R
    return np.stack([e, n], axis=1)


def _num(df: pd.DataFrame, i: int) -> np.ndarray:
    return pd.to_numeric(df.iloc[:, i], errors="coerce").to_numpy(dtype=float)


def _best_lag(phone_speed: np.ndarray, veh_speed: np.ndarray, max_lag: int = 6000) -> tuple[int, float]:
    """Lag L (samples) such that phone[t + L] ≈ vehicle[t], via FFT cross-correlation."""
    a = np.nan_to_num(phone_speed - np.nanmean(phone_speed))
    b = np.nan_to_num(veh_speed - np.nanmean(veh_speed))
    n = len(a)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
    vals = np.r_[xc[: max_lag + 1], xc[size - max_lag :]]
    L = int(lags[int(np.argmax(vals))])
    # exact Pearson correlation at the chosen lag
    x, y = (a[L:], b[: n - L]) if L >= 0 else (a[:L], b[-L:])
    if len(x) < 600 or x.std() == 0 or y.std() == 0:
        return L, -1.0
    return L, float(np.corrcoef(x, y)[0, 1])


def _phone_like_gnss(en: np.ndarray, speed: np.ndarray, heading: np.ndarray, seed: int) -> dict:
    """1 Hz fixes with ~3 m noise (slowly varying, like real multipath), from the vehicle receiver."""
    rng = np.random.default_rng(seed)
    T = len(en)
    fresh = np.zeros(T, bool)
    fresh[:: HZ] = True
    # random-walk-ish correlated error rather than white noise
    err = np.zeros((T, 2))
    e = rng.normal(0, 2.0, 2)
    for i in range(0, T, HZ):
        e = 0.8 * e + rng.normal(0, 1.2, 2)
        err[i : i + HZ] = e
    held = np.repeat(np.arange(0, T, HZ), HZ)[:T]
    return dict(
        gnss_en=(en + err)[held],
        gnss_speed=np.maximum(0, speed + rng.normal(0, 0.2, T))[held],
        gnss_acc=np.full(T, 4.0),
        gnss_course=((heading + rng.normal(0, 1.0, T)) % 360)[held],
        gnss_fresh=fresh,
    )


def _imu_lag(gyro: np.ndarray, heading: np.ndarray, speed: np.ndarray, max_lag: int = 300) -> tuple[int, float]:
    """Shift L so gyro[t + L] matches vehicle heading rate[t]. Returns (L, |corr|)."""
    k = np.ones(10) / 10
    hr = np.gradient(np.unwrap(np.radians(heading))) / DT
    hr[speed < 3] = 0
    hr = np.convolve(hr, k, "same")
    best = (0, 0.0)
    n = len(hr)
    for j in range(3):
        g = np.convolve(gyro[:, j], k, "same")
        a, b = g - g.mean(), hr - hr.mean()
        size = 1 << int(np.ceil(np.log2(2 * n)))
        x = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
        lags = np.r_[np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]
        v = np.r_[x[: max_lag + 1], x[size - max_lag :]] / (n * a.std() * b.std() + 1e-12)
        i = int(np.argmax(np.abs(v)))
        if abs(v[i]) > best[1]:
            best = (int(lags[i]), float(abs(v[i])))
    return best


MIN_IMU_CORR = 0.6

CHUNK = 12000  # 20 min: phone/vehicle offset is re-estimated per chunk


def _chunk_lag(p: np.ndarray, v: np.ndarray, v0: int, v1: int, max_lag: int = 6000) -> tuple[int, float]:
    """Best L so that phone[v0+L : v1+L] matches vehicle[v0:v1] (normalised cross-correlation)."""
    tpl = np.nan_to_num(v[v0:v1])
    m = len(tpl)
    tpl = tpl - tpl.mean()
    if tpl.std() < 0.5:
        return 0, -1.0
    lo, hi = max(0, v0 - max_lag), min(len(p), v1 + max_lag)
    seg = np.nan_to_num(p[lo:hi])
    if len(seg) < m:
        return 0, -1.0
    size = 1 << int(np.ceil(np.log2(len(seg) + m)))
    raw = np.fft.irfft(np.fft.rfft(seg, size) * np.conj(np.fft.rfft(tpl, size)), size)[: len(seg) - m + 1]
    cs, cs2 = np.r_[0, np.cumsum(seg)], np.r_[0, np.cumsum(seg**2)]
    mu = (cs[m:] - cs[:-m]) / m
    sd = np.sqrt(np.maximum((cs2[m:] - cs2[:-m]) / m - mu**2, 1e-9))
    corr = raw / (m * sd * tpl.std())
    k = int(np.argmax(corr))
    return lo + k - v0, float(corr[k])


def load_drive(folder: str) -> list[Drive]:
    """All usable, individually aligned segments of one recording."""
    s_files = glob.glob(os.path.join(folder, "S-*.csv"))
    v_files = glob.glob(os.path.join(folder, "V-*.csv"))
    if not s_files or not v_files:
        return []
    S = pd.read_csv(s_files[0], encoding="latin1", low_memory=False)
    V = pd.read_csv(v_files[0], encoding="latin1", low_memory=False)
    P = {i: _num(S, i) for i in (0, 1, 3, 4, 5) + tuple(range(9, 18))}
    Q = {i: _num(V, i) for i in (2, 3, 5, 15)}
    p_speed, v_speed = P[3] / 3.6, Q[15] / 3.6
    name = os.path.relpath(folder, RAW)
    out: list[Drive] = []
    for ci, v0 in enumerate(range(0, len(v_speed), CHUNK)):
        v1 = min(len(v_speed), v0 + CHUNK)
        if v1 - v0 < 3000:
            continue
        lag, corr = _chunk_lag(p_speed, v_speed, v0, v1)
        if corr < 0.9:
            continue
        vi = np.arange(v0, v1)
        pi = vi + lag
        keep = (pi >= 0) & (pi < len(p_speed))
        vi, pi = vi[keep], pi[keep]
        ok = np.isfinite(P[0][pi]) & np.isfinite(Q[2][vi]) & (np.abs(Q[2][vi]) > 1) & (np.abs(P[0][pi]) > 1)
        for i in range(9, 18):
            ok &= np.isfinite(P[i][pi])
        idx = np.flatnonzero(ok)
        if len(idx) < 3000:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(idx) - 1]
        k = int(np.argmax(ends - starts))
        sel = idx[starts[k] : ends[k] + 1]
        if len(sel) < 3000:
            continue
        vi, pi = vi[sel], pi[sel]
        # the IMU stream has its own offset: find it against vehicle turning
        gyro_raw = np.stack([P[i] for i in (15, 16, 17)], 1)
        probe = np.clip(pi, 0, len(p_speed) - 1)
        il, ic = _imu_lag(gyro_raw[probe], Q[5][vi], np.nan_to_num(v_speed[vi]))
        qi = pi + il
        keep = (qi >= 0) & (qi < len(p_speed))
        vi, pi, qi = vi[keep], pi[keep], qi[keep]
        if ic < MIN_IMU_CORR or len(vi) < 3000 or not all(np.isfinite(P[i][qi]).all() for i in range(9, 18)):
            continue
        p_lat, p_lon, v_lat, v_lon = P[0][pi], P[1][pi], Q[2][vi], Q[3][vi]
        lat0, lon0 = float(v_lat[0]), float(v_lon[0])
        ps = P[3][pi] / 3.6
        # some logs store phone speed in m/s rather than km/h
        vs = np.nan_to_num(v_speed[vi])
        scale = float(np.dot(ps, vs) / max(np.dot(ps, ps), 1e-9))
        if 2.8 < scale < 4.6:
            ps = ps * 3.6
        fresh = np.r_[True, (np.diff(p_lat) != 0) | (np.diff(p_lon) != 0) | (np.diff(ps) != 0)]
        out.append(
            Drive(
                name=f"{name}#{ci}",
                acc=np.stack([P[i][qi] for i in (9, 10, 11)], 1),
                grav=np.stack([P[i][qi] for i in (12, 13, 14)], 1),
                gyro=np.stack([P[i][qi] for i in (15, 16, 17)], 1),
                **_phone_like_gnss(to_en(v_lat, v_lon, lat0, lon0), vs, Q[5][vi], seed=hash(name) % 2**32 + ci),
                phone_gnss_en=to_en(p_lat, p_lon, lat0, lon0),
                phone_gnss_speed=ps,
                phone_gnss_fresh=fresh,
                truth_en=to_en(v_lat, v_lon, lat0, lon0),
                truth_speed=np.nan_to_num(v_speed[vi]),
                truth_heading=np.nan_to_num(Q[5][vi]),
                origin=(lat0, lon0),
                lag=lag,
                imu_lag=il,
                imu_corr=ic,
            )
        )
    return out


def drive_folders() -> list[str]:
    return sorted({os.path.dirname(p) for p in glob.glob(os.path.join(RAW, "**", "S-*.csv"), recursive=True)})


def split_of(name: str) -> str:
    """Deterministic split by *recording* (all chunks of a drive share a split)."""
    import hashlib

    h = int(hashlib.md5(name.split("#")[0].encode()).hexdigest(), 16) % 100
    return "test" if h < 20 else "val" if h < 35 else "train"


def save(d: Drive) -> str:
    os.makedirs(PROCESSED, exist_ok=True)
    path = os.path.join(PROCESSED, d.name.replace("/", "__").replace(" ", "_") + ".npz")
    np.savez_compressed(path, **{k: v for k, v in d.__dict__.items()})
    return path


def load_processed() -> list[Drive]:
    out = []
    for p in sorted(glob.glob(os.path.join(PROCESSED, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        kw = {k: z[k] for k in z.files}
        kw["name"] = str(kw["name"])
        kw["origin"] = tuple(kw["origin"])
        kw["lag"] = int(kw["lag"])
        kw["imu_lag"] = int(kw["imu_lag"])
        kw["imu_corr"] = float(kw["imu_corr"])
        out.append(Drive(**kw))
    return out
