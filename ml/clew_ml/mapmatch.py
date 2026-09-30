"""
Online HMM map matching (Newson & Krumm, 2009), adapted for dead reckoning.

Hidden states: candidate positions on nearby road segments.
Emission:      Gaussian in the distance from the estimate to the road,
               with σ taken from the filter's own position uncertainty.
Transition:    exp(−|d_network − d_travelled| / β), where d_travelled is the
               distance the filter says we moved (not straight-line distance,
               which is what makes this work on curvy roads during outages).
Decoding:      forward Viterbi, one step per second; the current best state
               is fed back into the UKF as a position measurement.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    seg: int
    t: float  # position along segment, 0..1
    point: np.ndarray  # (2,)
    dist: float  # distance from the estimate


class RoadGraph:
    """Undirected road network made of straight segments in local ENU metres."""

    def __init__(self, polylines: list[np.ndarray], snap: float = 0.5):
        nodes: dict[tuple[int, int], int] = {}
        self.xy: list[np.ndarray] = []

        def node(p: np.ndarray) -> int:
            key = (int(round(p[0] / snap)), int(round(p[1] / snap)))
            if key not in nodes:
                nodes[key] = len(self.xy)
                self.xy.append(np.asarray(p, float))
            return nodes[key]

        segs = []
        for line in polylines:
            for a, b in zip(line[:-1], line[1:]):
                u, v = node(a), node(b)
                if u != v:
                    segs.append((u, v))
        self.seg = np.array(segs, int)
        self.P = np.array(self.xy)
        self.A = self.P[self.seg[:, 0]]
        self.B = self.P[self.seg[:, 1]]
        self.len = np.linalg.norm(self.B - self.A, axis=1)
        self.adj: list[list[tuple[int, float]]] = [[] for _ in self.xy]
        for (u, v), L in zip(self.seg, self.len):
            self.adj[u].append((v, L))
            self.adj[v].append((u, L))

    def candidates(self, p: np.ndarray, radius: float, k: int = 8) -> list[Candidate]:
        d = self.B - self.A
        L2 = np.maximum((d**2).sum(1), 1e-9)
        t = np.clip(((p - self.A) * d).sum(1) / L2, 0, 1)
        q = self.A + t[:, None] * d
        dist = np.linalg.norm(q - p, axis=1)
        idx = np.flatnonzero(dist < radius)
        idx = idx[np.argsort(dist[idx])][:k]
        return [Candidate(int(i), float(t[i]), q[i], float(dist[i])) for i in idx]

    def _dijkstra(self, sources: dict[int, float], limit: float) -> dict[int, float]:
        best = dict(sources)
        pq = [(c, n) for n, c in sources.items()]
        heapq.heapify(pq)
        while pq:
            c, n = heapq.heappop(pq)
            if c > best.get(n, np.inf) or c > limit:
                continue
            for m, w in self.adj[n]:
                nc = c + w
                if nc < best.get(m, np.inf) and nc <= limit:
                    best[m] = nc
                    heapq.heappush(pq, (nc, m))
        return best

    def route_distance(self, a: Candidate, b: Candidate, limit: float) -> float:
        if a.seg == b.seg:
            return abs(a.t - b.t) * self.len[a.seg]
        u, v = self.seg[a.seg]
        src = {int(u): a.t * self.len[a.seg], int(v): (1 - a.t) * self.len[a.seg]}
        dist = self._dijkstra(src, limit)
        x, y = self.seg[b.seg]
        return min(
            dist.get(int(x), np.inf) + b.t * self.len[b.seg],
            dist.get(int(y), np.inf) + (1 - b.t) * self.len[b.seg],
        )


class OnlineHMM:
    def __init__(self, graph: RoadGraph, beta: float = 10.0, min_sigma: float = 5.0):
        self.g = graph
        self.beta = beta
        self.min_sigma = min_sigma
        self.prev: list[Candidate] = []
        self.logp = np.zeros(0)

    def reset(self):
        self.prev, self.logp = [], np.zeros(0)

    def step(self, estimate: np.ndarray, sigma: float, travelled: float) -> Candidate | None:
        """Advance one step. `travelled` = distance moved since the last step (from the filter)."""
        s = max(sigma, self.min_sigma)
        cands = self.g.candidates(estimate, radius=max(3 * s, 30.0))
        if not cands:
            self.reset()
            return None
        emit = np.array([-0.5 * (c.dist / s) ** 2 for c in cands])
        if not self.prev:
            self.prev, self.logp = cands, emit
            return cands[int(np.argmax(emit))]
        limit = 2 * travelled + 3 * s + 50
        trans = np.full((len(self.prev), len(cands)), -np.inf)
        for i, a in enumerate(self.prev):
            for j, b in enumerate(cands):
                d = self.g.route_distance(a, b, limit)
                if np.isfinite(d):
                    trans[i, j] = -abs(d - travelled) / self.beta
        score = self.logp[:, None] + trans
        best = score.max(0)
        if not np.isfinite(best).any():
            # no connected path (e.g. graph gap) — restart the chain here
            self.prev, self.logp = cands, emit
            return cands[int(np.argmax(emit))]
        self.logp = best + emit
        self.logp -= self.logp.max()
        self.prev = cands
        return cands[int(np.argmax(self.logp))]
