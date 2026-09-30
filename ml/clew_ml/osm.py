"""
python -m clew_ml.osm — fetch drivable OSM roads around every processed segment
(via the public Overpass API) into data/osm/*.json, for map-matched evaluation.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

from .data import Drive, load_processed, to_en

OSM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "osm")
OVERPASS = "https://overpass-api.de/api/interpreter"
DRIVABLE = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street"


def cache_path(d: Drive) -> str:
    return os.path.join(OSM_DIR, d.name.replace("/", "__").replace(" ", "_") + ".json")


def _bbox(d: Drive, margin_m: float = 300) -> tuple[float, float, float, float]:
    lat0, lon0 = d.origin
    R = 6371008.8
    e, n = d.truth_en[:, 0], d.truth_en[:, 1]
    lat = lat0 + np.degrees(n / R)
    lon = lon0 + np.degrees(e / (R * np.cos(np.radians(lat0))))
    dl = np.degrees(margin_m / R)
    return lat.min() - dl, lon.min() - dl / np.cos(np.radians(lat0)), lat.max() + dl, lon.max() + dl / np.cos(np.radians(lat0))


def fetch(d: Drive) -> None:
    s, w, n, e = _bbox(d)
    q = f'[out:json][timeout:120];way["highway"~"^({DRIVABLE})$"]({s},{w},{n},{e});out geom;'
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "clew-sih/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    lines = [[[p["lat"], p["lon"]] for p in el["geometry"]] for el in data.get("elements", []) if "geometry" in el]
    os.makedirs(OSM_DIR, exist_ok=True)
    with open(cache_path(d), "w") as f:
        json.dump({"origin": d.origin, "lines_latlon": lines}, f)


def load_roads(d: Drive) -> list[np.ndarray] | None:
    p = cache_path(d)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        data = json.load(f)
    lat0, lon0 = d.origin
    return [to_en(np.array(l)[:, 0], np.array(l)[:, 1], lat0, lon0) for l in data["lines_latlon"] if len(l) > 1]


def main():
    for d in load_processed():
        if os.path.exists(cache_path(d)):
            continue
        try:
            fetch(d)
            print(f"ok    {d.name}")
        except Exception as ex:  # noqa: BLE001
            print(f"fail  {d.name}: {ex}")
        time.sleep(2)  # be polite to the public Overpass server


if __name__ == "__main__":
    main()
