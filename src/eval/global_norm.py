"""
global_norm.py — global percentile normalization shared by training
preprocessing (src/data/preprocess_2.py) and Julie's eval load (src/eval/pipeline.py).

The scaling constants live in ONE file, configs/global_scaling.json, written by
fit_global_scaling.py. Both training and eval read it, so they cannot drift.
"""
from __future__ import annotations
import json
import numpy as np


def load_scaling(path: str) -> tuple[float, float]:
    d = json.load(open(path))
    return float(d["lo"]), float(d["hi"])


def global_normalize(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Clip to the FIXED [lo, hi] (from the reference set) and scale to [0, 1].
    Unlike per-sample min-max, lo/hi are constants across ALL craters, so
    absolute contrast — the fresh-vs-eroded cue — is preserved.
    """
    x = img.astype(np.float32)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo)
