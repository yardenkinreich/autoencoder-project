"""
load_holdout_dat must take the crop size from an eval set's
evalset_manifest.json (Kaguya sets are 224 px) and refuse anything that
would silently scramble crops.
Run with: PYTHONPATH=src:$(pwd) python -m pytest src/test/test_holdout_dat.py
"""
import json

import numpy as np
import pytest

from src.eval.holdout import load_holdout_dat


def _write(tmp_path, n, size, manifest=True):
    a = np.arange(n * size * size, dtype=np.float32).reshape(n, 1, size, size)
    p = tmp_path / "craters.dat"
    a.tofile(p)
    if manifest:
        json.dump({"size": size, "channels": 1}, open(tmp_path / "evalset_manifest.json", "w"))
    return str(p), a


def test_size_from_manifest(tmp_path):
    p, a = _write(tmp_path, 3, 224)
    d = load_holdout_dat(p)
    assert d.shape == (3, 1, 224, 224) and np.array_equal(d[2], a[2])


def test_explicit_size_contradicting_manifest_raises(tmp_path):
    p, _ = _write(tmp_path, 3, 224)
    with pytest.raises(ValueError):
        load_holdout_dat(p, size=128)


def test_legacy_default_128_and_ragged_check(tmp_path):
    p, _ = _write(tmp_path, 4, 128, manifest=False)
    assert load_holdout_dat(p).shape == (4, 1, 128, 128)
    p2, _ = _write(tmp_path, 2, 224, manifest=False)      # overwrite, no manifest
    with pytest.raises(ValueError):
        load_holdout_dat(p2)                               # 2*224^2 floats != k*128^2
