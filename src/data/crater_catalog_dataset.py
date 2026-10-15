"""
crater_catalog_dataset.py — the single, fixed entry point from a raw catalog
into a model.

A catalog (src/data/build_catalog.py) stores crops RAW and at NATIVE,
per-crater-variable size. That is deliberate - it keeps the percentile clip
out of the only stored copy and avoids upsampling the median 32x32 crop to
128 - but it means crops are NOT model-ready as stored. Everything needed to
make them model-ready lives here and nowhere else:

    which catalog  ->  root
    normalization  ->  norm  (see src/data/norm_stats.py for the candidates)
    resize         ->  resize (int or None)
    subsetting     ->  diam_min/diam_max, lat_abs_max  (index filters)

NORMALIZATION IS NOT A FREE CHOICE HERE
---------------------------------------
The constants live in norm_stats.json (fitted by src/data/norm_stats.py) and
the arithmetic lives in norm_stats.apply() - not here. That indirection is the
point: the dashboard's "what does each normalization do to the data" figures
and the tensors a model actually trains on go through the SAME function, so
they cannot drift apart.

`norm="global"` - the [0,1]-with-1st/99th-percentile-clip that every run in
this project's history used - is measurably NOT label-neutral on this data:
eta^2 = 0.52 between clipped-pixel-fraction and the degradation class, with
the freshest class losing 30.0% of its pixels to saturation against the most
degraded class's 0.07%. It is kept as a candidate for backward comparability,
not as a recommended default. See norm_stats.py's docstring.

WHY A POLICY FILE RATHER THAN A DATASET STRING
----------------------------------------------
dinov2's loaders._parse_dataset_str splits tokens on ":" then on "=" and
unpacks exactly two parts, and asserts the key is one of a fixed set - so an
inline option string can contain neither "=" nor ":". Rather than fight that
with positional encodings, the policy is a small JSON file. That is the
better artifact anyway: the exact normalization/resize used by a run becomes
a versioned file that can be recorded alongside the run, instead of a string
buried in a config line.

Resolution order (first wins): explicit kwargs > `extra` JSON path >
{root}/loader.json > module defaults.

THE ONE THING TO BE CAREFUL ABOUT
---------------------------------
norm="none" returns raw values - for morphology that is ~[0, 0.24], for a
sigma-100 high-pass it is ~[-0.05, +0.09]. Feeding either to a network that
expects [0,1] gives near-zero input variance and degenerate features, with
nothing raising. So `norm` has NO silent default when this class is
constructed from a dataset string: if no policy is found anywhere, __init__
raises rather than guessing. Constructed directly in Python, an explicit
norm=None (meaning raw) is honoured, because the artifact analysis genuinely
wants raw.
"""
from __future__ import annotations
import json
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.dinov2.dinov2.data.datasets.extended import ExtendedVisionDataset
from src.data import norm_stats as NS

# `norm` has no safe default, but it does have a defensible one: `robust`
# (median/MAD, no clip) is the only candidate that neither assumes a [0,1]
# range the data does not have nor discards the tails that carry this
# project's degradation cue. A policy file should still state it explicitly.
_DEFAULT_POLICY = {"norm": "robust", "resize": None, "resize_interp": "cubic", "stats_from": None,
                   "diam_min": None, "diam_max": None, "lat_abs_max": None,
                   "num_channels": 1}


class CraterCatalog(ExtendedVisionDataset):
    """Reads a build_catalog.py catalog. Same contract as
    dino_craters_dataset.Craters (get_image_data -> (C,H,W) float32 tensor,
    target -1, transforms applied in __getitem__) so it is a drop-in for
    dinov2's data pipeline."""

    def __init__(self, root: str, extra: str | None = None,
                 transform=None, target_transform=None, **overrides):
        super().__init__(root=root, transforms=None, transform=transform,
                         target_transform=target_transform)
        self.catalog_dir = root

        policy = dict(_DEFAULT_POLICY)
        found_policy = False
        cat_policy = os.path.join(root, "loader.json")
        if extra and os.path.exists(extra):
            policy.update(json.load(open(extra))); found_policy = True
        elif os.path.exists(cat_policy):
            policy.update(json.load(open(cat_policy))); found_policy = True
        # explicit kwargs win over any file
        explicit = {k: v for k, v in overrides.items() if k in _DEFAULT_POLICY}
        policy.update(explicit)
        if not found_policy and not explicit:
            raise ValueError(
                f"no loading policy for {root}: pass explicit kwargs, an "
                f"`extra=<policy.json>`, or place loader.json in the catalog. "
                f"Refusing to guess - a wrong normalization here feeds raw "
                f"values into a model silently.")

        self.norm = policy["norm"]
        self.resize = None if policy["resize"] in (None, "none") else int(policy["resize"])
        self.resize_interp = policy.get("resize_interp", "cubic")
        if self.resize_interp not in ("cubic", "auto"):
            raise ValueError(f"resize_interp={self.resize_interp!r}, expected cubic|auto")
        self.num_channels = int(policy["num_channels"])

        idx = pd.read_csv(os.path.join(root, "index.csv"), dtype={"crater_id": str})
        n_all = len(idx)

        # Drop crops whose stored statistics reveal nodata-sentinel pixels.
        # Catalogs built before build_catalog.py's magnitude check existed can
        # contain them: WAC_EMP_643's nodata = -3.4028e38 is FINITE, so the
        # old isfinite() guard passed it through. Five of emp_raw's 100,000
        # crops carry it, and feeding one to a model would put a -3.4e38 pixel
        # into a forward pass. Filtering here means such a catalog stays usable
        # without a multi-hour rebuild; new builds skip them at write time.
        sentinel = ((~np.isfinite(idx["mean"])) | (idx["mean"].abs() >= 1e30)
                    | (~np.isfinite(idx["std"])) | (idx["std"].abs() >= 1e30))
        n_sentinel = int(sentinel.sum())
        if n_sentinel:
            print(f"CraterCatalog({root}): dropping {n_sentinel} of {n_all} crops whose "
                  f"stored stats indicate nodata-sentinel pixels "
                  f"(|v|>=1e30). Crater ids: "
                  f"{', '.join(idx.loc[sentinel, 'crater_id'].astype(str)[:10])}")
            idx = idx[~sentinel]
        if policy["diam_min"] is not None:
            idx = idx[idx["diam_km"] >= float(policy["diam_min"])]
        if policy["diam_max"] is not None:
            idx = idx[idx["diam_km"] <= float(policy["diam_max"])]
        if policy["lat_abs_max"] is not None:
            idx = idx[idx["lat"].abs() <= float(policy["lat_abs_max"])]
        self.index = idx.reset_index(drop=True)
        if len(self.index) == 0:
            raise ValueError(f"index filters left 0 craters (of {n_all}) in {root}")

        # Normalization constants come from norm_stats.json, fitted on the
        # PARENT catalog's pixel sample. `stats_from` exists so a testset/
        # subdirectory can (and by default should) borrow its parent's
        # constants: fitting a test set's own percentiles would let the test
        # data set its own scale, which is a quiet train/test leak.
        self.norm = "none" if self.norm is None else self.norm
        if self.norm not in NS.NORMS:
            raise ValueError(f"unknown norm={self.norm!r}; known: {NS.NORMS}")
        self._stats = None
        self._stats_dir = None
        if self.norm != "none":
            sd = policy.get("stats_from") or root
            self._stats = NS.load(sd)
            self._stats_dir = sd

        # One flat memmap for the whole store; per-crop views are slices by
        # element offset. Far cheaper than an open+seek per __getitem__.
        crops = os.path.join(root, "crops.f32")
        self._mm = np.memmap(crops, dtype=np.float32, mode="r")
        self._offsets = (self.index["byte_offset"].to_numpy(dtype=np.int64) // 4)
        self._h = self.index["h"].to_numpy(dtype=np.int64)
        self._w = self.index["w"].to_numpy(dtype=np.int64)

        self.policy = {"norm": self.norm, "resize": self.resize,
                       "num_channels": self.num_channels,
                       "diam_min": policy["diam_min"], "diam_max": policy["diam_max"],
                       "lat_abs_max": policy["lat_abs_max"],
                       "n_craters": int(len(self.index)), "n_before_filter": int(n_all),
                       "n_dropped_sentinel": n_sentinel,
                       "catalog": root, "stats_from": self._stats_dir,
                       "norm_constants": (None if self._stats is None
                                          else self._stats.get(self.norm))}

    # ── the transformation chain, in one place ──────────────────────────────

    def _raw(self, i: int) -> np.ndarray:
        o, h, w = int(self._offsets[i]), int(self._h[i]), int(self._w[i])
        return np.asarray(self._mm[o:o + h * w]).reshape(h, w)

    def _normalize(self, a: np.ndarray) -> np.ndarray:
        if self.norm == "none":
            return a
        return NS.apply(a, self.norm, self._stats)

    def _resize(self, a: np.ndarray) -> np.ndarray:
        if self.resize is None:
            return a
        import cv2
        # "auto": area-average when SHRINKING (native-resolution Kaguya crops
        # are up to ~3,500 px; cubic would alias), cubic when enlarging.
        # "cubic" (default) keeps every existing WAC eval set reproducible.
        interp = cv2.INTER_CUBIC
        if self.resize_interp == "auto" and max(a.shape) > self.resize:
            interp = cv2.INTER_AREA
        return cv2.resize(a, (self.resize, self.resize), interpolation=interp)

    def get_image_data(self, index: int) -> torch.Tensor:
        # RESIZE BEFORE NORMALIZE, deliberately. The affine part of
        # normalization commutes with interpolation (both are linear), but the
        # CLIP does not: clipping to [0,1] and then resizing with INTER_CUBIC
        # lets cubic overshoot push values back outside the range (measured
        # -0.0097 .. 1.0281 when done the other way round). Resizing first
        # makes the clip the final operation, so [0,1] is actually guaranteed.
        a = self._normalize(self._resize(np.array(self._raw(index), dtype=np.float32)))
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]  # (1,H,W)
        if self.num_channels > 1:
            t = t.repeat(self.num_channels, 1, 1)
        return t

    def get_target(self, index: int) -> Any:
        return -1   # self-supervised: no labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.get_image_data(index)
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.index)
