"""
The eval contract: the single boundary between "your pipeline" and "the metrics".

Everything in this harness is computed from ONE table: predictions.csv.
That decoupling is the whole point — you can change preprocessing, the MAE
checkpoint, the clustering algorithm, k, anything — and as long as you can
produce this table, every metric below keeps working untouched.

predictions.csv schema (one row per labeled crater in Julie's test set)
-----------------------------------------------------------------------
crater_id        : str    unique id, joins back to Julie's labels + your metadata
true_label       : int    ground-truth erosion state, 0..K-1, ORDINAL (0 = freshest)
cluster          : int    raw cluster id assigned by the unsupervised pipeline
                          (NOT a class — needs alignment; see align.py)
dist_0..dist_{C-1}: float  OPTIONAL. distance/soft-assignment to each cluster
                          centroid. If present, enables calibration (ECE,
                          reliability) after we convert to class-probabilities.

You produce this table with pipeline.py's embed()/kmeans_cluster()/
build_predictions() (a thin adapter around YOUR repo). You consume it with
evaluate.py. Nothing else needs to know how craters became clusters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

REQUIRED_COLS = ["crater_id", "true_label", "cluster"]


@dataclass
class LabelScheme:
    """Describes the ordinal class set so metrics can be interpreted/printed."""
    names: list[str]                 # e.g. ["fresh", "slight", "moderate", "eroded"]
    # ordinal by construction: index 0 = freshest, increasing = more degraded

    @property
    def n_classes(self) -> int:
        return len(self.names)


def load_predictions(path: str, scheme: LabelScheme) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"predictions.csv missing required columns: {missing}")
    if df["true_label"].max() >= scheme.n_classes or df["true_label"].min() < 0:
        raise ValueError(
            f"true_label out of range for {scheme.n_classes}-class scheme; "
            f"got [{df['true_label'].min()}, {df['true_label'].max()}]"
        )
    # how many distinct clusters did the pipeline produce?
    df.attrs["n_clusters"] = int(df["cluster"].nunique())
    df.attrs["dist_cols"] = sorted(
        [c for c in df.columns if c.startswith("dist_")],
        key=lambda c: int(c.split("_")[1]),
    )
    return df
