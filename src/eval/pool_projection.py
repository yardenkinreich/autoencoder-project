"""
pool_projection.py — the whole (~98k) crater pool in the embedding, coloured
by the geographic map's own k-means clusters.

For a geographic_map variant directory (data_comparison/geomap/<cat>__<norm>/)
this reuses what earlier steps cached - pool_embeddings.npy (pool_labeling)
and label_knn/umap_coords.npy - and the per-crater cluster ids of
geographic_clusters_k{k}.csv, all in the pool's crater order, and writes
pca_kmeans_k{k}.png and umap_kmeans_k{k}.png in the SAME cluster colours as
the map (cluster_colors_k{k}.json), so a blob in the embedding and a region
on the map can be matched by colour. No GPU, no re-embedding.

Usage:
  PYTHONPATH=$(pwd) python -m src.eval.pool_projection --dir <variant dir> [--k 4 10]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def run(d: str, ks=(4, 10)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    emb_p = os.path.join(d, "pool_embeddings.npy")
    if not os.path.exists(emb_p):
        print(f"  skip {d}: no pool_embeddings.npy (run pool_labeling first)")
        return
    X = np.load(emb_p)
    P = PCA(n_components=2, random_state=0).fit(X)
    Z, ev = P.transform(X), P.explained_variance_ratio_
    U = None
    up = os.path.join(d, "label_knn", "umap_coords.npy")
    if os.path.exists(up):
        U = np.load(up)
    title = os.path.basename(d.rstrip("/"))
    for k in ks:
        cp = os.path.join(d, f"geographic_clusters_k{k}.csv")
        if not os.path.exists(cp):
            continue
        cl = pd.read_csv(cp)["cluster"].to_numpy()
        if len(cl) != len(X):
            print(f"  skip k={k}: {len(cl)} clusters vs {len(X)} embeddings")
            continue
        colp = os.path.join(d, f"cluster_colors_k{k}.json")
        cols = json.load(open(colp)) if os.path.exists(colp) else {}
        if isinstance(cols, dict) and "colors" in cols:      # {"k":..,"colors":{"0":"#hex",..}}
            cols = cols["colors"]
        def color(c):
            if isinstance(cols, dict):
                v = cols.get(str(c)) or cols.get(int(c)) if cols else None
                if isinstance(v, dict):
                    v = v.get("hex") or v.get("color_hex")
                return v or plt.cm.tab10(int(c) % 10)
            return cols[int(c) % len(cols)]
        for coords, name, xl, yl in [(Z, "pca", f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})"),
                                     (U, "umap", "UMAP 1", "UMAP 2")]:
            if coords is None:
                continue
            fig, ax = plt.subplots(figsize=(7.4, 6.4))
            for c in np.unique(cl):
                q = cl == c
                ax.scatter(coords[q, 0], coords[q, 1], s=1.5, color=color(c), alpha=0.4,
                           linewidths=0, rasterized=True,
                           label=(f"cluster {c} (n={int(q.sum()):,})" if k <= 12 else None))
            if k <= 12:
                ax.legend(fontsize=8, markerscale=5, frameon=False)
            ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.suptitle(f"{title} — {name.upper()} of all {len(X):,} pool craters, "
                         f"coloured by k-means k={k} (map colours)", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            out = os.path.join(d, f"{name}_kmeans_k{k}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="+", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[4, 10])
    a = ap.parse_args()
    for d in a.dir:
        run(d, a.k)
