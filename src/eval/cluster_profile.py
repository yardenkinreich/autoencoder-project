"""
cluster_profile.py — what IS each cluster, and one fixed colour per cluster
shared by every surface that shows it.

TWO JOBS, both driven by the same per-run colour table.

1. STABLE COLOURS ACROSS THE APP AND QGIS
   The PNG map, the dashboard and a QGIS layer must agree: if cluster 3 is
   red here it must be the same red there. Colour is therefore computed ONCE
   per (run, k) from geographic_map._cluster_colors - which is deterministic
   given k - frozen to `cluster_colors_k{k}.json`, written as a `color_hex`
   attribute on every exported feature, AND emitted as a QGIS `.qml` style
   file. QGIS auto-loads a .qml sitting beside a layer with the same
   basename, so the categorized renderer comes up already matching.

   Colours still carry NO ordering: k-means ids are arbitrary labels. Fixing
   them makes the same cluster look the same everywhere; it does not make
   cluster 1 "less" than cluster 2, and they remain incomparable across runs.

2. PER-CLUSTER CHARACTER
   A cluster id on a map says nothing about what the model put in it. This
   computes, in ONE pass over the crop memmap:

     - the AVERAGE CRATER: the mean image over the cluster's members. The
       single most informative thing here - a sharp bowl with a bright rim
       versus a flat smudge is immediately legible in a way no statistic is.
     - brightness and a gradient-magnitude (sobel-like) texture proxy, as
       mean and IQR, since those are the two confounds this project keeps
       finding.
     - size, diameter, and two geography numbers: spatial spread relative to
       the whole pool, and a KS statistic against the pool's longitude
       distribution - i.e. how much this particular cluster is a longitude
       artifact rather than a morphology group.

   Deliberately NOT everything measurable. The figures are capped and the
   table is a dozen columns, because a profile nobody reads is worse than a
   short one that gets read.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.eval.geographic_map import _cluster_colors

# Past this, per-cluster panels stop being readable and the mean-crater grid
# is capped to the largest clusters (the rest stay in the CSV).
_MAX_PANELS = 24


def cluster_colors_hex(k: int) -> list[str]:
    """The frozen colour table for k clusters, as hex. Same source as the map
    figure, so the two cannot disagree."""
    from matplotlib.colors import to_hex
    return [to_hex(c) for c in _cluster_colors(k)]


def write_qml(out_path: str, k: int, colors: list[str], *,
              field: str = "cluster", marker_size: float = 1.6,
              labels: list[str] | None = None) -> str:
    """A QGIS layer-style file giving a categorized renderer on `cluster`,
    one category per id with the exact colour used everywhere else.

    QGIS loads `<layer basename>.qml` automatically when the layer is added,
    so this is what makes "same red here and there" true without anyone
    styling anything by hand."""
    cats, syms = [], []
    for i, c in enumerate(colors):
        _lbl = labels[i] if labels and i < len(labels) else f"cluster {i}"
        cats.append(f'    <category symbol="{i}" value="{i}" label="{_lbl}" '
                    f'render="true"/>')
        syms.append(
            f'    <symbol type="marker" name="{i}" alpha="1" force_rhr="0" clip_to_extent="1">\n'
            f'      <layer class="SimpleMarker" enabled="1" pass="0" locked="0">\n'
            f'        <prop k="color" v="{_hex_to_rgba(c)}"/>\n'
            f'        <prop k="name" v="circle"/>\n'
            f'        <prop k="outline_style" v="no"/>\n'
            f'        <prop k="size" v="{marker_size}"/>\n'
            f'        <prop k="size_unit" v="MM"/>\n'
            f'      </layer>\n'
            f'    </symbol>')
    xml = ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
           '<qgis version="3.28" styleCategories="Symbology">\n'
           f'  <renderer-v2 type="categorizedSymbol" attr="{field}" '
           'forceraster="0" symbollevels="0" enableorderby="0">\n'
           '    <categories>\n' + "\n".join(cats) + '\n    </categories>\n'
           '    <symbols>\n' + "\n".join(syms) + '\n    </symbols>\n'
           '  </renderer-v2>\n'
           '</qgis>\n')
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path


def _hex_to_rgba(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"{r},{g},{b},255"


def profile(dat_path: str, df: pd.DataFrame, size: int | None = None,
            batch: int = 512) -> tuple[pd.DataFrame, dict]:
    """One pass over the memmap: per-cluster mean image plus the feature
    sums needed for the table. Batched so a multi-GB memmap never lands in
    RAM at once - the same reason seam_check.per_crater_stats batches."""
    from src.eval.holdout import load_holdout_dat
    data = load_holdout_dat(dat_path, size=size)
    size = data.shape[-1]
    n = len(data)
    if n != len(df):
        raise ValueError(f"{dat_path} holds {n} crops but the cluster table has "
                         f"{len(df)} rows - these are positionally joined, so a "
                         f"mismatch means they came from different runs.")
    clusters = df["cluster"].to_numpy()
    ids = sorted(pd.unique(clusters))
    idx_of = {c: i for i, c in enumerate(ids)}

    acc = np.zeros((len(ids), size, size), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.int64)
    bright = np.empty(n, dtype=np.float64)
    sobel = np.empty(n, dtype=np.float64)

    for i0 in range(0, n, batch):
        chunk = np.asarray(data[i0:i0 + batch], dtype=np.float32)[:, 0]   # (B,H,W)
        flat = chunk.reshape(len(chunk), -1)
        bright[i0:i0 + len(chunk)] = flat.mean(axis=1)
        # Gradient magnitude, vectorized over the batch. A true cv2.Sobel per
        # crop would mean 100k python-level calls; the forward-difference
        # magnitude answers the same question (is this crop crisp or washed
        # out) and runs in one numpy op.
        gx = np.diff(chunk, axis=2)
        gy = np.diff(chunk, axis=1)
        m = np.sqrt(gx[:, :-1, :] ** 2 + gy[:, :, :-1] ** 2)
        sobel[i0:i0 + len(chunk)] = m.reshape(len(chunk), -1).mean(axis=1)
        for j, c in enumerate(clusters[i0:i0 + len(chunk)]):
            a = idx_of[c]
            acc[a] += chunk[j]
            counts[a] += 1

    mean_images = {c: (acc[idx_of[c]] / max(counts[idx_of[c]], 1)) for c in ids}

    from scipy import stats as sps
    pool_lon = df["lon"].to_numpy(dtype=float)

    def _rms(g):
        return float(np.sqrt(((g["x"] - g["x"].mean()) ** 2
                              + (g["y"] - g["y"].mean()) ** 2).mean()))
    pool_rms = _rms(df)

    rows = []
    for c in ids:
        sel = clusters == c
        g = df[sel]
        b, s = bright[sel], sobel[sel]
        # KS against the POOL's longitude distribution: how unlike the whole
        # population is this cluster's geographic spread? Large = the cluster
        # is concentrated in particular longitudes, which is the artifact
        # signature.
        ks = float(sps.ks_2samp(g["lon"].to_numpy(dtype=float), pool_lon).statistic)
        rows.append({
            "cluster": int(c), "n": int(sel.sum()),
            "pct_of_pool": float(100.0 * sel.sum() / n),
            "brightness_mean": float(b.mean()), "brightness_std": float(b.std()),
            "brightness_p25": float(np.percentile(b, 25)),
            "brightness_p75": float(np.percentile(b, 75)),
            "sobel_mean": float(s.mean()), "sobel_std": float(s.std()),
            "sobel_p25": float(np.percentile(s, 25)),
            "sobel_p75": float(np.percentile(s, 75)),
            "diam_km_mean": float(g["diam"].mean() / 1000.0) if "diam" in g else np.nan,
            "diam_km_median": float(g["diam"].median() / 1000.0) if "diam" in g else np.nan,
            "mean_abs_lat": float(g["lat"].abs().mean()),
            "spatial_spread_vs_pool": _rms(g) / (pool_rms or 1.0),
            "lon_ks_vs_pool": ks,
        })
    return pd.DataFrame(rows), mean_images


_ORDER_KEYS = {
    "sobel_then_brightness": ["sobel_mean", "brightness_mean"],
    "brightness_then_sobel": ["brightness_mean", "sobel_mean"],
    "sobel": ["sobel_mean"],
    "brightness": ["brightness_mean"],
    "size": ["n"],
}


def add_ordering(prof: pd.DataFrame, by: str = "sobel_then_brightness") -> pd.DataFrame:
    """Rank the clusters along a chosen feature axis, as a DERIVED VIEW.

    READ THIS BEFORE USING IT. k-means produces unordered groups; this
    imposes an order on them from a pixel statistic. That is a hypothesis
    ("is there a 1-D degradation-like axis here?"), not a finding, and it is
    deliberately kept out of everything that defines identity: cluster ids
    and colours are untouched, so the map, the QGIS layer and the exports do
    not change. Only `order_rank` is added.

    The caution is specific and has bitten this project before: an earlier
    version of geographic_map.py renumbered clusters by mean brightness so
    two models' maps would share colours, which made every map look
    brightness-organized BY CONSTRUCTION. Ordering for a table is fine;
    ordering that feeds colour is not. See is_ordinal() for the test of
    whether the ordering reflects real structure or is just an imposed sort."""
    keys = _ORDER_KEYS.get(by)
    if keys is None:
        raise ValueError(f"unknown ordering {by!r}; known: {sorted(_ORDER_KEYS)}")
    out = prof.sort_values(keys, ascending=True).reset_index(drop=True)
    out["order_rank"] = np.arange(len(out))
    out["ordered_by"] = by
    return out


def is_ordinal(prof: pd.DataFrame) -> dict:
    """Do the clusters actually lie on ONE axis, or is the ordering imposed?

    Two independent checks, because "sort them and they look ordered" is
    true of any sort:

    1. LINEARITY. PCA on the cluster means in standardized
       (brightness, sobel, diameter, spatial-spread) space. If the first
       component explains most of the between-cluster variance, the clusters
       genuinely lie along a line and an ordinal reading is defensible. If
       variance is spread across components, they occupy a space with no
       single axis and any 1-D ordering discards real structure.

    2. AGREEMENT. Spearman between the brightness ordering and the texture
       ordering. If those two disagree, "order by sobel then brightness"
       is a tiebreak between conflicting axes rather than one axis seen twice
       - which is itself the answer to the question.

    Neither test can tell you the axis means DEGRADATION. Both are computed
    from pixel statistics that this project has repeatedly found to be
    confounded with acquisition geometry. A clean 1-D axis that turns out to
    be longitude is a warning, not a result - which is why lon_ks_vs_pool
    sits in the same table."""
    from scipy import stats as sps
    from sklearn.decomposition import PCA

    cols = ["brightness_mean", "sobel_mean", "diam_km_mean", "spatial_spread_vs_pool"]
    use = [c for c in cols if c in prof.columns and prof[c].notna().all()]
    out = {"features_used": use, "n_clusters": int(len(prof))}
    if len(prof) >= 3 and len(use) >= 2:
        X = prof[use].to_numpy(dtype=float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        p = PCA().fit(X)
        ev = p.explained_variance_ratio_
        out["pc1_explained_variance"] = float(ev[0])
        out["pc2_explained_variance"] = float(ev[1]) if len(ev) > 1 else 0.0
        out["verdict_linearity"] = (
            "strongly 1-D" if ev[0] >= 0.80 else
            "mostly 1-D" if ev[0] >= 0.60 else
            "not 1-D - a single ordering discards real structure")
    # A rank correlation over a handful of clusters carries almost no
    # information - at k=4 there are 4 points and rho can only take a few
    # values - so it is computed but explicitly marked uninformative rather
    # than reported as a finding.
    if {"brightness_mean", "sobel_mean"} <= set(prof.columns) and len(prof) >= 3:
        r, pv = sps.spearmanr(prof["brightness_mean"], prof["sobel_mean"])
        out["spearman_brightness_vs_sobel"] = float(r)
        out["spearman_p"] = float(pv)
        if len(prof) < 8:
            out["verdict_agreement"] = (
                f"only {len(prof)} clusters - a rank correlation over this few "
                f"points is uninformative; read the k=10/k=100 runs instead")
        else:
            out["verdict_agreement"] = (
                "brightness and texture order the clusters the same way"
                if abs(r) >= 0.7 else
                "brightness and texture disagree - the two-key sort is a tiebreak "
                "between different axes, not one axis")
    return out


def fig_mean_craters(mean_images: dict, prof: pd.DataFrame, colors: list[str],
                     out_png: str, k: int, order_by: str | None = None,
                     dat_path: str | None = None, clusters=None,
                     n_examples: int = 4, size: int | None = None, seed: int = 0) -> None:
    """The average crater of each cluster - the one panel that says what the
    model actually grouped. Bordered in that cluster's own colour so the grid
    reads against the map and the QGIS layer.

    order_by: lay the panels out along that feature axis instead of by size.
    This is the most direct test of whether an ordinal exists - if it does,
    the grid shows a visible progression from sharp-rimmed to smooth, and if
    it does not, the panels look shuffled. The eye is good at this."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if order_by and "order_rank" in prof.columns:
        ranked = prof.sort_values("order_rank")
        order = ranked["cluster"].tolist()[:_MAX_PANELS]
    else:
        order = prof.sort_values("n", ascending=False)["cluster"].tolist()[:_MAX_PANELS]

    # ONE ROW PER CLUSTER: the mean crater first, then real members beside it.
    # A grid of means alone hides within-cluster variance, which is exactly what
    # decides whether a cluster is a real group or a bag of leftovers - so the
    # average and its members are put on the same line where the eye compares
    # them without scrolling.
    # HALF closest-to-the-cluster-mean, HALF random. The two answer different
    # questions and their disagreement is the informative part: the closest
    # members show what a cluster is ABOUT, random members show whether typical
    # membership looks like that at all. With silhouette around 0.09 on this
    # data, random members from different clusters are near-indistinguishable -
    # so if even the closest members are alike too, the clustering is not
    # capturing anything visual, and that is worth being able to see.
    examples, n_near = {}, max(1, n_examples // 2)
    if dat_path and clusters is not None and n_examples > 0:
        from src.eval.holdout import load_holdout_dat
        d = load_holdout_dat(dat_path, size=size)
        size = d.shape[-1]
        rng = np.random.default_rng(seed)
        cl = np.asarray(clusters)
        for cid in order:
            sel = np.where(cl == cid)[0]
            if not len(sel):
                continue
            # distance to the cluster's own mean IMAGE - the same quantity the
            # average panel beside it displays, so "closest" means closest to
            # what is shown, not to a latent centroid the reader cannot see.
            mu = mean_images[cid]
            dist = np.array([np.abs(np.asarray(d[int(i)])[0] - mu).mean()
                             for i in sel])
            near = sel[np.argsort(dist)[:n_near]]
            rest = np.setdiff1d(sel, near)
            rnd = (rng.choice(rest, size=min(n_examples - n_near, len(rest)),
                              replace=False) if len(rest) else [])
            examples[cid] = [np.asarray(d[int(i)])[0]
                             for i in list(near) + list(rnd)]

    ncols = 1 + (n_examples if examples else 0)
    fig, axes = plt.subplots(len(order), ncols,
                             figsize=(1.95 * ncols, 2.15 * len(order)),
                             squeeze=False)
    for ri, cid in enumerate(order):
        r = prof.set_index("cluster").loc[cid]
        col = colors[cid % len(colors)]
        ax = axes[ri][0]
        ax.imshow(mean_images[cid], cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(3.5)
        ax.set_ylabel(f"cluster {cid}", fontsize=9, color=col)
        ax.set_xlabel(f"n={int(r['n']):,}  bright {r['brightness_mean']:.3f}\n"
                      f"sobel {r['sobel_mean']:.4f}", fontsize=6.5, labelpad=2)
        if ri == 0:
            ax.set_title("AVERAGE", fontsize=8, color=col)
        for e in range(ncols - 1):
            ax = axes[ri][1 + e]
            ax.set_xticks([]); ax.set_yticks([])
            ex = examples.get(cid, [])
            if e >= len(ex):
                ax.axis("off"); continue
            ax.imshow(ex[e], cmap="gray", interpolation="nearest")
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(1.2)
            if ri == 0:
                ax.set_title("closest" if e < n_near else "random",
                             fontsize=7.5, color="#555555")

    extra = "" if len(order) == len(prof) else \
        f"  ({len(order)} of {len(prof)} shown; all clusters are in the CSV)"
    how = (f"rows ordered by {order_by.replace('_', ' ')} — look for a "
           f"progression from sharp to smooth"
           if order_by and "order_rank" in prof.columns
           else "rows ordered by cluster size")
    fig.suptitle(f"Clusters, k={k}{extra}\n"
                 f"average crater then real members, {how}; border colour matches "
                 "the map and the QGIS layer", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.1)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_feature_profile(prof: pd.DataFrame, colors: list[str], out_png: str,
                        k: int, order_by: str | None = None) -> None:
    """Brightness and texture per cluster, with IQR, plus the two geography
    numbers. Point-and-range rather than histograms: at k=100 a hundred
    overlaid distributions is noise, while a hundred ordered points with
    spread is still readable.

    Ordered by `order_rank` - the SAME axis the mean-crater grid uses. An
    earlier version sorted this figure by brightness while the grid sorted by
    sobel, so two figures written side by side into one directory disagreed
    about the x-axis with nothing saying so."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "order_rank" in prof.columns:
        p = prof.sort_values("order_rank").reset_index(drop=True)
        order_label = (order_by or prof.get("ordered_by", pd.Series(["order_rank"])).iloc[0])
        order_label = str(order_label).replace("_", " ")
    else:
        p = prof.sort_values("brightness_mean").reset_index(drop=True)
        order_label = "mean brightness"
    x = np.arange(len(p))
    cols = [colors[int(c) % len(colors)] for c in p["cluster"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    for ax, (m, lo, hi, name) in zip(axes[:2], [
            ("brightness_mean", "brightness_p25", "brightness_p75", "Brightness"),
            ("sobel_mean", "sobel_p25", "sobel_p75", "Texture (gradient magnitude)")]):
        ax.vlines(x, p[lo], p[hi], color=cols, lw=2.0, alpha=0.55)
        ax.scatter(x, p[m], c=cols, s=26, zorder=3, edgecolors="white", linewidths=0.4)
        ax.set_title(f"{name} per cluster (point = mean, bar = IQR)", fontsize=10)
        ax.set_xlabel(f"clusters, ordered by {order_label}", fontsize=8)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    ax = axes[2]
    ax.scatter(p["spatial_spread_vs_pool"], p["lon_ks_vs_pool"], c=cols,
               s=np.clip(p["n"] / max(p["n"].max(), 1) * 260, 12, 260),
               edgecolors="white", linewidths=0.4, zorder=3)
    ax.axvline(1.0, color="#888888", ls=":", lw=1.0)
    ax.set_xlabel("spatial spread ÷ pool  (1.0 = spread like the whole pool)", fontsize=8)
    ax.set_ylabel("longitude KS vs pool", fontsize=8)
    ax.set_title("Geographic concentration (marker size = cluster size)", fontsize=10)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(f"Cluster profile, k={k} — ordered by {order_label}; "
                 f"colours match the map and the QGIS layer", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_for_dir(map_dir: str, *, size: int = 128, overwrite: bool = False,
                order_by: str = "sobel_then_brightness") -> list[int]:
    """Profile every k present in one geographic_map output directory."""
    done = []
    for f in sorted(os.listdir(map_dir)):
        if not (f.startswith("geographic_clusters_k") and f.endswith(".csv")):
            continue
        if f.endswith("_summary.csv"):
            continue
        k = int(f[len("geographic_clusters_k"):-len(".csv")])
        prof_csv = os.path.join(map_dir, f"cluster_profile_k{k}.csv")
        if os.path.exists(prof_csv) and not overwrite:
            print(f"  skip k={k} (exists)")
            continue

        df = pd.read_csv(os.path.join(map_dir, f), dtype={"id": str})
        meta_json = os.path.join(map_dir, f"geographic_map_k{k}.json")
        dat_path = None
        if os.path.exists(meta_json):
            dat_path = json.load(open(meta_json)).get("dat_path")
        if not dat_path or not os.path.exists(dat_path):
            print(f"  k={k}: dat_path missing ({dat_path}) - cannot profile pixels")
            continue

        colors = cluster_colors_hex(k)
        json.dump({"k": k, "colors": {str(i): colors[i] for i in range(k)},
                   "source": "eval/geographic_map._cluster_colors - deterministic "
                             "given k, so the map PNG, the app and QGIS agree",
                   "note": "cluster ids are arbitrary k-means labels; fixed colours "
                           "make the same cluster look the same everywhere, they do "
                           "NOT imply an ordering, and they are not comparable "
                           "across runs"},
                  open(os.path.join(map_dir, f"cluster_colors_k{k}.json"), "w"),
                  indent=2)
        # One .qml, matching the .gpkg's basename, so QGIS auto-applies it.
        # (Earlier versions also wrote _projected/_geographic variants for the
        # GeoJSON exports; those exports are gone, and an orphan .qml naming a
        # layer that does not exist is just clutter.)
        write_qml(os.path.join(map_dir, f"geographic_clusters_k{k}.qml"), k, colors)
        for orphan in (f"geographic_clusters_k{k}_projected.qml",
                       f"geographic_clusters_k{k}_geographic.qml"):
            op = os.path.join(map_dir, orphan)
            if os.path.exists(op):
                os.remove(op)

        prof, means = profile(dat_path, df, size=size)
        prof["color_hex"] = [colors[int(c) % len(colors)] for c in prof["cluster"]]
        prof = add_ordering(prof, by=order_by)
        ordinal = is_ordinal(prof)
        prof.to_csv(prof_csv, index=False)
        json.dump({"k": k, "ordered_by": order_by, **ordinal},
                  open(os.path.join(map_dir, f"cluster_ordinal_k{k}.json"), "w"),
                  indent=2)
        fig_mean_craters(means, prof, colors,
                         os.path.join(map_dir, f"cluster_mean_craters_k{k}.png"), k,
                         order_by=order_by, dat_path=dat_path,
                         clusters=df["cluster"].to_numpy(), size=size)
        fig_feature_profile(prof, colors,
                            os.path.join(map_dir, f"cluster_feature_profile_k{k}.png"), k,
                            order_by=order_by)
        print(f"  k={k}: profiled {len(prof)} clusters -> {prof_csv}")
        if "pc1_explained_variance" in ordinal:
            print(f"        ordinal check: PC1 explains "
                  f"{ordinal['pc1_explained_variance']:.1%} of between-cluster "
                  f"variance — {ordinal.get('verdict_linearity')}")
        if "spearman_brightness_vs_sobel" in ordinal:
            print(f"        brightness vs sobel rank: rho="
                  f"{ordinal['spearman_brightness_vs_sobel']:+.3f} — "
                  f"{ordinal.get('verdict_agreement')}")
        done.append(k)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default="logs",
                    help="Root to search for geographic_map output directories.")
    ap.add_argument("--dir", nargs="*", default=None)
    ap.add_argument("--size", type=int, default=None,
                    help="crop size of the memmap; default: from its evalset_manifest.json, else 128")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--order-by", default="sobel_then_brightness",
                    choices=sorted(_ORDER_KEYS),
                    help="Feature axis the clusters are RANKED along in the profile "
                         "table and the mean-crater grid. A derived view only - it "
                         "never touches cluster ids or colours. See add_ordering().")
    args = ap.parse_args()

    import glob
    dirs = list(args.dir or [])
    if not dirs:
        dirs = sorted({os.path.dirname(p) for p in glob.glob(
            os.path.join(args.scan, "**", "geographic_clusters_k*.csv"),
            recursive=True)})
    if not dirs:
        raise SystemExit("no geographic_map directories found")
    for d in dirs:
        print(f"\n=== {d}")
        run_for_dir(d, size=args.size, overwrite=args.overwrite,
                    order_by=args.order_by)


if __name__ == "__main__":
    main()
