"""
per_band_clustering.py — cluster WITHIN each acquisition band, then match the
clusters across bands, and ask whether the result is band-independent.

THE IDEA BEING TESTED
---------------------
The WAC_GLOBAL mosaic is assembled from three observation windows, and the
three longitude bands differ in illumination geometry - measured, diameter-
matched, against the middle band:

    contrast      west 1.23   mid 1.00   east 0.82     (solar elevation)
    azimuth (LR)  west FLIPPED, mid/east same sign     (solar azimuth)

Neither the subtractive high-pass nor a divisive local-contrast normalization
removes this (both were built and measured; the divisive one cut the contrast
step 28.6% -> 6.4% and left the band clusters marginally STRONGER). The
difference is where the light came from, not a gain or an offset.

So: cluster inside each band, where illumination is constant and band identity
is therefore unavailable as a feature, and combine afterwards.

WHY COMBINING IS THE HARD PART
------------------------------
Cluster ids are arbitrary, and k-means run separately on three subsets gives
three unrelated labelings - cluster 3 in west has nothing to do with cluster 3
in mid. The usual fix, aligning two embeddings via paired anchors, is not
available: every crater sits in exactly one band, so no object appears twice.

The way through is to match clusters by what they ARE rather than by where
they sit in a latent space. Each cluster has a profile that is defined in the
same physical units in every band - mean crater image, texture, brightness,
diameter - so clusters can be matched across bands by profile similarity with
a Hungarian assignment. That is a claim this module TESTS rather than assumes:
if matching works, the matched groups should look alike across bands and the
merged labeling should be far less predictable from band than global
clustering is.

THE METRIC
----------
Cramer's V between the final labeling and the band. Global clustering on this
data measures V = 0.224 (k=10); a band-independent labeling should be near 0.
V is used rather than the epsilon-squared elsewhere in this project because
both variables here are categorical.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

BANDS = ("west", "mid", "east")


def band_of(lon: np.ndarray) -> np.ndarray:
    """Acquisition bands, split at the measured boundaries. The -90 boundary
    is where the illumination azimuth flips sign within one 5-degree step; +90
    is where contrast steps. Both were located empirically, not assumed."""
    lon = np.where(lon > 180, lon - 360, lon)
    return np.where(lon < -90, "west", np.where(lon < 90, "mid", "east"))


def embed_pool(dat_path: str, checkpoint: str, model: str, cache: str) -> np.ndarray:
    if os.path.exists(cache):
        print(f"  reusing embeddings {cache}")
        return np.load(cache)
    from src.eval.holdout import embed_holdout
    X = np.asarray(embed_holdout(dat_path, checkpoint, model))
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.save(cache, X)
    print(f"  embedded {X.shape} -> {cache}")
    return X


def crop_profiles(dat_path: str, labels: np.ndarray, ids, size: int = 128,
                  batch: int = 512):
    """Mean image + scalar profile per group, in PHYSICAL units that mean the
    same thing in every band - which is what makes cross-band matching
    possible at all."""
    from src.eval.holdout import load_holdout_dat
    d = load_holdout_dat(dat_path, size=size)
    uniq = list(ids)
    pos = {g: i for i, g in enumerate(uniq)}
    acc = np.zeros((len(uniq), size, size)); cnt = np.zeros(len(uniq))
    br = np.zeros(len(uniq)); ct = np.zeros(len(uniq))
    for i0 in range(0, len(d), batch):
        c = np.asarray(d[i0:i0 + batch], dtype=np.float32)[:, 0]
        k = len(c)
        f = c.reshape(k, -1)
        b = f.mean(1)
        t = np.abs(np.diff(c, axis=2)).reshape(k, -1).mean(1)
        for j, lb in enumerate(labels[i0:i0 + k]):
            if lb not in pos:
                continue
            a = pos[lb]
            acc[a] += c[j]; cnt[a] += 1; br[a] += b[j]; ct[a] += t[j]
    cnt = np.maximum(cnt, 1)
    return (acc / cnt[:, None, None], br / cnt, ct / cnt, cnt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-dir", required=True,
                    help="geographic_map output dir (for the cluster CSV + dat_path).")
    ap.add_argument("--k", type=int, default=10, help="clusters PER BAND")
    ap.add_argument("--checkpoint", default="data/raw/pretrained/dinov2_vits14_pretrain.pth")
    ap.add_argument("--model", default="dino_pretrained")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", choices=("closest", "mixed"), default="closest",
                    help="which members to show per cluster. 'closest' (default) "
                         "shows the members nearest the cluster CENTROID - the "
                         "mean embedding k-means minimised distance to. 'mixed' "
                         "shows half nearest, half random members.")
    ap.add_argument("--map-marker-size", type=float, default=36.0,
                    help="Marker AREA in points^2 on the per-band maps. matplotlib's "
                         "`s` is an AREA, so diameter grows as its square root - the "
                         "earlier 2 -> 6 change would have widened a point by only "
                         "1.7x, which is why it read as almost no change. 36 is 6pt "
                         "across - 4x the diameter of the original 2.0 setting.")
    ap.add_argument("--map-marker-alpha", type=float, default=0.45,
                    help="Alpha drops as markers grow so overlapping points still let "
                         "the mosaic show through instead of forming a solid sheet.")
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment
    from scipy import stats as sps

    df = pd.read_csv(os.path.join(args.map_dir, f"geographic_clusters_k{args.k}.csv"),
                     dtype={"id": str})
    meta = json.load(open(os.path.join(args.map_dir, f"geographic_map_k{args.k}.json")))
    dat = meta["dat_path"]
    band = band_of(df["lon"].to_numpy(dtype=float))
    print(f"{len(df):,} craters | " + "  ".join(
        f"{b}={int((band == b).sum()):,}" for b in BANDS))

    X = embed_pool(dat, args.checkpoint, args.model,
                   os.path.join(args.map_dir, "embeddings_pool.npy"))

    def cramers_v(a, b) -> float:
        ct = pd.crosstab(pd.Series(a), pd.Series(b))
        if min(ct.shape) < 2:
            return float("nan")
        chi2 = sps.chi2_contingency(ct)[0]
        return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))

    # --- reference: ONE clustering over everything -------------------------
    glob = KMeans(args.k, random_state=args.seed, n_init=10).fit_predict(X)
    v_glob = cramers_v(glob, band)

    # --- the proposal: cluster INSIDE each band ----------------------------
    per = np.full(len(df), -1, dtype=int)
    for b in BANDS:
        m = band == b
        per[m] = KMeans(args.k, random_state=args.seed, n_init=10).fit_predict(X[m])
    # Per-band ids are only meaningful WITH their band, so the raw combination
    # is (band, id) - 3k groups that trivially encode band. The question is
    # whether MATCHING collapses them back to k band-independent groups.
    combo = np.array([f"{b}_{c}" for b, c in zip(band, per)])
    v_combo = cramers_v(combo, band)

    # --- match clusters across bands by PROFILE ----------------------------
    keys = [f"{b}_{c}" for b in BANDS for c in range(args.k)]
    mean_img, br, ct, cnt = crop_profiles(dat, combo, keys)
    prof = {k: (mean_img[i], br[i], ct[i]) for i, k in enumerate(keys)}

    # Match every band to the MIDDLE band (the largest, and the reference the
    # contrast ordering was measured against). Cost = distance between mean
    # crater images after per-image standardization, so a band-wide contrast
    # offset cannot dominate the matching it is supposed to see through.
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)
    mapping = {f"mid_{c}": c for c in range(args.k)}
    flip_used = {}
    for b in ("west", "east"):
        # Try BOTH orientations. The west band's illumination azimuth is
        # reversed (measured: left-right asymmetry +0.016 vs mid -0.012, a sign
        # flip inside one 5-degree step), so its mean crater images are MIRRORS
        # of the corresponding mid ones. Comparing them unflipped can never
        # align them - measured directly: west matching cost 1.0782 against a
        # random-pairing baseline of 1.0859, i.e. no better than chance, while
        # east scored 0.1346 against 0.2140.
        #
        # Taking the better of the two orientations is therefore both the fix
        # and a test: if flipping collapses west's cost, the azimuth reversal
        # is confirmed as the reason the bands look different.
        best = None
        for flip in (False, True):
            C = np.zeros((args.k, args.k))
            for i in range(args.k):
                a = prof[f"{b}_{i}"][0]
                a = a[:, ::-1] if flip else a
                for j in range(args.k):
                    C[i, j] = np.abs(z(a) - z(prof[f"mid_{j}"][0])).mean()
            r, c = linear_sum_assignment(C)
            cost = C[r, c].mean()
            if best is None or cost < best[0]:
                best = (cost, r, c, C.mean(), flip)
        cost, r, c, rand, flip = best
        flip_used[b] = bool(flip)
        for i, j in zip(r, c):
            mapping[f"{b}_{i}"] = int(j)
        print(f"  {b}->mid cost {cost:.4f} (random {rand:.4f})  "
              f"mirrored={flip}")

    merged = np.array([mapping[k] for k in combo])
    v_merged = cramers_v(merged, band)

    print()
    print(f"  {'labeling':38} {'Cramers V vs band':>18}")
    print(f"  {'global k-means (current approach)':38} {v_glob:18.4f}")
    print(f"  {'per-band, unmatched (band,id)':38} {v_combo:18.4f}")
    print(f"  {'per-band, matched across bands':38} {v_merged:18.4f}")
    print()
    print("  A matched labeling near 0 means the acquisition band is no longer")
    print("  recoverable from the cluster assignment - the point of the exercise.")
    print("  It does NOT by itself show the clusters are degradation-meaningful;")
    print("  that needs the labeled set, where west has only 30 craters.")

    # Defined here, before the figures: the match figure references it and an
    # earlier version assigned it only in the map block further down, which
    # raised UnboundLocalError after all the expensive work was done.
    label = os.path.basename(args.map_dir.rstrip("/"))

    # ── SHOW THE MATCHES ────────────────────────────────────────────────────
    # The V number says band is less predictable; it does not say the matching
    # is sensible. These two figures are how that gets judged by eye.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.geographic_map import _basemap, _DEFAULT_BASEMAP
    from src.eval.cluster_profile import cluster_colors_hex
    colors = cluster_colors_hex(args.k)

    # (a) mean crater of every (matched group x band). If the matching is
    #     real, a row looks like the same kind of crater three times.
    fig, axes = plt.subplots(args.k, 3, figsize=(3 * 2.3, args.k * 2.45),
                             squeeze=False)
    for g in range(args.k):
        for bi, b in enumerate(BANDS):
            ax = axes[g][bi]
            ax.set_xticks([]); ax.set_yticks([])
            src = [kk for kk in keys if mapping.get(kk) == g and kk.startswith(b + "_")]
            if not src:
                ax.axis("off"); continue
            kk = src[0]
            img = prof[kk][0]
            ax.imshow(img, cmap="gray", interpolation="nearest")
            for sp in ax.spines.values():
                sp.set_color(colors[g % len(colors)]); sp.set_linewidth(3)
            n_g = int(cnt[keys.index(kk)])
            mark = "  (mirrored to match)" if flip_used.get(b) else ""
            ax.set_title(f"{b} · n={n_g:,}{mark}", fontsize=7.5)
            if bi == 0:
                ax.set_ylabel(f"group {g}", fontsize=9)
    fig.suptitle(
        f"{label} — cross-band cluster MATCHES, k={args.k}\n"
        "each row is one matched group; if the match is real the three bands "
        "show the same kind of crater", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.3)
    mp = os.path.join(args.map_dir, f"per_band_matches_k{args.k}.png")
    fig.savefig(mp, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {mp}")

    # (b) three REAL craters per (group x band) - a mean image hides variance,
    #     and whether the members actually resemble each other is the question
    #     a mean cannot answer.
    from src.eval.holdout import load_holdout_dat
    dd = load_holdout_dat(dat)
    rng2 = np.random.default_rng(args.seed)
    n_ex = 3
    fig, axes = plt.subplots(args.k, 3 * n_ex, figsize=(3 * n_ex * 1.5, args.k * 1.75),
                             squeeze=False)
    for g in range(args.k):
        for bi, b in enumerate(BANDS):
            sel = np.where((merged == g) & (band == b))[0]
            # Nearest the centroid (mean embedding) - same rationale as the
            # per-band profiles: a cluster whose CENTRAL members are already
            # indistinguishable from another's is not a visual grouping.
            n_near_g = n_ex if args.examples == "closest" else max(1, n_ex // 2)
            if len(sel):
                dd_ = np.linalg.norm(X[sel] - X[sel].mean(0), axis=1)
                nearest = sel[np.argsort(dd_)[:n_near_g]]
                rest_ = np.setdiff1d(sel, nearest)
                pick = list(nearest) + list(
                    rng2.choice(rest_, size=min(n_ex - len(nearest), len(rest_)),
                                replace=False)
                    if len(rest_) and n_near_g < n_ex else [])
            else:
                pick = []
            for e in range(n_ex):
                ax = axes[g][bi * n_ex + e]
                ax.set_xticks([]); ax.set_yticks([])
                if e >= len(pick):
                    ax.axis("off"); continue
                ax.imshow(np.asarray(dd[int(pick[e])])[0], cmap="gray",
                          interpolation="nearest")
                for sp in ax.spines.values():
                    sp.set_color(colors[g % len(colors)]); sp.set_linewidth(2)
                if g == 0:
                    ax.set_title(f"{b}\n{'closest' if e < n_near_g else 'random'}",
                                 fontsize=8)
            if bi == 0:
                axes[g][0].set_ylabel(f"grp {g}", fontsize=8)
    fig.suptitle(f"{label} — {n_ex} example craters per matched group, per band "
                 f"(k={args.k})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ep = os.path.join(args.map_dir, f"per_band_examples_k{args.k}.png")
    fig.savefig(ep, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {ep}")

    # (c) geographic spread WITHIN each band, coloured by that band's own
    #     clusters. The matched map answers "did band structure survive"; this
    #     answers the prior question "what does clustering inside one band even
    #     look like", which the matched map cannot show because it has already
    #     merged the three labelings.
    #
    #     ONE FIGURE PER BAND, not three panels in one. With aspect="equal" a
    #     90-degree band is taller than wide while the 180-degree middle one is
    #     wider than tall; packing all three into a single figure squeezed the
    #     narrow bands into small squares where 25-33k points became a solid
    #     sheet of colour and the mosaic underneath was invisible. Sizing each
    #     figure to its own aspect ratio is what makes the basemap readable.
    bmi, bex = _basemap(_DEFAULT_BASEMAP)
    R = 1737400.0
    edges = {"west": (-180.0, -90.0), "mid": (-90.0, 90.0), "east": (90.0, 180.0)}
    band_pngs = {}
    for b in BANDS:
        lo, hi = edges[b]
        x0, x1 = R * np.deg2rad(lo), R * np.deg2rad(hi)
        y0, y1 = bex[2], bex[3]
        H = 9.0
        W = max(6.0, H * (x1 - x0) / (y1 - y0))
        fig, ax = plt.subplots(figsize=(W, H))
        ax.imshow(bmi, extent=bex, cmap="gray", interpolation="bilinear",
                  aspect="equal", zorder=0)
        m = band == b
        # Small and semi-transparent on purpose: at ~30k points in one panel a
        # marker large enough to see individually guarantees the basemap is
        # hidden, and the basemap is the only thing that says WHERE a cluster
        # is. Density is read from colour accumulation instead.
        for c in range(args.k):
            sel = m & (per == c)
            if sel.any():
                ax.scatter(df.loc[sel, "x"], df.loc[sel, "y"],
                           s=args.map_marker_size,
                           color=colors[c % len(colors)], linewidths=0,
                           alpha=args.map_marker_alpha, zorder=2, rasterized=True)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label} — {b} band  ({int(m.sum()):,} craters, "
                     f"{lo:g}° to {hi:g}°)\nclustered WITHIN this band, k={args.k}; "
                     f"colours are this band's own ids", fontsize=11)
        fig.tight_layout()
        bp = os.path.join(args.map_dir, f"per_band_map_{b}_k{args.k}.png")
        fig.savefig(bp, dpi=170, bbox_inches="tight"); plt.close(fig)
        band_pngs[b] = bp
        print(f"  wrote {bp}")

    # (d) PER BAND, the same "average then real members" layout the global
    #     cluster profile uses - rows ordered by texture, which is the axis the
    #     clusters actually separate on. This is what answers "do the clusters
    #     inside one band track degradation", by eye, BEFORE any merging: a
    #     band whose rows run smoothly from washed-out to sharp-rimmed is
    #     separating something morphological; one whose rows all look alike is
    #     not.
    from src.eval.holdout import load_holdout_dat
    dd2 = load_holdout_dat(dat)
    rng3 = np.random.default_rng(args.seed)
    n_ex2 = 4
    for b in BANDS:
        keys_b = [f"{b}_{c}" for c in range(args.k)]
        # order by the same contrast proxy used everywhere else in the project
        order_b = sorted(range(args.k), key=lambda c: prof[f"{b}_{c}"][2])
        fig, axes = plt.subplots(args.k, 1 + n_ex2,
                                 figsize=(1.95 * (1 + n_ex2), 2.15 * args.k),
                                 squeeze=False)
        for ri, c in enumerate(order_b):
            kk = f"{b}_{c}"
            col = colors[c % len(colors)]
            ax = axes[ri][0]
            ax.imshow(prof[kk][0], cmap="gray", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(col); sp.set_linewidth(3.5)
            ax.set_ylabel(f"{b} cl {c}", fontsize=9, color=col)
            ax.set_xlabel(f"n={int(cnt[keys.index(kk)]):,}\n"
                          f"bright {prof[kk][1]:.3f}  sobel {prof[kk][2]:.4f}",
                          fontsize=6.5, labelpad=2)
            if ri == 0:
                ax.set_title("AVERAGE", fontsize=8, color=col)
            sel = np.where((band == b) & (per == c))[0]
            # The NEAREST members to the cluster centroid, all of them - the
            # centroid being the mean EMBEDDING of the cluster, i.e. the thing
            # k-means actually minimised distance to, not the mean image.
            # Earlier this was half nearest, half random, on the argument that
            # their disagreement is informative. It is, but it costs half the
            # panels, and with silhouette near 0.09 the random draws were
            # mostly edge members that made every cluster look alike - which
            # answers "are the clusters fuzzy" (already answered, numerically:
            # yes) instead of "what is this cluster ABOUT", which is the
            # question these rows exist to answer. --examples mixed restores
            # the old behaviour.
            n_near = n_ex2 if args.examples == "closest" else n_ex2 // 2
            if len(sel):
                d2 = np.linalg.norm(X[sel] - X[sel].mean(0), axis=1)
                near = sel[np.argsort(d2)[:n_near]]
                rest = np.setdiff1d(sel, near, assume_unique=False)
                rnd = (rng3.choice(rest, size=min(n_ex2 - n_near, len(rest)),
                                   replace=False)
                       if len(rest) and n_near < n_ex2 else [])
                pick = list(near) + list(rnd)
            else:
                pick = []
            for e in range(n_ex2):
                ax = axes[ri][1 + e]
                ax.set_xticks([]); ax.set_yticks([])
                if e >= len(pick):
                    ax.axis("off"); continue
                ax.imshow(np.asarray(dd2[int(pick[e])])[0], cmap="gray",
                          interpolation="nearest")
                for sp in ax.spines.values():
                    sp.set_color(col); sp.set_linewidth(1.2)
                if ri == 0:
                    ax.set_title(f"#{e + 1} closest" if e < n_near else "random",
                                 fontsize=7.5, color="#555555")
        how_ex = ("the 4 members nearest the cluster centroid"
                  if args.examples == "closest"
                  else "2 nearest the centroid, 2 random members")
        fig.suptitle(f"{label} — {b} band, k={args.k}\n"
                     "clusters WITHIN this band, rows ordered by texture — "
                     "look for a progression from washed-out to sharp-rimmed\n"
                     f"examples are {how_ex} (centroid = mean embedding)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.1)
        pp = os.path.join(args.map_dir, f"per_band_profile_{b}_k{args.k}.png")
        fig.savefig(pp, dpi=145, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {pp}")

    # (e) LATENT SEPARATION. One PCA basis fitted on ALL embeddings, so the
    #     four panels share coordinates and can be compared directly - fitting
    #     a separate PCA per band would give each its own axes and make the
    #     panels look different for reasons that have nothing to do with the
    #     clustering.
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    pca = PCA(n_components=2, random_state=0).fit(X)
    P = pca.transform(X)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    panels = [("west", None), ("mid", None), ("east", None), ("MATCHED (all bands)", None)]
    sil = {}
    for ax, (name, _) in zip(axes.ravel(), panels):
        if name.startswith("MATCHED"):
            lab_v, mask = merged, np.ones(len(X), bool)
        else:
            lab_v, mask = per, band == name
        sub = np.where(mask)[0]
        if len(np.unique(lab_v[sub])) > 1:
            samp = (rng3.choice(sub, 8000, replace=False)
                    if len(sub) > 8000 else sub)     # silhouette is O(n^2)
            sil[name] = float(silhouette_score(X[samp], lab_v[samp]))
        for c in sorted(np.unique(lab_v[sub])):
            q = sub[lab_v[sub] == c]
            ax.scatter(P[q, 0], P[q, 1], s=1.5, color=colors[int(c) % len(colors)],
                       alpha=0.45, linewidths=0, rasterized=True)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{name}  (n={len(sub):,}"
                     + (f", silhouette {sil[name]:.3f}" if name in sil else "")
                     + ")", fontsize=10)
    ev = pca.explained_variance_ratio_
    fig.suptitle(f"{label} — latent separation, k={args.k}\n"
                 f"shared PCA basis (PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%} of variance); "
                 "first three panels are each band's OWN clusters, last is after "
                 "matching", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    lp = os.path.join(args.map_dir, f"per_band_latent_k{args.k}.png")
    fig.savefig(lp, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {lp}   silhouettes: "
          + "  ".join(f"{k}={v:.3f}" for k, v in sil.items()))

    # (f) EACH BAND IN ITS OWN BASIS. The shared-basis figure above asks
    #     whether the panels are comparable; this asks whether a band's
    #     embedding has different STRUCTURE when considered alone - its own
    #     principal axes, its own variance profile, its own separation.
    #
    #     Both are needed and they are not substitutes. A band can look
    #     identical in the shared basis while having quite different intrinsic
    #     structure, and vice versa.
    #
    #     The decisive number is not either picture but the PRINCIPAL ANGLES
    #     between the bands' top subspaces: if the three bands span nearly the
    #     same directions in the 384-dim embedding, the encoder is doing the
    #     same thing everywhere and only the cluster boundaries differ; if the
    #     subspaces are rotated apart, the bands are genuinely encoded
    #     differently and no relabeling can reconcile them.
    from scipy.linalg import subspace_angles
    NC = 10
    fits, fig = {}, None
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    def _draw_ownbasis(ax, b, Pb, sub, ev, eff, sil, standalone=False):
        """One band's own-basis scatter. Drawn twice - once into the 3-panel
        comparison figure, once standalone so the app can put each band's PCA
        inside that band's own tab, next to the clusters it describes."""
        for c in sorted(np.unique(per[sub])):
            q = per[sub] == c
            ax.scatter(Pb[q, 0], Pb[q, 1],
                       s=6.0 if standalone else 1.5,
                       color=colors[int(c) % len(colors)],
                       alpha=0.5 if standalone else 0.45,
                       linewidths=0, rasterized=True,
                       label=f"cluster {int(c)}" if standalone else None)
        ax.set_title(f"{b} — its OWN basis\nPC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}, "
                     f"effective rank {eff:.1f}, silhouette {sil:.3f}",
                     fontsize=10)
        ax.set_xlabel("PC1 (this band)", fontsize=8)
        ax.set_ylabel("PC2 (this band)", fontsize=8)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for ax, b in zip(axes, BANDS):
        m = band == b
        pb_ = PCA(n_components=min(NC, m.sum()), random_state=0).fit(X[m])
        fits[b] = pb_
        Pb = pb_.transform(X[m])
        sub = np.where(m)[0]
        ev = pb_.explained_variance_ratio_
        # participation ratio: a smooth "how many directions actually matter"
        # that does not depend on picking a variance cutoff
        full = PCA(random_state=0).fit(X[m]).explained_variance_ratio_
        eff = float(1.0 / np.sum(full ** 2))
        # Silhouette is computed in the FULL embedding, not in the 2 PCs being
        # drawn: the picture is a projection and can make well-separated
        # clusters overlap on screen, so the number has to come from the space
        # the clustering actually happened in or it would flatter the plot.
        try:
            sil = float(silhouette_score(X[m], per[sub],
                                         sample_size=min(10000, int(m.sum())),
                                         random_state=0))
        except Exception:
            sil = float("nan")
        _draw_ownbasis(ax, b, Pb, sub, ev, eff, sil)

        f1, a1 = plt.subplots(figsize=(6.4, 5.6))
        _draw_ownbasis(a1, b, Pb, sub, ev, eff, sil, standalone=True)
        a1.legend(fontsize=6.5, markerscale=2.2, ncol=2, frameon=False,
                  loc="best")
        f1.suptitle(f"{label} — {b} band clustered and PCA'd ON ITS OWN, "
                    f"k={args.k}", fontsize=10)
        f1.tight_layout(rect=(0, 0, 1, 0.94))
        p1 = os.path.join(args.map_dir, f"per_band_ownbasis_{b}_k{args.k}.png")
        f1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(f1)
        print(f"  wrote {p1}   (silhouette {sil:.4f})")
    fig.suptitle(f"{label} — each band encoded and clustered ON ITS OWN, k={args.k}\n"
                 "separate PCA per band: axes are NOT comparable between panels, "
                 "which is the point — this shows each band's intrinsic structure",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    op = os.path.join(args.map_dir, f"per_band_ownbasis_k{args.k}.png")
    fig.savefig(op, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {op}")

    ang = {}
    for a_, b_ in (("west", "mid"), ("east", "mid"), ("west", "east")):
        th = np.rad2deg(subspace_angles(fits[a_].components_.T, fits[b_].components_.T))
        ang[f"{a_}_vs_{b_}"] = {"first": float(th.min()), "median": float(np.median(th)),
                                "max": float(th.max())}
        print(f"  principal angles {a_:>4} vs {b_:<4} (top-{NC} subspaces): "
              f"min {th.min():5.1f}°  median {np.median(th):5.1f}°  max {th.max():5.1f}°")
    print("  0 deg = the two bands span the same directions; 90 deg = orthogonal.")

    # ── make it LOOKABLE ────────────────────────────────────────────────────
    # The V numbers say band is less predictable; only a map shows whether the
    # right-hand band has stopped being its own patch of colour. Rendered with
    # the same basemap, palette and export machinery as every other map here,
    # so it drops straight into the app and QGIS beside them.
    from src.eval.geographic_map import plot_map, write_gpkg, raster_crs_wkt
    from src.eval.cluster_profile import write_qml
    bm = (bmi, bex)
    _, pw2 = raster_crs_wkt(_DEFAULT_BASEMAP)
    # Each band's OWN clustering as its own layer, so a band can be loaded and
    # styled independently in QGIS. Colours are that band's own ids and do not
    # correspond between bands - the .qml is the same table for all of them,
    # which is correct only WITHIN a band.
    for b in BANDS:
        m = band == b
        d3 = df[m].copy(); d3["cluster"] = per[m]
        gp = os.path.join(args.map_dir, f"per_band_clusters_{b}_k{args.k}.gpkg")
        write_gpkg(d3, gp, label=f"{label} {b}", k=args.k, crs_wkt=pw2)
        write_qml(os.path.splitext(gp)[0] + ".qml", args.k, colors)
        print(f"  wrote {gp}  (+ .qml)  n={int(m.sum()):,}")

    for tag, lab in (("global", glob), ("bandmatched", merged)):
        d2 = df.copy(); d2["cluster"] = lab
        png = os.path.join(args.map_dir, f"per_band_map_{tag}_k{args.k}.png")
        plot_map(d2, args.k, png, _DEFAULT_BASEMAP, basemap_cache=bm,
                 title=(f"{label} — k={args.k} — "
                        + ("ONE clustering over all bands"
                           if tag == "global" else
                           "clustered WITHIN each band, then matched across bands")
                        + f"  (Cramer's V vs band = "
                          f"{(v_glob if tag=='global' else v_merged):.3f})"))
        gp = os.path.join(args.map_dir, f"per_band_clusters_{tag}_k{args.k}.gpkg")
        write_gpkg(d2, gp, label=f"{label} {tag}", k=args.k, crs_wkt=pw2)
        write_qml(os.path.splitext(gp)[0] + ".qml", args.k,
                  cluster_colors_hex(args.k))
        print(f"  wrote {png}")
        print(f"  wrote {gp}  (+ .qml)")

    out = os.path.join(args.map_dir, f"per_band_clustering_k{args.k}.json")
    json.dump({"k": args.k, "v_global": v_glob, "v_per_band_unmatched": v_combo,
               "v_per_band_matched": v_merged, "mirror_used": flip_used,
               "silhouette": sil, "principal_angles_deg": ang,
               "band_counts": {b: int((band == b).sum()) for b in BANDS},
               "mapping": mapping}, open(out, "w"), indent=2)
    pd.DataFrame({"id": df["id"], "lon": df["lon"], "lat": df["lat"],
                  "band": band, "cluster_global": glob,
                  "cluster_in_band": per, "cluster_matched": merged}
                 ).to_csv(os.path.join(args.map_dir,
                                       f"per_band_clusters_k{args.k}.csv"), index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
