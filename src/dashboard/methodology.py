"""
methodology.py — exactly how every number in this dashboard is computed.

One module, so a metric is described in one place and the description sits
next to nothing that can drift from it. Every entry names the FUNCTION that
produces the number, so the text can be checked against the code rather than
trusted.

WHY THIS EXISTS
---------------
Several numbers in this project are called "the longitude effect" or "the
correlation" while being computed in genuinely different ways - different
sample (642 labeled craters vs 100,000), different grouping (k-means clusters
vs three longitude zones vs a continuous PC), and different statistic
(epsilon-squared vs Spearman rho vs a relative difference in means). Those are
not interchangeable, and reading one as another has already produced wrong
statements. Each entry below therefore states, explicitly:

    what is measured | on which sample | grouped/correlated how | which statistic

USAGE
    from dashboard import methodology as M
    M.explain("confound_correlation")        # renders a Streamlit expander
    M.text("confound_correlation")           # the raw markdown
"""
from __future__ import annotations

# Effect-size convention used throughout the project (Kruskal-Wallis
# epsilon-squared): 0.01 small / 0.06 medium / 0.14 large.
EFFECT_SIZE_BARS = "0.01 = small, 0.06 = medium, 0.14 = large"

METHODS: dict[str, dict] = {

    # ── the two "longitude effect" numbers, which are NOT the same thing ──

    "confound_correlation": {
        "title": "Confound η² (longitude, latitude, brightness, sobel, diameter, saturation)",
        "source": "src/eval/metrics.py :: confound_correlation()",
        "md": """
**What it answers:** do the clusters the model drew differ systematically in some
property that is *not* degradation?

**Sample:** the labeled evaluation set (642 craters for the full test set, 321 for
`new_set_final_test`).

**Grouping:** by the **raw k-means cluster assignment** (`group_col="cluster"`) —
*not* by the true label, and *not* by the Hungarian-aligned predicted class. So
this asks "are the groups the model invented geographically/photometrically
structured", which is the question that matters for a confound.

**Statistic:** Kruskal-Wallis H (a non-parametric one-way ANOVA on ranks) across
the cluster groups, converted to an effect size

```
eta_sq = (H - k + 1) / (n - k)
```

where `k` = number of clusters, `n` = number of craters. Bars: **{bars}**.

**Naming caveat:** the output field is `eta_sq`, but this formula is
**epsilon-squared** — the bounded, H-based analogue of ANOVA's η². Read it as
"share of rank variance attributable to cluster membership".

**Because it is rank-based, any positive rescaling of a feature leaves it
unchanged.** That matters here: the intensity features are multiplied by 255
(see *Intensity features*), and that factor cannot affect these numbers.

**Features tested** come from two sources — `image_intensity_features()`
(brightness, sobel_mean, frac_saturated, straight off the crop pixels) and
`robbins_lookup_features()` (lat, lon, diam_km, looked up from the Robbins 2018
catalog by crater ID).
""",
    },

    "seam_zone_comparison": {
        "title": "Seam / consistency jumps across longitude zones",
        "source": "src/eval/seam_check.py :: per_crater_stats(), zone_of(), compare_zones()",
        "md": """
**What it answers:** do craters near a mosaic tile boundary look measurably
different from craters in the middle of the retained region?

**Sample:** the **100,000-crater** geographic-map set — not the labeled 642. A
completely different sample from the confound η² above.

**Grouping:** three fixed longitude zones, by distance to the edges of the
retained band (`_SEAMS = (-90.0, 180.0)` in signed longitude, i.e. 270°E and
180°E), with a default `edge_width` of **15°**:

- `left-seam`   : lon ≤ −90 + 15
- `right-seam`  : lon ≥ 180 − 15
- `middle`      : everything else

Measured zone counts: middle 89,024 / right-seam 7,092 / left-seam 3,884.

**Measures, computed per crater off the memmap:**

- `brightness` = mean pixel value of the crop
- `contrast` = **mean absolute horizontal neighbour difference**,
  `|diff(crop, axis=-1)|.mean()` — a cheap stand-in for Sobel magnitude that
  answers the same question (is this crop crisp or washed out)
- `diam` = crater diameter in metres

**Statistic:** the same Kruskal-Wallis epsilon-squared as the confound test, so
the two are read on the same scale.

**TWO DIFFERENT QUANTITIES, DO NOT CONFLATE THEM.** Measured on the frozen
off-the-shelf DINOv2 set:

| measure | left-seam | middle | right-seam | H | η² |
|---|---|---|---|---|---|
| contrast | 0.00926 | 0.01152 | 0.01294 | 980.1 | **0.0098** |
| brightness | 0.46685 | 0.46094 | 0.45879 | 42.2 | 0.0004 |
| diam (m) | 2018 | 2065 | 2284 | 230.7 | 0.0023 |

- As a **relative difference in means**, contrast is **−19.6%** left-vs-middle and
  **+12.4%** right-vs-middle (−28.4% left-vs-right). That is a large, visible step.
- As **variance explained**, η² = 0.0098 — *small* on the bars above.

Both are true. Zone means differ by ~20% of the mean, while zone membership
explains only ~1% of the total variance, because within-zone spread is large.
`p = 1.5e-213` reflects `n = 100,000`, not importance: at this sample size a
negligible effect is still overwhelmingly significant. **Quote the relative
step or the η², and say which.**
""",
    },

    "latent_geo_correlation": {
        "title": "Latent-space vs geography (PC1/PC2 against lat/lon)",
        "source": "src/eval/metrics.py :: latent_geo_correlation()",
        "md": """
**What it answers:** does the embedding itself vary *smoothly* with geography,
independent of where cluster boundaries happen to fall?

**It is computed on the PRINCIPAL COMPONENTS, not the raw latent space.**
`PCA(n_components=2, random_state=0)` is fitted to the embedding matrix, and
each of `pc1`, `pc2` is then correlated against `lat` and `lon` separately —
giving `pc1_vs_lat`, `pc1_vs_lon`, `pc2_vs_lat`, `pc2_vs_lon`.

PC1 specifically is the axis `latent_separation.png` plots on its x-axis, so a
strong `pc1_vs_lon` means the picture you are looking at is partly a map.

**Statistic: SPEARMAN rank correlation** (ρ), not Pearson. Rank-based to match
this module's non-parametric convention, so it assumes no linear relationship
and tolerates outlier craters. Reported as `r` in the output — but it is a
Spearman ρ.

**Sample:** only craters whose ID matches the Robbins format `NN-N-NNNNNN`,
since lat/lon are looked up from the Robbins 2018 catalog. Returns `None`
entirely for sets with non-Robbins IDs (Julie's set uses sequential filename
indices).

**How it complements the confound test:** the confound η² asks a *discrete*
question (do the k-means groups differ in longitude?); this asks a *continuous*
one (does the embedding drift with longitude?). A model can score low on one and
high on the other.
""",
    },

    # ── the representation-quality metrics ──

    "silhouette": {
        "title": "Latent separation (silhouette) and purity",
        "source": "src/eval/metrics.py :: separation()",
        "md": """
**`silhouette_true`** — `sklearn.metrics.silhouette_score(latents, true_label)`.
Computed on the **full latent space**, not on the PCA projection, and against the
**true labels**. For each crater: (mean distance to other classes' members −
mean distance within its own class) / max of the two, averaged over all craters.
Range −1 to +1; 0 means classes overlap completely.

This is the primary "does the representation organize itself by degradation"
number, because it needs no classifier, no cluster alignment, and no threshold.

**`silhouette_cluster`** — the same score against the **k-means cluster** labels
instead. It measures how well-separated the clusters are *as geometry*,
regardless of whether they correspond to the classes. A high cluster silhouette
with a low true silhouette means the model found crisp structure that is not
degradation.

**`purity`** — from the alignment step (see *Cluster→class alignment*): the share
of craters whose aligned cluster matches their true class.

**Scale note:** silhouette values in this project are small (typically 0.04–0.13).
That is the honest range for 4 overlapping ordinal classes in a 384-dim
embedding, not a bug.
""",
    },

    "map": {
        "title": "mAP (mean average precision)",
        "source": "src/eval/metrics.py :: mean_average_precision(), _class_probabilities()",
        "md": """
**What it answers:** how well does the model *rank* craters by class membership —
independent of any decision threshold?

**Where the scores come from.** There is no classifier head. Each crater has a
distance to every k-means centroid; those become soft scores:

```
Z = softmax(-(D - D.min(axis=1)))        # per-crater, over clusters
P[:, class] = sum of Z over clusters assigned to that class
```

i.e. a **softmax over negative distance to centroid**, with mass summed across
whichever clusters the Hungarian step mapped to each class.

**Statistic:** per class, `sklearn.metrics.average_precision_score` of
"is this crater class c?" against that class's score — the area under the
precision-recall curve — then **averaged unweighted across classes** (macro).
Classes with zero true members are skipped.

**Why it is reported alongside accuracy:** accuracy only scores the argmax
decision. mAP scores the whole ranking, so a model that puts genuinely degraded
craters near the top of its "degraded" list scores well even with no usable
threshold. It is also far less flattered by the 63%-majority class imbalance.

**Precondition:** requires distance-to-centroid columns; returns `None` without
them (the same precondition as ECE).
""",
    },

    "clustering_agreement": {
        "title": "ARI / NMI / V-measure",
        "source": "src/eval/metrics.py :: clustering_agreement()",
        "md": """
Agreement between the **aligned predicted labels** and the true labels:

- **ARI** — `adjusted_rand_score`. Chance-corrected: 0 means "no better than a
  random partition of the same shape", 1 means identical. Can go negative.
- **NMI** — `normalized_mutual_info_score`. Shared information, normalized to
  [0, 1]. Not chance-corrected, so it reads higher than ARI at equal quality.
- **V-measure** — the harmonic mean of homogeneity and completeness.

These are computed on `true_label` vs `pred_label`, so they *do* depend on the
alignment step; the silhouette scores do not.
""",
    },

    "alignment": {
        "title": "Cluster→class alignment (Hungarian)",
        "source": "src/eval/align.py",
        "md": """
k-means produces unlabelled clusters. To score them against classes, each cluster
must be assigned a class. When `n_clusters == n_classes`, this project uses
**one-to-one Hungarian assignment** (`scipy.optimize.linear_sum_assignment`)
maximizing total agreement — reported as `alignment: hungarian_1to1`.

One-to-one matters: the looser alternative (assign each cluster to whichever
class is most common in it) can map several clusters onto the same class,
inflating accuracy by letting the model avoid ever predicting a hard class.
Hungarian forbids that, so every class must be predicted by exactly one cluster.

If `n_clusters != n_classes`, the code falls back to many-to-one — which is why
`n_clusters` is derived per labeled set from that set's own class count rather
than from the config's top-level default.
""",
    },

    "bootstrap_ci": {
        "title": "Confidence intervals",
        "source": "src/eval/metrics.py :: bootstrap_ci()",
        "md": """
**Percentile bootstrap**, default **2,000 resamples**, seed 0.

Craters are resampled **with replacement** to the same sample size, the metric is
recomputed on each resample, and the interval is the **2.5th and 97.5th
percentiles** of the resulting distribution.

**These are not standard deviations, and not standard errors.** They describe how
much the metric moves when the *sample of craters* changes. They do not capture
model-training variance (a different seed for the same model is not resampled
here), so two checkpoints with overlapping intervals are "not separated by this
test", which is weaker than "equivalent".

**Applies to:** accuracy, QWK, ordinal MAE, macro-F1, mAP. Not to silhouette,
ARI/NMI, or the confound η² values, which are reported as point estimates.

**Cross-validation intervals are a different thing.** Where a table reports a CI
across CV folds (e.g. the catalog probe), that describes split variability, not
sampling variability, and the two should not be compared.
""",
    },

    "qwk_ordinal": {
        "title": "QWK and ordinal MAE",
        "source": "src/eval/metrics.py :: quadratic_weighted_kappa(), ordinal_mae()",
        "md": """
The four classes are **ordinal** (v-fr → r-fr → r-dr → v-dr, freshest to most
degraded), so confusing neighbours is a smaller error than confusing extremes —
something accuracy and macro-F1 cannot see.

- **QWK** — Cohen's kappa with quadratic weights: an error is penalized by the
  *square* of the distance between true and predicted class index, then
  chance-corrected. 1 = perfect, 0 = chance-level.
- **ordinal MAE** — mean absolute difference between true and predicted class
  *index*. In units of classes, so 0.5 means "off by half a class on average".

Class order comes from each labeled set's own `class_names`, index 0 = freshest
(`contract.LabelScheme`).
""",
    },

    "ece": {
        "title": "ECE (expected calibration error)",
        "source": "src/eval/metrics.py :: ece()",
        "md": """
Confidence (the max class probability from the same softmax-over-negative-
distance scores mAP uses) is binned into **10 equal-width bins on [0, 1]**. In
each bin, |accuracy − mean confidence| is computed, and ECE is the
**sample-weighted average** of those gaps.

0 = perfectly calibrated. Note the "confidence" here is derived from k-means
distances, not from a trained probabilistic head, so it should be read as "is
distance-to-centroid a usable confidence proxy" rather than as a claim about a
calibrated classifier.
""",
    },

    # ── the data-side computations ──

    "intensity_features": {
        "title": "Intensity features (brightness, sobel, saturation)",
        "source": "src/eval/metrics.py :: image_intensity_features()",
        "md": """
Computed **directly from each crater's display crop**, with no metadata
dependency:

- **brightness** = `img.mean()`
- **sobel_mean** = mean of `sqrt(gx² + gy²)`, with `cv2.Sobel(..., ksize=3)` —
  a texture/sharpness proxy
- **frac_saturated** = share of pixels `>= 254.0`

**Scale caveat, important for the catalog comparison.** For a PNG-based set the
crop is 8-bit, so these are on [0, 255]. For a memmap-based set the crop is a
`.npy` float tensor and the code applies `np.load(path)[0] * 255.0` — which
assumes the stored values are in **[0, 1]**.

That assumption held for the historical `global` normalization. It does **not**
hold for `zscore`, `robust`, or `asinh`, whose values are centred near 0 and can
be negative or exceed 1. Consequences:

- **brightness** and **sobel_mean** stay valid for the confound tests, because
  Kruskal-Wallis is **rank-based** and a positive scale factor cannot change
  ranks. Their absolute values are not comparable across normalizations.
- **frac_saturated** does **not** mean saturation for those normalizations. It
  becomes "share of pixels above 254/255 ≈ 0.996 in normalized units", whose
  meaning differs per normalization. Do not compare it across `norm` values.
""",
    },

    "clip_by_class": {
        "title": "Clipped-pixel share and its class dependence",
        "source": "src/eval/processing_compare.py :: norm_stats_for_variant(), _eta_sq_kruskal()",
        "md": """
**What it answers:** does a normalization destroy pixels *in proportion to the
label*?

**Per class:** the share of that class's pooled pixels falling **below the low
bound or above the high bound** of the normalization window, before clipping
maps them onto the bounds. Zero by construction for `zscore`, `robust` and
`asinh`, which have no bounds.

**Effect size:** the clipped fraction is computed **per crater**, then
Kruskal-Wallis is run across the four class groups with the same
`(H - k + 1)/(n - k)` epsilon-squared as everywhere else. Bars: **{bars}**.

**Constants:** fitted on the same 642 labeled craters being measured, so every
variant is treated identically and the comparison is self-contained. A real
training run instead uses the parent catalog's `norm_stats.json`, fitted on its
100,000-crater sample — and the two differ substantially (measured skew 3.53 on
the labeled set vs 0.708 on the catalog), because the labeled craters were
chosen for visual review and are larger and fresher than the population.

**Stored saturation** is a separate column: the share of pixels sitting on
*exactly* the stored product's min or max. A high-pass residual is a difference
of two smooth fields and should have no value spike at its extremes, so any
nonzero value there means an upstream step clipped the product itself.
""",
    },

    "terrain_shortcut": {
        "title": "Terrain identifiability from brightness",
        "source": "ad-hoc analysis (see the Data Processing page)",
        "md": """
**What it answers:** how much of the maria/highland albedo difference survives a
given processing choice — i.e. how large a non-morphological shortcut is on offer.

**Method:** a logistic regression predicting the labels file's `area` column
(`mare` vs `highland`) from a **single feature**, the crop's mean value, taken
from each catalog testset's `index.csv`. Scored by accuracy under
**stratified 5-fold CV × 4 seeds**, with `class_weight="balanced"` and a
`StandardScaler`. Chance is 0.508 (the majority terrain share).

**Result:** 0.921 (morph_raw) and 0.946 (emp_raw) versus 0.583–0.620 for the
σ-filtered variants. The high-pass filter removes most of the albedo signal, as
designed.

**Why it matters:** terrain type is itself associated with the degradation label
— χ² = 20.7, p = 1.2e-4, Cramér's V = 0.18 on the 642 craters, with highlands
holding 11.0% very-fresh craters against mare's 4.0%. So absolute brightness
carries real label information through a route that has nothing to do with
crater shape.
""",
    },

    "supervised_probe_caveat": {
        "title": "Why probe accuracy is not the headline",
        "source": "src/eval/catalog_probe.py (superseded by the eval suite)",
        "md": """
A supervised linear probe was used at one point to rank catalogs. It should be
read with care, and it is **not** the primary instrument.

**What it did:** multinomial logistic regression on `StandardScaler`'d frozen
embeddings, stratified 5-fold × 4 seeds, `class_weight="balanced"`, scored by
macro recall.

**Why it misleads here:** it *fits a classifier* to whatever information is
present, however it is arranged. The project's question is whether the
representation **organizes itself** by degradation without supervision. On the
same data and the same encoder the probe read **accuracy 0.759 / QWK 0.833**
where the unsupervised suite read **0.530 / 0.658** — an inflation of ~0.23
accuracy. It also cannot report mAP, ARI/NMI, silhouette, or any confound
correlation.

**And it is beatable by trivial statistics.** Three hand-crafted scalars per
crop (mean, std, mean absolute horizontal neighbour difference) reach 0.624–0.657
macro recall against the frozen embedding's 0.635–0.659 — on `emp_raw` the
scalars *win*. So probe scores largely track how much intensity-statistic signal
a variant carries, not representation quality.

Prefer `silhouette_true`, `map` and `ari`, each read next to the confound
columns.
""",
    },

    "separation_tradeoff": {
        "title": "Why separation and the longitude artifact must be read together",
        "source": "aggregate over logs/**/eval_results/**/summary.json",
        "md": """
Across **67 unique (labeled set × embedding) evaluations** already on disk:

```
corr(lon_eta, silhouette_true) = +0.248   p = 0.043   n = 67
```

Pearson correlation across runs. Representations with **more** longitude
confound score **slightly better** class separation — part of the apparent class
structure is riding on the geographic artifact.

**Consequence for reading results:** an intervention that genuinely removes the
artifact should be expected to *lower* `silhouette_true` somewhat. That is the
confound leaving, not the method failing. Any table that reports one column
without the other can be read to support the opposite conclusion, which is why
they are always shown side by side and never combined into a single score.

Recorded range across those runs: `lon_eta` 0.018–0.343, median 0.196. By family,
MAE averages 0.051 and DINO 0.212 at essentially equal silhouette (0.080 vs
0.084) — though with n=7 vs n=60 and differing sets, that is suggestive rather
than controlled.
""",
    },

    "catalog_storage": {
        "title": "How the catalogs store data (and what is applied when)",
        "source": "src/data/build_catalog.py, crater_catalog_dataset.py, norm_stats.py",
        "md": """
**Stored:** RAW values, exactly as read from the source mosaic, at **native
per-crater size** (span = diameter × (1 + fov)), as a concatenated C-order
float32 blob plus an `index.csv` carrying byte offset, height, width, lat, lon,
diameter and per-crater raw statistics.

Nothing is normalized on disk. That is deliberate: it keeps every clipping and
scaling decision out of the only stored copy, so changing it is a loader edit
rather than a multi-GB rebuild.

**Geometry caveat:** `crop_crater()` stretches each crop by **1/cos(lat)** to undo
equirectangular distortion. This is a geometric interpolation applied even in
"raw" mode — the pixels are raw in *value*, not untouched.

**Applied at load time, in this order** (`crater_catalog_dataset.get_image_data`):

1. read the native-size raw crop
2. **resize** to the model's input size (`cv2.INTER_CUBIC`)
3. **normalize** via `norm_stats.apply()`

**Resize precedes normalize deliberately.** The affine part of normalization
commutes with interpolation (both linear), but **clipping does not**: clipping to
[0,1] first and then resizing lets cubic overshoot push values back outside the
range — measured −0.0097 to 1.0281 when done the other way round. Resizing first
makes the clip the final operation, so the stated range is actually guaranteed.

**Constants** come from `norm_stats.json`, fitted on a 20,000-crater sample of the
**parent catalog**. A `testset/` inherits its parent's constants via
`stats_from`, so evaluation data never defines its own scaling.
""",
    },

    "normalizations": {
        "title": "The normalizations, exactly",
        "source": "src/data/norm_stats.py :: fit(), apply()",
        "md": """
All constants are fitted on a **20,000-crater subsample** (craters, not pixels —
matching the historical `fit_global_scaling.py` convention so percentiles stay
comparable), with pixels of non-finite or sentinel magnitude (|v| ≥ 1e30)
excluded.

| norm | formula | clips? |
|---|---|---|
| `none` | *x* | no |
| `global` | (x − p1) / (p99 − p1), then clip to [0,1] | **yes** |
| `global_wide` | (x − p0.01) / (p99.99 − p0.01), then clip to [0,1] | yes, ~inert |
| `zscore` | (x − mean) / std | no |
| `robust` | (x − median) / (1.4826 · MAD) | no |
| `asinh` | asinh( robust(x) / softening ), softening = 1.0 | no |

**MAD** = median absolute deviation, `median(|x − median|)`, computed on a
subsample capped at 20 million pixels. The **1.4826** factor makes it a
σ-equivalent for Gaussian data, so `robust` and `zscore` agree on well-behaved
data and diverge only where tails are heavy.

**`tail_ratio_std_over_madstd`** = std / (1.4826 · MAD) is recorded as a
diagnostic: 1.00 for Gaussian data, higher with heavier tails. It answers whether
an affine map suffices or a compression is warranted. Measured 1.86 (raw
morphology) and 2.45 (raw EMP) on the labeled set.

**`asinh`** is linear for |z| < 1 and logarithmic beyond (asinh(33) ≈ 4.19). Its
essential property versus clipping is that it is **strictly monotone** — the
brightest pixel stays brightest, so the ordering survives instead of collapsing
onto a single saturated value. Standard practice for this in astronomy (Lupton
et al. 1999, the SDSS asinh magnitude).
""",
    },

    "highpass": {
        "title": "The Gaussian high-pass filter",
        "source": "src/data/full_mosaic_processing.py, src/data/highpass_mosaic.py",
        "md": """
```
output = image - gaussian_blur(image, sigma)
```

σ is in **pixels**; at ~100 m/pixel, σ=100 → **10 km** and σ=200 → **20 km**.

**Purpose:** remove the regional albedo difference between dark basaltic maria and
bright anorthositic highlands, so the same crater cut from each does not arrive at
a different absolute brightness. Measured effect: terrain identifiability from
crop mean brightness drops from 0.92–0.95 to 0.58–0.62 (chance 0.508).

**What it cannot remove.** Subtracting a local mean removes an *additive offset*.
The tile-boundary artifact is largely a *multiplicative* difference — a contrast
or gain difference, because adjacent tiles were imaged at different incidence
angles. Subtracting a mean does not change a gain, so the seam survives this
filter by construction, which is what the before/after measurements show. A
divisive step (dividing by a local standard deviation) is the matched tool for
that, and is not currently applied.

**The clip that used to be here, and is not any more.** Both filter
implementations previously ended with an unconditional clip of the residual to
the **whole mosaic's 0.1/99.9 percentiles**. Those bounds are set by the smooth
terrain that is most of the Moon, while craters are the high-contrast minority —
so crater rims and shadows were flattened onto the bounds and **stored that way**.
Share of pixels pinned to exactly a bound, on the 642 labeled craters:

| product | overall | v-fr | r-fr | r-dr | v-dr |
|---|---|---|---|---|---|
| morphology σ100 (legacy) | 0.99% | **6.04%** | 2.24% | 0.16% | **0.00%** |
| EMP σ100 (legacy) | **11.87%** | — | — | — | — |
| unfiltered raw | 0.0003% | | | | |

Monotonic in degradation and exactly zero for the most degraded class: the clip
removed signal in direct proportion to crater freshness. It is now off
(`clip_pct=None`), a saturation gate refuses to save a product whose pixels pile
up at its extremes, and the rebuilt products measure 2 pixels at the extremes
instead. Legacy products are retained under `data/raw/**/sigma_legacyclip/`.
""",
    },

    "sources": {
        "title": "The two source mosaics",
        "source": "LROC WAC product documentation",
        "md": """
Both are the **same instrument** — the Wide Angle Camera on NASA's Lunar
Reconnaissance Orbiter — at ~**100 m/pixel**, covering **±60° latitude** (the
mosaic bounds are ±1,819,401 m, exactly ±60.0°) and all longitudes.

**`morph_*` — WAC_GLOBAL morphology mosaic.** Assembled from three observation
windows chosen to hold incidence angle at roughly **55–75°** — low sun, long
shadows, strong topographic shading. Built for morphologic interpretation and
deliberately *retains* the shading the degradation cue rides on. Cost: brightness
depends on when and where each pixel was imaged, producing tile-boundary steps.

**`emp_*` — WAC_EMP_643 empirically normalized mosaic.** Every pixel
photometrically corrected to one fixed geometry — **incidence 30°, emission 0°,
phase 30°** — using GLD100 topography, then median-stacked over ~142 observations
(Boyd et al. 2012). The 643 nm band at 304 pixels/degree. Removes much of the
acquisition-time effect, but incidence 30° is near-overhead sun, which *minimizes*
topographic shading.

**Why not WAC_HAPKE.** The Hapke-normalized product (Sato et al. 2014) is more
principled photometrically but is published only at **400 m/pixel**, and its
Hapke parameters were fitted **per 1°×1° tile** (~30 km), so the correction itself
only varies at that scale. Serving it finer would interpolate a coarse correction
rather than add information. Rejected on resolution grounds.

**The 642 craters.** The labels file holds 770; 128 lie beyond ±60° latitude and
so fall outside every mosaic. 770 − 128 = 642, identically for every variant.
""",
    },
}


def text(key: str) -> str:
    m = METHODS.get(key)
    if m is None:
        return f"*No methodology entry for `{key}`.*"
    return m["md"].format(bars=EFFECT_SIZE_BARS).strip()


def explain(key: str, expanded: bool = False) -> None:
    """Render one methodology entry as a Streamlit expander."""
    import streamlit as st
    m = METHODS.get(key)
    if m is None:
        st.caption(f"No methodology entry for `{key}`.")
        return
    with st.expander(f"How this is calculated — {m['title']}", expanded=expanded):
        st.markdown(text(key))
        st.caption(f"Source: `{m['source']}`")
