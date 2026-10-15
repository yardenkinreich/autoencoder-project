"""
supcon_loss.py — ordinal supervised-contrastive auxiliary loss for the DINOv2
finetune (configs/dino_craters_finetune.yaml `supcon:` section; wired into
ssl_meta_arch.SSLMetaArch exactly like the prototypical loss).

WHY, ON TOP OF / INSTEAD OF THE PROTOTYPICAL LOSS
-------------------------------------------------
The prototypical loss compares each query crater with 4 class means. A
supervised-contrastive loss (Khosla et al. 2020) compares every labelled
crater in the batch with every other one, so each labelled crater is used
many times per step and same-class craters are pulled together directly -
the property the k-means evaluation rewards.

ORDINAL WEIGHTING
-----------------
The classes are a scale, v-fr < r-fr < r-dr < v-dr, and plain SupCon treats
v-fr-vs-v-dr like v-fr-vs-r-fr. Here the target over the other craters in the
batch is a label-distance kernel instead of a same-class indicator
(the "y-aware" InfoNCE of Dufumier et al. 2021, on class indices):

    t_ij  ∝  exp(-(y_i - y_j)^2 / (2 sigma^2))        (j != i)
    loss  =  mean_i  sum_j  -t_ij * log softmax_j(sim_ij / tau)

sigma=0.75 -> weights 1 (same class), 0.41 (adjacent), 0.03 (two apart),
~0 (three apart): neighbours stay close, far classes are pushed apart.
sigma=0 reduces exactly to standard SupCon.

DATA
----
Draws ONLY train_pool craters of the split (configs/liorr_labels_split.csv),
via the same LabeledEpisodePool the prototypical loss uses - never final_test.
Batches are class-balanced (per_class craters of every class, with
replacement for the rarest) and every crater appears in TWO augmented views,
so each anchor always has a same-crater positive.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def ordinal_supcon_loss(emb: torch.Tensor, y: torch.Tensor, tau: float = 0.1,
                        sigma: float = 0.75) -> torch.Tensor:
    """emb: (N, D) embeddings (normalized here). y: (N,) integer class ids."""
    z = F.normalize(emb.float(), dim=1)
    logits = z @ z.t() / tau
    n = len(y)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, -1e9)          # never your own sample
    log_p = F.log_softmax(logits, dim=1)
    d = (y[:, None] - y[None, :]).float()
    w = (d == 0).float() if sigma <= 0 else torch.exp(-d.pow(2) / (2 * sigma ** 2))
    w = w.masked_fill(self_mask, 0.0)
    keep = w.sum(1) > 0                                    # anchors with any positive
    t = w[keep] / w[keep].sum(1, keepdim=True)
    return -(t * log_p[keep]).sum(1).mean()


def _augment(x: torch.Tensor, gen: torch.Generator, zoom_min: float,
             jitter: float) -> torch.Tensor:
    """Crater-preserving GPU augmentation of a (B, C, H, W) batch: rot90 +
    flips, a random centred zoom between the full image (image_fov, 2xD)
    and zoom_min of its side (0.5 -> the 1xD crater), and mild contrast /
    brightness jitter in the catalog's normalized units."""
    b, c, h, w = x.shape
    dev = x.device
    out = torch.empty_like(x)
    k = torch.randint(0, 4, (b,), generator=gen, device="cpu")
    fl = torch.rand(b, generator=gen) < 0.5
    fr = torch.rand(b, generator=gen, device="cpu") * (1 - zoom_min) + zoom_min
    for i in range(b):
        s = max(8, int(round(h * float(fr[i]))))
        o = (h - s) // 2
        v = x[i:i + 1, :, o:o + s, o:o + s]
        if s != h:
            v = F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False,
                              antialias=False)
        v = torch.rot90(v, int(k[i]), dims=(2, 3))
        out[i] = torch.flip(v, dims=(3,))[0] if fl[i] else v[0]
    if jitter > 0:
        a = 1 + (torch.rand(b, 1, 1, 1, generator=gen) * 2 - 1).to(dev) * jitter
        s = (torch.rand(b, 1, 1, 1, generator=gen) * 2 - 1).to(dev) * jitter
        mu = out.mean(dim=(1, 2, 3), keepdim=True)
        out = (out - mu) * a + mu + s
    return out


class SupConSampler:
    """Class-balanced two-view batches from a LabeledEpisodePool's in-memory
    train_pool images (already at crop_size, channel-replicated, on GPU)."""

    def __init__(self, pool, per_class: int = 16, zoom_min: float = 0.5,
                 jitter: float = 0.15, seed: int = 0, views=None):
        """views: a labeled_views.TeacherStyleViews - if given, the two views
        are the teacher's context + zoom global crops (with its blur /
        solarize / jitter / flips, plus rot90) instead of _augment()."""
        self.views = views
        self.x, self.y = pool.pool_x, np.asarray(pool.pool_y)
        self.n_classes = pool.n_classes
        self.per_class, self.zoom_min, self.jitter = per_class, zoom_min, jitter
        self.rng = np.random.RandomState(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.by_class = [np.flatnonzero(self.y == c) for c in range(self.n_classes)]

    def sample(self):
        idx = np.concatenate([
            self.rng.choice(ix, self.per_class, replace=len(ix) < self.per_class)
            for ix in self.by_class if len(ix)])
        x = self.x[torch.as_tensor(idx, device=self.x.device)]
        y = torch.as_tensor(self.y[idx], dtype=torch.long, device=self.x.device)
        if self.views is not None:
            v1, v2 = self.views(x)
        else:
            v1 = _augment(x, self.gen, self.zoom_min, self.jitter)
            v2 = _augment(x, self.gen, self.zoom_min, self.jitter)
        return torch.cat([v1, v2]), torch.cat([y, y])
