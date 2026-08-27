"""
prototypical_loss.py — architecture-agnostic prototypical-network auxiliary
loss (Snell et al. 2017, "Prototypical Networks for Few-Shot Learning"),
for injecting a small amount of label supervision into an otherwise
self-supervised training loop without adding any separate, persistent
classifier weights: a class prototype is just the mean of that class's
support embeddings, so the only parameters that ever receive gradients
from this loss are embed_fn's own (the backbone being trained) - there is
no separate classifier head that would cap the model to a fixed class
count the way an LDA/k-NN head would (see session discussion - that's
exactly why this approach was chosen over eval/head.py's heads for this
purpose: those measure whether adapting frozen features helps, this
actually shapes the embedding space itself while training).

Deliberately architecture-agnostic (embed_fn is the only backbone-specific
piece) so the same function can hook into DINO's self-distillation loop
(src/models/dinov2/dinov2/train/ssl_meta_arch.py) and, if this proves
useful, MAE's reconstruction loop (src/train/train.py) later, without
duplicating the episode-sampling/prototype math in two places.

Data: draws from configs/new_test_set_split.csv's "train_pool" rows only
(321 craters, stratified half of the new test set) via LabeledEpisodePool
below - the "final_test" half and Julie's set are never touched by
anything in this module, by construction (see eval/new_test_set_split.py's
docstring for why the split is fixed once and shared, not recomputed here).
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.eval.holdout import load_holdout_dat


def prototypical_loss(embed_fn, support_x: torch.Tensor, support_y: torch.Tensor,
                      query_x: torch.Tensor, query_y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    embed_fn: callable, image batch -> (N, D) embedding, differentiable
    w.r.t. whatever backbone parameters produced it (e.g. self.student.backbone
    for DINO, or a CLS-token extraction closure for MAE later).
    support_x/query_x: image tensors, already at the model's expected input
    size/channels. support_y/query_y: integer class indices (0..n_classes-1),
    same convention as this project's true_label columns everywhere else.

    Caller must ensure every class has >=1 support example in this episode
    (see sample_episode's stratification) - a class with zero support
    examples gets an all-NaN prototype (mean of an empty tensor), which
    would poison the whole batch's loss.
    """
    support_emb = embed_fn(support_x)
    query_emb = embed_fn(query_x)

    prototypes = torch.stack([support_emb[support_y == c].mean(dim=0) for c in range(n_classes)])
    logits = -torch.cdist(query_emb, prototypes)
    return F.cross_entropy(logits, query_y)


def sample_episode(pool_x: torch.Tensor, pool_y: np.ndarray, n_classes: int,
                   n_support_per_class: int, rng: np.random.RandomState):
    """
    Fresh, stratified support/query split drawn from pool_x/pool_y - a NEW
    random split every call, not one fixed split reused every step
    ("episodic" sampling, standard practice in the few-shot literature, so
    prototypes see varied support sets across training instead of
    overfitting to one particular subset). Every class contributes exactly
    n_support_per_class support examples (or all-but-one of them, if a
    class has fewer than that + 1 available) - everything else in the pool
    for that class becomes query.
    """
    support_idx, query_idx = [], []
    for c in range(n_classes):
        class_idx = np.flatnonzero(pool_y == c)
        rng.shuffle(class_idx)
        n_sup = min(n_support_per_class, max(len(class_idx) - 1, 0))
        support_idx.extend(class_idx[:n_sup].tolist())
        query_idx.extend(class_idx[n_sup:].tolist())
    support_idx = torch.tensor(support_idx, dtype=torch.long)
    query_idx = torch.tensor(query_idx, dtype=torch.long)
    device = pool_x.device
    support_y_t = torch.as_tensor(pool_y[support_idx.numpy()], dtype=torch.long, device=device)
    query_y_t = torch.as_tensor(pool_y[query_idx.numpy()], dtype=torch.long, device=device)
    return pool_x[support_idx], support_y_t, pool_x[query_idx], query_y_t


class LabeledEpisodePool:
    """Loads configs/new_test_set_split.csv's train_pool craters (from the
    wide-FOV new_test_set variant - see Snakefile's preprocess_new_test_set
    {fov} wildcard) once, keeps them in memory (321 craters x 128x128
    floats = ~21MB, trivial), and hands out fresh episodes on request.

    Resizes to crop_size and replicates the single grayscale channel if
    in_chans>1, matching CraterAugmentationDINO's own convention exactly
    (src/data/dino_craters_augmentation.py) - so the prototype loss sees
    images framed the same way the self-distillation objective's global
    crops do, not a mismatched preprocessing path."""

    def __init__(self, dat_path: str, metadata_csv: str, split_csv: str,
                class_names: list[str], crop_size: int = 224, in_chans: int = 3,
                device: str = "cuda"):
        split = pd.read_csv(split_csv)
        pool_ids = set(split.loc[split["split"] == "train_pool", "crater_id"])

        meta = pd.read_csv(metadata_csv)
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        labels_by_id = dict(zip(split["crater_id"], split["degree"].map(class_to_idx)))

        row_idx = [i for i, crater_id in enumerate(meta["id"]) if crater_id in pool_ids]
        if not row_idx:
            raise ValueError(f"no train_pool craters found in {metadata_csv} - "
                            f"does it match {split_csv}'s crater_id set?")

        data = load_holdout_dat(dat_path)
        imgs = torch.from_numpy(np.asarray(data[row_idx])).float()  # (N, 1, 128, 128), already [0,1]
        imgs = F.interpolate(imgs, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        if in_chans > 1 and imgs.shape[1] == 1:
            imgs = imgs.repeat(1, in_chans, 1, 1)
        self.pool_x = imgs.to(device)
        self.pool_y = np.array([labels_by_id[meta["id"].iloc[i]] for i in row_idx])
        self.n_classes = len(class_names)

        counts = {class_names[c]: int((self.pool_y == c).sum()) for c in range(self.n_classes)}
        print(f"LabeledEpisodePool: {len(row_idx)} train_pool craters loaded from {dat_path}, "
             f"per-class counts: {counts}")

    def sample(self, n_support_per_class: int, rng: np.random.RandomState):
        return sample_episode(self.pool_x, self.pool_y, self.n_classes, n_support_per_class, rng)
