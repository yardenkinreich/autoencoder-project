"""
new_test_set_split.py — a fixed, stratified 50/50 split of the new test set
(configs/correct_crater_with_labels_final.csv, materialized via
`snakemake preprocess_new_test_set`) into a training-episode pool and a
final held-out test set, for the planned prototypical-loss auxiliary
training (episodic support/query sampling drawn from the pool at each
training step - see session discussion) and its end-of-model evaluation.

Two disjoint sets, both fixed once and written to disk so every consumer
(the eventual training script's episodic sampler, and any final-eval
script) references the exact same split - no risk of the two drifting
apart or a script accidentally recomputing its own random split and
leaking test craters into training.

    train_pool  (321 craters) - ONLY ever read by the training loop's
                episodic support/query sampler. Never touched by anything
                that reports final numbers.
    final_test  (321 craters) - ONLY ever read once, at the very end, to
                report the in-distribution generalization numbers. Never
                seen during training or during any in-training monitoring.

Stratified by class (not a plain random shuffle) because new_test_set's
classes are badly imbalanced (v-fr=44, r-fr=88, r-dr=103, v-dr=407) - an
unstratified split could easily starve one side of the rarest class by
chance. A stratified 50/50 still only gives ~22 v-fr craters per side, so
aggregate metrics (accuracy, QWK) are trustworthy from this split; v-fr-
specific numbers should be read as lower-confidence, same caveat that
already applies to this class in the full 642-crater set.

Usage:
    PYTHONPATH=src python -m eval.new_test_set_split
    writes configs/new_test_set_split.csv (crater_id, degree, split)
"""
from __future__ import annotations
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from eval import pipeline as P

DEFAULT_METADATA_CSV = "data/processed_wac_100m_new/sigma/100/new_test_set/metadata.csv"
DEFAULT_LABELS_CSV = "configs/correct_crater_with_labels_final.csv"
DEFAULT_CLASS_NAMES = ["v-fr", "r-fr", "r-dr", "v-dr"]
DEFAULT_OUT = "configs/new_test_set_split.csv"


def build_split(metadata_csv: str = DEFAULT_METADATA_CSV, labels_csv: str = DEFAULT_LABELS_CSV,
                class_names: list[str] = DEFAULT_CLASS_NAMES, seed: int = 0) -> pd.DataFrame:
    labels = P.labels_from_memmap_csv(metadata_csv, labels_csv, class_names)
    train_idx, test_idx = train_test_split(
        labels.index, test_size=0.5, stratify=labels["true_label"], random_state=seed)

    out = labels[["crater_id", "true_label"]].copy()
    out["degree"] = out["true_label"].map(dict(enumerate(class_names)))
    out["split"] = "final_test"
    out.loc[train_idx, "split"] = "train_pool"
    return out[["crater_id", "degree", "split"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    ap.add_argument("--labels-csv", default=DEFAULT_LABELS_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = build_split(args.metadata_csv, args.labels_csv, seed=args.seed)

    print("Per-class counts by split:")
    print(split.groupby(["degree", "split"]).size().unstack(fill_value=0))
    print(f"\ntotal train_pool: {(split['split'] == 'train_pool').sum()}")
    print(f"total final_test: {(split['split'] == 'final_test').sum()}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    split.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
