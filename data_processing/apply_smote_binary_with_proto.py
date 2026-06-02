#!/usr/bin/env python3
"""Balance the reduced binary dataset using SMOTE.

SMOTE is applied to the benign class until both binary classes contain the
requested number of samples. The protocol field is encoded numerically before
resampling.

Example:
    python data_processing/apply_smote_binary_with_proto.py \\
        --input results/datasets/dataset_binary_reduced_before_smote.csv \\
        --output results/datasets/dataset_binary_balanced_100k_per_class_withproto.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE


FINAL_CLASSES = ["Benign", "Attack"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input reduced binary CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output balanced binary CSV.")
    parser.add_argument("--target", type=int, default=100_000,
                        help="Target number of rows per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def protocol_to_integer(series: pd.Series) -> pd.Series:
    """Encode supported protocol names as their IP protocol numbers."""
    return (
        series.astype(str).str.strip().str.lower()
        .map({"tcp": 6, "udp": 17, "icmp": 1})
        .fillna(-1)
        .astype("int16")
    )


def main() -> None:
    """Create the balanced binary dataset."""
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    dataframe = pd.read_csv(args.input, sep=";")
    for required_column in ("label", "proto"):
        if required_column not in dataframe.columns:
            raise ValueError(f"Required column not found: {required_column}")

    dataframe["label"] = dataframe["label"].astype(str).str.strip()
    dataframe = dataframe[dataframe["label"].isin(FINAL_CLASSES)].copy()
    dataframe["proto"] = protocol_to_integer(dataframe["proto"])

    features = dataframe.drop(columns=["flow_id", "capture", "label"], errors="ignore")
    labels = dataframe["label"]
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("[INFO] Initial class distribution:")
    print(labels.value_counts())

    smote = SMOTE(
        sampling_strategy={"Benign": args.target},
        random_state=args.seed,
        k_neighbors=3,
    )
    resampled_features, resampled_labels = smote.fit_resample(features, labels)

    balanced = pd.DataFrame(resampled_features, columns=features.columns)
    balanced["label"] = resampled_labels

    final_parts = []
    for class_name in FINAL_CLASSES:
        class_rows = balanced[balanced["label"] == class_name]
        if len(class_rows) < args.target:
            raise RuntimeError(
                f"Class {class_name} has {len(class_rows)} rows after SMOTE; "
                f"{args.target} were requested."
            )
        if len(class_rows) > args.target:
            class_rows = class_rows.sample(n=args.target, random_state=args.seed)
        final_parts.append(class_rows)

    final_dataset = pd.concat(final_parts, ignore_index=True)
    final_dataset = final_dataset.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(args.output, sep=";", index=False)

    print("[INFO] Final class distribution:")
    print(final_dataset["label"].value_counts())
    print(f"[INFO] Balanced binary dataset created: {args.output}")


if __name__ == "__main__":
    main()
