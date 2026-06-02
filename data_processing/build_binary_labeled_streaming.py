#!/usr/bin/env python3
"""Transform the labelled multiclass flow dataset into a binary dataset.

The benign class is retained and all attack classes are grouped as ``Attack``.
The ``#`` label is treated as benign traffic according to the labelling decision
adopted for the corresponding scenario in the experiment.

Example:
    python data_processing/build_binary_labeled_streaming.py \\
        --input results/datasets/dataset_expanded_labeled_raw_byScenario_streaming.csv \\
        --output results/datasets/dataset_binary_labeled_raw_byScenario_streaming.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input labelled CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output binary-labelled CSV.")
    parser.add_argument("--chunksize", type=int, default=200_000,
                        help="Rows processed per chunk.")
    return parser.parse_args()


def map_to_binary(label: object) -> str:
    """Map a multiclass label to either ``Benign`` or ``Attack``."""
    cleaned_label = str(label).strip()
    if cleaned_label in {"Benign", "#"}:
        return "Benign"
    return "Attack"


def main() -> None:
    """Create the binary-labelled dataset using chunked processing."""
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_counts: Counter[str] = Counter()
    binary_counts: Counter[str] = Counter()
    rows_written = 0
    first_chunk = True

    reader = pd.read_csv(args.input, sep=";", chunksize=args.chunksize, low_memory=False)
    for chunk_index, dataframe in enumerate(reader, start=1):
        if "label" not in dataframe.columns:
            raise ValueError("The input dataset must contain a 'label' column.")

        original_counts.update(dataframe["label"].astype(str).str.strip())
        dataframe["label"] = dataframe["label"].apply(map_to_binary)
        binary_counts.update(dataframe["label"])

        dataframe.to_csv(
            args.output,
            sep=";",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        rows_written += len(dataframe)

        if chunk_index % 10 == 0:
            print(f"[INFO] Chunks processed={chunk_index}, rows written={rows_written:,}")

    print(f"[INFO] Binary-labelled dataset created: {args.output}")
    print(f"[INFO] Rows written: {rows_written:,}")
    print("[INFO] Original label distribution:")
    for label, count in original_counts.most_common():
        print(f"  {label:30s} {count}")
    print("[INFO] Binary label distribution:")
    for label, count in binary_counts.most_common():
        print(f"  {label:30s} {count}")


if __name__ == "__main__":
    main()
