#!/usr/bin/env python3
"""Normalise size and inter-packet-time histogram features.

Packet-size bins are divided by the number of packets in their direction. IPT
bins are divided by the number of possible intervals, defined as packets minus
one when at least two packets are available.

Example:
    python data_processing/normalize_histograms_replace.py \\
        --input results/datasets/dataset_balanced_100k_per_class_withproto_fixed.csv \\
        --output results/datasets/dataset_balanced_100k_per_class_withproto_fixed_norm.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SIZE_FWD = [f"fwd_size_bin{i}" for i in range(1, 9)]
SIZE_BWD = [f"bwd_size_bin{i}" for i in range(1, 9)]
IPT_FWD = [f"fwd_ipt_bin{i}" for i in range(1, 9)]
IPT_BWD = [f"bwd_ipt_bin{i}" for i in range(1, 9)]
HISTOGRAM_COLUMNS = SIZE_FWD + SIZE_BWD + IPT_FWD + IPT_BWD


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input corrected CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output normalised CSV.")
    parser.add_argument("--chunksize", type=int, default=200_000,
                        help="Rows processed per chunk.")
    parser.add_argument("--float-format", default="%.6f",
                        help="Output format for normalised histogram values.")
    return parser.parse_args()


def normalise_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    denominator: pd.Series,
) -> None:
    """Normalise histogram columns by a non-negative row-wise denominator."""
    valid_denominator = denominator > 0
    safe_denominator = denominator.where(valid_denominator, 1)
    for column in columns:
        dataframe[column] = dataframe[column].where(valid_denominator, 0) / safe_denominator


def main() -> None:
    """Create a dataset with normalised histogram features."""
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    rows_written = 0

    reader = pd.read_csv(args.input, sep=";", chunksize=args.chunksize, low_memory=False)
    for chunk_index, dataframe in enumerate(reader, start=1):
        required_columns = {"pkts_fwd", "pkts_bwd", *HISTOGRAM_COLUMNS}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        dataframe["pkts_fwd"] = pd.to_numeric(dataframe["pkts_fwd"], errors="coerce").fillna(0).astype("int64")
        dataframe["pkts_bwd"] = pd.to_numeric(dataframe["pkts_bwd"], errors="coerce").fillna(0).astype("int64")
        for column in HISTOGRAM_COLUMNS:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").fillna(0)

        normalise_columns(dataframe, SIZE_FWD, dataframe["pkts_fwd"])
        normalise_columns(dataframe, SIZE_BWD, dataframe["pkts_bwd"])
        normalise_columns(dataframe, IPT_FWD, (dataframe["pkts_fwd"] - 1).clip(lower=0))
        normalise_columns(dataframe, IPT_BWD, (dataframe["pkts_bwd"] - 1).clip(lower=0))

        dataframe.to_csv(
            args.output,
            sep=";",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            float_format=args.float_format,
        )
        first_chunk = False
        rows_written += len(dataframe)

        if chunk_index % 5 == 0:
            print(f"[INFO] Chunks processed={chunk_index}, rows written={rows_written:,}")

    print(f"[INFO] Normalised dataset created: {args.output}")
    print(f"[INFO] Rows written: {rows_written:,}")


if __name__ == "__main__":
    main()
