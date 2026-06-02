#!/usr/bin/env python3
"""Restore consistency constraints in count-based histogram features.

Packet-size histogram counts define the packet counts in each direction. IPT
histogram counts are capped to the maximum number of intervals available for
the corresponding number of packets.

Example:
    python data_processing/fix_bins_consistency.py \\
        --input results/datasets/dataset_balanced_100k_per_class_withproto.csv \\
        --output results/datasets/dataset_balanced_100k_per_class_withproto_fixed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FWD_SIZE = [f"fwd_size_bin{i}" for i in range(1, 9)]
BWD_SIZE = [f"bwd_size_bin{i}" for i in range(1, 9)]
FWD_IPT = [f"fwd_ipt_bin{i}" for i in range(1, 9)]
BWD_IPT = [f"bwd_ipt_bin{i}" for i in range(1, 9)]
HISTOGRAM_COLUMNS = FWD_SIZE + BWD_SIZE + FWD_IPT + BWD_IPT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output corrected CSV.")
    parser.add_argument("--chunksize", type=int, default=200_000,
                        help="Rows processed per chunk.")
    parser.add_argument("--sep", default=";", help="CSV separator.")
    return parser.parse_args()


def ensure_nonnegative_integers(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert histogram columns to non-negative integer counts."""
    for column in columns:
        dataframe[column] = (
            pd.to_numeric(dataframe[column], errors="coerce")
            .fillna(0)
            .round()
            .astype("int64")
            .clip(lower=0)
        )
    return dataframe


def cap_bins_rowwise(
    dataframe: pd.DataFrame,
    bin_columns: list[str],
    cap_series: pd.Series,
) -> pd.DataFrame:
    """Reduce bins from highest to lowest until each row satisfies its cap."""
    bins = dataframe[bin_columns].to_numpy(dtype=np.int64, copy=True)
    caps = cap_series.to_numpy(dtype=np.int64, copy=False)
    excess = bins.sum(axis=1) - caps

    for row_index in np.where(excess > 0)[0]:
        remaining_excess = int(excess[row_index])
        for bin_index in range(len(bin_columns) - 1, -1, -1):
            if remaining_excess <= 0:
                break
            reduction = min(remaining_excess, int(bins[row_index, bin_index]))
            bins[row_index, bin_index] -= reduction
            remaining_excess -= reduction

    dataframe[bin_columns] = bins
    return dataframe


def main() -> None:
    """Correct histogram consistency constraints using chunked processing."""
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    total_rows = 0
    packet_count_changes = 0
    ipt_changes = 0

    reader = pd.read_csv(args.input, sep=args.sep, chunksize=args.chunksize, low_memory=False)
    for chunk_index, dataframe in enumerate(reader, start=1):
        required_columns = {"pkts_fwd", "pkts_bwd", *HISTOGRAM_COLUMNS}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        dataframe = ensure_nonnegative_integers(dataframe, HISTOGRAM_COLUMNS)

        old_fwd_packets = pd.to_numeric(dataframe["pkts_fwd"], errors="coerce").fillna(0).round().astype("int64")
        old_bwd_packets = pd.to_numeric(dataframe["pkts_bwd"], errors="coerce").fillna(0).round().astype("int64")
        new_fwd_packets = dataframe[FWD_SIZE].sum(axis=1).astype("int64")
        new_bwd_packets = dataframe[BWD_SIZE].sum(axis=1).astype("int64")

        packet_count_changes += int(
            (old_fwd_packets != new_fwd_packets).sum()
            + (old_bwd_packets != new_bwd_packets).sum()
        )
        dataframe["pkts_fwd"] = new_fwd_packets
        dataframe["pkts_bwd"] = new_bwd_packets

        fwd_cap = (dataframe["pkts_fwd"] - 1).clip(lower=0).astype("int64")
        bwd_cap = (dataframe["pkts_bwd"] - 1).clip(lower=0).astype("int64")
        fwd_ipt_before = dataframe[FWD_IPT].sum(axis=1)
        bwd_ipt_before = dataframe[BWD_IPT].sum(axis=1)

        dataframe = cap_bins_rowwise(dataframe, FWD_IPT, fwd_cap)
        dataframe = cap_bins_rowwise(dataframe, BWD_IPT, bwd_cap)

        ipt_changes += int(
            (fwd_ipt_before != dataframe[FWD_IPT].sum(axis=1)).sum()
            + (bwd_ipt_before != dataframe[BWD_IPT].sum(axis=1)).sum()
        )

        if "duration" in dataframe.columns:
            dataframe["duration"] = pd.to_numeric(dataframe["duration"], errors="coerce").fillna(0.0)
            dataframe.loc[dataframe["duration"] < 0, "duration"] = 0.0

        dataframe.to_csv(
            args.output,
            sep=args.sep,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        total_rows += len(dataframe)

        if chunk_index % 5 == 0:
            print(f"[INFO] Chunks processed={chunk_index}, rows written={total_rows:,}")

    print(f"[INFO] Corrected dataset created: {args.output}")
    print(f"[INFO] Rows={total_rows:,}, packet-count edits={packet_count_changes:,}, IPT edits={ipt_changes:,}")


if __name__ == "__main__":
    main()
