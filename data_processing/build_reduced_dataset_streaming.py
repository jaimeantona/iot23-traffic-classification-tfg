#!/usr/bin/env python3
"""Build the reduced five-class dataset before SMOTE balancing.

All samples from the minority classes Benign and C&C are retained. The three
larger classes are reduced through reservoir sampling to a configurable target
size. Label variants are consolidated before sampling.

Example:
    python data_processing/build_reduced_dataset_streaming.py \\
        --input results/datasets/dataset_expanded_labeled_raw_byScenario_streaming.csv \\
        --output results/datasets/dataset_reduced_before_smote.csv \\
        --target 100000
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


PORTSCAN = "PartOfAHorizontalPortScan"
OKIRU = "Okiru"
DDOS = "DDoS"
CC = "C&C"
BENIGN = "Benign"
FINAL_CLASSES = (BENIGN, CC, DDOS, OKIRU, PORTSCAN)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input labelled CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output reduced CSV.")
    parser.add_argument("--target", type=int, default=100_000,
                        help="Maximum samples retained for large classes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def reservoir_update(
    reservoir: list[list[str]],
    row: list[str],
    seen_count: int,
    target: int,
    random_generator: random.Random,
) -> None:
    """Maintain a uniform sample of at most ``target`` observed rows."""
    if len(reservoir) < target:
        reservoir.append(row)
        return

    position = random_generator.randrange(seen_count)
    if position < target:
        reservoir[position] = row


def consolidate_label(label: str) -> str:
    """Map label variants into the final classes used in the experiment."""
    if label == "C&C-Torii":
        return CC
    if label == "#":
        return BENIGN
    return label


def main() -> None:
    """Create the reduced five-class dataset."""
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    rng = random.Random(args.seed)
    reservoirs = {DDOS: [], OKIRU: [], PORTSCAN: []}
    seen_large = Counter()
    retained_small = {BENIGN: [], CC: []}
    seen_counts: Counter[str] = Counter()

    with args.input.open("r", newline="", encoding="utf-8", errors="replace") as input_file:
        reader = csv.reader(input_file, delimiter=";")
        header = next(reader)

        if "label" not in header:
            raise ValueError("The input dataset must contain a 'label' column.")
        label_index = header.index("label")

        for row in reader:
            if not row or len(row) <= label_index:
                continue

            label = consolidate_label(row[label_index].strip())
            if label not in FINAL_CLASSES:
                continue

            row[label_index] = label
            seen_counts[label] += 1

            if label in retained_small:
                retained_small[label].append(row)
            else:
                seen_large[label] += 1
                reservoir_update(reservoirs[label], row, seen_large[label], args.target, rng)

    output_rows = (
        retained_small[BENIGN]
        + retained_small[CC]
        + reservoirs[DDOS]
        + reservoirs[OKIRU]
        + reservoirs[PORTSCAN]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, delimiter=";")
        writer.writerow(header)
        writer.writerows(output_rows)

    retained_counts = {
        BENIGN: len(retained_small[BENIGN]),
        CC: len(retained_small[CC]),
        DDOS: len(reservoirs[DDOS]),
        OKIRU: len(reservoirs[OKIRU]),
        PORTSCAN: len(reservoirs[PORTSCAN]),
    }

    print(f"[INFO] Reduced dataset created: {args.output}")
    print("[INFO] Observed samples after label consolidation:")
    for label in FINAL_CLASSES:
        print(f"  {label:30s} {seen_counts[label]}")
    print("[INFO] Retained samples before SMOTE:")
    for label in FINAL_CLASSES:
        print(f"  {label:30s} {retained_counts[label]}")


if __name__ == "__main__":
    main()
