#!/usr/bin/env python3
"""Build the reduced binary dataset before SMOTE balancing.

All benign samples are retained and the attack class is reduced through
reservoir sampling to the requested target size. Auxiliary identifier columns
are excluded from the generated dataset.

Example:
    python data_processing/build_binary_reduced_before_smote.py \\
        --input results/datasets/dataset_binary_labeled_raw_byScenario_streaming.csv \\
        --output results/datasets/dataset_binary_reduced_before_smote.csv \\
        --target-attack 100000
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


DROP_COLUMNS = {"flow_id", "capture"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input binary-labelled CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output reduced binary CSV.")
    parser.add_argument("--target-attack", type=int, default=100_000,
                        help="Number of attack samples retained.")
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


def main() -> None:
    """Create the reduced binary dataset."""
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    rng = random.Random(args.seed)
    benign_rows: list[list[str]] = []
    attack_reservoir: list[list[str]] = []
    attack_seen = 0
    original_counts: Counter[str] = Counter()

    with args.input.open("r", newline="", encoding="utf-8", errors="replace") as input_file:
        reader = csv.reader(input_file, delimiter=";")
        header = next(reader)

        if "label" not in header:
            raise ValueError("The input dataset must contain a 'label' column.")

        label_index = header.index("label")
        keep_indices = [index for index, column in enumerate(header) if column not in DROP_COLUMNS]
        output_header = [header[index] for index in keep_indices]
        output_label_index = output_header.index("label")

        for row in reader:
            if not row or len(row) <= label_index:
                continue

            label = row[label_index].strip()
            original_counts[label] += 1
            output_row = [row[index] if index < len(row) else "" for index in keep_indices]

            if label == "Benign":
                benign_rows.append(output_row)
            elif label == "Attack":
                attack_seen += 1
                reservoir_update(
                    attack_reservoir,
                    output_row,
                    attack_seen,
                    args.target_attack,
                    rng,
                )

    if len(attack_reservoir) < args.target_attack:
        raise RuntimeError(
            f"Only {len(attack_reservoir)} attack samples were found; "
            f"{args.target_attack} were requested."
        )

    final_rows = benign_rows + attack_reservoir
    rng.shuffle(final_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, delimiter=";")
        writer.writerow(output_header)
        writer.writerows(final_rows)

    reduced_counts = Counter(row[output_label_index] for row in final_rows)
    print(f"[INFO] Reduced binary dataset created: {args.output}")
    print("[INFO] Original binary distribution:")
    for label, count in original_counts.items():
        print(f"  {label:20s} {count}")
    print("[INFO] Retained binary distribution:")
    for label, count in reduced_counts.items():
        print(f"  {label:20s} {count}")


if __name__ == "__main__":
    main()
