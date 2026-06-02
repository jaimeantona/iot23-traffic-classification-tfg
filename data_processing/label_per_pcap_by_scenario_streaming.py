#!/usr/bin/env python3
"""Assign scenario-level labels to per-PCAP flow feature files.

The input directory must contain one subdirectory per IoT-23 scenario. Each
scenario subdirectory contains the CSV files produced during flow extraction.
The mapping of malware captures to attack categories is derived from the
official assigned-labels CSV; honeypot captures are labelled as benign traffic.

Example:
    python data_processing/label_per_pcap_by_scenario_streaming.py \\
        --input-dir results/flow_extraction/per_pcap_by_scenario \\
        --assigned-labels data/assigned_labels.csv \\
        --output results/datasets/dataset_expanded_labeled_raw_byScenario_streaming.csv \\
        --report results/datasets/labeling_report_byScenario_streaming.csv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


CAPTURE_PATTERN = r"CTU-(?:IoT-Malware|Honeypot)-Capture-"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing per-scenario CSV directories.")
    parser.add_argument("--assigned-labels", type=Path, required=True,
                        help="Official IoT-23 assigned-labels CSV.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output labelled flow dataset.")
    parser.add_argument("--report", type=Path, required=True,
                        help="Output processing report.")
    return parser.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a feature file accepting the delimiters used during extraction."""
    last_error: Exception | None = None
    for separator in ("\t", ";", ",", r"\s+"):
        try:
            dataframe = pd.read_csv(path, sep=separator, engine="python")
            if dataframe.shape[1] > 1:
                return dataframe
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Unable to read {path}. Last error: {last_error}")


def clean_label(label: object) -> str | None:
    """Remove line breaks and surrounding whitespace from a label."""
    if label is None:
        return None
    return str(label).replace("\n", "").replace("\r", "").strip()


def load_capture_to_category(labels_csv: Path) -> dict[str, str]:
    """Map malware capture names to their predominant non-benign category."""
    dataframe = pd.read_csv(labels_csv)

    capture_column = next(
        (
            column for column in dataframe.columns
            if dataframe[column].astype(str).str.contains(CAPTURE_PATTERN, regex=True).any()
        ),
        None,
    )
    if capture_column is None:
        raise RuntimeError("No IoT-23 capture-name column was found in the assigned-labels CSV.")

    category_columns = [
        column for column in dataframe.columns
        if column != capture_column and pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if not category_columns:
        raise RuntimeError("No numeric attack-category columns were found in the assigned-labels CSV.")

    mapping: dict[str, str] = {}
    for _, row in dataframe.iterrows():
        capture = str(row[capture_column]).strip()
        if "CTU-IoT-Malware" not in capture:
            continue

        candidates = [column for column in category_columns if column.lower() != "benign"]
        values = {
            column: int(row[column])
            for column in candidates
            if pd.notna(row[column])
        }
        if values:
            mapping[capture] = clean_label(max(values, key=values.get)) or "Attack"
        else:
            mapping[capture] = "Attack"

    return mapping


def normalise_folder_to_capture(folder_name: str) -> str:
    """Return the parent capture name for scenario folders with suffixes."""
    match = re.match(r"^(CTU-(?:IoT-Malware|Honeypot)-Capture-\d+-\d+)", folder_name)
    return match.group(1) if match else folder_name


def main() -> None:
    """Create the scenario-labelled dataset and its processing report."""
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.assigned_labels.exists():
        raise FileNotFoundError(f"Assigned-labels CSV not found: {args.assigned_labels}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    capture_to_category = load_capture_to_category(args.assigned_labels)
    scenario_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not scenario_dirs:
        raise RuntimeError(f"No scenario directories found in: {args.input_dir}")

    report_rows: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    wrote_header = False

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        capture_name = normalise_folder_to_capture(scenario_name)
        csv_files = sorted(scenario_dir.glob("*.csv"))

        if capture_name.startswith("CTU-Honeypot"):
            label = "Benign"
        else:
            label = clean_label(capture_to_category.get(capture_name))

        if label is None:
            report_rows.append({
                "scenario_folder": scenario_name,
                "capture_name": capture_name,
                "label": None,
                "csv_files": len(csv_files),
                "rows_total": 0,
                "status": "NO_LABEL_FOR_CAPTURE",
            })
            print(f"[WARNING] Scenario not labelled: {scenario_name}")
            continue

        rows_total = 0
        for file_path in csv_files:
            dataframe = read_csv_robust(file_path)
            dataframe["capture"] = capture_name
            dataframe["label"] = label
            dataframe.to_csv(
                args.output,
                index=False,
                sep=";",
                mode="a",
                header=not wrote_header,
            )
            wrote_header = True
            rows_total += len(dataframe)

        report_rows.append({
            "scenario_folder": scenario_name,
            "capture_name": capture_name,
            "label": label,
            "csv_files": len(csv_files),
            "rows_total": rows_total,
            "status": "OK",
        })
        label_counts[label] += rows_total
        print(f"[INFO] {scenario_name}: label={label}, files={len(csv_files)}, rows={rows_total}")

    pd.DataFrame(report_rows).to_csv(args.report, index=False)

    print(f"[INFO] Labelled dataset created: {args.output}")
    print(f"[INFO] Processing report created: {args.report}")
    print("[INFO] Flow distribution by label:")
    for label, count in label_counts.most_common():
        print(f"  {label:30s} {count}")


if __name__ == "__main__":
    main()
