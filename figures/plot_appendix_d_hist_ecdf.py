#!/usr/bin/env python3
"""Generate the complementary histogram and ECDF figures included in Appendix D.

The script compares the Okiru and PartOfAHorizontalPortScan classes using the
normalised histogram features and the aggregate packet and byte variables.

Example:
    python figures/plot_appendix_d_hist_ecdf.py \\
        --dataset data/dataset_balanced_100k_per_class_withproto_fixed_norm.csv \\
        --output-dir results/figures/appendix_d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_A = "Okiru"
CLASS_B = "PartOfAHorizontalPortScan"
DISPLAY_NAMES = {
    CLASS_A: "Okiru",
    CLASS_B: "PartOfAHorizontalPortScan",
}

HISTOGRAM_GROUPS = {
    "Forward Packet Size Histogram": [f"fwd_size_bin{i}" for i in range(1, 9)],
    "Backward Packet Size Histogram": [f"bwd_size_bin{i}" for i in range(1, 9)],
    "Forward IPT Histogram": [f"fwd_ipt_bin{i}" for i in range(1, 9)],
    "Backward IPT Histogram": [f"bwd_ipt_bin{i}" for i in range(1, 9)],
}

ECDF_COLUMNS = ["pkts_fwd", "pkts_bwd", "bytes_fwd", "bytes_bwd"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the normalised five-class dataset containing histogram features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/appendix_d"),
        help="Directory where PDF figures are written.",
    )
    parser.add_argument(
        "--separator",
        default=";",
        help="CSV delimiter used by the dataset.",
    )
    return parser.parse_args()


def load_dataset(path: Path, separator: str) -> pd.DataFrame:
    """Load the input dataset and verify that it contains class labels."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path, sep=separator)
    if "label" not in dataframe.columns:
        raise ValueError("The input dataset must contain a 'label' column.")

    return dataframe


def require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    """Raise an error when required columns are not present."""
    missing_columns = [column for column in columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def select_compared_classes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return only the samples belonging to the compared attack classes."""
    selected = dataframe[dataframe["label"].isin([CLASS_A, CLASS_B])].copy()
    if selected.empty:
        raise ValueError(
            "No samples were found for Okiru or PartOfAHorizontalPortScan."
        )
    return selected


def plot_histograms(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Generate the mean histogram comparison figure."""
    histogram_columns = [
        column
        for columns in HISTOGRAM_GROUPS.values()
        for column in columns
    ]
    require_columns(dataframe, histogram_columns)

    selected = select_compared_classes(dataframe)
    bins = np.arange(1, 9)
    bar_width = 0.35
    offsets = {
        CLASS_A: -bar_width / 2,
        CLASS_B: bar_width / 2,
    }

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for axis, (title, columns) in zip(axes.flatten(), HISTOGRAM_GROUPS.items()):
        for class_name in (CLASS_A, CLASS_B):
            mean_values = selected.loc[selected["label"] == class_name, columns].mean().to_numpy()
            axis.bar(
                bins + offsets[class_name],
                mean_values,
                width=bar_width,
                label=DISPLAY_NAMES[class_name],
            )

        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Bin", fontsize=10)
        axis.set_ylabel("Average probability", fontsize=10)
        axis.set_xticks(bins)
        axis.grid(axis="y", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=True)
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def compute_ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted observations and empirical cumulative probabilities."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    finite_values.sort()

    if finite_values.size == 0:
        return np.array([]), np.array([])

    cumulative_probabilities = np.arange(1, finite_values.size + 1) / finite_values.size
    return finite_values, cumulative_probabilities


def plot_ecdf(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Generate ECDF comparisons for aggregate packet and byte variables."""
    require_columns(dataframe, ECDF_COLUMNS)
    selected = select_compared_classes(dataframe)

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for axis, column in zip(axes.flatten(), ECDF_COLUMNS):
        for class_name in (CLASS_A, CLASS_B):
            x_values, y_values = compute_ecdf(
                selected.loc[selected["label"] == class_name, column].to_numpy()
            )
            positive_values = x_values > 0
            axis.plot(
                x_values[positive_values],
                y_values[positive_values],
                label=DISPLAY_NAMES[class_name],
                linewidth=1.6,
            )

        axis.set_xscale("log")
        axis.set_title(f"ECDF de {column}", fontsize=11)
        axis.set_xlabel(column, fontsize=10)
        axis.set_ylabel("ECDF", fontsize=10)
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=True)
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate the two complementary figures used in Appendix D."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(args.dataset, args.separator)

    histogram_path = args.output_dir / "hist_okiru_portscan.pdf"
    ecdf_path = args.output_dir / "ecdf_okiru_portscan.pdf"

    plot_histograms(dataframe, histogram_path)
    print(f"[INFO] Figure created: {histogram_path}")

    plot_ecdf(dataframe, ecdf_path)
    print(f"[INFO] Figure created: {ecdf_path}")


if __name__ == "__main__":
    main()
