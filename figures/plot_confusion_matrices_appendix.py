#!/usr/bin/env python3
"""Generate the confusion-matrix figures included in Appendix C.

The matrices stored in this script correspond to the final experimental results
reported in the TFG. Keeping the numerical results in the script makes it
possible to reproduce the PDF figures without distributing the processed
datasets or prediction files.

Example:
    python figures/plot_confusion_matrices_appendix.py \\
        --output-dir results/figures/confusion_matrices
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BINARY_LABELS = ["Attack", "Benign"]
MULTICLASS_LABELS = ["Benign", "C&C", "DDoS", "Okiru", "PortScan"]

# Rows correspond to actual classes and columns to predicted classes.
CONFUSION_MATRICES = {
    "cm_binario_rf.pdf": {
        "matrix": np.array([
            [19910, 90],
            [77, 19923],
        ]),
        "labels": BINARY_LABELS,
        "title": "Random Forest",
    },
    "cm_binario_lgbm.pdf": {
        "matrix": np.array([
            [19860, 140],
            [2, 19998],
        ]),
        "labels": BINARY_LABELS,
        "title": "LightGBM",
    },
    "cm_binario_mlp.pdf": {
        "matrix": np.array([
            [19792, 208],
            [8, 19992],
        ]),
        "labels": BINARY_LABELS,
        "title": "MLP",
    },
    "cm_multiclase_hist_rf.pdf": {
        "matrix": np.array([
            [18872, 373, 753, 0, 2],
            [207, 19785, 8, 0, 0],
            [671, 16, 19123, 142, 48],
            [5, 0, 10, 19984, 1],
            [2, 1, 5, 13383, 6609],
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "Random Forest",
    },
    "cm_multiclase_hist_lgbm.pdf": {
        "matrix": np.array([
            [17802, 485, 1711, 0, 2],
            [185, 19805, 10, 0, 0],
            [8, 12, 19782, 139, 59],
            [3, 1, 11, 19984, 1],
            [1, 1, 8, 13384, 6606],
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "LightGBM",
    },
    "cm_multiclase_hist_mlp.pdf": {
        "matrix": np.array([
            [17116, 1125, 1750, 1, 8],
            [6018, 13929, 53, 0, 0],
            [14, 8, 19791, 139, 48],
            [5, 6, 12, 19977, 0],
            [8, 57, 26, 13381, 6528],
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "MLP",
    },
    "cm_multiclase_nohist_rf.pdf": {
        "matrix": np.array([
            [18871, 374, 754, 0, 1],
            [218, 19766, 16, 0, 0],
            [667, 21, 19122, 142, 48],
            [5, 0, 10, 19984, 1],
            [2, 1, 6, 13383, 6608],
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "Random Forest",
    },
    "cm_multiclase_nohist_lgbm.pdf": {
        "matrix": np.array([
            [17796, 493, 1710, 0, 1],
            [186, 19800, 14, 0, 0],
            [6, 13, 19782, 139, 60],
            [3, 1, 11, 19985, 0],
            [2, 1, 6, 13384, 6607],
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "LightGBM",
    },
    "cm_multiclase_nohist_mlp.pdf": {
        "matrix": np.array([
            [16931, 1296, 1735, 33, 5],
            [6141, 13799, 40, 20, 0],
            [22, 101, 19379, 294, 204],
            [2, 69, 11, 19903, 15],
            [91, 280, 110, 13378, 6141],
        ]),
        "labels": MULTICLASS_LABELS,
        "title": "MLP",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/confusion_matrices"),
        help="Directory where PDF figures are written.",
    )
    return parser.parse_args()


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Render one confusion matrix and save it as a PDF figure."""
    class_count = len(labels)
    if class_count == 2:
        figure_size = (4.2, 3.8)
        rotation = 0
        value_font_size = 9
    else:
        figure_size = (5.8, 5.0)
        rotation = 35
        value_font_size = 8

    figure, axis = plt.subplots(figsize=figure_size)
    image = axis.imshow(matrix, cmap="viridis")

    axis.set_title(title, fontsize=12)
    axis.set_xlabel("Predicción", fontsize=10)
    axis.set_ylabel("Clase real", fontsize=10)
    axis.set_xticks(np.arange(class_count))
    axis.set_yticks(np.arange(class_count))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_yticklabels(labels, fontsize=8)

    plt.setp(
        axis.get_xticklabels(),
        rotation=rotation,
        ha="right" if rotation else "center",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2.0
    for row in range(class_count):
        for column in range(class_count):
            value = int(matrix[row, column])
            text_colour = "black" if value > threshold else "white"
            axis.text(
                column,
                row,
                f"{value}",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=value_font_size,
            )

    colour_bar = figure.colorbar(image, ax=axis)
    colour_bar.ax.tick_params(labelsize=8)

    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Generate all Appendix C confusion-matrix figures."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, configuration in CONFUSION_MATRICES.items():
        output_path = args.output_dir / filename
        plot_confusion_matrix(
            matrix=configuration["matrix"],
            labels=configuration["labels"],
            title=configuration["title"],
            output_path=output_path,
        )
        print(f"[INFO] Figure created: {output_path}")

    print("[INFO] All confusion-matrix figures were generated successfully.")


if __name__ == "__main__":
    main()
