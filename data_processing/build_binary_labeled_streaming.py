#!/usr/bin/env python3
import argparse
import pandas as pd
from collections import Counter

def map_to_binary(label: str) -> str:
    """
    Mapea las etiquetas originales/agrupadas a dos clases:
    - Benign se mantiene como Benign
    - # se considera Benign, siguiendo la decisión tomada para el escenario Trojan
    - El resto de etiquetas se agrupan como Attack
    """
    label = str(label).strip()

    if label == "Benign":
        return "Benign"

    if label == "#":
        return "Benign"

    return "Attack"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="out/dataset_expanded_labeled_raw_byScenario_streaming.csv",
        help="CSV de entrada etiquetado por escenario"
    )
    parser.add_argument(
        "--output",
        default="out/dataset_binary_labeled_raw_byScenario_streaming.csv",
        help="CSV de salida con etiquetas binarias"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Tamaño de chunk para procesamiento streaming"
    )

    args = parser.parse_args()

    first = True
    rows = 0

    original_counter = Counter()
    binary_counter = Counter()

    print("Input:", args.input)
    print("Output:", args.output)
    print("Building binary labeled dataset...")

    reader = pd.read_csv(
        args.input,
        sep=";",
        chunksize=args.chunksize,
        low_memory=False
    )

    for chunk_idx, df in enumerate(reader, start=1):
        if "label" not in df.columns:
            raise RuntimeError("No se ha encontrado la columna 'label' en el CSV.")

        original_counter.update(df["label"].astype(str).str.strip())

        df["label"] = df["label"].apply(map_to_binary)

        binary_counter.update(df["label"])

        df.to_csv(
            args.output,
            sep=";",
            index=False,
            mode="w" if first else "a",
            header=first
        )

        first = False
        rows += len(df)

        if chunk_idx % 10 == 0:
            print(f"[PROGRESS] chunks={chunk_idx}, rows_written={rows:,}")

    print("\n[DONE] Binary labeled dataset created.")
    print("Rows:", f"{rows:,}")

    print("\nOriginal label distribution:")
    for k, v in original_counter.most_common():
        print(k, v)

    print("\nBinary label distribution:")
    for k, v in binary_counter.most_common():
        print(k, v)


if __name__ == "__main__":
    main()