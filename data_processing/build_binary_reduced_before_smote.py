#!/usr/bin/env python3
import csv
import random
from pathlib import Path
from collections import Counter

INPUT = Path("pipeline_hist/out/dataset_binary_labeled_raw_byScenario_streaming.csv")
OUTPUT = Path("pipeline_hist/out/dataset_binary_reduced_before_smote.csv")

TARGET_ATTACK = 100_000
SEED = 42

# Columnas auxiliares que no se usarán como características
DROP_COLS = {"flow_id", "capture"}

def reservoir_update(reservoir, row, seen_count, k, rng):
    """
    Reservoir sampling:
    mantiene una muestra aleatoria uniforme de tamaño k sin cargar todo el CSV.
    """
    if len(reservoir) < k:
        reservoir.append(row)
    else:
        j = rng.randrange(seen_count)
        if j < k:
            reservoir[j] = row

def main():
    if not INPUT.exists():
        raise SystemExit(f"No existe el archivo de entrada: {INPUT}")

    rng = random.Random(SEED)

    benign_rows = []
    attack_reservoir = []

    seen_attack = 0
    total_rows = 0
    counts_original = Counter()

    print("Input:", INPUT)
    print("Output:", OUTPUT)
    print("Reducing Attack with streaming reservoir sampling...")
    print("Keeping all Benign rows...")

    with INPUT.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)

        if "label" not in header:
            raise SystemExit("No se encontró la columna 'label'.")

        label_idx = header.index("label")

        # Quitamos flow_id y capture para reducir memoria y tamaño.
        keep_indices = [i for i, col in enumerate(header) if col not in DROP_COLS]
        out_header = [header[i] for i in keep_indices]
        out_label_idx = out_header.index("label")

        for row in reader:
            if not row:
                continue

            total_rows += 1

            # Por si alguna fila rara viniera incompleta
            if len(row) <= label_idx:
                continue

            label = row[label_idx].strip()
            counts_original[label] += 1

            # Filtrar columnas que sí queremos conservar
            row_out = [row[i] if i < len(row) else "" for i in keep_indices]

            if label == "Benign":
                benign_rows.append(row_out)

            elif label == "Attack":
                seen_attack += 1
                reservoir_update(
                    attack_reservoir,
                    row_out,
                    seen_attack,
                    TARGET_ATTACK,
                    rng
                )

            if total_rows % 5_000_000 == 0:
                print(
                    f"[PROGRESS] rows={total_rows:,} | "
                    f"Attack seen={seen_attack:,} | "
                    f"Attack sample={len(attack_reservoir):,} | "
                    f"Benign={len(benign_rows):,}"
                )

    if len(attack_reservoir) < TARGET_ATTACK:
        raise SystemExit(
            f"No hay suficientes muestras Attack: {len(attack_reservoir)}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Mezclamos Benign + Attack
    final_rows = benign_rows + attack_reservoir
    rng.shuffle(final_rows)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(out_header)
        writer.writerows(final_rows)

    print("\n[DONE] Reduced binary dataset created.")
    print("\nOriginal binary distribution:")
    for k, v in counts_original.items():
        print(k, v)

    print("\nReduced distribution:")
    reduced_counts = Counter(row[out_label_idx] for row in final_rows)
    for k, v in reduced_counts.items():
        print(k, v)

    print("\nRows written:", len(final_rows))
    print("Columns written:", len(out_header))
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()