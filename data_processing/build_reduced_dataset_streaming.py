#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
from pathlib import Path

INPUT_FILE = Path("pipeline_hist/out/dataset_expanded_labeled_raw_byScenario_streaming.csv")
OUTPUT_FILE = Path("pipeline_hist/out/dataset_reduced_before_smote.csv")

# Clases finales
PORTSCAN = "PartOfAHorizontalPortScan"
OKIRU = "Okiru"
DDOS = "DDoS"
CC = "C&C"
BENIGN = "Benign"

# Downsample objetivo para clases grandes
K = 100_000
RANDOM_SEED = 42

def reservoir_update(reservoir, item, seen_count, k, rng):
    """
    Reservoir sampling:
      - reservoir: list con muestras
      - item: fila actual (list)
      - seen_count: cuántos elementos de esa clase hemos visto (incluyendo este)
    """
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        j = rng.randrange(seen_count)  # 0..seen_count-1
        if j < k:
            reservoir[j] = item

def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"No existe: {INPUT_FILE}")

    rng = random.Random(RANDOM_SEED)

    # Reservorios para clases grandes
    res_portscan = []
    res_okiru = []
    res_ddos = []
    seen = {PORTSCAN: 0, OKIRU: 0, DDOS: 0}

    # Para clases pequeñas (guardamos todas; son pocas)
    rows_benign = []
    rows_cc = []

    # Conteos (para imprimir al final)
    counts_after = {BENIGN: 0, CC: 0, DDOS: 0, OKIRU: 0, PORTSCAN: 0}
    counts_seen_raw = {}

    with INPUT_FILE.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)

        # localizar índice de label
        try:
            label_idx = header.index("label")
        except ValueError:
            raise SystemExit("No encuentro columna 'label' en el CSV de entrada.")

        for row in reader:
            if not row:
                continue

            # label original
            label = row[label_idx].strip()

            # Relabels:
            # - C&C-Torii -> C&C
            # - # -> Benign
            if label == "C&C-Torii":
                label = CC
            elif label == "#":
                label = BENIGN

            # Guardar label relabelado en la fila
            row[label_idx] = label

            # (solo para info)
            counts_seen_raw[label] = counts_seen_raw.get(label, 0) + 1

            # Distribución / muestreo
            if label == BENIGN:
                rows_benign.append(row)
                counts_after[BENIGN] += 1

            elif label == CC:
                rows_cc.append(row)
                counts_after[CC] += 1

            elif label == DDOS:
                seen[DDOS] += 1
                reservoir_update(res_ddos, row, seen[DDOS], K, rng)

            elif label == OKIRU:
                seen[OKIRU] += 1
                reservoir_update(res_okiru, row, seen[OKIRU], K, rng)

            elif label == PORTSCAN:
                seen[PORTSCAN] += 1
                reservoir_update(res_portscan, row, seen[PORTSCAN], K, rng)

            else:
                # Si apareciese alguna etiqueta inesperada, la ignoramos
                # (o puedes imprimir warning)
                pass

    # Conteos finales de grandes (reservorio)
    counts_after[DDOS] = len(res_ddos)
    counts_after[OKIRU] = len(res_okiru)
    counts_after[PORTSCAN] = len(res_portscan)

    # Escribir dataset reducido
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="") as out:
        writer = csv.writer(out, delimiter=";")
        writer.writerow(header)

        # Escribimos: benign + cc + (reservorios grandes)
        # (si prefieres mezclar, luego lo barajamos en el script 2)
        writer.writerows(rows_benign)
        writer.writerows(rows_cc)
        writer.writerows(res_ddos)
        writer.writerows(res_okiru)
        writer.writerows(res_portscan)

    print("✅ Reduced dataset saved:", OUTPUT_FILE)
    print("\nDistribución reducida (antes de SMOTE):")
    for k, v in counts_after.items():
        print(f"{k:25s} {v}")

    print("\nInfo (cuántas filas vistas por clase tras relabel, antes de muestrear):")
    for k, v in sorted(counts_seen_raw.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:25s} {v}")

if __name__ == "__main__":
    main()