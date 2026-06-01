#!/usr/bin/env python3
import pandas as pd
from imblearn.over_sampling import SMOTE

INPUT = "pipeline_hist/out/dataset_binary_reduced_before_smote.csv"
OUTPUT = "pipeline_hist/out/dataset_binary_balanced_100k_per_class_withproto.csv"

TARGET = 100000
SEED = 42

FINAL_CLASSES = ["Benign", "Attack"]

def proto_to_int(s):
    s = s.astype(str).str.strip().str.lower()
    return s.map({
        "tcp": 6,
        "udp": 17,
        "icmp": 1
    }).fillna(-1).astype("int16")

def main():
    print("Loading reduced binary dataset...")
    df = pd.read_csv(INPUT, sep=";")

    if "label" not in df.columns:
        raise RuntimeError("No existe la columna label.")

    if "proto" not in df.columns:
        raise RuntimeError("No existe la columna proto.")

    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin(FINAL_CLASSES)].copy()

    print("\nInitial distribution:")
    print(df["label"].value_counts())

    # proto -> numérico, igual que en el multiclase final
    df["proto"] = proto_to_int(df["proto"])

    # No usamos identificadores ni campos auxiliares como features
    drop_cols = ["flow_id", "capture"]
    X = df.drop(columns=drop_cols + ["label"], errors="ignore")
    y = df["label"]

    # Conversión robusta a numérico
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("\nApplying SMOTE to Benign...")

    smote = SMOTE(
        sampling_strategy={"Benign": TARGET},
        random_state=SEED,
        k_neighbors=3
    )

    X_res, y_res = smote.fit_resample(X, y)

    out = pd.DataFrame(X_res, columns=X.columns)
    out["label"] = y_res

    # Seguridad: dejar exactamente TARGET por clase
    final_parts = []
    for cls in FINAL_CLASSES:
        part = out[out["label"] == cls]
        if len(part) < TARGET:
            raise RuntimeError(f"La clase {cls} quedó con {len(part)} muestras (< {TARGET}).")
        if len(part) > TARGET:
            part = part.sample(n=TARGET, random_state=SEED)
        final_parts.append(part)

    final = pd.concat(final_parts, ignore_index=True)
    final = final.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("\nFinal distribution:")
    print(final["label"].value_counts())
    print("\nFinal shape:", final.shape)

    final.to_csv(OUTPUT, sep=";", index=False)
    print("\nSaved:", OUTPUT)

if __name__ == "__main__":
    main()