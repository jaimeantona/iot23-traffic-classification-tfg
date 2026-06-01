#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

INPUT = "pipeline_hist/out/dataset_reduced_before_smote.csv"
OUTPUT = "pipeline_hist/out/dataset_balanced_100k_per_class_withproto.csv"

TARGET = 100000
SEED = 42

FINAL_CLASSES = [
    "Benign",
    "C&C",
    "DDoS",
    "Okiru",
    "PartOfAHorizontalPortScan"
]

def proto_to_int(s):
    s = s.astype(str).str.strip().str.lower()
    return s.map({
        "tcp":6,
        "udp":17,
        "icmp":1
    }).fillna(-1).astype("int16")

def clean_label(s):
    return (
        s.astype(str)
        .str.replace("\n","",regex=False)
        .str.replace("\r","",regex=False)
        .str.strip()
        .replace({
            "C&C-Torii":"C&C",
            "#":"Benign"
        })
    )

def main():

    print("Loading dataset...")
    df = pd.read_csv(INPUT, sep=";")

    df["label"] = clean_label(df["label"])
    df = df[df["label"].isin(FINAL_CLASSES)]

    print("Initial distribution:")
    print(df["label"].value_counts())

    # proto → numeric
    df["proto"] = proto_to_int(df["proto"])

    # separar features
    drop_cols = ["flow_id","capture"]
    X = df.drop(columns=drop_cols + ["label"], errors="ignore")
    y = df["label"]

    # numeric conversion
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("\nApplying SMOTE...")

    smote = SMOTE(
        sampling_strategy={
            "Benign":TARGET,
            "C&C":TARGET
        },
        random_state=SEED,
        k_neighbors=3
    )

    X_res, y_res = smote.fit_resample(X, y)

    out = pd.DataFrame(X_res, columns=X.columns)
    out["label"] = y_res

    # recortar por seguridad
    final = []
    for c in FINAL_CLASSES:
        part = out[out["label"] == c]
        if len(part) > TARGET:
            part = part.sample(n=TARGET, random_state=SEED)
        final.append(part)

    final = pd.concat(final)

    print("\nFinal distribution:")
    print(final["label"].value_counts())

    final.to_csv(OUTPUT, sep=";", index=False)

    print("\nSaved:", OUTPUT)

if __name__ == "__main__":
    main()