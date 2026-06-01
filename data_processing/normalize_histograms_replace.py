#!/usr/bin/env python3
import argparse
import pandas as pd

SIZE_FWD = [f"fwd_size_bin{i}" for i in range(1, 9)]
SIZE_BWD = [f"bwd_size_bin{i}" for i in range(1, 9)]
IPT_FWD  = [f"fwd_ipt_bin{i}"  for i in range(1, 9)]
IPT_BWD  = [f"bwd_ipt_bin{i}"  for i in range(1, 9)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--float_fmt", default="%.6f", help="Formato para bins normalizados")
    args = ap.parse_args()

    inp = args.input
    out = args.output
    chunk = args.chunksize

    first = True
    rows = 0

    reader = pd.read_csv(inp, sep=";", chunksize=chunk, low_memory=False)

    for df in reader:
        # Column checks (fail-fast si algo raro)
        for col in ["pkts_fwd", "pkts_bwd"]:
            if col not in df.columns:
                raise RuntimeError(f"Falta columna requerida: {col}")
        for col in (SIZE_FWD + SIZE_BWD + IPT_FWD + IPT_BWD):
            if col not in df.columns:
                raise RuntimeError(f"Falta columna bin requerida: {col}")

        # Asegura numéricos (por si acaso)
        df["pkts_fwd"] = pd.to_numeric(df["pkts_fwd"], errors="coerce").fillna(0).astype("int64")
        df["pkts_bwd"] = pd.to_numeric(df["pkts_bwd"], errors="coerce").fillna(0).astype("int64")

        for c in (SIZE_FWD + SIZE_BWD + IPT_FWD + IPT_BWD):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # ===== Normalización SIZE bins =====
        # fwd: dividir por pkts_fwd si pkts_fwd>0, si no -> 0
        denom_fwd = df["pkts_fwd"].where(df["pkts_fwd"] > 0, 0)
        denom_bwd = df["pkts_bwd"].where(df["pkts_bwd"] > 0, 0)

        # Evitar división por 0 con where
        for c in SIZE_FWD:
            df[c] = df[c].where(denom_fwd > 0, 0) / denom_fwd.where(denom_fwd > 0, 1)
        for c in SIZE_BWD:
            df[c] = df[c].where(denom_bwd > 0, 0) / denom_bwd.where(denom_bwd > 0, 1)

        # ===== Normalización IPT bins =====
        # IPT tiene (pkts - 1) intervalos si pkts>=2, si no -> 0
        denom_fwd_ipt = (df["pkts_fwd"] - 1).where(df["pkts_fwd"] >= 2, 0)
        denom_bwd_ipt = (df["pkts_bwd"] - 1).where(df["pkts_bwd"] >= 2, 0)

        for c in IPT_FWD:
            df[c] = df[c].where(denom_fwd_ipt > 0, 0) / denom_fwd_ipt.where(denom_fwd_ipt > 0, 1)
        for c in IPT_BWD:
            df[c] = df[c].where(denom_bwd_ipt > 0, 0) / denom_bwd_ipt.where(denom_bwd_ipt > 0, 1)

        # Escribe incremental (streaming)
        df.to_csv(out, sep=";", index=False, mode="w" if first else "a",
                  header=first, float_format=args.float_fmt)
        first = False
        rows += len(df)
        if rows % (chunk * 5) == 0:
            print(f"[PROGRESS] rows_written={rows:,}")

    print(f"[DONE] Output: {out}")
    print(f"[STATS] rows={rows:,}")

if __name__ == "__main__":
    main()
