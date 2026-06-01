#!/usr/bin/env python3
import argparse
import pandas as pd

FWD_SIZE = [f"fwd_size_bin{i}" for i in range(1, 9)]
BWD_SIZE = [f"bwd_size_bin{i}" for i in range(1, 9)]
FWD_IPT  = [f"fwd_ipt_bin{i}"  for i in range(1, 9)]
BWD_IPT  = [f"bwd_ipt_bin{i}"  for i in range(1, 9)]

def _ensure_int_nonneg(df, cols):
    # Convierte a entero no negativo (por si acaso)
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round().astype("int64")
            df[c] = df[c].clip(lower=0)
    return df

def _cap_bins_rowwise(df, bin_cols, cap_series):
    """
    Asegura sum(bin_cols) <= cap_series por fila.
    Si se excede, recorta restando desde el bin más alto hacia abajo.
    """
    # Trabajamos en numpy para velocidad
    import numpy as np
    bins = df[bin_cols].to_numpy(dtype=np.int64, copy=True)
    cap  = cap_series.to_numpy(dtype=np.int64, copy=False)

    sums = bins.sum(axis=1)
    excess = sums - cap
    mask = excess > 0
    if not mask.any():
        df[bin_cols] = bins
        return df

    idxs = np.where(mask)[0]
    for r in idxs:
        e = int(excess[r])
        # restar desde el último bin hacia el primero
        for j in range(len(bin_cols)-1, -1, -1):
            if e <= 0:
                break
            take = min(e, int(bins[r, j]))
            if take > 0:
                bins[r, j] -= take
                e -= take
        # Si aún quedara exceso (raro), ya no hay más que quitar

    df[bin_cols] = bins
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV de entrada (sep=';')")
    ap.add_argument("--output", required=True, help="CSV de salida arreglado (sep=';')")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--sep", default=";")
    args = ap.parse_args()

    inp = args.input
    out = args.output
    sep = args.sep
    chunksize = args.chunksize

    first = True
    total_rows = 0
    fixed_pkts = 0
    fixed_ipt  = 0

    reader = pd.read_csv(inp, sep=sep, chunksize=chunksize, low_memory=False)
    for chunk_idx, df in enumerate(reader, start=1):
        # Sanear bins a enteros no negativos
        df = _ensure_int_nonneg(df, FWD_SIZE + BWD_SIZE + FWD_IPT + BWD_IPT)

        # Recalcular pkts_* como sum(size_bins)
        if "pkts_fwd" not in df.columns or "pkts_bwd" not in df.columns:
            raise RuntimeError("No encuentro pkts_fwd/pkts_bwd en el CSV.")

        old_pf = pd.to_numeric(df["pkts_fwd"], errors="coerce").fillna(0).round().astype("int64")
        old_pb = pd.to_numeric(df["pkts_bwd"], errors="coerce").fillna(0).round().astype("int64")

        new_pf = df[FWD_SIZE].sum(axis=1).astype("int64")
        new_pb = df[BWD_SIZE].sum(axis=1).astype("int64")

        fixed_pkts += int((old_pf != new_pf).sum() + (old_pb != new_pb).sum())
        df["pkts_fwd"] = new_pf
        df["pkts_bwd"] = new_pb

        # Ajustar IPT bins para que no excedan pkts-1
        cap_fwd = (df["pkts_fwd"] - 1).clip(lower=0).astype("int64")
        cap_bwd = (df["pkts_bwd"] - 1).clip(lower=0).astype("int64")

        sum_fipt_before = df[FWD_IPT].sum(axis=1)
        sum_bipt_before = df[BWD_IPT].sum(axis=1)

        df = _cap_bins_rowwise(df, FWD_IPT, cap_fwd)
        df = _cap_bins_rowwise(df, BWD_IPT, cap_bwd)

        sum_fipt_after = df[FWD_IPT].sum(axis=1)
        sum_bipt_after = df[BWD_IPT].sum(axis=1)

        fixed_ipt += int((sum_fipt_before != sum_fipt_after).sum() + (sum_bipt_before != sum_bipt_after).sum())

        # (Opcional) asegurar duration >= 0 si existiera algún -0.000 o similar
        if "duration" in df.columns:
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
            df.loc[df["duration"] < 0, "duration"] = 0.0

        # Escribir
        df.to_csv(out, sep=sep, index=False, mode="w" if first else "a", header=first)
        first = False

        total_rows += len(df)
        if chunk_idx % 5 == 0:
            print(f"[PROGRESS] chunks={chunk_idx} rows_written={total_rows:,} fixed_pkts_so_far={fixed_pkts:,} fixed_ipt_so_far={fixed_ipt:,}")

    print(f"[DONE] Output: {out}")
    print(f"[STATS] rows={total_rows:,} fixed_pkts_edits={fixed_pkts:,} fixed_ipt_edits={fixed_ipt:,}")

if __name__ == "__main__":
    main()
