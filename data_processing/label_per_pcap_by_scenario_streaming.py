#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import pandas as pd

# ========== RUTAS (AJUSTA SI HACE FALTA) ==========
PER_PCAP_BY_SCENARIO = Path("/home/jaimeant/iot23_dataset/pipeline_hist/out/per_pcap_by_scenario")

# CSV oficial IoT-23 Assigned Labels (pon aquí la ruta REAL en tu WSL)
ASSIGNED_LABELS_CSV = Path("/home/jaimeant/iot23_dataset/assigned_labels.csv")

OUT_DIR = Path("/home/jaimeant/iot23_dataset/pipeline_hist/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DATASET = OUT_DIR / "dataset_expanded_labeled_raw_byScenario_streaming.csv"
OUT_REPORT  = OUT_DIR / "labeling_report_byScenario_streaming.csv"
# ================================================


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Tus per_pcap suelen ser TSV (tab). Probamos varios separadores.
    """
    seps = ["\t", ";", ",", r"\s+"]
    last_err = None
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No pude leer {path}. Último error: {last_err}")


def clean_label(label: str) -> str:
    """
    Normaliza labels raros: quita saltos de línea/espacios.
    """
    if label is None:
        return None
    return str(label).replace("\n", "").replace("\r", "").strip()


def load_capture_to_category_expanded(labels_csv: Path) -> dict:
    """
    Devuelve: capture_name -> category_expanded

    Regla:
      - Malware -> categoría con mayor conteo (ignorando Benign)
      - Honeypot NO se toma de aquí: lo forzamos a Benign en el bucle principal.
    """
    df = pd.read_csv(labels_csv)

    # Evitamos warning usando regex sin grupos capturables: (?: ...)
    pattern = r"CTU-(?:IoT-Malware|Honeypot)-Capture-"

    capture_col = None
    for c in df.columns:
        if df[c].astype(str).str.contains(pattern, regex=True).any():
            capture_col = c
            break
    if capture_col is None:
        raise RuntimeError("No encontré columna con CTU-...-Capture-... en el CSV oficial.")

    # Columnas numéricas = categorías (Benign, C&C, DDoS, FileDownload, Okiru, ...)
    numeric_cols = [c for c in df.columns if c != capture_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise RuntimeError("No encontré columnas numéricas de categorías en el CSV oficial.")

    capture_to_cat = {}

    for _, row in df.iterrows():
        capture = str(row[capture_col]).strip()
        if not capture.startswith("CTU-"):
            continue

        # Solo mapeamos malware aquí; honeypot lo forzamos luego a Benign
        if "CTU-IoT-Malware" not in capture:
            continue

        candidates = [c for c in numeric_cols if c.lower() != "benign"]

        best_col, best_val = None, -1
        for c in candidates:
            try:
                v = int(row[c])
            except Exception:
                continue
            if v > best_val:
                best_val = v
                best_col = c

        capture_to_cat[capture] = clean_label(best_col) if best_col is not None else "Attack"

    return capture_to_cat


def normalize_folder_to_capture(folder_name: str) -> str:
    """
    Soporta carpetas tipo:
      CTU-Honeypot-Capture-7-1__Somfy-01
    -> CTU-Honeypot-Capture-7-1
    """
    m = re.match(r"^(CTU-(?:IoT-Malware|Honeypot)-Capture-\d+-\d+)", folder_name)
    return m.group(1) if m else folder_name


def main():
    if not PER_PCAP_BY_SCENARIO.exists():
        raise SystemExit(f"No existe: {PER_PCAP_BY_SCENARIO}")

    if not ASSIGNED_LABELS_CSV.exists():
        raise SystemExit(
            f"No existe: {ASSIGNED_LABELS_CSV}\n"
            f"➡️ Pon el CSV oficial Assigned Labels ahí o cambia la ruta en el script."
        )

    capture_to_cat = load_capture_to_category_expanded(ASSIGNED_LABELS_CSV)

    scenario_dirs = sorted([p for p in PER_PCAP_BY_SCENARIO.iterdir() if p.is_dir()])
    if not scenario_dirs:
        raise SystemExit("No hay carpetas dentro de per_pcap_by_scenario.")

    # Borramos salida previa si existe
    if OUT_DATASET.exists():
        OUT_DATASET.unlink()

    report_rows = []
    label_row_counts = {}  # conteo de filas por label (sin cargar todo)

    wrote_header = False

    for scen_dir in scenario_dirs:
        scen_name = scen_dir.name
        capture_name = normalize_folder_to_capture(scen_name)

        csv_files = sorted([p for p in scen_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])

        # ✅ FIX: Honeypot siempre Benign
        if capture_name.startswith("CTU-Honeypot"):
            label = "Benign"
        else:
            label = clean_label(capture_to_cat.get(capture_name))

        if label is None:
            report_rows.append({
                "scenario_folder": scen_name,
                "capture_name": capture_name,
                "label": None,
                "csv_files": len(csv_files),
                "rows_total": 0,
                "status": "NO_LABEL_FOR_CAPTURE"
            })
            print(f"[SKIP] {scen_name} -> (no label) | files={len(csv_files)}")
            continue

        rows_total = 0

        for f in csv_files:
            df = read_csv_robust(f)
            df["capture"] = capture_name
            df["label"] = label

            # streaming write
            df.to_csv(
                OUT_DATASET,
                index=False,
                sep=";",
                mode="a",
                header=(not wrote_header)
            )
            wrote_header = True
            rows_total += len(df)

        report_rows.append({
            "scenario_folder": scen_name,
            "capture_name": capture_name,
            "label": label,
            "csv_files": len(csv_files),
            "rows_total": rows_total,
            "status": "OK"
        })

        label_row_counts[label] = label_row_counts.get(label, 0) + rows_total

        print(f"[OK] {scen_name} -> {label} | files={len(csv_files)} rows={rows_total}")

    rep = pd.DataFrame(report_rows)
    rep.to_csv(OUT_REPORT, index=False)

    print("\n✅ Dataset streaming guardado en:", OUT_DATASET)
    print("🧾 Reporte guardado en:", OUT_REPORT)

    print("\nDistribución aproximada (filas) por label:")
    for k, v in sorted(label_row_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:30s} {v}")


if __name__ == "__main__":
    main()