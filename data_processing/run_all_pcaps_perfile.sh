


#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config (override con env)
# =========================
ROOT="${ROOT:-/home/jaimeant/iot23_dataset/dataset/opt/Malware-Project/BigDataset/IoTScenarios}"
MAPPING="${MAPPING:-/home/jaimeant/iot23_dataset/mapping_manual_abs.csv}"
OUT_DIR="${OUT_DIR:-/home/jaimeant/iot23_dataset/pipeline_hist/out}"

# (aunque digas que ya no quedan chunks, lo dejamos por si acaso)
CHUNK100="${CHUNK100:-chunks_100mb}"
CHUNK500="${CHUNK500:-chunks_500mb}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PCAP2CSV="$SCRIPT_DIR/pcap_to_flow_features.sh"

PER_PCAP_DIR="$OUT_DIR/per_pcap"
LOG_DIR="$OUT_DIR/logs"
TMP_DIR="$OUT_DIR/tmp"
MANIFEST="$OUT_DIR/manifest_pcaps.csv"

mkdir -p "$PER_PCAP_DIR" "$LOG_DIR" "$TMP_DIR"

echo "ROOT=$ROOT"
echo "MAPPING=$MAPPING"
echo "OUT_DIR=$OUT_DIR"
echo "CHUNK_DIRNAMEs=$CHUNK100,$CHUNK500"
echo

if [[ ! -f "$MAPPING" ]]; then
  echo "ERROR: no existe mapping: $MAPPING" >&2
  exit 2
fi

if [[ ! -x "$PCAP2CSV" ]]; then
  echo "ERROR: no es ejecutable: $PCAP2CSV (haz chmod +x)" >&2
  exit 2
fi

# Reset manifest (si quieres append en vez de reset, comenta estas 2 líneas)
echo "pcap_path;scenario_path;label;out_csv;status;message" > "$MANIFEST"

# =========================
# Helpers
# =========================
trim() { sed 's/^[[:space:]]*//; s/[[:space:]]*$//'; }

# =========================
# Loop mapping
# mapping_manual_abs.csv: header + filas "scenario_path,label"
# =========================
tail -n +2 "$MAPPING" | while IFS=',' read -r SCENARIO_PATH LABEL; do
  SCENARIO_PATH="$(printf "%s" "${SCENARIO_PATH:-}" | tr -d '\r' | trim)"
  LABEL="$(printf "%s" "${LABEL:-}" | tr -d '\r' | trim)"

  [[ -z "$SCENARIO_PATH" ]] && continue

  # Asegurar que es ruta absoluta dentro de ROOT (si tu mapping ya es absoluto, no toca nada)
  # Si alguien metió algo relativo por error:
  if [[ "$SCENARIO_PATH" != /* ]]; then
    SCENARIO_PATH="$ROOT/$SCENARIO_PATH"
  fi

  echo
  echo "=== Escenario: $SCENARIO_PATH | label=$LABEL ==="

  if [[ ! -d "$SCENARIO_PATH" ]]; then
    echo "[WARN] No existe directorio: $SCENARIO_PATH"
    continue
  fi

  # -------------------------
  # Selección de PCAPs (prioridad):
  # 1) chunks_100mb si existe
  # 2) chunks_500mb si existe
  # 3) *_fixed.pcap si existe
  # 4) *.pcap (excluyendo only15000000 y test.pcap)
  # -------------------------
  PCAPS=()

  if [[ -d "$SCENARIO_PATH/$CHUNK100" ]]; then
    mapfile -t PCAPS < <(find "$SCENARIO_PATH/$CHUNK100" -type f -name "*.pcap" 2>/dev/null | sort)
    echo "[INFO] Usando chunks: $CHUNK100 -> ${#PCAPS[@]}"
  elif [[ -d "$SCENARIO_PATH/$CHUNK500" ]]; then
    mapfile -t PCAPS < <(find "$SCENARIO_PATH/$CHUNK500" -type f -name "*.pcap" 2>/dev/null | sort)
    echo "[INFO] Usando chunks: $CHUNK500 -> ${#PCAPS[@]}"
  else
    mapfile -t FIXED < <(find "$SCENARIO_PATH" -type f -name "*_fixed.pcap" 2>/dev/null | sort)
    if (( ${#FIXED[@]} > 0 )); then
      PCAPS=("${FIXED[@]}")
      echo "[INFO] Usando *_fixed.pcap: ${#PCAPS[@]}"
    else
      mapfile -t PCAPS < <(find "$SCENARIO_PATH" -type f -name "*.pcap" \
        ! -name "*only15000000.pcap" \
        ! -name "test.pcap" \
        2>/dev/null | sort)
      echo "[INFO] Usando *.pcap normal: ${#PCAPS[@]}"
    fi
  fi

  if (( ${#PCAPS[@]} == 0 )); then
    echo "[WARN] No hay PCAPs/chunks que procesar en: $SCENARIO_PATH"
    continue
  fi

  # -------------------------
  # Procesado secuencial 1 a 1 (RAM friendly)
  # -------------------------
  for PCAP in "${PCAPS[@]}"; do
    BASE="$(basename "$PCAP")"
    ID="$(printf "%s" "$PCAP" | sha1sum | awk '{print substr($1,1,10)}')"

    OUT_CSV="$PER_PCAP_DIR/${ID}__${BASE}.csv"
    LOG="$LOG_DIR/${ID}__${BASE}.log"

    # Si ya existe un CSV "decente", lo saltamos (evita rehacer)
    if [[ -s "$OUT_CSV" ]]; then
      # opcional: si solo tiene cabecera (1 línea), lo re-procesamos
      LINES=$(wc -l < "$OUT_CSV" | tr -d ' ')
      if (( LINES > 1 )); then
        echo "[SKIP] Ya existe: $OUT_CSV"
        echo "$PCAP;$SCENARIO_PATH;$LABEL;$OUT_CSV;OK;skip_exists" >> "$MANIFEST"
        continue
      fi
      # si existe pero vacío o solo cabecera, seguimos y lo re-generamos
    fi

    echo "Procesando PCAP: $PCAP"

    set +e
    "$PCAP2CSV" "$PCAP" "$OUT_CSV" "$LOG"
    RC=$?
    set -e

    if (( RC == 0 )); then
      echo "$PCAP;$SCENARIO_PATH;$LABEL;$OUT_CSV;OK;-" >> "$MANIFEST"
    else
      echo "$PCAP;$SCENARIO_PATH;$LABEL;$OUT_CSV;FAIL;pcap_to_flow_features_rc=$RC" >> "$MANIFEST"
      echo "[WARN] FAIL rc=$RC (ver log: $LOG)"
    fi
  done
done

echo
echo "DONE. Manifest: $MANIFEST"
