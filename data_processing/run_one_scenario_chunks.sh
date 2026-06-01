#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:?Uso: run_one_scenario_chunks.sh /ruta/al/escenario [label]}"
LABEL="${2:-UNKNOWN}"

OUT_DIR="${OUT_DIR:-/home/jaimeant/iot23_dataset/pipeline_hist/out}"

# Prioridad: chunks más pequeños primero (más estable RAM)
CHUNK25="${CHUNK25:-chunks_25mb}"
CHUNK50="${CHUNK50:-chunks_50mb}"
CHUNK100="${CHUNK100:-chunks_100mb}"
CHUNK500="${CHUNK500:-chunks_500mb}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PCAP2CSV="$SCRIPT_DIR/pcap_to_flow_features.sh"

PER_PCAP_DIR="$OUT_DIR/per_pcap"
LOG_DIR="$OUT_DIR/logs"

# Un manifest por escenario (más limpio)
SCEN_NAME="$(basename "$SCENARIO")"
MANIFEST="$OUT_DIR/manifest_${SCEN_NAME}.csv"

mkdir -p "$PER_PCAP_DIR" "$LOG_DIR"
echo "pcap_path;scenario_path;label;out_csv;status;message" > "$MANIFEST"

echo "=== Escenario: $SCENARIO | label=$LABEL ==="
echo "[INFO] Salida CSV: $PER_PCAP_DIR"
echo "[INFO] Logs:       $LOG_DIR"
echo "[INFO] Manifest:   $MANIFEST"
echo

pick_chunks_dir() {
  local d
  for d in "$CHUNK25" "$CHUNK50" "$CHUNK100" "$CHUNK500"; do
    if [[ -d "$SCENARIO/$d" ]]; then
      echo "$SCENARIO/$d"
      return 0
    fi
  done
  return 1
}

# 1) Si hay chunks, usar los chunks (prioridad a chunks más pequeños)
# 2) Si no, usar *_fixed.pcap si existe
# 3) Si no, usar *.pcap (excluyendo only15000000/test)
PCAPS=()

if CHDIR="$(pick_chunks_dir)"; then
  mapfile -t PCAPS < <(find "$CHDIR" -type f -name "*.pcap" 2>/dev/null | sort)
  echo "[INFO] Usando chunks: $(basename "$CHDIR") -> ${#PCAPS[@]}"
else
  mapfile -t FIXED < <(find "$SCENARIO" -type f -name "*_fixed.pcap" 2>/dev/null | sort)
  if (( ${#FIXED[@]} > 0 )); then
    PCAPS=("${FIXED[@]}")
    echo "[INFO] Usando *_fixed.pcap: ${#PCAPS[@]}"
  else
    mapfile -t PCAPS < <(find "$SCENARIO" -type f -name "*.pcap" \
      ! -name "*only15000000.pcap" \
      ! -name "test.pcap" \
      2>/dev/null | sort)
    echo "[INFO] Usando *.pcap normal: ${#PCAPS[@]}"
  fi
fi

if (( ${#PCAPS[@]} == 0 )); then
  echo "[WARN] No hay PCAPs/chunks en $SCENARIO"
  exit 0
fi

echo

is_bad_csv() {
  local f="$1"
  # CSV inexistente o vacío -> malo
  [[ ! -f "$f" ]] && return 0
  [[ ! -s "$f" ]] && return 0
  # Si no empieza por header esperado -> malo
  head -n 1 "$f" | grep -q '^flow_id;proto;duration;' || return 0
  return 1
}

# Procesar 1 a 1 (RAM-friendly)
for PCAP in "${PCAPS[@]}"; do
  BASE="$(basename "$PCAP")"
  ID="$(printf "%s" "$PCAP" | sha1sum | awk '{print substr($1,1,10)}')"

  OUT_CSV="$PER_PCAP_DIR/${ID}__${BASE}.csv"
  LOG="$LOG_DIR/${ID}__${BASE}.log"

  # Skip si ya está bien
  if [[ -f "$OUT_CSV" ]] && ! is_bad_csv "$OUT_CSV"; then
    echo "[SKIP] Ya existe OK: $OUT_CSV"
    echo "$PCAP;$SCENARIO;$LABEL;$OUT_CSV;OK;SKIPPED_EXISTS" >> "$MANIFEST"
    continue
  fi

  # Si existe pero está mal, borramos para reintentar limpio
  if [[ -f "$OUT_CSV" ]] && is_bad_csv "$OUT_CSV"; then
    echo "[RETRY] CSV existente pero malo -> borro y reproceso: $OUT_CSV"
    rm -f "$OUT_CSV"
  fi

  echo "Procesando: $PCAP"

  set +e
  "$PCAP2CSV" "$PCAP" "$OUT_CSV" "$LOG"
  RC=$?
  set -e

  if (( RC == 0 )); then
    # sanity: header
    if is_bad_csv "$OUT_CSV"; then
      echo "[FAIL] Generó CSV inválido (sin header): $OUT_CSV"
      echo "$PCAP;$SCENARIO;$LABEL;$OUT_CSV;FAIL;csv_invalid_no_header" >> "$MANIFEST"
      # lo dejamos para investigar
    else
      echo "$PCAP;$SCENARIO;$LABEL;$OUT_CSV;OK;-" >> "$MANIFEST"
    fi
  else
    echo "[FAIL] RC=$RC (ver log: $LOG)"
    echo "$PCAP;$SCENARIO;$LABEL;$OUT_CSV;FAIL;pcap_to_flow_features_rc=$RC" >> "$MANIFEST"
  fi
done

echo
echo "DONE -> $MANIFEST"

