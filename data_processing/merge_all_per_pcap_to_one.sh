#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-/home/jaimeant/iot23_dataset/pipeline_hist/out}"
PER_PCAP_DIR="${PER_PCAP_DIR:-$OUT_DIR/per_pcap}"

FINAL_CSV="${1:-$OUT_DIR/dataset_final.csv}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

# Si quieres además un backup comprimido (recomendado), pon GZIP_COPY=1
GZIP_COPY="${GZIP_COPY:-0}"   # 0/1

if [[ ! -d "$PER_PCAP_DIR" ]]; then
  echo "ERROR: no existe PER_PCAP_DIR=$PER_PCAP_DIR" >&2
  exit 2
fi

echo "[INFO] PER_PCAP_DIR=$PER_PCAP_DIR"
echo "[INFO] FINAL_CSV=$FINAL_CSV"
echo "[INFO] PROGRESS_EVERY=$PROGRESS_EVERY"
echo "[INFO] GZIP_COPY=$GZIP_COPY"

tmp_out="${FINAL_CSV}.tmp"
rm -f "$tmp_out"
mkdir -p "$(dirname "$FINAL_CSV")"

# Lista todos los CSVs (orden estable)
mapfile -t CSVs < <(find "$PER_PCAP_DIR" -maxdepth 1 -type f -name "*.csv" | sort)

if (( ${#CSVs[@]} == 0 )); then
  echo "ERROR: no hay CSVs en $PER_PCAP_DIR" >&2
  exit 3
fi

# Encuentra el primer CSV "bueno" para sacar header
HEADER_SRC=""
for f in "${CSVs[@]}"; do
  # skip vacíos
  if [[ ! -s "$f" ]]; then
    continue
  fi
  # debe tener al menos 2 líneas (header + 1 data)
  nlines="$(wc -l < "$f" || echo 0)"
  if (( nlines >= 2 )); then
    HEADER_SRC="$f"
    break
  fi
done

if [[ -z "$HEADER_SRC" ]]; then
  echo "ERROR: no encuentro ningún CSV con >=2 líneas (todos parecen vacíos o solo header)" >&2
  exit 4
fi

echo "[INFO] Header tomado de: $HEADER_SRC"
head -n 1 "$HEADER_SRC" > "$tmp_out"

# Merge streaming: append sin header
ok=0
skip_empty=0
skip_header_only=0
total=${#CSVs[@]}

i=0
for f in "${CSVs[@]}"; do
  ((i++)) || true

  if [[ ! -s "$f" ]]; then
    ((skip_empty++)) || true
    continue
  fi

  nlines="$(wc -l < "$f" || echo 0)"
  if (( nlines <= 1 )); then
    ((skip_header_only++)) || true
    continue
  fi

  # (Opcional) Comprobar que el header coincide con el header global.
  # Lo dejamos OFF por velocidad; si quieres activarlo:
  # if ! head -n 1 "$f" | cmp -s - "$tmp_out"; then
  #   echo "[WARN] Header distinto en $f -> lo salto"
  #   continue
  # fi

  tail -n +2 "$f" >> "$tmp_out"
  ((ok++)) || true

  if (( ok % PROGRESS_EVERY == 0 )); then
    echo "[PROGRESS] appended=$ok / total_files=$total (iter=$i)  skips: empty=$skip_empty header_only=$skip_header_only"
  fi
done

mv -f "$tmp_out" "$FINAL_CSV"

echo
echo "[DONE] CSV final creado: $FINAL_CSV"
echo "[STATS] total_files=$total appended=$ok skipped_empty=$skip_empty skipped_header_only=$skip_header_only"
echo "[INFO] Líneas totales (incluye header): $(wc -l < "$FINAL_CSV")"
echo "[INFO] Tamaño: $(du -h "$FINAL_CSV" | awk '{print $1}')"

if [[ "$GZIP_COPY" == "1" ]]; then
  echo
  echo "[INFO] Generando copia comprimida (backup): ${FINAL_CSV}.gz"
  gzip -c "$FINAL_CSV" > "${FINAL_CSV}.gz"
  echo "[DONE] Backup gzip: ${FINAL_CSV}.gz  size=$(du -h "${FINAL_CSV}.gz" | awk '{print $1}')"
fi
