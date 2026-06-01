#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   pcap_to_flow_features.sh <pcap> <out_csv> <log_file>
PCAP="${1:?PCAP requerido}"
OUT_CSV="${2:?OUT_CSV requerido}"
LOG="${3:?LOG requerido}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AWK_SCRIPT="$SCRIPT_DIR/packets_to_flow_features.awk"

echo "[INFO] PCAP=$PCAP"            >  "$LOG"
echo "[INFO] OUT_CSV=$OUT_CSV"      >> "$LOG"
echo "[INFO] AWK_SCRIPT=$AWK_SCRIPT">> "$LOG"

# Checks rápidos
if [[ ! -f "$PCAP" ]]; then
  echo "[ERROR] No existe PCAP: $PCAP" >> "$LOG"
  exit 2
fi
if [[ ! -f "$AWK_SCRIPT" ]]; then
  echo "[ERROR] No existe AWK script: $AWK_SCRIPT" >> "$LOG"
  exit 2
fi
command -v tshark >/dev/null 2>&1 || { echo "[ERROR] tshark no está instalado/en PATH" >> "$LOG"; exit 2; }
command -v awk    >/dev/null 2>&1 || { echo "[ERROR] awk no está instalado/en PATH" >> "$LOG"; exit 2; }

# Asegura carpeta destino
mkdir -p "$(dirname "$OUT_CSV")"

echo "[INFO] Ejecutando tshark | awk ..." >> "$LOG"

# Importante: NO usamos -E separator=, si luego AWK espera coma.
# Aquí usamos coma y AWK FS="," como en tu script.
tshark -r "$PCAP" \
  -Y "ip && (tcp || udp)" \
  -T fields \
  -E header=y \
  -E separator=, \
  -E occurrence=f \
  -e frame.time_epoch \
  -e ip.src \
  -e ip.dst \
  -e tcp.srcport \
  -e tcp.dstport \
  -e udp.srcport \
  -e udp.dstport \
  -e frame.len \
  -e ip.proto \
  2>>"$LOG" \
| awk -f "$AWK_SCRIPT" \
  > "$OUT_CSV"

# Sanity check salida
if [[ ! -s "$OUT_CSV" ]]; then
  echo "[ERROR] OUT_CSV vacío o no creado: $OUT_CSV" >> "$LOG"
  exit 3
fi

echo "[INFO] OK: generado $(wc -l < "$OUT_CSV") líneas" >> "$LOG"
