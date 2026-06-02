#!/usr/bin/env bash
#
# Convert a PCAP file into a semicolon-separated flow feature CSV.
#
# Usage:
#   ./pcap_to_flow_features.sh <input.pcap> <output.csv> <log_file>
#
# Requirements:
#   tshark, awk, and packets_to_flow_features.awk in the same directory.

set -euo pipefail

PCAP="${1:?Usage: pcap_to_flow_features.sh <input.pcap> <output.csv> <log_file>}"
OUT_CSV="${2:?Usage: pcap_to_flow_features.sh <input.pcap> <output.csv> <log_file>}"
LOG_FILE="${3:?Usage: pcap_to_flow_features.sh <input.pcap> <output.csv> <log_file>}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AWK_SCRIPT="$SCRIPT_DIR/packets_to_flow_features.awk"

mkdir -p "$(dirname "$OUT_CSV")" "$(dirname "$LOG_FILE")"

{
  echo "[INFO] Input PCAP: $PCAP"
  echo "[INFO] Output CSV: $OUT_CSV"
  echo "[INFO] AWK feature extractor: $AWK_SCRIPT"
} > "$LOG_FILE"

if [[ ! -f "$PCAP" ]]; then
  echo "[ERROR] Input PCAP file not found: $PCAP" >> "$LOG_FILE"
  exit 2
fi

if [[ ! -f "$AWK_SCRIPT" ]]; then
  echo "[ERROR] AWK feature extractor not found: $AWK_SCRIPT" >> "$LOG_FILE"
  exit 2
fi

command -v tshark >/dev/null 2>&1 || {
  echo "[ERROR] tshark is not installed or is not available in PATH." >> "$LOG_FILE"
  exit 2
}

command -v awk >/dev/null 2>&1 || {
  echo "[ERROR] awk is not installed or is not available in PATH." >> "$LOG_FILE"
  exit 2
}

echo "[INFO] Extracting packet fields and aggregating flows." >> "$LOG_FILE"

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
  2>> "$LOG_FILE" \
| awk -f "$AWK_SCRIPT" > "$OUT_CSV"

if [[ ! -s "$OUT_CSV" ]]; then
  echo "[ERROR] Output CSV was not created or is empty: $OUT_CSV" >> "$LOG_FILE"
  exit 3
fi

line_count="$(wc -l < "$OUT_CSV" | tr -d ' ')"
echo "[INFO] Completed successfully. Output lines: $line_count" >> "$LOG_FILE"
