#!/usr/bin/env bash
#
# Split large PCAP files into smaller chunks using editcap.
#
# Usage:
#   ./split_dir_por_tamano_editcap.sh <input_root> [threshold_kb] [target_kb] [output_dir_name]
#
# The input root may be a single scenario directory or a directory containing
# several IoT-23 scenarios. Repaired *_fixed.pcap files are preferred when
# both the original and repaired versions exist.

set -euo pipefail

ROOT="${1:?Usage: split_dir_por_tamano_editcap.sh <input_root> [threshold_kb] [target_kb] [output_dir_name]}"
THRESHOLD_KB="${2:-1000000}"
TARGET_KB="${3:-500000}"
OUTDIR_NAME="${4:-chunks_500mb}"

command -v capinfos >/dev/null 2>&1 || {
  echo "[ERROR] capinfos is not installed or is not available in PATH." >&2
  exit 2
}
command -v editcap >/dev/null 2>&1 || {
  echo "[ERROR] editcap is not installed or is not available in PATH." >&2
  exit 2
}
command -v du >/dev/null 2>&1 || {
  echo "[ERROR] du is not installed or is not available in PATH." >&2
  exit 2
}

if [[ ! -d "$ROOT" ]]; then
  echo "[ERROR] Input root directory not found: $ROOT" >&2
  exit 2
fi

normalise_packet_count() {
  local raw="$1"
  raw="${raw// /}"

  case "$raw" in
    *k|*K) echo $(( ${raw%[kK]} * 1000 )) ;;
    *m|*M) echo $(( ${raw%[mM]} * 1000000 )) ;;
    *g|*G) echo $(( ${raw%[gG]} * 1000000000 )) ;;
    *)     echo "$raw" ;;
  esac
}

echo "[INFO] Input root: $ROOT"
echo "[INFO] Split threshold: ${THRESHOLD_KB} KB"
echo "[INFO] Target chunk size: ${TARGET_KB} KB"
echo "[INFO] Chunk directory name: $OUTDIR_NAME"

find "$ROOT" -type f -name "*.pcap" \
  ! -name "*only15000000.pcap" \
  ! -name "test.pcap" \
| while read -r pcap; do
  pcap_to_use="$pcap"

  if [[ "$pcap" != *_fixed.pcap ]]; then
    repaired_pcap="${pcap%.pcap}_fixed.pcap"
    if [[ -f "$repaired_pcap" ]]; then
      pcap_to_use="$repaired_pcap"
    fi
  fi

  size_kb="$(du -k "$pcap_to_use" | awk '{print $1}')"
  if [[ -z "${size_kb:-}" ]]; then
    echo "[WARNING] Unable to obtain file size: $pcap_to_use" >&2
    continue
  fi

  if (( size_kb <= THRESHOLD_KB )); then
    continue
  fi

  input_dir="$(dirname "$pcap_to_use")"
  base_name="$(basename "$pcap_to_use")"
  output_dir="$input_dir/$OUTDIR_NAME"
  mkdir -p "$output_dir"

  raw_packets="$(capinfos -c "$pcap_to_use" 2>/dev/null \
    | awk -F: '/Number of packets/ {print $2}' | xargs || true)"

  if [[ -z "${raw_packets:-}" ]]; then
    echo "[WARNING] Unable to obtain packet count: $pcap_to_use" >&2
    continue
  fi

  packets="$(normalise_packet_count "$raw_packets")"
  if [[ -z "${packets:-}" || ! "$packets" =~ ^[0-9]+$ || "$packets" -le 0 ]]; then
    echo "[WARNING] Invalid packet count for $pcap_to_use: $raw_packets" >&2
    continue
  fi

  size_bytes=$(( size_kb * 1024 ))
  target_bytes=$(( TARGET_KB * 1024 ))
  average_packet_size=$(( size_bytes / packets ))
  if (( average_packet_size < 1 )); then
    average_packet_size=1
  fi

  packets_per_chunk=$(( target_bytes / average_packet_size ))
  if (( packets_per_chunk < 50000 )); then
    packets_per_chunk=50000
  fi

  prefix="$output_dir/${base_name%.pcap}_chunkpkts"

  echo "[INFO] Splitting: $pcap_to_use"
  echo "[INFO] Packets=$packets | estimated packets/chunk=$packets_per_chunk | output=$output_dir"

  if ! editcap -c "$packets_per_chunk" "$pcap_to_use" "${prefix}.pcap" >/dev/null 2>&1; then
    echo "[WARNING] editcap failed for: $pcap_to_use" >&2
    continue
  fi

  chunk_count="$(find "$output_dir" -maxdepth 1 -type f \
    -name "${base_name%.pcap}_chunkpkts_*.pcap" | wc -l | awk '{print $1}')"
  echo "[INFO] Chunks generated: $chunk_count"
done

echo "[INFO] Splitting process completed."
