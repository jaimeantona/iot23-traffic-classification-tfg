#!/usr/bin/env bash
#
# Process all PCAP chunks associated with a single IoT-23 scenario.
#
# Usage:
#   ./run_one_scenario_chunks.sh <scenario_dir> <label> <output_dir>
#
# The script prioritises available chunk directories from the smallest
# configured chunk size to the largest one. If no chunks are available,
# it processes repaired ``*_fixed.pcap`` files or ordinary PCAP files.

set -euo pipefail

SCENARIO_DIR="${1:?Usage: run_one_scenario_chunks.sh <scenario_dir> <label> <output_dir>}"
LABEL="${2:?Usage: run_one_scenario_chunks.sh <scenario_dir> <label> <output_dir>}"
OUTPUT_DIR="${3:?Usage: run_one_scenario_chunks.sh <scenario_dir> <label> <output_dir>}"

CHUNK_DIRS=("chunks_25mb" "chunks_50mb" "chunks_100mb" "chunks_500mb")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PCAP_TO_CSV="$SCRIPT_DIR/pcap_to_flow_features.sh"

PER_PCAP_DIR="$OUTPUT_DIR/per_pcap"
LOG_DIR="$OUTPUT_DIR/logs"
SCENARIO_NAME="$(basename "$SCENARIO_DIR")"
MANIFEST="$OUTPUT_DIR/manifest_${SCENARIO_NAME}.csv"

mkdir -p "$PER_PCAP_DIR" "$LOG_DIR"

if [[ ! -d "$SCENARIO_DIR" ]]; then
  echo "[ERROR] Scenario directory not found: $SCENARIO_DIR" >&2
  exit 2
fi

if [[ ! -x "$PCAP_TO_CSV" ]]; then
  echo "[ERROR] Script is not executable: $PCAP_TO_CSV" >&2
  echo "        Run: chmod +x \"$PCAP_TO_CSV\"" >&2
  exit 2
fi

find_scenario_pcaps() {
  local chunk_dir
  local -a repaired_pcaps

  for chunk_dir in "${CHUNK_DIRS[@]}"; do
    if [[ -d "$SCENARIO_DIR/$chunk_dir" ]]; then
      find "$SCENARIO_DIR/$chunk_dir" -type f -name "*.pcap" 2>/dev/null | sort
      return 0
    fi
  done

  mapfile -t repaired_pcaps < <(
    find "$SCENARIO_DIR" -type f -name "*_fixed.pcap" 2>/dev/null | sort
  )

  if (( ${#repaired_pcaps[@]} > 0 )); then
    printf '%s\n' "${repaired_pcaps[@]}"
    return 0
  fi

  find "$SCENARIO_DIR" -type f -name "*.pcap" \
    ! -name "*only15000000.pcap" \
    ! -name "test.pcap" \
    2>/dev/null | sort
}

is_invalid_csv() {
  local file_path="$1"

  [[ ! -f "$file_path" ]] && return 0
  [[ ! -s "$file_path" ]] && return 0
  head -n 1 "$file_path" | grep -q '^flow_id;proto;duration;' || return 0
  return 1
}

echo "pcap_path;scenario_path;label;out_csv;status;message" > "$MANIFEST"

echo "[INFO] Scenario: $SCENARIO_DIR | label=$LABEL"
echo "[INFO] Output directory: $OUTPUT_DIR"

mapfile -t pcaps < <(find_scenario_pcaps)

if (( ${#pcaps[@]} == 0 )); then
  echo "[WARNING] No PCAP files found in: $SCENARIO_DIR" >&2
  exit 0
fi

for pcap in "${pcaps[@]}"; do
  base_name="$(basename "$pcap")"
  identifier="$(printf "%s" "$pcap" | sha1sum | awk '{print substr($1, 1, 10)}')"
  output_csv="$PER_PCAP_DIR/${identifier}__${base_name}.csv"
  log_file="$LOG_DIR/${identifier}__${base_name}.log"

  if [[ -f "$output_csv" ]] && ! is_invalid_csv "$output_csv"; then
    echo "[INFO] Skipping existing output: $output_csv"
    echo "$pcap;$SCENARIO_DIR;$LABEL;$output_csv;OK;skipped_existing_output" >> "$MANIFEST"
    continue
  fi

  if [[ -f "$output_csv" ]] && is_invalid_csv "$output_csv"; then
    echo "[INFO] Removing invalid output before reprocessing: $output_csv"
    rm -f "$output_csv"
  fi

  echo "[INFO] Processing PCAP: $pcap"

  if "$PCAP_TO_CSV" "$pcap" "$output_csv" "$log_file"; then
    if is_invalid_csv "$output_csv"; then
      echo "$pcap;$SCENARIO_DIR;$LABEL;$output_csv;FAIL;invalid_csv_header" >> "$MANIFEST"
      echo "[WARNING] Invalid output CSV generated: $output_csv" >&2
    else
      echo "$pcap;$SCENARIO_DIR;$LABEL;$output_csv;OK;-" >> "$MANIFEST"
    fi
  else
    return_code=$?
    echo "$pcap;$SCENARIO_DIR;$LABEL;$output_csv;FAIL;pcap_to_flow_features_rc=$return_code" >> "$MANIFEST"
    echo "[WARNING] Processing failed. See log: $log_file" >&2
  fi
done

echo
echo "[INFO] Processing completed. Manifest: $MANIFEST"

