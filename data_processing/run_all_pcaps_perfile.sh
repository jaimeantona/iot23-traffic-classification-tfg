#!/usr/bin/env bash
#
# Process the PCAP files listed in a scenario-to-label mapping file.
#
# Usage:
#   ./run_all_pcaps_perfile.sh <scenarios_root> <mapping.csv> <output_dir>
#
# Mapping format:
#   scenario_path,label
#
# ``scenario_path`` may be absolute or relative to ``scenarios_root``.
# For each scenario, the script prioritises chunk directories, repaired
# ``*_fixed.pcap`` files, and then ordinary PCAP files.

set -euo pipefail

SCENARIOS_ROOT="${1:?Usage: run_all_pcaps_perfile.sh <scenarios_root> <mapping.csv> <output_dir>}"
MAPPING_FILE="${2:?Usage: run_all_pcaps_perfile.sh <scenarios_root> <mapping.csv> <output_dir>}"
OUTPUT_DIR="${3:?Usage: run_all_pcaps_perfile.sh <scenarios_root> <mapping.csv> <output_dir>}"

CHUNK_DIRS=("chunks_25mb" "chunks_50mb" "chunks_100mb" "chunks_500mb")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PCAP_TO_CSV="$SCRIPT_DIR/pcap_to_flow_features.sh"

PER_PCAP_DIR="$OUTPUT_DIR/per_pcap"
LOG_DIR="$OUTPUT_DIR/logs"
MANIFEST="$OUTPUT_DIR/manifest_pcaps.csv"

mkdir -p "$PER_PCAP_DIR" "$LOG_DIR"

if [[ ! -d "$SCENARIOS_ROOT" ]]; then
  echo "[ERROR] Scenarios root directory not found: $SCENARIOS_ROOT" >&2
  exit 2
fi

if [[ ! -f "$MAPPING_FILE" ]]; then
  echo "[ERROR] Mapping file not found: $MAPPING_FILE" >&2
  exit 2
fi

if [[ ! -x "$PCAP_TO_CSV" ]]; then
  echo "[ERROR] Script is not executable: $PCAP_TO_CSV" >&2
  echo "        Run: chmod +x \"$PCAP_TO_CSV\"" >&2
  exit 2
fi

trim() {
  sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

find_scenario_pcaps() {
  local scenario_path="$1"
  local chunk_dir
  local -a repaired_pcaps

  for chunk_dir in "${CHUNK_DIRS[@]}"; do
    if [[ -d "$scenario_path/$chunk_dir" ]]; then
      find "$scenario_path/$chunk_dir" -type f -name "*.pcap" 2>/dev/null | sort
      return 0
    fi
  done

  mapfile -t repaired_pcaps < <(
    find "$scenario_path" -type f -name "*_fixed.pcap" 2>/dev/null | sort
  )

  if (( ${#repaired_pcaps[@]} > 0 )); then
    printf '%s\n' "${repaired_pcaps[@]}"
    return 0
  fi

  find "$scenario_path" -type f -name "*.pcap" \
    ! -name "*only15000000.pcap" \
    ! -name "test.pcap" \
    2>/dev/null | sort
}

echo "pcap_path;scenario_path;label;out_csv;status;message" > "$MANIFEST"

echo "[INFO] Scenarios root: $SCENARIOS_ROOT"
echo "[INFO] Mapping file: $MAPPING_FILE"
echo "[INFO] Output directory: $OUTPUT_DIR"

tail -n +2 "$MAPPING_FILE" | while IFS=',' read -r raw_scenario_path raw_label; do
  scenario_path="$(printf "%s" "${raw_scenario_path:-}" | tr -d '\r' | trim)"
  label="$(printf "%s" "${raw_label:-}" | tr -d '\r' | trim)"

  [[ -z "$scenario_path" ]] && continue

  if [[ "$scenario_path" != /* ]]; then
    scenario_path="$SCENARIOS_ROOT/$scenario_path"
  fi

  echo
  echo "[INFO] Processing scenario: $scenario_path | label=$label"

  if [[ ! -d "$scenario_path" ]]; then
    echo "[WARNING] Scenario directory not found: $scenario_path" >&2
    continue
  fi

  mapfile -t pcaps < <(find_scenario_pcaps "$scenario_path")

  if (( ${#pcaps[@]} == 0 )); then
    echo "[WARNING] No PCAP files found in: $scenario_path" >&2
    continue
  fi

  for pcap in "${pcaps[@]}"; do
    base_name="$(basename "$pcap")"
    identifier="$(printf "%s" "$pcap" | sha1sum | awk '{print substr($1, 1, 10)}')"
    output_csv="$PER_PCAP_DIR/${identifier}__${base_name}.csv"
    log_file="$LOG_DIR/${identifier}__${base_name}.log"

    if [[ -s "$output_csv" ]] && (( $(wc -l < "$output_csv") > 1 )); then
      echo "[INFO] Skipping existing output: $output_csv"
      echo "$pcap;$scenario_path;$label;$output_csv;OK;skipped_existing_output" >> "$MANIFEST"
      continue
    fi

    echo "[INFO] Processing PCAP: $pcap"

    if "$PCAP_TO_CSV" "$pcap" "$output_csv" "$log_file"; then
      echo "$pcap;$scenario_path;$label;$output_csv;OK;-" >> "$MANIFEST"
    else
      return_code=$?
      echo "$pcap;$scenario_path;$label;$output_csv;FAIL;pcap_to_flow_features_rc=$return_code" >> "$MANIFEST"
      echo "[WARNING] Processing failed. See log: $log_file" >&2
    fi
  done
done

echo
echo "[INFO] Processing completed. Manifest: $MANIFEST"

