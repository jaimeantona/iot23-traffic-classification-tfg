#!/usr/bin/env bash
#
# Merge the flow-feature CSV files produced for individual PCAP files.
#
# Usage:
#   ./merge_all_per_pcap_to_one.sh <per_pcap_dir> <output_csv> [gzip_copy]
#
# Arguments:
#   per_pcap_dir  Directory containing the per-PCAP CSV files.
#   output_csv    Destination of the merged semicolon-separated CSV.
#   gzip_copy     Optional: 1 to create a compressed copy, 0 otherwise.

set -euo pipefail

PER_PCAP_DIR="${1:?Usage: merge_all_per_pcap_to_one.sh <per_pcap_dir> <output_csv> [gzip_copy]}"
FINAL_CSV="${2:?Usage: merge_all_per_pcap_to_one.sh <per_pcap_dir> <output_csv> [gzip_copy]}"
GZIP_COPY="${3:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

if [[ ! -d "$PER_PCAP_DIR" ]]; then
  echo "[ERROR] Input directory not found: $PER_PCAP_DIR" >&2
  exit 2
fi

if [[ "$GZIP_COPY" != "0" && "$GZIP_COPY" != "1" ]]; then
  echo "[ERROR] gzip_copy must be 0 or 1." >&2
  exit 2
fi

temporary_output="${FINAL_CSV}.tmp"
mkdir -p "$(dirname "$FINAL_CSV")"
rm -f "$temporary_output"

mapfile -t csv_files < <(find "$PER_PCAP_DIR" -maxdepth 1 -type f -name "*.csv" | sort)
if (( ${#csv_files[@]} == 0 )); then
  echo "[ERROR] No CSV files found in: $PER_PCAP_DIR" >&2
  exit 3
fi

header_source=""
for file_path in "${csv_files[@]}"; do
  if [[ -s "$file_path" ]] && (( $(wc -l < "$file_path") >= 2 )); then
    header_source="$file_path"
    break
  fi
done

if [[ -z "$header_source" ]]; then
  echo "[ERROR] No CSV file contains both a header and data rows." >&2
  exit 4
fi

echo "[INFO] Input directory: $PER_PCAP_DIR"
echo "[INFO] Output CSV: $FINAL_CSV"
echo "[INFO] Header source: $header_source"

head -n 1 "$header_source" > "$temporary_output"

appended=0
skipped_empty=0
skipped_header_only=0
total_files=${#csv_files[@]}

for file_path in "${csv_files[@]}"; do
  if [[ ! -s "$file_path" ]]; then
    ((skipped_empty++)) || true
    continue
  fi

  line_count="$(wc -l < "$file_path" || echo 0)"
  if (( line_count <= 1 )); then
    ((skipped_header_only++)) || true
    continue
  fi

  tail -n +2 "$file_path" >> "$temporary_output"
  ((appended++)) || true

  if (( appended % PROGRESS_EVERY == 0 )); then
    echo "[INFO] Files appended: $appended/$total_files"
  fi
done

mv -f "$temporary_output" "$FINAL_CSV"

echo "[INFO] Merge completed: $FINAL_CSV"
echo "[INFO] Files appended=$appended | empty=$skipped_empty | header_only=$skipped_header_only"
echo "[INFO] Total output lines: $(wc -l < "$FINAL_CSV")"

if [[ "$GZIP_COPY" == "1" ]]; then
  gzip -c "$FINAL_CSV" > "${FINAL_CSV}.gz"
  echo "[INFO] Compressed copy created: ${FINAL_CSV}.gz"
fi
