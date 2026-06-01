#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Uso: split_dir_por_tamano_editcap.sh /ruta/al/escenario_o_root [THRESHOLD_KB] [TARGET_KB] [OUTDIR_NAME]}"
THRESHOLD_KB="${2:-1000000}"     # si el PCAP pesa más de esto, lo partimos (1.000.000 KB)
TARGET_KB="${3:-500000}"         # tamaño objetivo por chunk (ej: 500000 KB = ~500MB, 100000 KB = ~100MB)
OUTDIR_NAME="${4:-chunks_500mb}" # carpeta dentro del mismo dir del pcap

echo "ROOT=$ROOT"
echo "THRESHOLD_KB=$THRESHOLD_KB KB | TARGET_KB=$TARGET_KB KB | OUTDIR_NAME=$OUTDIR_NAME"
echo

command -v capinfos >/dev/null 2>&1 || { echo "ERROR: falta capinfos. Instala: sudo apt install wireshark-common"; exit 1; }
command -v editcap  >/dev/null 2>&1 || { echo "ERROR: falta editcap. Instala: sudo apt install wireshark-common"; exit 1; }
command -v du       >/dev/null 2>&1 || { echo "ERROR: falta du (coreutils)"; exit 1; }

# Recorre todos los PCAPs en ROOT (puede ser un escenario o el root de escenarios)
find "$ROOT" -type f -name "*.pcap" \
  ! -name "*only15000000.pcap" \
  ! -name "test.pcap" \
| while read -r PCAP; do

  # Preferir *_fixed.pcap si existe para el mismo nombre base
  # (si el PCAP ya es _fixed, se usa tal cual)
  PCAP_USE="$PCAP"
  if [[ "$PCAP" != *_fixed.pcap ]]; then
    FIXED="${PCAP%.pcap}_fixed.pcap"
    if [[ -f "$FIXED" ]]; then
      PCAP_USE="$FIXED"
    fi
  fi

  # Tamaño en KB
  SIZE_KB="$(du -k "$PCAP_USE" | awk '{print $1}')"
  if [[ -z "${SIZE_KB:-}" ]]; then
    echo "  [WARN] No pude leer tamaño con du: $PCAP_USE"
    continue
  fi

  # Solo split si supera umbral
  if (( SIZE_KB <= THRESHOLD_KB )); then
    continue
  fi

  DIR="$(dirname "$PCAP_USE")"
  BASE="$(basename "$PCAP_USE")"
  OUTDIR="$DIR/$OUTDIR_NAME"
  mkdir -p "$OUTDIR"

  echo "[SPLIT] $PCAP_USE  (${SIZE_KB} KB) -> $OUTDIR"

  # ---- PKTS robusto (convierte 73 M / 1264 k / etc a entero) ----
  PKTS_RAW="$(capinfos -c "$PCAP_USE" 2>/dev/null | awk -F: '/Number of packets/ {print $2}' | xargs || true)"
  # Normaliza: "73 M" -> "73M", "1264 k"->"1264k"
  PKTS_RAW="${PKTS_RAW// /}"

  if [[ -z "${PKTS_RAW:-}" ]]; then
    echo "  [WARN] No pude leer 'Number of packets' con capinfos. Saltando."
    continue
  fi

  case "$PKTS_RAW" in
    *k) PKTS=$(( ${PKTS_RAW%k} * 1000 )) ;;
    *K) PKTS=$(( ${PKTS_RAW%K} * 1000 )) ;;
    *m) PKTS=$(( ${PKTS_RAW%m} * 1000000 )) ;;
    *M) PKTS=$(( ${PKTS_RAW%M} * 1000000 )) ;;
    *g) PKTS=$(( ${PKTS_RAW%g} * 1000000000 )) ;;
    *G) PKTS=$(( ${PKTS_RAW%G} * 1000000000 )) ;;
    *)  PKTS=$PKTS_RAW ;;
  esac

  if [[ -z "${PKTS:-}" || "$PKTS" -le 0 ]]; then
    echo "  [WARN] PKTS inválido (raw='$PKTS_RAW' -> '$PKTS'). Saltando."
    continue
  fi
  # ---------------------------------------------------------------

  SIZE_BYTES=$(( SIZE_KB * 1024 ))
  TARGET_BYTES=$(( TARGET_KB * 1024 ))

  # Tamaño medio de paquete y pkts por chunk
  AVG_PKT=$(( SIZE_BYTES / PKTS ))
  if (( AVG_PKT < 1 )); then AVG_PKT=1; fi

  PKTS_PER_CHUNK=$(( TARGET_BYTES / AVG_PKT ))
  # Evitar generar millones de ficheros
  if (( PKTS_PER_CHUNK < 50000 )); then PKTS_PER_CHUNK=50000; fi

  echo "  pkts_total=$PKTS (raw=$PKTS_RAW) | avg_pkt=${AVG_PKT}B | pkts_per_chunk=$PKTS_PER_CHUNK (~${TARGET_KB}KB)"

  # Prefijo de salida
  PREFIX="$OUTDIR/${BASE%.pcap}_chunkpkts"
  # Limpieza opcional: si ya existían chunks previos con el mismo prefijo, bórralos
  # (descomenta si quieres regenerar siempre)
  # rm -f "$OUTDIR/${BASE%.pcap}_chunkpkts_"*.pcap 2>/dev/null || true

  # Ejecuta split por nº de paquetes
  if ! editcap -c "$PKTS_PER_CHUNK" "$PCAP_USE" "${PREFIX}.pcap" >/dev/null 2>&1; then
    echo "  [WARN] editcap falló en $PCAP_USE"
    continue
  fi

  # Conteo correcto de chunks (editcap crea *_chunkpkts_00000_YYYY...pcap)
  COUNT="$(find "$OUTDIR" -maxdepth 1 -type f -name "${BASE%.pcap}_chunkpkts_*.pcap" | wc -l | awk '{print $1}')"
  echo "  chunks_generados=$COUNT"
done

echo
echo "DONE."
