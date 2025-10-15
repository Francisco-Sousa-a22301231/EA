#!/usr/bin/env bash
set -euo pipefail

# Config
SYMBOL="${1:-BTCUSDT}"
INTERVAL="${2:-15m}"
START_YEAR="${3:-2017}"
END_YEAR="${4:-2023}"   # inclusive; we stop at 2023 to keep 2024–2025 as OOS
BASE_URL="https://data.binance.vision/data/spot/monthly/klines"
WORKDIR="${PWD}/binance_${SYMBOL}_${INTERVAL}_to_${END_YEAR}"
RAW_DIR="${WORKDIR}/raw_zips"
CSV_DIR="${WORKDIR}/csvs"

mkdir -p "$RAW_DIR" "$CSV_DIR"

echo "Downloading ${SYMBOL} ${INTERVAL} klines ${START_YEAR}..${END_YEAR} into $WORKDIR"

exists() {
  curl -sfI "$1" >/dev/null
}

for Y in $(seq "$START_YEAR" "$END_YEAR"); do
  for M in $(seq -w 01 12); do
    ZIP="${SYMBOL}-${INTERVAL}-${Y}-${M}.zip"
    URL="${BASE_URL}/${SYMBOL}/${INTERVAL}/${ZIP}"
    if exists "$URL"; then
      echo "Fetching $ZIP"
      curl -fSL "$URL" -o "${RAW_DIR}/${ZIP}"
      unzip -o "${RAW_DIR}/${ZIP}" -d "${CSV_DIR}" >/dev/null
    else
      echo "Missing on server (ok): ${ZIP}"
    fi
  done
done

# Merge monthly CSVs into one raw file (Binance format columns)
MERGED_RAW="${WORKDIR}/${SYMBOL}_${INTERVAL}_raw.csv"
echo "Merging monthly CSVs → ${MERGED_RAW}"
# All monthly CSVs share the same header; use the first header then append tail of others
first=1
: > "$MERGED_RAW"
for f in $(ls -1 "${CSV_DIR}/${SYMBOL}-${INTERVAL}-"*.csv | sort); do
  if [[ $first -eq 1 ]]; then
    cat "$f" >> "$MERGED_RAW"
    first=0
  else
    tail -n +2 "$f" >> "$MERGED_RAW"
  fi
done

# Normalize to required columns for the backtester
NORMALIZED="${WORKDIR}/${SYMBOL}_${INTERVAL}_normalized.csv"
echo "Normalizing → ${NORMALIZED}"
python3 normalize_ohlcv.py \
  --in "$MERGED_RAW" \
  --out "$NORMALIZED" \
  --format binance \
  --timeframe "${INTERVAL}"

# Quick sanity report
echo "Inspecting normalized file:"
python3 inspect_csv.py --csv "$NORMALIZED"

echo "Done."
echo "Normalized CSV: $NORMALIZED"
