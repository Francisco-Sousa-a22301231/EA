# make_dataset.py
import argparse
from pathlib import Path
import pandas as pd
from volume_breakout_bot import read_config, load_prices, to_session_tz
from features import build_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Normalized OHLCV CSV (timestamp,open,high,low,close,volume UTC)")
    ap.add_argument("--config", required=True, help="config.yaml (for session_tz)")
    ap.add_argument("--out", required=True, help="Output dataset path (.csv or .parquet)")
    args = ap.parse_args()

    cfg = read_config(Path(args.config))
    df = load_prices(Path(args.csv))
    df = to_session_tz(df, cfg["session_tz"])  # adds ts_local, hour_local, etc.

    feat = build_features(df)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        feat.to_parquet(out_path, index=False)
    else:
        feat.to_csv(out_path, index=False)
    print(f"Saved features → {out_path}  rows={len(feat)}  cols={len(feat.columns)}")

if __name__ == "__main__":
    main()
