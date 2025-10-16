# make_labels.py
"""
Create modeling labels from the feature set using a triple-barrier scheme.

- Entry price: close[t]
- Barriers at t created with ATR[t]:
    TP = close[t] + tp_atr_mult * ATR[t]
    SL = close[t] - sl_atr_mult * ATR[t]
- Horizon: H bars into the future (excludes current bar)
- Label:
    +1 if TP is hit before SL within H bars
    -1 if SL is hit before TP within H bars
     0 if neither is hit within H, sign of horizon return (>=0 -> +0, <0 -> -0) can be optional
       (Here we set 0 if neither hit; we also provide ret_h for regression/meta-labels.)
- Also outputs:
    ret_h        : (close[t+H] / close[t] - 1), NaN if not enough bars
    hit_type     : "TP","SL","NONE"
    time_to_hit  : bars to first hit (if any)
    tp_price/sl_price for reference

Leakage notes:
- Only ATR[t], close[t] are used to set barriers.
- Horizon scans use future bars strictly after t (shifted by -1 window).
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

REQUIRED_COLS = {
    "timestamp","open","high","low","close","volume",
    "feat_atr_14"  # produced by features.py
}

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def write_any(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            print(f"Saved → {path}  rows={len(df)}  cols={len(df.columns)}")
            return
        except Exception as e:
            csv_fallback = path.with_suffix(".csv")
            df.to_csv(csv_fallback, index=False)
            print(f"Parquet engine missing; wrote CSV fallback → {csv_fallback}  "
                  f"(rows={len(df)} cols={len(df.columns)} error={e})")
            return
    else:
        df.to_csv(path, index=False)
        print(f"Saved → {path}  rows={len(df)}  cols={len(df.columns)}")

def build_labels(df: pd.DataFrame, horizon:int, tp_mult:float, sl_mult:float) -> pd.DataFrame:
    # basic checks
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = df.reset_index(drop=True).copy()

    # Barriers at t
    atr = df["feat_atr_14"].astype(float)
    close = df["close"].astype(float)

    tp_price = close + tp_mult * atr
    sl_price = close - sl_mult * atr

    n = len(df)
    H = int(horizon)
    # Pre-allocate outputs
    hit_type = np.full(n, "NONE", dtype=object)
    time_to_hit = np.full(n, np.nan, dtype=float)

    # We need to know which barrier is touched first in (t+1 ... t+H)
    # Vector approach: for each offset k=1..H, check TP/SL hit arrays; then pick first True.
    # This is O(n*H) but H is typically small (e.g., 96) and fine for multi-year 15m data.

    # Future highs/lows per offset
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    tp    = tp_price.to_numpy()
    sl    = sl_price.to_numpy()

    # Store the first-hit offsets; init as +inf
    first_tp = np.full(n, np.inf)
    first_sl = np.full(n, np.inf)

    for k in range(1, H+1):
        # shift highs/lows by -k to align future bar t+k with index t
        hi_k = np.r_[highs[k:], np.full(k, np.nan)]
        lo_k = np.r_[lows[k:],  np.full(k, np.nan)]

        tp_hit_k = hi_k >= tp  # if high >= tp at t+k
        sl_hit_k = lo_k <= sl  # if low  <= sl at t+k

        # set first occurrence offset if not set yet
        mask_tp_set = (first_tp == np.inf) & tp_hit_k
        mask_sl_set = (first_sl == np.inf) & sl_hit_k
        first_tp[mask_tp_set] = k
        first_sl[mask_sl_set] = k

    # Decide label by which occurs first
    # If both inf → NONE
    # If both finite → compare offsets
    both_inf = (np.isinf(first_tp) & np.isinf(first_sl))
    only_tp  = (np.isfinite(first_tp) & np.isinf(first_sl))
    only_sl  = (np.isfinite(first_sl) & np.isinf(first_tp))
    both_fin = (np.isfinite(first_tp) & np.isfinite(first_sl))

    hit_type[only_tp] = "TP"
    time_to_hit[only_tp] = first_tp[only_tp]

    hit_type[only_sl] = "SL"
    time_to_hit[only_sl] = first_sl[only_sl]

    # both hit: pick the earlier
    bf_idx = np.where(both_fin)[0]
    earlier_tp = first_tp[bf_idx] < first_sl[bf_idx]
    earlier_sl = ~earlier_tp
    hit_type[bf_idx[earlier_tp]] = "TP"
    time_to_hit[bf_idx[earlier_tp]] = first_tp[bf_idx[earlier_tp]]
    hit_type[bf_idx[earlier_sl]] = "SL"
    time_to_hit[bf_idx[earlier_sl]] = first_sl[bf_idx[earlier_sl]]

    # Horizon return for auxiliary targets (classification/regression)
    # ret over exactly H bars ahead (include t+H close, exclude current)
    close_fwd_H = np.r_[close[H:], np.full(H, np.nan)]
    ret_h = (close_fwd_H / close) - 1.0

    # Primary label: +1/ -1 / 0
    label = np.zeros(n, dtype=np.int8)
    label[hit_type == "TP"] = 1
    label[hit_type == "SL"] = -1
    # If NONE: keep 0 (you can choose to sign(ret_h) instead if you want dense labels)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "tp_price": tp_price,
        "sl_price": sl_price,
        "hit_type": hit_type,
        "time_to_hit": time_to_hit,
        "label": label,
        "ret_h": ret_h,
        "horizon_bars": H,
        "tp_atr_mult": float(tp_mult),
        "sl_atr_mult": float(sl_mult),
    })

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to features dataset (.parquet or .csv)")
    ap.add_argument("--out", dest="out", required=True, help="Output labeled dataset (.parquet or .csv)")
    ap.add_argument("--horizon", type=int, default=96, help="Horizon in bars (e.g., 96 ~ 1 day for 15m)")
    ap.add_argument("--tp_atr_mult", type=float, default=4.0, help="TP barrier multiple of ATR[t]")
    ap.add_argument("--sl_atr_mult", type=float, default=2.0, help="SL barrier multiple of ATR[t]")
    ap.add_argument("--meta_from_signal_csv", type=str, default=None,
                    help="Optional: path to debug_reasons.csv (or any file with timestamp,long_signal) to meta-label only when base signal==True.")
    ap.add_argument("--label_mode", type=str, default="triple_barrier",
                    choices=["triple_barrier","fwd_sign","fwd_qcut"],
                    help="Alternative label modes for experimentation.")
    ap.add_argument("--qcut", type=int, default=5,
                    help="Number of quantiles for fwd_qcut mode.")
    args = ap.parse_args()

    inp = Path(args.inp); outp = Path(args.out)

    df = read_any(inp)

    # Sanity: ensure chronological order and tz-preserving timestamp column
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # --- Meta-labeling support (optional) ---
    meta_mask = None
    if args.meta_from_signal_csv:
        sig = pd.read_csv(args.meta_from_signal_csv)
        sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True, errors="coerce")
        # Keep only the fields we need and align on timestamp
        sig = sig[["timestamp", "long_signal"]].dropna()
        merged = df[["timestamp"]].merge(sig, on="timestamp", how="left")
        # Boolean vector aligned to df rows
        meta_mask = merged["long_signal"].fillna(False).astype(bool).values

    labels = build_labels(df, args.horizon, args.tp_atr_mult, args.sl_atr_mult)
    if args.label_mode != "triple_barrier":
        H = int(args.horizon)
        close = df["close"].astype(float).values
        close_fwd_H = np.r_[close[H:], np.full(H, np.nan)]
        ret_h = (close_fwd_H / close) - 1.0
        if args.label_mode == "fwd_sign":
            label_alt = np.zeros(len(ret_h), dtype=np.int8)
            label_alt[ret_h > 0] = 1
            label_alt[ret_h < 0] = -1
        else:  # fwd_qcut
            sr = pd.Series(ret_h)
            label_alt = pd.qcut(sr.rank(method="first"), args.qcut, labels=False)
            # map lowest→-2 ... highest→+2 for 5-quantiles, for example
            mid = (args.qcut - 1) / 2.0
            label_alt = (label_alt - mid).astype("int8")
        labels["label"] = label_alt
        labels["ret_h"] = ret_h
        labels["hit_type"] = np.where(label_alt > 0, "TP", np.where(label_alt < 0, "SL", "NONE"))

    # Merge back onto features for a single ML table
    full = df.merge(labels, on="timestamp", how="left")

    if meta_mask is not None:
        full["meta_base_signal"] = meta_mask.astype(int)

    # Drop rows with NaNs at the tail due to horizon (optional: keep if you want inference-only)
    full_clean = full[full["ret_h"].notna()].reset_index(drop=True)

    write_any(full_clean, outp)

    # Quick class balance / stats to console
    vc = full_clean["label"].value_counts(dropna=False)
    n = len(full_clean)
    pos = int(vc.get(1, 0)); neg = int(vc.get(-1, 0)); neu = int(vc.get(0, 0))
    print(f"\nLabel counts (H={args.horizon}, TP={args.tp_atr_mult}*ATR, SL={args.sl_atr_mult}*ATR):")
    print(f"  +1 (TP first): {pos}  ({pos/n:.2%})")
    print(f"  -1 (SL first): {neg}  ({neg/n:.2%})")
    print(f"   0 (NONE)    : {neu}  ({neu/n:.2%})")
    print(f"Rows kept after horizon drop: {n}")
    print("Done.")

if __name__ == "__main__":
    main()
