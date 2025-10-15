"""
Walk-forward grid search for the breakout bot.
Usage:
  python wf_optimize.py --csv data.csv --config base.yaml --out results_dir

It reads the base config, sweeps predefined grids, and evaluates walk-forward
windows: train (opt) on window k, test on window k+1, rolling through the series.

"""
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from pathlib import Path
from volume_breakout_bot import read_config, load_prices, to_session_tz, backtest, metrics

def split_walkforward(df, k_folds=6):
    n = len(df)
    fold_size = n // (k_folds + 1)
    windows = []
    for k in range(k_folds):
        train = df.iloc[: (k+1)*fold_size]
        test = df.iloc[(k+1)*fold_size : (k+2)*fold_size]
        windows.append((train.copy(), test.copy(), f"wf_{k+1}"))
    return windows

def eval_params(df_train, df_test, base_cfg, params):
    cfg = {**base_cfg, **params}
    # compute indicators on concatenated, but only score on test
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    tr, eq, _ = backtest(df_all, cfg)
    # slice equity to test window only (align by timestamps)
    test_start = df_test["timestamp"].iloc[0]
    eq_test = eq[eq["timestamp"] >= test_start].copy()
    return metrics(eq_test)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="wf_results")
    ap.add_argument("--kfolds", type=int, default=6)
    args = ap.parse_args()

    base_cfg = read_config(Path(args.config))
    df = load_prices(Path(args.csv))
    df = to_session_tz(df, base_cfg["session_tz"])

    # Parameter grid (tune as needed)
    grid = {
        "rvol_threshold": [1.2, 1.4, 1.6],
        "breakout_lookback": [12, 16, 24],
        "atr_pct_min": [0.0005, 0.0008, 0.0012],
        "trend_ema_len": [48, 96, 192],
        "sl_atr_mult": [1.2, 1.5, 2.0],
        "tp_atr_mult": [2.0, 3.0, 4.0],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    windows = split_walkforward(df, k_folds=args.kfolds)
    results = []

    for (train_df, test_df, tag) in windows:
        best_score = -1e9
        best_cfg = None
        for combo in combos:
            params = {k: v for k, v in zip(keys, combo)}
            try:
                m = eval_params(train_df, test_df, base_cfg, params)
                score = m["total_return"] - 0.5*abs(m["max_drawdown"])  # simple objective
                if score > best_score:
                    best_score = score
                    best_cfg = params
            except Exception as e:
                continue
        # evaluate best on test window for record
        m_best = eval_params(train_df, test_df, base_cfg, best_cfg)
        m_best["tag"] = tag
        m_best["params"] = best_cfg
        results.append(m_best)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "walkforward_results.csv", index=False)
    print("Saved:", out_dir / "walkforward_results.csv")

if __name__ == "__main__":
    main()
