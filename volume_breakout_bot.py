
import argparse
import numpy as np
import pandas as pd
import yaml
import pytz
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json

# ================= Utilities =================

def read_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_prices(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"CSV missing columns: {missing}")
    return df

def to_session_tz(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    tz = pytz.timezone(tz_name)
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.tz_convert(pytz.UTC)
    df["ts_local"] = df["timestamp"].dt.tz_convert(tz)
    df["session_day"] = df["ts_local"].dt.floor("D")  # NY midnight boundary
    df["hour_local"] = df["ts_local"].dt.hour
    df["minute_local"] = df["ts_local"].dt.minute
    return df

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

# ================= Fees/Slippage =================

def fee_for_side(size: float, price: float, side: str, cfg: dict, venue: str) -> float:
    model = cfg.get("fee_model", "flat")
    # per-venue overrides
    v = cfg.get("venues", {}).get(venue, {})
    if model == "maker_taker":
        entry_is_taker = v.get("entry_is_taker", cfg.get("entry_is_taker", True))
        exit_is_taker = v.get("exit_is_taker", cfg.get("exit_is_taker", True))
        taker = v.get("taker_fee_pct", cfg.get("taker_fee_pct", 0.0006))
        maker_rebate = v.get("maker_rebate_pct", cfg.get("maker_rebate_pct", 0.0002))
        if side == "entry":
            rate = taker if entry_is_taker else -maker_rebate
        else:
            rate = taker if exit_is_taker else -maker_rebate
        return rate * abs(size) * price
    else:
        rate = v.get("commission_pct", cfg.get("commission_pct", 0.0005))
        return rate * abs(size) * price

def slippage_for(size: float, price: float, cfg: dict, venue: str) -> float:
    v = cfg.get("venues", {}).get(venue, {})
    slip = v.get("slippage_pct", cfg.get("slippage_pct", 0.0008))
    return slip * abs(size) * price

# ================= Strategy =================

from dataclasses import dataclass
@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop: float
    take: float
    trailing_mult: float = None
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = None
    reason: str = ""
    session_day: pd.Timestamp = None

def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["atr"] = atr(df, cfg["atr_lookback"])
    df["atr_pct"] = df["atr"] / df["close"]
    df["atr_ok"] = df["atr_pct"] >= cfg.get("atr_pct_min", 0.0)

    df["avg_vol"] = df["volume"].rolling(cfg["rvol_lookback"]).mean()
    df["rvol"] = df["volume"] / (df["avg_vol"] + 1e-12)

    df["prior_high"] = df["high"].rolling(cfg["breakout_lookback"]).max().shift(1)
    df["breakout"] = df["close"] > df["prior_high"]

    # Multi-timeframe trend
    ema_len = cfg.get("trend_ema_len", 0)
    if ema_len and ema_len > 0:
        df["ema_trend"] = ema(df["close"], ema_len)
        df["trend_ok_base"] = df["close"] > df["ema_trend"]
    else:
        df["trend_ok_base"] = True

    # Higher timeframe confirmation via resample (e.g., 60m EMA over 15m bars)
    ht_minutes = cfg.get("htf_minutes", 0)
    htf_ema_len = cfg.get("htf_ema_len", 0)
    if ht_minutes and htf_ema_len:
        # resample by local time index (use timestamp as index)
        tmp = df.set_index("timestamp")[["close"]].copy()
        htf = tmp.resample(f"{ht_minutes}min").last().dropna()
        htf["ema_htf"] = ema(htf["close"], htf_ema_len)
        # forward fill to base timeframe
        htf["trend_ok_htf"] = htf["close"] > htf["ema_htf"]
        joined = df.join(htf["trend_ok_htf"], on="timestamp", how="left")
        s = joined["trend_ok_htf"].astype("boolean").ffill().fillna(False)
        df["trend_ok_htf"] = s.astype(bool)
    else:
        df["trend_ok_htf"] = True

    # Intraday window (NY time)
    start_h, start_m = cfg.get("intraday_start_h", 0), cfg.get("intraday_start_m", 0)
    end_h, end_m     = cfg.get("intraday_end_h", 23), cfg.get("intraday_end_m", 59)
    def in_window(hr, mn):
        after = (hr > start_h) or (hr == start_h and mn >= start_m)
        before = (hr < end_h) or (hr == end_h and mn <= end_m)
        return after and before
    df["in_window"] = [in_window(h, m) for h, m in zip(df["hour_local"], df["minute_local"])]

    # Master signal
    df["long_signal"] = df["breakout"] & (df["rvol"] >= cfg["rvol_threshold"]) & df["atr_ok"] & df["trend_ok_base"] & df["trend_ok_htf"] & df["in_window"]

    # Debug columns
    df["dbg_breakout_ok"] = df["breakout"].fillna(False)
    df["dbg_rvol_ok"] = (df["rvol"] >= cfg["rvol_threshold"]).fillna(False)
    df["dbg_atr_ok"] = df["atr_ok"].fillna(False)
    df["dbg_trend_ok_base"] = df["trend_ok_base"].fillna(False)
    df["dbg_trend_ok_htf"] = df["trend_ok_htf"].fillna(False)
    df["dbg_window_ok"] = df["in_window"].fillna(False)

    # Optional ML gate (probability governor)
    if "proba_tp_first" in df.columns:
        df["ml_ok"] = True  # default pass-through
    else:
        df["ml_ok"] = True

    return df

def backtest(df: pd.DataFrame, cfg: dict, venue: str = "default"):
    df = df.copy()
    df = compute_indicators(df, cfg)

    equity = cfg["initial_equity"]
    pos = None
    trades = []
    equity_curve = []
    cooldown_bars = 0
    cooldown_after_exit = cfg.get("cooldown_bars", 0)

    # daily risk cap tracking
    max_daily_loss = cfg.get("max_daily_loss_pct", None)
    day_start_equity = equity
    paused_until_next_day = False
    current_day = None

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        price = float(row["close"])

        # day boundary check
        if (current_day is None) or (row["session_day"] != current_day):
            current_day = row["session_day"]
            day_start_equity = equity
            paused_until_next_day = False

        # mark-to-market
        if pos is None:
            equity_curve.append({"timestamp": ts, "equity": equity})
        else:
            mtm = pos.size * (price - pos.entry_price)
            equity_curve.append({"timestamp": ts, "equity": equity + mtm})

        # apply daily risk cap
        if (max_daily_loss is not None) and (equity < day_start_equity * (1 - max_daily_loss)):
            paused_until_next_day = True

        # manage exits
        if pos is not None:
            # trailing stop updates
            if cfg.get("use_trailing", False) and pos.trailing_mult is not None:
                atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else None
                if atr_val and atr_val > 0:
                    new_stop = price - pos.trailing_mult * atr_val
                    pos.stop = max(pos.stop, new_stop)

            hit_sl = row["low"] <= pos.stop
            hit_tp = row["high"] >= pos.take
            exit_price = None
            reason = None
            if hit_sl and hit_tp:
                exit_price = pos.stop
                reason = "SL&TP"
            elif hit_sl:
                exit_price = pos.stop
                reason = "SL"
            elif hit_tp:
                exit_price = pos.take
                reason = "TP"
            if exit_price is not None:
                fee = fee_for_side(pos.size, exit_price, "exit", cfg, venue)
                slip = slippage_for(pos.size, exit_price, cfg, venue)
                pnl = pos.size * (exit_price - pos.entry_price) - fee - slip
                equity += pnl
                pos.exit_time = ts
                pos.exit_price = exit_price
                pos.pnl = pnl
                pos.reason = reason
                trades.append(pos)
                pos = None
                cooldown_bars = cooldown_after_exit
                continue

        # update cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1

            # consider entries
        if (not paused_until_next_day) and pos is None and cooldown_bars == 0 and bool(df["long_signal"].iat[i]):

            # ML probability governor (if present + threshold configured)
            proba = float(row["proba_tp_first"]) if "proba_tp_first" in df.columns and not pd.isna(row["proba_tp_first"]) else None
            th = cfg.get("ml_threshold", None)  # allow config-based default too
            # prefer CLI over config if provided
            if th is None:
                th = globals().get("_cli_ml_threshold", None)
            ml_gate_ok = True
            if (proba is not None) and (th is not None):
                ml_gate_ok = proba >= float(th)
            if not ml_gate_ok:
                continue

            atr_val = float(row["atr"]) if not np.isnan(row["atr"]) else None
            if atr_val is None or atr_val <= 0:
                continue
            sl = price - cfg["sl_atr_mult"] * atr_val
            tp = price + cfg["tp_atr_mult"] * atr_val
            if sl <= 0 or (price - sl) <= 0:
                continue

            # Base risk
            base_risk = equity * cfg["risk_per_trade"]

            # Probability → size multiplier
            size_mode = cfg.get("ml_size_mode", "fixed")
            cli_mode = globals().get("_cli_ml_size_mode", None)
            if cli_mode is not None:
                size_mode = cli_mode
            size_mult = 1.0
            if (proba is not None):
                if size_mode == "linear":
                    # map [th,1] → [1, cap]
                    cap = float(cfg.get("ml_size_cap", 2.0))
                    p0 = float(th) if th is not None else 0.5
                    if proba <= p0:
                        size_mult = 1.0
                    else:
                        size_mult = 1.0 + (cap - 1.0) * (proba - p0) / max(1e-9, 1.0 - p0)
                elif size_mode == "kelly_cap":
                    # approximate edge via calibrated proba against 1:1 R (your TP:SL is ~2R, even better)
                    # raw kelly = 2p-1 (bounded), then clip → [0.25, cap]
                    cap = float(cfg.get("ml_size_cap", 2.0))
                    k = max(0.0, min(2*proba - 1.0, 1.0))  # [0,1]
                    size_mult = min(cap, max(0.25, 0.25 + 0.75*k))
                # fixed → 1.0

            risk_amount = base_risk * size_mult
            stop_distance = price - sl
            size = risk_amount / stop_distance

            # fees/slip on entry
            fee = fee_for_side(size, price, "entry", cfg, venue)
            slip = slippage_for(size, price, cfg, venue)
            equity -= (fee + slip)

            trailing_mult = cfg.get("trail_atr_mult", None) if cfg.get("use_trailing", False) else None
            pos = Trade(entry_time=ts, entry_price=price, size=size, stop=sl, take=tp, trailing_mult=trailing_mult, session_day=row["session_day"])


    # close at last
    if pos is not None:
        last = df.iloc[-1]
        exit_price = float(last["close"])
        fee = fee_for_side(pos.size, exit_price, "exit", cfg, venue)
        slip = slippage_for(pos.size, exit_price, cfg, venue)
        pnl = pos.size * (exit_price - pos.entry_price) - fee - slip
        equity += pnl
        pos.exit_time = last["timestamp"]
        pos.exit_price = exit_price
        pos.pnl = pnl
        pos.reason = "CloseEnd"
        trades.append(pos)

    eq = pd.DataFrame(equity_curve)
    tr = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(columns=["entry_time","entry_price","size","stop","take","trailing_mult","exit_time","exit_price","pnl","reason","session_day"])

    debug_df = df[[
        "timestamp","ts_local","hour_local","minute_local","close","volume","rvol","prior_high",
        "breakout","atr","atr_pct","atr_ok","trend_ok_base","trend_ok_htf","in_window","long_signal",
        "dbg_breakout_ok","dbg_rvol_ok","dbg_atr_ok","dbg_trend_ok_base","dbg_trend_ok_htf","dbg_window_ok","session_day"
    ]].copy()
    return tr, eq, debug_df

def metrics(equity: pd.DataFrame) -> dict:
    equity = equity.dropna().copy()
    equity["ret"] = equity["equity"].pct_change().fillna(0)
    sharpe = float(equity["ret"].mean()) / (float(equity["ret"].std()) + 1e-9)
    roll_max = equity["equity"].cummax()
    dd = equity["equity"]/roll_max - 1.0
    max_dd = float(dd.min())
    total_ret = float(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1.0)
    return {"total_return": total_ret, "sharpe_naive": sharpe, "max_drawdown": max_dd}

def plot_equity(equity: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10,6))
    plt.plot(equity["timestamp"], equity["equity"])
    plt.title("Equity Curve (Volume-Activated Breakout)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)

def _merge_ml_preds(df, path):
    preds = pd.read_csv(path)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True, errors="coerce")
    preds = preds.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.merge(preds[["timestamp","proba_tp_first"]], on="timestamp", how="left")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to OHLCV CSV")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Write debug_reasons.csv")
    parser.add_argument("--venue", type=str, default="default", help="Venue key in config.venues")
    parser.add_argument("--ml_preds", type=str, default=None,
                        help="Path to oos_predictions.csv (timestamp,proba_tp_first[,y_true])")
    parser.add_argument("--ml_threshold", type=float, default=None,
                        help="If set, require proba>=threshold to allow entries")
    parser.add_argument("--ml_size_mode", type=str, default="fixed",
                        choices=["fixed","linear","kelly_cap"],
                        help="How to map proba to size multiplier")
    parser.add_argument("--ml_size_cap", type=float, default=2.0,
                        help="Cap on size multiplier when using probability-based sizing")
    args = parser.parse_args()

    globals()["_cli_ml_threshold"] = args.ml_threshold
    globals()["_cli_ml_size_mode"] = args.ml_size_mode

    cfg = read_config(Path(args.config))
    df = load_prices(Path(args.csv))
    df = to_session_tz(df, cfg["session_tz"])

    if args.ml_preds:
        df = _merge_ml_preds(df, args.ml_preds)

    try:
        from features import build_features
        _feat = build_features(df)
        # align on timestamp for a slice of key features; avoid heavy joins in production backtests
        key_cols = ["feat_rvol_l","feat_ema_dist_96","feat_bb_width","feat_atr_pct","feat_trend_bias_96"]
        df = df.merge(_feat[["timestamp"] + key_cols], on="timestamp", how="left")
    except Exception as _e:
        pass

    trades, equity, debug_df = backtest(df, cfg, venue=args.venue)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "signals.csv", index=False)
    equity.to_csv(out_dir / "equity.csv", index=False)
    plot_equity(equity, out_dir / "equity_curve.png")
    if args.debug:
        debug_df.to_csv(out_dir / "debug_reasons.csv", index=False)

    m = metrics(equity)
    print(f"Total Return: {m['total_return']:.2%}")
    print(f"Max Drawdown: {m['max_drawdown']:.2%}")
    print(f"Naive Sharpe: {m['sharpe_naive']:.2f}")
    print(f"Outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
