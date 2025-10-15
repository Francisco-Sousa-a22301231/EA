# features.py
import numpy as np
import pandas as pd

# ---- small utils ----
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    hi_lo = df["high"] - df["low"]
    hi_cl = (df["high"] - df["close"].shift()).abs()
    lo_cl = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hi_lo, hi_cl, lo_cl], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    chg = close.diff()
    up = chg.clip(lower=0.0)
    dn = -chg.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast=12, slow=26, sig=9):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd = ema_f - ema_s
    signal = _ema(macd, sig)
    hist = macd - signal
    return macd, signal, hist

def _bb(close: pd.Series, n=20, k=2.0):
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()
    upper = ma + k*sd
    lower = ma - k*sd
    width = (upper - lower) / (ma + 1e-12)
    pctb = (close - lower) / ((upper - lower) + 1e-12)
    return ma, upper, lower, width, pctb

def _realized_vol(close: pd.Series, n: int = 96):
    ret = np.log(close).diff()
    return (ret.rolling(n, min_periods=n).std() * np.sqrt(96))  # dailyized-ish for 15m

def _percentile_rank(x: pd.Series, n: int):
    # Rolling percentile rank of the last value in the window (in [0,1]).
    # Robust to NaNs; uses raw=True for speed and to avoid label/iloc issues.

    def ranker(a):
        # a is a 1D numpy array when raw=True
        last = a[-1]
        if np.isnan(last):
            return np.nan
        valid = ~np.isnan(a)
        if valid.sum() == 0:
            return np.nan
        return np.mean(a[valid] <= last)

    return x.rolling(n, min_periods=n).apply(ranker, raw=True)

def _safe_div(a, b):
    return a / (b + 1e-12)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # session borders already built in your pipeline; use ts_local
    hr = df["hour_local"]
    mn = df["minute_local"]
    dow = df["ts_local"].dt.weekday  # 0=Mon
    # circular encodings to avoid ordinal traps
    df["feat_hour_sin"] = np.sin(2*np.pi*(hr + mn/60)/24)
    df["feat_hour_cos"] = np.cos(2*np.pi*(hr + mn/60)/24)
    df["feat_dow_sin"]  = np.sin(2*np.pi*dow/7)
    df["feat_dow_cos"]  = np.cos(2*np.pi*dow/7)
    return df

def add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    body = (df["close"] - df["open"]).abs()
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = (df["high"] - df[["close","open"]].max(axis=1))
    lower_wick = (df[["close","open"]].min(axis=1) - df["low"])
    df["feat_wick_upper_ratio"] = _safe_div(upper_wick, range_)
    df["feat_wick_lower_ratio"] = _safe_div(lower_wick, range_)
    df["feat_body_range_ratio"]  = _safe_div(body, range_)
    df["feat_close_loc"] = _safe_div(df["close"] - df["low"], (df["high"] - df["low"]).replace(0,np.nan))
    return df

def add_volume_features(df: pd.DataFrame, rvol_n_short=24, rvol_n_long=96) -> pd.DataFrame:
    df["feat_vol_ma_s"] = df["volume"].rolling(rvol_n_short, min_periods=rvol_n_short).mean()
    df["feat_vol_ma_l"] = df["volume"].rolling(rvol_n_long,  min_periods=rvol_n_long).mean()
    df["feat_rvol_s"]   = _safe_div(df["volume"], df["feat_vol_ma_s"])
    df["feat_rvol_l"]   = _safe_div(df["volume"], df["feat_vol_ma_l"])
    # deciles of RVOL (ranked in rolling window)
    df["feat_rvol_rank_l"] = _percentile_rank(df["feat_rvol_l"], rvol_n_long)
    return df

def add_trend_volatility(df: pd.DataFrame) -> pd.DataFrame:
    # EMAs across multiple scales (execution and HTF-alike)
    for n in (12, 24, 48, 96, 144, 288):
        df[f"feat_ema_{n}"] = _ema(df["close"], n)
        df[f"feat_ema_dist_{n}"] = _safe_div(df["close"] - df[f"feat_ema_{n}"], df["close"])
    # MACD
    macd, sig, hist = _macd(df["close"])
    df["feat_macd"] = macd; df["feat_macd_sig"] = sig; df["feat_macd_hist"] = hist
    # RSI
    df["feat_rsi_14"] = _rsi(df["close"], 14)
    df["feat_rsi_28"] = _rsi(df["close"], 28)
    # Bollinger features
    ma, up, lo, width, pctb = _bb(df["close"], 20, 2.0)
    df["feat_bb_width"] = width
    df["feat_bb_pctb"]  = pctb
    # ATR & regimes
    df["feat_atr_14"] = _atr(df, 14)
    df["feat_atr_pct"] = _safe_div(df["feat_atr_14"], df["close"])
    df["feat_atr_pct_prank_14d"] = _percentile_rank(df["feat_atr_pct"], 96*14)  # ~14 NY days on 15m
    # realized vol
    df["feat_rvol_daily"] = _realized_vol(df["close"], 96)  # ~1 day
    return df

def add_breakout_context(df: pd.DataFrame, lookbacks=(24, 48, 96, 128, 192)):
    for n in lookbacks:
        prior_high = df["high"].rolling(n, min_periods=n).max().shift(1)
        prior_low  = df["low"].rolling(n, min_periods=n).min().shift(1)
        df[f"feat_brk_dist_hi_{n}"] = _safe_div(df["close"] - prior_high, df["close"])
        df[f"feat_brk_dist_lo_{n}"] = _safe_div(prior_low - df["close"], df["close"])
        df[f"feat_above_hi_{n}"] = (df["close"] > prior_high).astype(int)
        df[f"feat_below_lo_{n}"] = (df["close"] < prior_low ).astype(int)
    return df

def add_return_features(df: pd.DataFrame):
    df["feat_ret_1"]  = df["close"].pct_change(1)
    df["feat_ret_4"]  = df["close"].pct_change(4)
    df["feat_ret_12"] = df["close"].pct_change(12)
    df["feat_ret_24"] = df["close"].pct_change(24)
    # forward-looking returns are NOT included here to avoid leakage (they go in labels later)
    return df

def finalize_features(df: pd.DataFrame, min_history_bars=288):
    # Drop rows until all rolling features have valid history
    return df.iloc[min_history_bars:].copy()

def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df_in with:
      timestamp (tz-aware UTC), open, high, low, close, volume,
      ts_local, hour_local, minute_local, session_day
    Returns df with feature columns; no future leakage.
    """
    df = df_in.copy()

    # Core blocks
    df = add_time_features(df)
    df = add_microstructure(df)
    df = add_volume_features(df, rvol_n_short=24, rvol_n_long=96)
    df = add_trend_volatility(df)
    df = add_breakout_context(df, lookbacks=(24, 48, 96, 128, 192))
    df = add_return_features(df)

    # Optional: regime flags (simple)
    df["feat_trend_bias_96"] = (df["close"] > df["feat_ema_96"]).astype(int)
    df["feat_vol_high"] = (df["feat_atr_pct"] > df["feat_atr_pct"].rolling(96*7, min_periods=96*7).median()).astype(int)

    # Clean up starting NaNs from rolling windows
    df = finalize_features(df, min_history_bars=288)
    return df
