
import argparse
import pandas as pd
from pathlib import Path

def parse_time(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def load_raw(path: Path, fmt: str, args) -> pd.DataFrame:
    df = pd.read_csv(path)
    fmt = fmt.lower()

    if fmt == "binance":
        # normalize headers: strip BOM/whitespace, lower-case
        clean_cols = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        df.columns = clean_cols
        cols_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}

        # possible keys for "open time"
        time_key = None
        for cand in ["opentime", "closetime", "time", "date"]:
            if cand in cols_map:
                time_key = cols_map[cand]
                break

        # price/volume column resolution (handle variations)
        def pick(name_opts):
            for opt in name_opts:
                k = opt.lower().replace(" ", "").replace("_", "")
                if k in cols_map:
                    return cols_map[k]
            return None

        o = pick(["open"])
        h = pick(["high"])
        l = pick(["low"])
        c = pick(["close"])
        v = pick(["volume"])

        # Fallback to positional columns if headers are odd but order is standard
        # Binance kline CSV order: open_time, open, high, low, close, volume, close_time, ...
        if time_key is None or not all([o, h, l, c, v]):
            if df.shape[1] >= 6:
                df = df.iloc[:, :6].copy()
                df.columns = ["timestamp","open","high","low","close","volume"]
            else:
                raise ValueError("Binance: could not resolve columns and no positional fallback available.")
        else:
            df = df[[time_key, o, h, l, c, v]].copy()
            df.columns = ["timestamp","open","high","low","close","volume"]

        # convert timestamp: epoch ms or s, else ISO
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            ts = pd.to_numeric(df["timestamp"], errors="coerce")
            # detect ms vs s
            if ts.dropna().median() > 1e12:
                df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
            else:
                df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    elif fmt == "bybit":
        col = None
        for cand in ["startTime","timestamp","time","start_time"]:
            if cand in df.columns:
                col = cand; break
        if col is None:
            raise ValueError("Bybit: couldn't find time column")
        df = df[[col,"open","high","low","close","volume"]].copy()
        df.columns = ["timestamp","open","high","low","close","volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    elif fmt == "coinbase":
        req = ["time","open","high","low","close","volume"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"Coinbase: missing {miss}")
        df = df[req].copy()
        df.columns = ["timestamp","open","high","low","close","volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    elif fmt == "kraken":
        col = "time" if "time" in df.columns else None
        if col is None:
            raise ValueError("Kraken: missing 'time' column")
        df = df[[col,"open","high","low","close","volume"]].copy()
        df.columns = ["timestamp","open","high","low","close","volume"]
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    elif fmt == "generic":
        cols = {
            "timestamp": args.ts or "timestamp",
            "open": args.open or "open",
            "high": args.high or "high",
            "low": args.low or "low",
            "close": args.close or "close",
            "volume": args.volume or "volume",
        }
        miss = [v for v in cols.values() if v not in df.columns]
        if miss:
            raise ValueError(f"Generic: missing columns {miss}")
        df = df[[cols["timestamp"], cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()
        df.columns = ["timestamp","open","high","low","close","volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    df = df.dropna(subset=["timestamp"])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df

def resample_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    # Map Binance-style minutes to pandas minutes
    import re
    tf = timeframe
    m = re.fullmatch(r"(\d+)[mM]", str(tf).strip())
    if m:
        tf = f"{m.group(1)}min"   # e.g., 15m -> 15min

    o = df.set_index("timestamp")[["open","high","low","close","volume"]]
    r = o.resample(tf).agg({
        "open":"first",
        "high":"max",
        "low":"min",
        "close":"last",
        "volume":"sum"
    }).dropna().reset_index()
    return r
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="Path to raw CSV export")
    ap.add_argument("--out", dest="outfile", required=True, help="Output normalized CSV")
    ap.add_argument("--format", dest="fmt", required=True, choices=["binance","bybit","coinbase","kraken","generic"], help="Source CSV format")
    ap.add_argument("--timeframe", default="15min", help="Resample timeframe (e.g., 1min,5min,15min,1H)")
    ap.add_argument("--ts"); ap.add_argument("--open"); ap.add_argument("--high"); ap.add_argument("--low"); ap.add_argument("--close"); ap.add_argument("--volume")
    args = ap.parse_args()

    raw = load_raw(Path(args.infile), args.fmt, args)
    out = resample_df(raw, args.timeframe)
    out.to_csv(args.outfile, index=False)
    print(f"Saved normalized OHLCV to: {args.outfile}  (rows={len(out)})")

if __name__ == "__main__":
    main()
