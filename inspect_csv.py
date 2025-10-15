
import argparse
import pandas as pd

def infer_timeframe(df: pd.DataFrame) -> str:
    s = df["timestamp"].sort_values().diff().dropna()
    if s.empty:
        return "unknown"
    dt = s.mode().iloc[0]
    minutes = int(dt.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}min"
    else:
        hours = round(dt.total_seconds() / 3600, 2)
        return f"{hours}H"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    print("Rows:", len(df))
    if not df.empty:
        print("Start:", df["timestamp"].iloc[0])
        print("End  :", df["timestamp"].iloc[-1])
        print("Inferred timeframe:", infer_timeframe(df))
        gaps = df["timestamp"].diff().dt.total_seconds().fillna(0)
        big_gaps = (gaps > gaps.median() * 3).sum()
        print("Large gaps (approx):", int(big_gaps))

if __name__ == "__main__":
    main()
