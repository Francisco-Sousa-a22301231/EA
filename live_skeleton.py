
# Live/Paper trading skeleton (outline). Not executing trades here—structure only.
# You can fill in CCXT Pro streaming and exchange credentials.
#
# Usage idea:
#   python live_skeleton.py --config live.yaml
#
import argparse
import asyncio
import time
from pathlib import Path
import pandas as pd
from volume_breakout_bot import read_config, compute_indicators, fee_for_side, slippage_for

async def run_live(cfg_path: Path):
    cfg = read_config(cfg_path)
    # TODO: connect to exchange websocket for OHLCV stream (e.g., CCXT Pro)
    # TODO: maintain rolling DataFrame of recent bars
    # TODO: on each new bar, recompute indicators, decide orders, and route via REST
    # TODO: enforce New York session boundaries and daily risk cap
    print("Live skeleton loaded. Fill in exchange connectivity.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    asyncio.run(run_live(Path(args.config)))

if __name__ == "__main__":
    main()
