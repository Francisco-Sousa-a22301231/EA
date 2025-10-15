# Volume-Activated Breakout Bot (Backtester)

An **enhanced volume-activated breakout** backtester for OHLCV data.  
Sessions align to **New York (America/New_York)**, **00:00 → 23:59**.

> Research tool only — not live trading code.

---

## What it does

- Converts timestamps to **America/New_York** and anchors sessions at **NY midnight**.
- Signals require **breakout + RVOL confirmation + ATR regime filter**.
- **Multi-timeframe trend** (e.g., 15m execution with 1h EMA confirmation).
- **Risk-based position sizing**, **ATR stops/targets**, optional **trailing stop**.
- **Venue-aware fees/slippage** (maker/taker), **cooldown**, **daily circuit breaker**.
- Rich **debug trace**: why a bar did/didn’t qualify.

---

## Data: fastest way (Binance Vision, automatic)

Download BTCUSDT **15m** klines from first availability (2017-08) through **2023** (keep 2024–2025 as OOS):

```bash
chmod +x download_binance_klines.sh
./download_binance_klines.sh BTCUSDT 15m 2017 2023
```

This produces:

```
./binance_BTCUSDT_15m_to_2023/
  raw_zips/                # monthly .zip files
  csvs/                    # unzipped monthly CSVs
  BTCUSDT_15m_raw.csv      # merged raw (Binance columns)
  BTCUSDT_15m_normalized.csv  # normalized: timestamp,open,high,low,close,volume (UTC)
```

### Sanity-check the dataset

```bash
python3 inspect_csv.py --csv ./binance_BTCUSDT_15m_to_2023/BTCUSDT_15m_normalized.csv
# Expect:
# Start: ~2017-08-17
# End  : 2023-12-31 23:45:00
# Inferred timeframe: 15min
```

> Tip: Binance filenames use **`15m`**; pandas resampling uses **`15min`**. The normalizer maps correctly.

---

## Run a backtest

Use the **pro** config and write outputs to **./output**:

```bash
python3 volume_breakout_bot.py   --csv ./binance_BTCUSDT_15m_to_2023/BTCUSDT_15m_normalized.csv   --config config.yaml   --out ./output   --debug   --venue binance
```

### Outputs

- `./output/signals.csv` — entries/exits with PnL
- `./output/equity.csv` — equity curve
- `./output/equity_curve.png` — chart
- `./output/debug_reasons.csv` — per-bar diagnostics
- Console: Total Return, Max Drawdown, Naive Sharpe

---

## Config (key knobs)

```yaml
# config.pro.yaml (excerpt of important fields)
session_tz: America/New_York

# Signal gates
rvol_lookback: 96          # baseline volume window
rvol_threshold: 2.6        # require strong volume
breakout_lookback: 128     # meaningful prior-high breakouts
atr_lookback: 14
atr_pct_min: 0.006         # block quiet regimes (tiny stops)

# Trend filters
trend_ema_len: 144
htf_minutes: 60
htf_ema_len: 96

# Risk/exits
risk_per_trade: 0.0015
sl_atr_mult: 2.2
tp_atr_mult: 4.4
use_trailing: false
cooldown_bars: 24
max_daily_loss_pct: 0.01

# Fees/slippage + venue overrides
fee_model: maker_taker
entry_is_taker: true
exit_is_taker: true
taker_fee_pct: 0.0006
slippage_pct: 0.0008
venues:
  binance:
    taker_fee_pct: 0.0004
    maker_rebate_pct: 0.0001
    slippage_pct: 0.0007

initial_equity: 10000
seed: 42
```

---

## Common gotchas

- **Too many trades / fee drag** → raise `rvol_threshold`/`breakout_lookback`, increase `atr_pct_min`, reduce `risk_per_trade`, increase `cooldown_bars`.

---

## Optional: different timeframe

Generate 1h bars from the same raw:

```bash
python3 normalize_ohlcv.py   --in ./binance_BTCUSDT_15m_to_2023/BTCUSDT_15m_raw.csv   --out ./bt_BTCUSDT_1h.csv   --format binance   --timeframe 1H
```

Run the same backtest with `--csv ./bt_BTCUSDT_1h.csv`.  
Higher TF often reduces noise and fees.

---

## Roadmap

- Walk-forward splits & grid search
- Multi-symbol portfolio runs
- Partial/intrabar fill modeling
- CCXT Pro live/paper skeleton
