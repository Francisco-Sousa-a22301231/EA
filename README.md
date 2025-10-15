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
````

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

* `./output/signals.csv` — entries/exits with PnL
* `./output/equity.csv` — equity curve
* `./output/equity_curve.png` — chart
* `./output/debug_reasons.csv` — per-bar diagnostics
* Console: Total Return, Max Drawdown, Naive Sharpe

---

## Config (key knobs)

```yaml
# config.yaml — parameter reference
# --- Session & timezone ---
session_tz: Defines the timezone used for session segmentation (default: America/New_York).

# --- Signal gates ---
rvol_lookback: Number of bars used to calculate the average volume baseline for relative volume (RVOL).
rvol_threshold: Minimum RVOL value required for trade activation — filters out weak volume breakouts.
breakout_lookback: Number of past bars to determine the “previous high” or “breakout level”.
atr_lookback: Number of bars used for ATR (Average True Range) smoothing.
atr_pct_min: Minimum acceptable ATR as a percentage of price to avoid trading during quiet or low-volatility periods.

# --- Trend filters ---
trend_ema_len: Length of EMA used on the execution timeframe to determine base trend direction.
htf_minutes: Length of the higher timeframe in minutes (e.g., 60 = 1h) used for additional trend confirmation.
htf_ema_len: EMA length for higher timeframe trend confirmation.

# --- Risk management & exits ---
risk_per_trade: Fraction of total equity risked per trade (e.g., 0.01 = 1% of account equity).
sl_atr_mult: Stop-loss distance expressed in multiples of ATR.
tp_atr_mult: Take-profit distance expressed in multiples of ATR.
use_trailing: Enables or disables trailing stop mode.
trail_atr_mult: If trailing is active, defines the trailing distance in ATR multiples.
cooldown_bars: Minimum number of bars to wait after a trade before allowing another entry.
max_daily_loss_pct: Daily drawdown limit — halts trading if this percentage of equity is lost in a day.

# --- Fees & slippage ---
fee_model: Fee model type, e.g., "maker_taker" or "flat".
entry_is_taker: Whether entry orders execute as takers (incurring taker fees).
exit_is_taker: Whether exit orders execute as takers.
taker_fee_pct: Percentage fee applied to taker orders.
maker_rebate_pct: Rebate applied when acting as a maker (if supported by the venue).
commission_pct: Generic fallback commission rate (if no maker/taker split is defined).
slippage_pct: Expected price deviation due to slippage per trade.

# --- Venue-specific overrides ---
venues:
  binance:
    taker_fee_pct: Taker fee specific to Binance.
    maker_rebate_pct: Maker rebate specific to Binance.
    slippage_pct: Expected Binance slippage model.
  bybit:
    taker_fee_pct: Taker fee specific to Bybit.
    maker_rebate_pct: Maker rebate for Bybit.
    slippage_pct: Expected Bybit slippage model.

# --- Backtest settings ---
initial_equity: Starting capital for the backtest (in USD or quote currency).
seed: Random seed used to ensure deterministic results.
```

---

## Common gotchas

* **Too many trades / fee drag** → raise `rvol_threshold`/`breakout_lookback`, increase `atr_pct_min`, reduce `risk_per_trade`, increase `cooldown_bars`.

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

### 🧭 Phase 1 — Treat It Like a Machine Learning Problem
*(unchanged)*

### 🔬 Phase 2 — Expand the Dataset & Feature Space
*(unchanged)*

### 🧠 Phase 3 — Predictive Modeling
*(unchanged)*

### ⚙️ Phase 4 — Walk-Forward + Out-of-Sample Validation
*(unchanged)*

### ⚡ Phase 5 — Portfolio Construction & Leverage
*(unchanged)*

### 💸 Phase 6 — Deployment for Passive Income
*(unchanged)*

### 🔑 Tooling Stack for Research
*(unchanged)*

---

# 🧩 NEW — Machine Learning Dataset Builder + Walk-Forward Training

Recent updates turn the backtester into a full **ML pipeline** for predictive modeling.

---

## 📦 1. Build dataset features

```bash
python3 make_dataset.py   --csv ./binance_BTCUSDT_15m_to_2025/BTCUSDT_15m_normalized.csv   --config config.yaml   --out ./output/dataset_features_2017_2024.parquet
```

Creates engineered features such as:
- EMAs, RSI, MACD, Bollinger Bands, ATR ratios  
- Relative volume metrics (`feat_rvol_s`, `feat_rvol_l`)  
- Wick/body/volatility structure ratios  
- Multi-timeframe breakout distances  

✅ Output: `dataset_features_2017_2024.parquet` (`~72` columns)

---

## 🏷️ 2. Label outcomes (TP-first / SL-first)

```bash
python3 make_labels.py   --in ./output/dataset_features_2017_2024.parquet   --out ./output/dataset_ml_2017_2024.parquet   --horizon 96   --tp_atr_mult 4.4   --sl_atr_mult 2.2
```

This labels each bar:
- **+1** → Take Profit hit first  
- **−1** → Stop Loss hit first  
- **0** → Neither within horizon  

✅ Output: `dataset_ml_2017_2024.parquet` (`~81` columns)

---

## 🤖 3. Train and evaluate models

### Option A — Single-split (default)
```bash
python3 train_xgb.py   --in ./output/dataset_ml_2017_2024.parquet   --outdir ./output/model_xgb   --drop_none   --calibrate
```

✅ Saves:
- `metrics.json` → train/test AUC & AP
- `model_xgb.json` or `model_sklearn.joblib`
- `oos_predictions.csv`

---

### Option B — Walk-forward training (NEW 🔥)

```bash
python3 train_xgb.py   --in ./output/dataset_ml_2017_2024.parquet   --outdir ./output/model_xgb_walk   --drop_none   --calibrate   --walk_forward   --wf_start 2020   --wf_end 2024   --save_per_fold
```

Each fold:
- Trains on data **before that year**
- Tests on **that year only**
- Reports AUC/AP and saves models/predictions per fold

✅ Outputs:
```
output/model_xgb_walk/
  fold_2020/
  fold_2021/
  fold_2022/
  fold_2023/
  fold_2024/
  walk_forward_results.json
  oos_predictions_all.csv
```

This allows realistic **out-of-sample validation** and performance tracking over time.

---

## ⚙️ 4. Parameters and notes

- `--drop_none` → exclude samples with label `0`  
- `--calibrate` → uses `CalibratedClassifierCV(cv=5)` for probability reliability  
- `--wf_start / --wf_end` → define the walk-forward range  
- `--save_per_fold` → export per-year models/predictions  
- Works with both **XGBoost** (preferred) and **RandomForest** (fallback)

---

## 🧰 5. Requirements
Inside a Python 3.9+ virtualenv:
```bash
pip install -U pandas numpy pyarrow pyyaml scikit-learn xgboost matplotlib
```

---

## ✅ Summary of new features
- Modular **dataset builder** (`make_dataset.py`)
- **Automated labeling** via ATR-based horizon logic (`make_labels.py`)
- **Improved model trainer** (`train_xgb.py`)
  - Supports `--walk_forward`
  - Supports per-fold saving
  - Supports calibrated probabilities
- Compatible with **XGBoost** or **scikit-learn RandomForest**
