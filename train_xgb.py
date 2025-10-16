# train_xgb.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import warnings
# Try XGBoost; fall back to RandomForest if missing
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, KFold

# Columns that are NOT model features
NON_FEATURE_COLS = {
    # identifiers / time / raw OHLCV
    "timestamp", "ts_local", "session_day", "hour_local", "minute_local",
    "open", "high", "low", "close", "volume",
    # labels and aux
    "tp_price", "sl_price", "hit_type", "time_to_hit", "label", "ret_h",
    "horizon_bars", "tp_atr_mult", "sl_atr_mult"
}

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def pick_features(df: pd.DataFrame):
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # numeric only
    num = df[cols].select_dtypes(include=["number"]).columns.tolist()
    return num

def split_time_default(df: pd.DataFrame):
    """Time-split: train 2017–2023 inclusive, OOS 2024–..."""
    train_mask = (df["timestamp"] < pd.Timestamp("2024-01-01", tz="UTC"))
    test_mask  = ~train_mask
    return df[train_mask].copy(), df[test_mask].copy()

def make_model(spw: float, seed: int):
    """Create a base classifier with reasonable regularization."""
    if HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            scale_pos_weight=spw,
            n_jobs=-1
        )
    else:
        # Fallback RF with class_weight balanced
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        )

def fit_with_optional_calibration(clf, X_train, y_train, calibrate: bool):
    """Fit model; if calibrate=True, wrap with CalibratedClassifierCV (cv=5)."""
    if calibrate:
        # Fit calibration on TRAIN ONLY to avoid leakage
        model = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        return model, True
    else:
        clf.fit(X_train, y_train)
        return clf, False

def predict_proba01(model, X):
    return model.predict_proba(X)[:, 1]

def walk_forward_eval(df: pd.DataFrame,
                      X_cols: list,
                      outdir: Path,
                      start_year: int = 2020,
                      end_year:   int = 2024,
                      seed: int = 42,
                      calibrate: bool = True,
                      save_per_fold: bool = False):
    """
    For each test year in [start_year, end_year]:
      - Train on all years strictly < test_year
      - Test on exactly test_year
    Saves metrics summary and (optionally) per-fold models/preds.
    """
    results = []
    all_oos_rows = []

    for year in range(start_year, end_year + 1):
        train_df = df[df["timestamp"].dt.year < year]
        test_df  = df[df["timestamp"].dt.year == year]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Binary target: TP-first vs not
        y_train = (train_df["label"] == 1).astype(int).values
        y_test  = (test_df["label"]  == 1).astype(int).values

        X_train = train_df[X_cols].astype(float).values
        X_test  = test_df[X_cols].astype(float).values

        # Class imbalance
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        spw = float(neg) / max(1.0, float(pos))

        # Build & fit model
        clf = make_model(spw=spw, seed=seed)
        model, was_calibrated = fit_with_optional_calibration(clf, X_train, y_train, calibrate)

        # Evaluate
        p_test = predict_proba01(model, X_test)
        auc = float(roc_auc_score(y_test, p_test))
        ap  = float(average_precision_score(y_test, p_test))

        print(f"Year {year}: AUC={auc:.3f}  AP={ap:.3f}  "
              f"train={len(train_df):,}  test={len(test_df):,}  spw={spw:.2f}")

        fold_res = {
            "year": year,
            "auc": auc,
            "ap": ap,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "spw": spw,
            "calibrated": was_calibrated
        }
        results.append(fold_res)

        # Collect OOS predictions
        oos = test_df[["timestamp"]].copy()
        oos["proba_tp_first"] = p_test
        oos["y_true"] = y_test
        all_oos_rows.append(oos)

        # Optionally save per-fold artifacts
        if save_per_fold:
            fold_dir = outdir / f"fold_{year}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            oos.to_csv(fold_dir / "oos_predictions.csv", index=False)
            if HAS_XGB and not was_calibrated:
                model.get_booster().save_model(str(fold_dir / "model_xgb.json"))
                (fold_dir / "model_type.txt").write_text("xgboost")
            else:
                import joblib
                joblib.dump(model, fold_dir / "model_sklearn.joblib")
                (fold_dir / "model_type.txt").write_text("sklearn")
            save_json({"feature_list": X_cols}, fold_dir / "features.json")
            save_json(fold_res, fold_dir / "metrics.json")

    # Save summary
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(results, outdir / "walk_forward_results.json")

    # Save concatenated OOS predictions across all years
    if len(all_oos_rows):
        oos_all = pd.concat(all_oos_rows, axis=0).reset_index(drop=True)
        oos_all.sort_values("timestamp").to_csv(outdir / "oos_predictions_all.csv", index=False)

    # Print aggregate stats
    if results:
        mean_auc = float(np.mean([r["auc"] for r in results]))
        mean_ap  = float(np.mean([r["ap"]  for r in results]))
        print(json.dumps({
            "wf_years": [r["year"] for r in results],
            "wf_mean_auc": mean_auc,
            "wf_mean_ap":  mean_ap,
            "folds": results
        }, indent=2))

def purged_kfold_indices(n, folds, embargo):
    idx = np.arange(n)
    fold_sizes = np.full(folds, n // folds, dtype=int)
    fold_sizes[:n % folds] += 1
    current = 0
    for k in range(folds):
        start, stop = current, current + fold_sizes[k]
        test_idx = idx[start:stop]
        left  = max(0, start - embargo)
        right = min(n, stop + embargo)
        train_mask = np.ones(n, dtype=bool)
        train_mask[left:right] = False
        train_idx = idx[train_mask]
        current = stop
        yield train_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="dataset_ml.parquet/csv from make_labels.py")
    ap.add_argument("--outdir", required=True, help="output dir for models and metrics")
    ap.add_argument("--drop_none", action="store_true", help="drop rows with label=0 (NONE)")
    ap.add_argument("--calibrate", action="store_true", help="apply probability calibration (cv=5) on train data")
    ap.add_argument("--seed", type=int, default=42)
    # Walk-forward options
    ap.add_argument("--walk_forward", action="store_true", help="enable walk-forward evaluation instead of single split")
    ap.add_argument("--wf_start", type=int, default=2020, help="first test year (inclusive)")
    ap.add_argument("--wf_end",   type=int, default=2024, help="last  test year (inclusive)")
    ap.add_argument("--save_per_fold", action="store_true", help="save per-fold models and predictions")
    ap.add_argument("--purged_cv", action="store_true", help="Do purged KFold with embargo for validation metrics")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--cv_embargo_bars", type=int, default=96)
    ap.add_argument("--export_feature_importance", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = read_any(Path(args.inp))

    # Ensure timestamp sorting / tz
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Optional: drop NONE class to make binary TP vs SL task
    if args.drop_none:
        df = df[df["label"].isin([1, -1])].reset_index(drop=True)

    X_cols = pick_features(df)

    # Walk-forward mode
    if args.walk_forward:
        walk_forward_eval(
            df=df,
            X_cols=X_cols,
            outdir=outdir,
            start_year=args.wf_start,
            end_year=args.wf_end,
            seed=args.seed,
            calibrate=args.calibrate,
            save_per_fold=args.save_per_fold
        )
        print(f"Walk-forward artifacts saved to: {outdir}")
        return

    # ----- Single split path (original behavior) -----
    df_train, df_test = split_time_default(df)
    X_train, y_train = df_train[X_cols].astype(float).values, (df_train["label"]==1).astype(int).values
    X_test,  y_test  = df_test[X_cols].astype(float).values,  (df_test["label"]==1).astype(int).values

    # Guard: ensure non-empty test
    if len(y_test) == 0:
        raise ValueError("Test set is empty. Use --walk_forward or include OOS data (e.g., 2024).")

    if args.purged_cv:
        n = len(df_train)
        X_all = df_train[X_cols].astype(float).values
        y_all = (df_train["label"] == 1).astype(int).values
        aucs, aps = [], []
        embargo = int(args.cv_embargo_bars)
        for tr_idx, te_idx in purged_kfold_indices(n, args.cv_folds, embargo):
            Xtr, Xte = X_all[tr_idx], X_all[te_idx]
            ytr, yte = y_all[tr_idx], y_all[te_idx]
            pos = int(ytr.sum()); neg = int(len(ytr) - pos)
            spw = float(neg) / max(1.0, float(pos))
            clf_cv = make_model(spw=spw, seed=args.seed)
            model_cv, _ = fit_with_optional_calibration(clf_cv, Xtr, ytr, args.calibrate)
            p_cv = predict_proba01(model_cv, Xte)
            aucs.append(float(roc_auc_score(yte, p_cv)))
            aps.append(float(average_precision_score(yte, p_cv)))
        cv_report = {
            "cv_auc_mean": float(np.mean(aucs)),
            "cv_ap_mean": float(np.mean(aps)),
            "fold_aucs": aucs,
            "fold_aps": aps,
            "folds": args.cv_folds,
            "embargo_bars": embargo
        }
        save_json(cv_report, outdir / "cv_report.json")

    # Class imbalance: compute scale_pos_weight on TRAIN
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = float(neg) / max(1.0, float(pos))

    clf = make_model(spw=spw, seed=args.seed)

    model, was_calibrated = fit_with_optional_calibration(clf, X_train, y_train, args.calibrate)

    p_train = predict_proba01(model, X_train)
    p_test  = predict_proba01(model, X_test)

    metrics = {
        "train": {
            "auc": float(roc_auc_score(y_train, p_train)),
            "ap" : float(average_precision_score(y_train, p_train))
        },
        "test": {
            "auc": float(roc_auc_score(y_test, p_test)),
            "ap" : float(average_precision_score(y_test, p_test))
        },
        "class_balance_train": {"pos": pos, "neg": neg, "spw": spw},
        "calibrated": was_calibrated,
        "features": X_cols,
        "rows": {"train": int(len(y_train)), "test": int(len(y_test)), "total": int(len(df))}
    }
    save_json(metrics, outdir / "metrics.json")

    if args.export_feature_importance:
        try:
            if HAS_XGB and not was_calibrated:
                booster = model.get_booster()
                imp = booster.get_score(importance_type="gain")
                imp_df = pd.DataFrame({
                    "feature": list(imp.keys()),
                    "gain": list(imp.values())
                }).sort_values("gain", ascending=False)
            else:
                fi = getattr(model, "feature_importances_", None)
                if fi is not None:
                    imp_df = pd.DataFrame({"feature": X_cols, "gain": fi}).sort_values("gain", ascending=False)
                else:
                    imp_df = pd.DataFrame({"feature": X_cols, "gain": 0.0})
            imp_df.to_csv(outdir / "feature_importance.csv", index=False)
        except Exception as _e:
            pass

    # Save predictions (OOS) aligned with timestamps
    oos = df_test[["timestamp"]].copy()
    oos["proba_tp_first"] = p_test
    oos["y_true"] = y_test
    oos.to_csv(outdir / "oos_predictions.csv", index=False)

    # Save model
    if HAS_XGB and not was_calibrated:
        model.get_booster().save_model(str(outdir / "model_xgb.json"))
        (outdir / "model_type.txt").write_text("xgboost")
    else:
        import joblib
        joblib.dump(model, outdir / "model_sklearn.joblib")
        (outdir / "model_type.txt").write_text("sklearn")
    save_json({"feature_list": X_cols}, outdir / "features.json")

    print(json.dumps(metrics, indent=2))
    print(f"Saved OOS predictions → {outdir / 'oos_predictions.csv'}")
    print(f"Model + features saved in → {outdir}")
    print("Done.")

if __name__ == "__main__":
    main()
