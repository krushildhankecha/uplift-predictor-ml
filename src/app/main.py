"""
criteo_uplift_pipeline.py
A compact end-to-end script to load the `criteo/criteo-uplift` dataset from Hugging Face,
preprocess it, train a simple T-learner uplift model (two LightGBM models), and evaluate
using uplift metrics (uplift@k and qini curve).

Usage example
--------------
python criteo_uplift_pipeline.py --nrows 200000 --model-output model.pkl --log-mlflow

Requirements
------------
pip install datasets pandas scikit-learn lightgbm mlflow joblib

Notes
-----
- This is intentionally single-file and minimal to follow the structure of the referenced
  YT-MLOPS repo. Replace components with your project's modules as needed.
- The script uses a T-learner: two separate models trained on treatment and control.
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
import os
import mlflow


def load_criteo_uplift(nrows: int | None = None) -> pd.DataFrame:
    """Load the criteo/criteo-uplift dataset from Hugging Face and return a DataFrame.
    If nrows is provided, sample that many rows for faster local development.
    """
    ds = load_dataset("criteo/criteo-uplift", split="train")
    df = ds.to_pandas()
    if nrows is not None and nrows < len(df):
        df = df.sample(n=nrows, random_state=42).reset_index(drop=True)
    return df


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing:
    - drop columns not useful for modeling
    - ensure types
    - fill na
    - create X, treatment, y
    """
    df = df.copy()
    # The dataset has continuous features named f0..f11 and treatment, conversion, visit, exposure
    # For simplicity we will use f0..f11 as features. If more features exist, adapt accordingly.
    feature_cols = [c for c in df.columns if c.startswith("f")]
    # fill missing
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    df[feature_cols] = df[feature_cols].astype(float)
    # target
    if "conversion" not in df.columns:
        raise ValueError("expected 'conversion' column in dataset")
    y = df["conversion"].astype(int)
    treatment = df["treatment"].astype(int)  # 1 = treated, 0 = control
    X = df[feature_cols]
    return X, treatment, y


def train_t_learner(X_train, t_train, y_train, X_val, t_val, y_val, params=None):
    """Train two LightGBM models: one on treated group, one on control group.
    Return fitted models and validation uplift predictions.
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
        }
    # split train into treated and control
    X_treat = X_train[t_train == 1]
    y_treat = y_train[t_train == 1]
    X_ctrl = X_train[t_train == 0]
    y_ctrl = y_train[t_train == 0]

    dtrain_t = lgb.Dataset(X_treat, label=y_treat)
    dtrain_c = lgb.Dataset(X_ctrl, label=y_ctrl)

    # small num_boost_round for speed; increase for better results
    num_boost_round = 100
    model_t = lgb.train(params, dtrain_t, num_boost_round=num_boost_round)
    model_c = lgb.train(params, dtrain_c, num_boost_round=num_boost_round)

    # validation predictions: probability of conversion if treated / control
    p_treated = model_t.predict(X_val)
    p_control = model_c.predict(X_val)
    uplift_pred = p_treated - p_control

    return {
        "model_t": model_t,
        "model_c": model_c,
        "uplift_val": uplift_pred,
        "p_treated": p_treated,
        "p_control": p_control,
    }


def uplift_at_k(y_true, treatment, uplift_score, k=0.1):
    """Compute uplift@k: average treatment effect among top k fraction scored for targeting.
    Steps:
      - rank by uplift_score descending
      - take top k fraction
      - compute difference in observed conversion rates between treated and control in that top subset
    """
    assert len(y_true) == len(treatment) == len(uplift_score)
    n = len(y_true)
    top_n = max(1, int(n * k))
    order = np.argsort(-uplift_score)
    sel = order[:top_n]
    y_sel = y_true[sel]
    t_sel = treatment.values[sel] if hasattr(treatment, "values") else treatment[sel]
    treated_mask = t_sel == 1
    control_mask = t_sel == 0
    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        return 0.0
    conv_t = y_sel[treated_mask].mean()
    conv_c = y_sel[control_mask].mean()
    return conv_t - conv_c


def qini_curve(y_true, treatment, uplift_score):
    """Compute a simple Qini curve as cumulative incremental gains.
    Returns x (fraction targeted) and incremental gain values.
    """
    # DataFrame for sorting
    df = pd.DataFrame({"y": y_true, "t": treatment, "score": uplift_score})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    # cumulative treated & control conversions
    df["cum_n"] = np.arange(1, len(df) + 1)
    df["cum_treated"] = (df["t"] == 1).cumsum()
    df["cum_control"] = (df["t"] == 0).cumsum()
    df["cum_conv_treated"] = (df["y"] * (df["t"] == 1)).cumsum()
    df["cum_conv_control"] = (df["y"] * (df["t"] == 0)).cumsum()

    # avoid div by zero
    treated_rate = df["cum_conv_treated"] / df["cum_treated"].replace(0, np.nan)
    control_rate = df["cum_conv_control"] / df["cum_control"].replace(0, np.nan)
    # incremental gain (fillna with 0 when denominator was zero)
    incremental = (treated_rate.fillna(0) - control_rate.fillna(0)) * (df["cum_treated"] + df["cum_control"])
    frac = df["cum_n"] / len(df)
    return frac.values, incremental.values


def evaluate_and_log(y_val, t_val, uplift_val, p_treated, p_control, run_name=None, log_mlflow=False):
    """Compute metrics and optionally log to MLflow."""
    # simple metrics
    # AUC for treated and control models on their subsets
    mask_t = t_val == 1
    mask_c = t_val == 0
    auc_t = roc_auc_score(y_val[mask_t], p_treated[mask_t]) if mask_t.sum() > 0 else np.nan
    auc_c = roc_auc_score(y_val[mask_c], p_control[mask_c]) if mask_c.sum() > 0 else np.nan
    u_at_1 = uplift_at_k(y_val.values, t_val, uplift_val, k=0.01)
    u_at_5 = uplift_at_k(y_val.values, t_val, uplift_val, k=0.05)

    frac, qini = qini_curve(y_val.values, t_val.values, uplift_val)
    qini_auc = np.trapz(qini, frac)

    metrics = {
        "auc_treated": float(auc_t) if not np.isnan(auc_t) else None,
        "auc_control": float(auc_c) if not np.isnan(auc_c) else None,
        "uplift_at_1pct": float(u_at_1),
        "uplift_at_5pct": float(u_at_5),
        "qini_auc": float(qini_auc),
    }

    if log_mlflow:
        mlflow.set_experiment("criteo_uplift_experiment")
        with mlflow.start_run(run_name=run_name):
            for k, v in metrics.items():
                mlflow.log_metric(k, v if v is not None else -1)
    return metrics


def run_pipeline(args):
    print("Loading dataset from Hugging Face...")
    df = load_criteo_uplift(nrows=args.nrows)
    print(f"Loaded {len(df)} rows")

    X, t, y = basic_preprocess(df)

    # split
    X_train, X_val, t_train, t_val, y_train, y_val = train_test_split(
        X, t, y, test_size=0.2, random_state=42, stratify=None
    )

    print("Training T-learner (LightGBM models)...")
    trained = train_t_learner(X_train, t_train, y_train, X_val, t_val, y_val)

    metrics = evaluate_and_log(
        y_val, t_val, trained["uplift_val"], trained["p_treated"], trained["p_control"], run_name=args.run_name, log_mlflow=args.log_mlflow
    )
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # save models
    if args.model_output:
        os.makedirs(os.path.dirname(args.model_output) or ".", exist_ok=True)
        joblib.dump({"model_t": trained["model_t"], "model_c": trained["model_c"]}, args.model_output)
        print(f"Saved models to {args.model_output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=100000, help="sample n rows for quick runs")
    p.add_argument("--model-output", type=str, default="models/criteo_tlearner.joblib")
    p.add_argument("--log-mlflow", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()
    run_pipeline(args)
