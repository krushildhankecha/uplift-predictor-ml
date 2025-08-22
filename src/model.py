import joblib
from sklearn.ensemble import RandomForestClassifier
from sklift.datasets import fetch_criteo
from sklearn.model_selection import train_test_split
import numpy as np

MODEL_TREATED = "rf_treated.pkl"
MODEL_CONTROL = "rf_control.pkl"

def train_and_save(n_samples=100_000):
    X, y, treatment = fetch_criteo(target_col='conversion', treatment_col='treatment', return_X_y_t=True)
    # for speed, downsample
    idx = np.random.choice(len(y), size=n_samples, replace=False)
    X, y, treatment = X.iloc[idx], y.iloc[idx], treatment.iloc[idx]

    X_t = X[treatment == 1]
    y_t = y[treatment == 1]
    X_c = X[treatment == 0]
    y_c = y[treatment == 0]

    rf_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_c = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    rf_t.fit(X_t, y_t)
    rf_c.fit(X_c, y_c)

    joblib.dump(rf_t, MODEL_TREATED)
    joblib.dump(rf_c, MODEL_CONTROL)
    print("Models saved.")

def load_models():
    rf_t = joblib.load(MODEL_TREATED)
    rf_c = joblib.load(MODEL_CONTROL)
    return rf_t, rf_c

def predict_uplift(X_new):
    rf_t, rf_c = load_models()
    return rf_t.predict_proba(X_new)[:,1] - rf_c.predict_proba(X_new)[:,1]
