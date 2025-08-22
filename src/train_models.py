# train_models.py
import os
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)

def train_and_save_models(n_samples=200_000):
    # 1. Load dataset
    dataset = load_dataset("criteo/criteo-uplift", split="train[:5%]")  # use 5% subset for speed
    df = pd.DataFrame(dataset)

    features = [col for col in df.columns if col.startswith("f")]
    X = df[features]

    # 2. Treatment model
    y_treatment = df["treatment"]
    rf_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_t.fit(X, y_treatment)
    joblib.dump(rf_t, "models/rf_treatment.pkl")
    print("✅ Treatment model saved.")

    # 3. Visit model
    if "visit" in df.columns:
        y_visit = df["visit"]
        X_visit = X.copy()
        X_visit["treatment"] = y_treatment  # treatment is feature for visit
        rf_v = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_v.fit(X_visit, y_visit)
        joblib.dump(rf_v, "models/rf_visit.pkl")
        print("✅ Visit model saved.")

    # 4. Conversion model
    if "conversion" in df.columns:
        y_conv = df["conversion"]
        X_conv = X.copy()
        X_conv["treatment"] = y_treatment
        rf_c = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_c.fit(X_conv, y_conv)
        joblib.dump(rf_c, "models/rf_conversion.pkl")
        print("✅ Conversion model saved.")

if __name__ == "__main__":
    train_and_save_models()
