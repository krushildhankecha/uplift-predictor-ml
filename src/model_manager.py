import joblib
import pandas as pd

MODEL_PATHS = {
    "treatment": "src/models/rf_treatment.pkl",
    "visit": "src/models/rf_visit.pkl",
    "conversion": "src/models/rf_conversion.pkl"
}

_models = {}

def load_model(name):
    if name not in _models:
        _models[name] = joblib.load(MODEL_PATHS[name])
    return _models[name]

def predict_pipeline(target: str, features: dict):
    df = pd.DataFrame([features])

    # Handle dependencies
    if target == "conversion":
        if "treatment" not in df.columns:
            treatment_model = load_model("treatment")
            df["treatment"] = treatment_model.predict(df)
        model = load_model("conversion")
        pred = model.predict(df)[0]

    elif target == "visit":
        if "treatment" not in df.columns:
            treatment_model = load_model("treatment")
            df["treatment"] = treatment_model.predict(df)
        model = load_model("visit")
        pred = model.predict(df)[0]

    elif target == "treatment":
        model = load_model("treatment")
        pred = model.predict(df)[0]

    else:
        raise ValueError(f"Unknown target: {target}")

    return pred
