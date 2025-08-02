import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load processed data
X_train_treat = pd.read_csv("../data/processed/X_train_treat.csv")
y_train_treat = pd.read_csv("../data/processed/y_train_treat.csv").values.ravel()

X_train_control = pd.read_csv("../data/processed/X_train_control.csv")
y_train_control = pd.read_csv("../data/processed/y_train_control.csv").values.ravel()

X_test = pd.read_csv("../data/processed/X_test.csv")
y_test = pd.read_csv("../data/processed/y_test.csv").values.ravel()
treatment_test = pd.read_csv("../data/processed/treatment_test.csv")["treatment"].values

# Train models
model_treat = RandomForestClassifier(random_state=42)
model_treat.fit(X_train_treat, y_train_treat)

model_control = RandomForestClassifier(random_state=42)
model_control.fit(X_train_control, y_train_control)

# Predict probabilities
proba_treat = model_treat.predict_proba(X_test)[:, 1]
proba_control = model_control.predict_proba(X_test)[:, 1]

# Calculate uplift
uplift = proba_treat - proba_control

# Save predictions and uplift
results = pd.DataFrame({
    "treatment": treatment_test,
    "actual": y_test,
    "pred_treatment": proba_treat,
    "pred_control": proba_control,
    "uplift": uplift
})

results.to_csv("../data/output/uplift_predictions.csv", index=False)

print("✅ Uplift predictions saved to ../data/output/uplift_predictions.csv")
