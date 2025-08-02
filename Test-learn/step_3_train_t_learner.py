# step_3_train_t_learner.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from step_2_preprocess_and_split import X_train_treat, y_train_treat, X_train_control, y_train_control, X_test, treatment_test, y_test

# Load training data
X_train_treat = pd.read_csv("../data/processed/X_train_treat.csv")
y_train_treat = pd.read_csv("../data/processed/y_train_treat.csv").squeeze()

# Train treatment model
model_treat = RandomForestClassifier(n_estimators=100, random_state=42)
model_treat.fit(X_train_treat, y_train_treat)

# Train control model
model_control = RandomForestClassifier(n_estimators=100, random_state=42)
model_control.fit(X_train_control, y_train_control)

# Predict both outcomes
proba_treat = model_treat.predict_proba(X_test)[:, 1]
proba_control = model_control.predict_proba(X_test)[:, 1]

# Calculate uplift
uplift = proba_treat - proba_control

# Save models
joblib.dump(model_treat, "models/t_learner_treat.pkl")
joblib.dump(model_control, "models/t_learner_control.pkl")

# Print results
results_df = pd.DataFrame({
    "treatment": treatment_test,
    "actual": y_test,
    "pred_treat": proba_treat,
    "pred_control": proba_control,
    "uplift": uplift
})

print(results_df.head(10))
print("\nAverage uplift score (top 10):", results_df.sort_values("uplift", ascending=False).head(10)["uplift"].mean())
