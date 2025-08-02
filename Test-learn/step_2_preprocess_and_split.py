import pandas as pd
from sklearn.model_selection import train_test_split

# Load sample data
df = pd.read_csv("../data/raw/sample_balanced_2000.csv")

# Step 1: Basic check for missing values
print("🔍 Missing values per column:")
print(df.isnull().sum())

# Step 2: Separate feature columns
feature_cols = [col for col in df.columns if col.startswith("f")]
target_col = "visit"
treatment_col = "treatment"

# Step 3: Split into treatment and control groups
df_treatment = df[df[treatment_col] == 1]
df_control = df[df[treatment_col] == 0]

print(f"\n📌 Treatment group size: {len(df_treatment)}")
print(f"📌 Control group size: {len(df_control)}")

# Check before split
if len(df_treatment) < 2 or len(df_control) < 2:
    raise ValueError("❌ Not enough samples in treatment or control group. Try increasing dataset size.")

# Step 4: Train/Test split for both groups
X_train_treat, X_test_treat, y_train_treat, y_test_treat = train_test_split(
    df_treatment[feature_cols], df_treatment[target_col], test_size=0.3, random_state=42
)

X_train_control, X_test_control, y_train_control, y_test_control = train_test_split(
    df_control[feature_cols], df_control[target_col], test_size=0.3, random_state=42
)

# Combine test sets for final evaluation
X_test = pd.concat([X_test_treat, X_test_control], ignore_index=True)
y_test = pd.concat([y_test_treat, y_test_control], ignore_index=True)
treatment_test = pd.concat([
    pd.Series([1] * len(X_test_treat)),
    pd.Series([0] * len(X_test_control))
], ignore_index=True)

# ✅ Sanity check
print(f"\n🧪 Treatment train size: {X_train_treat.shape[0]}")
print(f"🧪 Control train size: {X_train_control.shape[0]}")
print(f"🧪 Test set size: {X_test.shape[0]}")

# Save to CSV for Step 3
X_train_treat.to_csv("../data/processed/X_train_treat.csv", index=False)
y_train_treat.to_frame().to_csv("../data/processed/y_train_treat.csv", index=False)

X_train_control.to_csv("../data/processed/X_train_control.csv", index=False)
y_train_control.to_frame().to_csv("../data/processed/y_train_control.csv", index=False)

X_test.to_csv("../data/processed/X_test.csv", index=False)
y_test.to_frame().to_csv("../data/processed/y_test.csv", index=False)
treatment_test.to_frame(name="treatment").to_csv("../data/processed/treatment_test.csv", index=False)

# Export for next step (optional if using import)
# Now these variables are globally available
