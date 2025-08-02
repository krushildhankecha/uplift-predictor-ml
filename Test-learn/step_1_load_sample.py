from datasets import load_dataset
import pandas as pd

# Load full dataset as a streaming generator to avoid memory overload
dataset = load_dataset("criteo/criteo-uplift", split="train", streaming=True)

# Filter for treatment = 1 and 0 separately
treatment_1 = dataset.filter(lambda x: x["treatment"] == 1)
treatment_0 = dataset.filter(lambda x: x["treatment"] == 0)

# Convert streaming datasets to lists and sample 1000 of each
print("⚙️ Sampling treatment = 1...")
df_treat = pd.DataFrame(list(treatment_1.take(1000)))

print("⚙️ Sampling treatment = 0...")
df_control = pd.DataFrame(list(treatment_0.take(1000)))

# Combine and shuffle
df_balanced = pd.concat([df_treat, df_control]).sample(frac=1, random_state=42).reset_index(drop=True)

# Final check
print("\n✅ Balanced subset distribution:")
print(df_balanced['treatment'].value_counts())

# Save
df_balanced.to_csv("../data/raw/sample_balanced_2000.csv", index=False)
