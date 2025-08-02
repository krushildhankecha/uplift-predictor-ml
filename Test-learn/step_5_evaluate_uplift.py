import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load predictions
df = pd.read_csv("../data/output/uplift_predictions.csv")

# Sort by predicted uplift (descending)
df_sorted = df.sort_values(by="uplift", ascending=False).reset_index(drop=True)

# Cumulative treatment & control
df_sorted["cumulative_treat"] = df_sorted["treatment"].cumsum()
df_sorted["cumulative_control"] = (~df_sorted["treatment"].astype(bool)).cumsum()

# Cumulative gain
df_sorted["gain_treat"] = (df_sorted["actual"] * df_sorted["treatment"]).cumsum()
df_sorted["gain_control"] = (df_sorted["actual"] * (1 - df_sorted["treatment"])).cumsum()

# Avoid divide-by-zero
df_sorted["cumulative_gain"] = df_sorted["gain_treat"] - df_sorted["gain_control"]

# Qini Curve Plot
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(df_sorted)) / len(df_sorted), df_sorted["cumulative_gain"], label="Qini Curve", color="blue")
plt.plot([0, 1], [0, df_sorted["cumulative_gain"].iloc[-1]], label="Random Model", linestyle="--", color="gray")
plt.xlabel("Proportion of Population (sorted by uplift)")
plt.ylabel("Cumulative Gain")
plt.title("Qini Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../data/output/qini_curve.png")
plt.show()

# Qini coefficient
qini_auc = np.trapz(df_sorted["cumulative_gain"]) / len(df_sorted)
qini_random = df_sorted["cumulative_gain"].iloc[-1] / 2
qini_coefficient = qini_auc - qini_random

print(f"✅ Qini Coefficient: {qini_coefficient:.4f}")
