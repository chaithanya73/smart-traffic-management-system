import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the trained model
clf = joblib.load("congestion_predictor.pkl")

# Feature names (must match the training code!)
features = [
    "Lane ID (encoded)", "Density", "Entering Rate", "Exit Rate", "Vehicle Count",
    "Net Flow", "Flow Ratio", "Flow vs Count"
]

# Get feature importances from model
importances = clf.feature_importances_

# Ensure matching lengths
if len(features) != len(importances):
    raise ValueError(f"Mismatch: {len(features)} features vs {len(importances)} importances.")

# Create a DataFrame for plotting
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="crest")
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("🔥 Feature Importance in Congestion Prediction", fontsize=14, weight='bold')
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()