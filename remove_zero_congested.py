import pandas as pd

# Load your dataset
df = pd.read_csv("filtered_dataset_density_below_0.55_1.csv")

# Keep only rows where congested is not 0
df_filtered = df[df["congested"] != 0]

# Save the filtered dataset
df_filtered.to_csv("congested_only_dataset.csv", index=False)
print(f"✅ Removed rows where congested == 0. Remaining rows: {len(df_filtered)}")
