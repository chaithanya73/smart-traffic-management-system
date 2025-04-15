import pandas as pd

# Load your dataset
df = pd.read_csv("network_dataset_with_flow_1.csv")

# Step 1: Filter for rows where density < 0.55
df_filtered = df[df["density"] < 0.55].copy()

# Step 2: Remove rows where all columns (except 'time' and 'lane_id') are zero
columns_to_check = [col for col in df_filtered.columns if col not in ['time', 'lane_id']]
df_filtered = df_filtered[~(df_filtered[columns_to_check] == 0).all(axis=1)]

# Step 3: Save to a new CSV file
df_filtered.to_csv("filtered_dataset_density_below_0.55_1.csv", index=False)

print("✅ Filtered dataset saved as 'filtered_dataset_density_below_0.55.csv'")
