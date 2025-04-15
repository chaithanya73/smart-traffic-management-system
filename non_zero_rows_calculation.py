import pandas as pd

# Load your dataset
df = pd.read_csv("dataset_network_mixed.csv")  # Replace with your actual file name

# Total rows
total_rows = len(df)

# Non-zero counts
non_zero_exit_speed = (df["entering_rate"] != 0).sum()
non_zero_entering_speed = (df["exit_rate"] != 0).sum()

# Percentages
exit_speed_percent = (non_zero_exit_speed / total_rows) * 100
entering_speed_percent = (non_zero_entering_speed / total_rows) * 100

# Output
print(f"✅ Non-zero exit rate rows: {non_zero_exit_speed} ({exit_speed_percent:.2f}%)")
print(f"✅ Non-zero entering rate rows: {non_zero_entering_speed} ({entering_speed_percent:.2f}%)")
