import pandas as pd

# Load your dataset
df = pd.read_csv("last_copy.csv")  # replace with your actual file name if different

# Filter for congested rows
congested_rows = df[df['congested'] == 1]

# Sort by density in ascending order
least_density_congested = congested_rows.sort_values(by='density', ascending=True)

# Display the result
print(least_density_congested)

# Optionally, save to CSV
least_density_congested.to_csv("least_density_congested_rows.csv", index=False)
