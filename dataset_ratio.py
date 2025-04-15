import pandas as pd

# Load dataset
df = pd.read_csv("last_copy.csv")

# Calculate value counts and normalize to get ratio
congestion_ratios = df["congested"].value_counts(normalize=True)

print("📊 Congestion Ratios:")
print(congestion_ratios)
