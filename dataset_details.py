import pandas as pd

# Load your dataset
df = pd.read_csv("last_copy.csv")  # Change filename if needed

# Dataset shape
print("🔢 Shape of dataset:")
print(df.shape)

# Columns and data types
print("\n🧱 Column Data Types:")
print(df.dtypes)

# Missing values
print("\n❌ Missing Values per Column:")
print(df.isnull().sum())

# Basic statistics
print("\n📊 Descriptive Statistics (Numerical Features):")
print(df.describe())

# Unique values
print("\n🔍 Unique Values per Column:")
for col in df.columns:
    unique_vals = df[col].nunique()
    print(f"{col}: {unique_vals} unique values")

# Value counts for categorical columns (optional: skip long ones)
print("\n📦 Value Counts for Categorical Columns:")
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head(10))  # top 10 only

# Sample rows
print("\n🔎 First 5 Rows:")
print(df.head())