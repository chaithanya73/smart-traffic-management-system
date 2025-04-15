import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load your updated dataset
df = pd.read_csv("traffic_training_data_with_lane_id_entering_speed.csv")

# Encode lane_id again (or load previously saved encoder if preferred)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["lane_id_encoded"] = le.fit_transform(df["lane_id"])
joblib.dump(le, "lane_encoder.pkl")

# Use all 5 features now
X = df[["lane_id_encoded", "density", "speed_exit", "vehicle_count", "entering_speed"]]

# Retrain scaler
scaler = StandardScaler()
scaler.fit(X)

# Save updated scaler
joblib.dump(scaler, "scaler.pkl")
