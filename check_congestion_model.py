import joblib
import numpy as np
import pandas as pd

# Load model and preprocessing tools
clf = joblib.load("congestion_predictor.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("lane_encoder.pkl")

# Define lane_id and encode
lane_id = "2149_0"
encoded_lane = encoder.transform([lane_id])[0]

# Define test scenarios
test_cases = [
    {"density": 0.3, "entering_rate": 7000, "exit_rate": 0, "vehicle_count": 70000},
    {"density": 0.10, "entering_rate": 700, "exit_rate": 100, "vehicle_count": 30000},
    {"density": 0.30, "entering_rate": 300, "exit_rate": 250, "vehicle_count": 40000},
    {"density": 0.40, "entering_rate": 500, "exit_rate": 50,  "vehicle_count": 10000},
    {"density": 0.05, "entering_rate": 10, "exit_rate": 900, "vehicle_count": 5000},
    {"density": 0.20, "entering_rate": 100, "exit_rate": 0,   "vehicle_count": 70000},
    {"density": 0.19, "entering_rate": 0, "exit_rate": 500, "vehicle_count": 3},
    {"density": 0.19, "entering_rate": 700, "exit_rate": 100, "vehicle_count": 50000}
]

# Expected feature order
feature_columns = [
    "density", "entering_rate", "exit_rate", "vehicle_count",
    "net_flow", "flow_ratio", "flow_vs_count"
]

# Run test cases
results = []
for case in test_cases:
    # Create interaction features
    entering = case["entering_rate"]
    exiting = case["exit_rate"]
    count = case["vehicle_count"]
    
    interaction_features = {
        "density": case["density"],
        "entering_rate": entering,
        "exit_rate": exiting,
        "vehicle_count": count,
        "net_flow": entering - exiting,
        "flow_ratio": (entering + 1) / (exiting + 1),  # +1 to avoid division by 0
        "flow_vs_count": (entering - exiting) / (count + 1)
    }

    # Prepare input and scale
    input_df = pd.DataFrame([interaction_features])[feature_columns]
    scaled_features = scaler.transform(input_df)

    # Combine with encoded lane
    final_input = np.hstack([[encoded_lane], scaled_features[0]])

    # Predict
    pred = clf.predict([final_input])[0]
    prob = clf.predict_proba([final_input])[0][1]

    # Store results
    results.append({
        **case,
        "predicted_congestion": "Yes" if pred else "No",
        "congestion_probability": round(prob, 4)
    })

# Show results
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
