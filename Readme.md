# Intelligent Traffic Management System using SUMO and Machine Learning

This project simulates an intelligent traffic control system in **SUMO** (Simulation of Urban Mobility), using **machine learning** to predict congestion, estimate clearance times, reroute vehicles, and dynamically update traffic flows.

---

## 🚦 Features

- **Congestion Prediction** using a trained ML model (`congestion_predictor.pkl`)
- **Clearance Time Estimation** for congested lanes using `congestion_clearance_model.pkl`
- **Route Time Estimation** to compare current vs. alternative routes
- **Dynamic Rerouting** of vehicles waiting at red lights
- **Live Lane Speed Updates** to improve route decisions
- Fully integrated with **SUMO and TraCI**

---

## 📁 Project Structure

```bash
.
├── traffic_simulation.py         # Main simulation script
├── congestion_predictor.pkl      # ML model to predict congestion
├── clearance_model.pkl           # ML model to estimate congestion clearance time
├── scaler.pkl                    # Pre-fitted scaler for feature normalization
├── your_sumo_config.sumocfg      # Your SUMO configuration file
├── net.net.xml                   # SUMO network file
├── routes.rou.xml                # Vehicle routes
├── data/                         # Optional: historical traffic data
└── README.md                     # This file

