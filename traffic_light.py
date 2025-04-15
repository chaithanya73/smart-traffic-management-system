import os
import sys
import traci
import math
import joblib
import numpy as np
from collections import defaultdict, deque

# === SUMO Initialization ===
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("❌ Please set the 'SUMO_HOME' environment variable.")

sumo_cmd = ["sumo-gui", "-c", "copy6_traffic_lights.sumocfg", "--step-length", "1"]
traci.start(sumo_cmd)

# === Load ML Models ===
clf = joblib.load("congestion_predictor.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("lane_encoder.pkl")

# === Data Buffers ===
lane_flow_history = defaultdict(lambda: deque(maxlen=5))  # Store recent lane features
previous_phase = {}

# === Congestion Predictor Function ===
def is_congested_lane(lane_id):
    try:
        # Extract features from the buffer
        if len(lane_flow_history[lane_id]) < 1:
            return False  # Not enough data

        # Use the most recent flow data
        latest = lane_flow_history[lane_id][-1]
        (
            density, entering_rate, exit_rate,
            vehicle_count, net_flow, flow_ratio, flow_vs_count
        ) = latest

        lane_encoded = encoder.transform([lane_id])[0]
        X = np.array([[lane_encoded, density, entering_rate, exit_rate,
                       vehicle_count, net_flow, flow_ratio, flow_vs_count]])

        X[:, 1:] = scaler.transform(X[:, 1:])
        pred = clf.predict(X)
        return pred[0] == 1
    except:
        return False

# === Dummy Models ===
def predict_time_to_reach(vehicle_id, target_lane):
    return 25.0  # stub

def predict_time_to_clear(lane_id):
    return 40.0  # stub

# === Update Flow History ===
def update_lane_flow_data(lane_id):
    if lane_id.startswith(":"):
        return

    try:
        density = traci.lane.getLastStepOccupancy(lane_id)
        entering_rate = len(traci.lane.getLastStepVehicleIDs(lane_id))  # estimate
        exit_rate = 0  # optional placeholder
        vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)

        net_flow = entering_rate - exit_rate
        flow_ratio = entering_rate / (exit_rate + 1)
        flow_vs_count = net_flow / (vehicle_count + 1)

        lane_flow_history[lane_id].append((
            density, entering_rate, exit_rate,
            vehicle_count, net_flow, flow_ratio, flow_vs_count
        ))
    except:
        pass

# === Main Adaptive Signal Logic ===
def adjust_traffic_lights(tl_id):
    global previous_phase

    current_phase = traci.trafficlight.getPhase(tl_id)
    num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)

    if tl_id in previous_phase and previous_phase[tl_id] > current_phase:
        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tl_id)))
        current_state = traci.trafficlight.getRedYellowGreenState(tl_id)
        lane_signal_map = {lane: current_state[i] for i, lane in enumerate(controlled_lanes)}

        # print(f"\n🔄 Traffic light '{tl_id}' changing phase:")
        # print(f"🔧 Controlled lanes: {controlled_lanes}")

        for lane, signal in lane_signal_map.items():
            if signal.lower() != 'r':
                continue

            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            lane_len = traci.lane.getLength(lane)

            for veh_id in vehicle_ids:
                pos = traci.vehicle.getLanePosition(veh_id)
                if lane_len - pos <= 20:
                    try:
                        route = traci.vehicle.getRoute(veh_id)
                        curr_edge = traci.vehicle.getRoadID(veh_id)
                        i = route.index(curr_edge)
                        future_edges = route[i+1:]

                        for edge in future_edges:
                            for j in range(traci.edge.getLaneNumber(edge)):
                                next_lane = f"{edge}_{j}"
                                if is_congested_lane(next_lane):
                                    # time_reach = predict_time_to_reach(veh_id, next_lane)
                                    # time_clear = predict_time_to_clear(next_lane)

                                    # if time_clear > time_reach:
                                        traci.vehicle.rerouteTraveltime(veh_id)
                                        print(f"🚘 Rerouted {veh_id} to avoid {next_lane}")
                                    # break
                    except Exception as e:
                        print(f"⚠️ Error rerouting {veh_id}: {e}")

        # === Adjust Traffic Light Durations ===
        distance_threshold = math.inf
        lane_vehicle_counts = {}
        for lane in controlled_lanes:
            update_lane_flow_data(lane)
            count_near = sum(
                1 for vid in traci.lane.getLastStepVehicleIDs(lane)
                if traci.lane.getLength(lane) - traci.vehicle.getLanePosition(vid) <= distance_threshold
            )
            lane_vehicle_counts[lane] = count_near

        total_vehicles = sum(lane_vehicle_counts.values())
        min_green_time = 5
        total_cycle_time = 100
        remain_time = total_cycle_time - (min_green_time * num_phases)
        extra_time = {lane: 0 for lane in controlled_lanes}

       # print(f"📊 Vehicle count per lane: {lane_vehicle_counts}")

        if total_vehicles > 0:
            for lane, count in lane_vehicle_counts.items():
                extra_time[lane] = (count / total_vehicles) * remain_time

        flow_rate = 0.3
        new_phases = []

        for i, phase in enumerate(logic.phases):
            if "g" in phase.state:
                lane = controlled_lanes[i] if i < len(controlled_lanes) else None
                if not lane:
                    print(f"⚠️ Skipping phase {i}, no matching lane.")
                  #  new_phases.append(traci.trafficlight.Phase(phase.duration, phase.state))
                    continue

                count = lane_vehicle_counts.get(lane, 0)
                extra = extra_time.get(lane, 0)
                expected_time = count / flow_rate if flow_rate > 0 else min_green_time
                new_duration = max(min_green_time, min(expected_time, min_green_time + extra))

              #  print(f"🟢 Lane {lane}: vehicles={count}, extra={extra:.2f}, new green={new_duration:.2f}s")
            else:
                new_duration = phase.duration

            new_phases.append(traci.trafficlight.Phase(new_duration, phase.state))

        updated_logic = traci.trafficlight.Logic(logic.programID, 0, 0, new_phases)
        traci.trafficlight.setProgramLogic(tl_id, updated_logic)
        traci.trafficlight.setPhase(tl_id, 0)
       #   print(f"✅ Updated traffic light '{tl_id}' program.\n")

    previous_phase[tl_id] = current_phase


# === Main Simulation Loop ===
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    for tl in traci.trafficlight.getIDList():
        adjust_traffic_lights(tl)

traci.close()