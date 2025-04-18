# # # # # # #program of trafiic light setting based on the vehicle counts of lane 

# # # # # # # import os
# # # # # # # import sys
# # # # # # # import traci

# # # # # # # # Initialize SUMO
# # # # # # # if "SUMO_HOME" in os.environ:
# # # # # # #     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
# # # # # # #     sys.path.append(tools)
# # # # # # # else:
# # # # # # #     sys.exit("Please declare environment variable 'SUMO_HOME'")

# # # # # # # # Define SUMO configuration
# # # # # # # sumo_cmd = ["sumo-gui", "-c", "routes.sumocfg", "--step-length", "1"]
# # # # # # # traci.start(sumo_cmd)

# # # # # # # # Store the last phase to prevent duplicate updates
# # # # # # # previous_phase = {}

# # # # # # # def adjust_traffic_lights(tl):
# # # # # # #     global previous_phase

# # # # # # #     # Get current phase and total number of phases
# # # # # # #     current_phase = traci.trafficlight.getPhase(tl)
# # # # # # #     num_phases = len(traci.trafficlight.getAllProgramLogics(tl)[0].phases)

# # # # # # #     # Check if a **full cycle has completed**
# # # # # # #     if tl in previous_phase and previous_phase[tl] > current_phase:
# # # # # # #         logic = traci.trafficlight.getAllProgramLogics(tl)[0]
# # # # # # #         phases = logic.phases
# # # # # # #         controlled_lanes = traci.trafficlight.getControlledLanes(tl)

# # # # # # #         print(logic)
# # # # # # #         print("\n")
# # # # # # #         print(phases)
# # # # # # #         print("\n")
# # # # # # #         print(controlled_lanes)
# # # # # # #         print("\n ------------------")

# # # # # # #         # **Step 1: Remove Duplicate Lanes**
# # # # # # #         unique_lanes = list(dict.fromkeys(controlled_lanes))  # Keeps order, removes duplicates

# # # # # # #         # Get vehicle counts per lane
# # # # # # #         lane_vehicle_counts = {lane: traci.lane.getLastStepVehicleNumber(lane) for lane in unique_lanes}
# # # # # # #         print("lane vehicle counts ", lane_vehicle_counts)
# # # # # # #         total_vehicles = sum(lane_vehicle_counts.values())
# # # # # # #         print("total vehicles ", total_vehicles)

# # # # # # #         # Base timings
# # # # # # #         min_green_time = 10
# # # # # # #         total_cycle_time = 100
# # # # # # #         remaining_time = total_cycle_time - (min_green_time * num_phases)
# # # # # # #         print("remaining time ", remaining_time)

# # # # # # #         # Allocate extra time based on vehicle count
# # # # # # #         extra_time_per_lane = {lane: 0 for lane in unique_lanes}
# # # # # # #         if total_vehicles > 0:
# # # # # # #             for lane, count in lane_vehicle_counts.items():
# # # # # # #                 print("lane ", lane, "count ", count)
# # # # # # #                 extra_time_per_lane[lane] = (count / total_vehicles) * remaining_time
# # # # # # #                 print(f"extra time per lane {lane} is {extra_time_per_lane[lane]}")
        
# # # # # # #         # **Step 2: Create New Phase Logic**
# # # # # # #         updated_phases = []
# # # # # # #         for i, phase in enumerate(phases):
# # # # # # #             if "g" in phase.state:  # ✅ Fix: Match lowercase 'g' in phase states
# # # # # # #                 lane = unique_lanes[i] if i < len(unique_lanes) else None
# # # # # # #                 new_duration = min_green_time + extra_time_per_lane.get(lane, 0) if lane else phase.duration
# # # # # # #             else:
# # # # # # #                 new_duration = phase.duration  # Keep red/yellow phases unchanged

# # # # # # #             updated_phases.append(traci.trafficlight.Phase(new_duration, phase.state))

# # # # # # #         # **Step 3: Apply Updated Logic**
# # # # # # #         new_logic = traci.trafficlight.Logic(logic.programID, 0, 0, updated_phases)
# # # # # # #         traci.trafficlight.setProgramLogic(tl, new_logic)  # ✅ Correct way to apply changes
# # # # # # #         traci.trafficlight.setPhase(tl, 0)  # ✅ Immediately reset to Phase 0

# # # # # # #         print(f"Updated timings for {tl}")

# # # # # # #     # Store phase for next iteration
# # # # # # #     previous_phase[tl] = current_phase


# # # # # # # # Simulation loop
# # # # # # # while traci.simulation.getMinExpectedNumber() > 0:
# # # # # # #     traci.simulationStep()

# # # # # # #     # Check all traffic lights
# # # # # # #     for tl in traci.trafficlight.getIDList():
# # # # # # #         adjust_traffic_lights(tl)

# # # # # # # # Close SUMO
# # # # # # # traci.close()

  

# # # # # #time claculated infront of traffic light

# # # # # # import os
# # # # # # import sys
# # # # # # import traci

# # # # # # # Initialize SUMO
# # # # # # if "SUMO_HOME" in os.environ:
# # # # # #     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
# # # # # #     sys.path.append(tools)
# # # # # # else:
# # # # # #     sys.exit("Please declare environment variable 'SUMO_HOME'")

# # # # # # # Define SUMO configuration
# # # # # # sumo_cmd = ["sumo-gui", "-c", "routes.sumocfg", "--step-length", "1"]
# # # # # # traci.start(sumo_cmd)

# # # # # # # Store the last phase to prevent duplicate updates
# # # # # # previous_phase = {}

# # # # # # def adjust_traffic_lights(tl):
# # # # # #     global previous_phase

# # # # # #     # Get current phase and total number of phases
# # # # # #     current_phase = traci.trafficlight.getPhase(tl)
# # # # # #     num_phases = len(traci.trafficlight.getAllProgramLogics(tl)[0].phases)

# # # # # #     # Check if a **full cycle has completed**
# # # # # #     if tl in previous_phase and previous_phase[tl] > current_phase:
# # # # # #         logic = traci.trafficlight.getAllProgramLogics(tl)[0]
# # # # # #         phases = logic.phases
# # # # # #         controlled_lanes = traci.trafficlight.getControlledLanes(tl)

# # # # # #         print(logic)
# # # # # #         print("\n")
# # # # # #         print(phases)
# # # # # #         print("\n")
# # # # # #         print(controlled_lanes)
# # # # # #         print("\n ------------------")

# # # # # #         # **Step 1: Remove Duplicate Lanes**
# # # # # #         unique_lanes = list(dict.fromkeys(controlled_lanes))  # Keeps order, removes duplicates

# # # # # #         # Only count vehicles close to the junction
# # # # # #         distance_threshold = 150 # meters near the stop line
# # # # # #         lane_vehicle_counts = {}

# # # # # #         for lane in unique_lanes:
# # # # # #             vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
# # # # # #             count_near = 0
# # # # # #             lane_length = traci.lane.getLength(lane)

# # # # # #             for veh_id in vehicle_ids:
# # # # # #                 pos = traci.vehicle.getLanePosition(veh_id)
# # # # # #                 distance_to_tls = lane_length - pos
# # # # # #                 if distance_to_tls <= distance_threshold:
# # # # # #                     count_near += 1

# # # # # #             lane_vehicle_counts[lane] = count_near

# # # # # #         print("lane vehicle counts ", lane_vehicle_counts)
# # # # # #         total_vehicles = sum(lane_vehicle_counts.values())
# # # # # #         print("total vehicles ", total_vehicles)

# # # # # #         # Base timings
# # # # # #         min_green_time = 10
# # # # # #         total_cycle_time = 100
# # # # # #         remaining_time = total_cycle_time - (min_green_time * num_phases)
# # # # # #         print("remaining time ", remaining_time)

# # # # # #         # Allocate extra time based on vehicle count
# # # # # #         extra_time_per_lane = {lane: 0 for lane in unique_lanes}
# # # # # #         if total_vehicles > 0:
# # # # # #             for lane, count in lane_vehicle_counts.items():
# # # # # #                 print("lane ", lane, "count ", count)
# # # # # #                 extra_time_per_lane[lane] = (count / total_vehicles) * remaining_time
# # # # # #                 print(f"extra time per lane {lane} is {extra_time_per_lane[lane]}")

# # # # # #         # **Step 2: Create New Phase Logic**
# # # # # #         updated_phases = []
# # # # # #         for i, phase in enumerate(phases):
# # # # # #             if "g" in phase.state:  # ✅ lowercase 'g'
# # # # # #                 lane = unique_lanes[i] if i < len(unique_lanes) else None
# # # # # #                 new_duration = min_green_time + extra_time_per_lane.get(lane, 0) if lane else phase.duration
# # # # # #             else:
# # # # # #                 new_duration = phase.duration  # Keep red/yellow phases unchanged

# # # # # #             updated_phases.append(traci.trafficlight.Phase(new_duration, phase.state))

# # # # # #         # **Step 3: Apply Updated Logic**
# # # # # #         new_logic = traci.trafficlight.Logic(logic.programID, 0, 0, updated_phases)
# # # # # #         traci.trafficlight.setProgramLogic(tl, new_logic)
# # # # # #         traci.trafficlight.setPhase(tl, 0)  # Reset phase index

# # # # # #         print(f"Updated timings for {tl}")

# # # # # #     # Store phase for next iteration
# # # # # #     previous_phase[tl] = current_phase

# # # # # # # Simulation loop
# # # # # # while traci.simulation.getMinExpectedNumber() > 0:
# # # # # #     traci.simulationStep()

# # # # # #     # Check all traffic lights
# # # # # #     for tl in traci.trafficlight.getIDList():
# # # # # #         adjust_traffic_lights(tl)

# # # # # # # Close SUMO
# # # # # # traci.close()


# # # # flow


# # # # # <?xml version="1.0" encoding="UTF-8"?>

# # # # # <!-- generated on 2025-04-02 22:40:42 by Eclipse SUMO netedit Version 1.22.0
# # # # # -->

# # # # # <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
# # # # #     <!-- Vehicles, persons and containers (sorted by depart) -->
# # # # #     <flow id="f_0" begin="0.00" fromJunction="1914" toJunction="1" end="100.00" vehsPerHour="1000"/>
# # # # #     <flow id="f_1" begin="0.00" fromJunction="130" toJunction="1993" end="100.00" vehsPerHour="500"/>
# # # # # </routes>


# # # # import os
# # # # import sys
# # # # import traci
# # # # import pandas as pd
# # # # from csv import DictWriter

# # # # # Prepare CSV file
# # # # output_file = "traffic_training_data.csv"
# # # # write_header = not os.path.exists(output_file)  # Only write header if file is new

# # # # # Set up SUMO environment
# # # # if "SUMO_HOME" in os.environ:
# # # #     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
# # # #     sys.path.append(tools)
# # # # else:
# # # #     sys.exit("Declare 'SUMO_HOME'!")

# # # # sumo_cmd = ["sumo-gui", "-c", "copy6.sumocfg", "--step-length", "1"]
# # # # traci.start(sumo_cmd)

# # # # # Open file in append mode
# # # # with open(output_file, mode='a', newline='') as csvfile:
# # # #     fieldnames = ["time", "lane_id", "density", "speed_exit", "vehicle_count", "congested"]
# # # #     writer = DictWriter(csvfile, fieldnames=fieldnames)

# # # #     if write_header:
# # # #         writer.writeheader()

# # # #     # Run the SUMO simulation
# # # #     while traci.simulation.getMinExpectedNumber() > 0:
# # # #         traci.simulationStep()

# # # #         for lane_id in traci.lane.getIDList():
# # # #             if not lane_id.startswith(":"):  # Skip internal lanes
# # # #                 lane_length = traci.lane.getLength(lane_id)
# # # #                 vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

# # # #                 speeds_near_exit = []
# # # #                 for veh_id in vehicle_ids:
# # # #                     pos = traci.vehicle.getLanePosition(veh_id)
# # # #                     if lane_length - pos <= 20:  # Last 20 meters
# # # #                         speed = traci.vehicle.getSpeed(veh_id)
# # # #                         speeds_near_exit.append(speed)

# # # #                 exit_speed = sum(speeds_near_exit) / len(speeds_near_exit) if speeds_near_exit else 0
# # # #                 density = traci.lane.getLastStepOccupancy(lane_id)
# # # #                 vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)

# # # #                 print(f"[{lane_id}] Exit Speed: {exit_speed:.2f}, Density: {density:.2f}, Vehicles: {vehicle_count}")

# # # #                 writer.writerow({
# # # #                     "time": traci.simulation.getTime(),
# # # #                     "lane_id": lane_id,
# # # #                     "density": density,
# # # #                     "speed_exit": exit_speed,
# # # #                     "vehicle_count": vehicle_count,
# # # #                     "congested": 1 if True and density > 0.5 else 0
# # # #                 })

# # # # traci.close()
# # # # print("✅ CSV saved incrementally during simulation.")




# # # import os
# # # import sys
# # # import traci
# # # import math

# # # # Initialize SUMO
# # # if "SUMO_HOME" in os.environ:
# # #     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
# # #     sys.path.append(tools)
# # # else:
# # #     sys.exit("Please declare environment variable 'SUMO_HOME'")

# # # # Define SUMO configuration
# # # sumo_cmd = ["sumo-gui", "-c", "routes.sumocfg", "--step-length", "1"]
# # # traci.start(sumo_cmd)

# # # # Store the last phase to prevent duplicate updates
# # # previous_phase = {}

# # # # Dummy set of congested lanes for now (replace with real model later)
# # # congested_lanes = set(["lane1", "lane2"])

# # # def adjust_traffic_lights(tl):
# # #     global previous_phase

# # #     current_phase = traci.trafficlight.getPhase(tl)
# # #     num_phases = len(traci.trafficlight.getAllProgramLogics(tl)[0].phases)

# # #     if tl in previous_phase and previous_phase[tl] > current_phase:
# # #         logic = traci.trafficlight.getAllProgramLogics(tl)[0]
# # #         phases = logic.phases
# # #         controlled_lanes = traci.trafficlight.getControlledLanes(tl)

# # #         print(logic)
# # #         print("\n")
# # #         print(phases)
# # #         print("\n")
# # #         print(controlled_lanes)
# # #         print("\n ------------------")

# # #         unique_lanes = list(dict.fromkeys(controlled_lanes))
# # #         current_state = traci.trafficlight.getRedYellowGreenState(tl)

# # #         lane_phase_map = {lane: current_state[i] for i, lane in enumerate(unique_lanes)}

# # #         for lane, signal in lane_phase_map.items():
# # #             if signal.lower() != 'r':
# # #                 continue  # only consider red signal lanes

# # #             vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
# # #             lane_length = traci.lane.getLength(lane)

# # #             for veh_id in vehicle_ids:
# # #                 pos = traci.vehicle.getLanePosition(veh_id)
# # #                 distance_to_tls = lane_length - pos

# # #                 if distance_to_tls <= 20:  # Only vehicles near junction
# # #                     try:
# # #                         route = traci.vehicle.getRoute(veh_id)
# # #                         current_index = route.index(lane) if lane in route else 0
# # #                         remaining_route = route[current_index:]
# # #                     except ValueError:
# # #                         remaining_route = []

# # #                     # You should replace this with actual congestion prediction from your model
# # #                     will_be_congested = any(l in congested_lanes for l in remaining_route)

# # #                     if will_be_congested:
# # #                         print(f"🚧 Vehicle {veh_id} on {lane} rerouted due to predicted congestion")
# # #                         traci.vehicle.rerouteTraveltime(veh_id)

# # #         distance_threshold = math.inf
# # #         lane_vehicle_counts = {}

# # #         for lane in unique_lanes:
# # #             vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
# # #             count_near = 0
# # #             lane_length = traci.lane.getLength(lane)

# # #             for veh_id in vehicle_ids:
# # #                 pos = traci.vehicle.getLanePosition(veh_id)
# # #                 distance_to_tls = lane_length - pos
# # #                 print(veh_id, "    ", distance_to_tls)
# # #                 if distance_to_tls <= distance_threshold:
# # #                     count_near += 1

# # #             lane_vehicle_counts[lane] = count_near

# # #         print("lane vehicle counts ", lane_vehicle_counts)
# # #         total_vehicles = sum(lane_vehicle_counts.values())
# # #         print("total vehicles ", total_vehicles)

# # #         min_green_time = 5
# # #         total_cycle_time = 100
# # #         remaining_time = total_cycle_time - (min_green_time * num_phases)
# # #         print("remaining time ", remaining_time)

# # #         extra_time_per_lane = {lane: 0 for lane in unique_lanes}
# # #         if total_vehicles > 0:
# # #             for lane, count in lane_vehicle_counts.items():
# # #                 print("lane ", lane, "count ", count)
# # #                 extra_time_per_lane[lane] = (count / total_vehicles) * remaining_time
# # #                 print(f"extra time per lane {lane} is {extra_time_per_lane[lane]}")

# # #         flow_rate = 0.3  # vehicles per second

# # #         updated_phases = []
# # #         for i, phase in enumerate(phases):
# # #             if "g" in phase.state:
# # #                 lane = unique_lanes[i] if i < len(unique_lanes) else None
# # #                 count = lane_vehicle_counts.get(lane, 0)
# # #                 extra = extra_time_per_lane.get(lane, 0)

# # #                 expected_time = count / flow_rate if flow_rate > 0 else min_green_time
# # #                 if expected_time < min_green_time + extra:
# # #                     actual_time = max(min_green_time, expected_time)
# # #                 else:
# # #                     actual_time = min_green_time + extra

# # #                 print(f"Final green time for lane {lane} is {actual_time}")
# # #                 new_duration = actual_time
# # #             else:
# # #                 new_duration = phase.duration

# # #             updated_phases.append(traci.trafficlight.Phase(new_duration, phase.state))

# # #         new_logic = traci.trafficlight.Logic(logic.programID, 0, 0, updated_phases)
# # #         traci.trafficlight.setProgramLogic(tl, new_logic)
# # #         traci.trafficlight.setPhase(tl, 0)

# # #         print(f"Updated timings for {tl}")

# # #     previous_phase[tl] = current_phase

# # # # Simulation loop
# # # while traci.simulation.getMinExpectedNumber() > 0:
# # #     traci.simulationStep()
# # #     for tl in traci.trafficlight.getIDList():
# # #         adjust_traffic_lights(tl)

# # # traci.close()





















# # import os
# # import sys
# # import traci
# # import pandas as pd
# # from csv import DictWriter
# # from collections import defaultdict, deque

# # # Prepare CSV
# # output_file = "traffic_training_data.csv"
# # write_header = not os.path.exists(output_file)

# # # SUMO setup
# # if "SUMO_HOME" in os.environ:
# #     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
# #     sys.path.append(tools)
# # else:
# #     sys.exit("Declare 'SUMO_HOME'!")

# # sumo_cmd = ["sumo-gui", "-c", "copy6.sumocfg", "--step-length", "1"]
# # traci.start(sumo_cmd)

# # # Buffer to store past lane data
# # buffer = defaultdict(deque)  # lane_id -> deque of (time, density, speed_exit, vehicle_count)

# # # Start simulation
# # with open(output_file, mode='a', newline='') as csvfile:
# #     fieldnames = ["time", "lane_id", "density", "speed_exit", "vehicle_count", "congested"]
# #     writer = DictWriter(csvfile, fieldnames=fieldnames)
# #     if write_header:
# #         writer.writeheader()
# #         csvfile.flush()  # Ensure header is saved immediately

# #     while traci.simulation.getMinExpectedNumber() > 0:
# #         current_time = traci.simulation.getTime()
# #         traci.simulationStep()

# #         for lane_id in traci.lane.getIDList():
# #             if lane_id.startswith(":"):
# #                 continue  # skip internal lanes

# #             lane_length = traci.lane.getLength(lane_id)
# #             vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)

# #             speeds_near_exit = [
# #                 traci.vehicle.getSpeed(veh_id)
# #                 for veh_id in vehicle_ids
# #                 if lane_length - traci.vehicle.getLanePosition(veh_id) <= 20
# #             ]
# #             speed_exit = sum(speeds_near_exit) / len(speeds_near_exit) if speeds_near_exit else 0
# #             density = traci.lane.getLastStepOccupancy(lane_id)
# #             vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)

# #             # Store current data in buffer
# #             buffer[lane_id].append((current_time, density, speed_exit, vehicle_count))

# #             # Check if we have old data from 30 seconds ago
# #             while buffer[lane_id] and buffer[lane_id][0][0] <= current_time - 30:
# #                 old_time, old_density, old_speed, old_count = buffer[lane_id].popleft()
                
# #                 # Define "congestion" based on current values (30s later)
# #                 congested = 1 if density > 0.5 else 0

# #                 # Write labeled old record
# #                 writer.writerow({
# #                     "time": old_time,
# #                     "lane_id": lane_id,
# #                     "density": old_density,
# #                     "speed_exit": old_speed,
# #                     "vehicle_count": old_count,
# #                     "congested": congested
# #                 })
# #                 csvfile.flush()  # ✅ Write immediately

# # traci.close()
# # print("✅ Labeled CSV created with 30s future congestion evaluation and live saving.")

# <!DOCTYPE html>
# <html>
# <head>
#   <title>Digital Junction Map</title>
#   <meta charset="utf-8" />
#   <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
#   <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
#   <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
# </head>
# <body>
#   <div class="header">
#     <div class="junction-name">{{ junction }}</div>
#     <div class="time">Time: {{ time }}</div>
#   </div>

#   <div class="container">
#     <div id="map"></div>

#     <div class="info-panel">
#       <h2>Alternate Routes</h2>
#       {% for route in diversions %}
#         <div class="route">
#           <strong>{{ route.destination }}</strong><br>
#           via {{ route.via }}
#         </div>
#       {% endfor %}
#     </div>
#   </div>

#   <script>
#     const imgWidth = 768;
#     const imgHeight = 795;
#     const bounds = [[0, 0], [imgHeight, imgWidth]];

#     const map = L.map('map', {
#       crs: L.CRS.Simple,
#       minZoom: -2,
#       maxZoom: 4,
#       maxBounds: bounds,
#       maxBoundsViscosity: 1.0
#     });

#     const image = L.imageOverlay('/static/image.png', bounds).addTo(map);
#     map.fitBounds(bounds);

#     const routes = [
#       { coords: [[142.25, 440.25], [142.75, 556.75]], congested: true },
#       { coords: [[160.875, 428.625], [520,429.5]], congested: false },
#       { coords: [[150, 412.625], [149.5, 159.75]], congested: true },
#       { coords: [[157, 147.5], [334.5, 148.5]], congested: false },
#       { coords: [[347.5, 158], [526.5, 413]], congested: true },
#       { coords: [[553.625, 417.875], [732.625,162.875]], congested: false },
#       { coords: [[163, 569.75], [524.5, 569]], congested: true },
#       { coords: [[129.5, 569.5], [0.5, 614]], congested: false },
#       { coords: [[139, 583.5], [138.5, 764]], congested: true },
#       { coords: [[536.5, 580], [538,704]], congested: false },
#       { coords: [[558, 568.5], [792.5, 569.5]], congested: true },
#       { coords: [[364.25, 148.75], [716.5, 148.5]], congested: false },
#       { coords: [[125, 429.5], [3, 429]], congested: true },
#     ];

#     routes.forEach(route => {
#       const color = route.congested ? 'red' : 'green';
#       L.polyline(route.coords, {
#         color: color,
#         weight: 8,
#         opacity: 0.8
#       }).addTo(map);
#     });

#     map.on('click', e => console.log(e.latlng));
#   </script>
# </body>
# </html>


# body, html {
#   margin: 0;
#   padding: 0;
#   font-family: "Segoe UI", sans-serif;
#   background-color: #f4f4f4;
#   height: 100%;
# }

# .header {
#   background-color: #2c3e50;
#   color: white;
#   padding: 1em 2em;
#   display: flex;
#   justify-content: space-between;
#   align-items: center;
# }

# .junction-name {
#   font-size: 1.5em;
# }

# .time {
#   font-size: 1.1em;
# }

# .container {
#   display: flex;
#   height: calc(100vh - 80px); /* adjust for header height */
# }

# #map {
#   flex: 2;
#   height: 100%;
# }

# .info-panel {
#   flex: 1;
#   background-color: white;
#   padding: 20px;
#   overflow-y: auto;
#   box-shadow: -2px 0 6px rgba(0, 0, 0, 0.1);
# }

# .info-panel h2 {
#   margin-top: 0;
#   color: #333;
# }

# .route {
#   background: #ecf0f1;
#   padding: 10px;
#   margin-bottom: 10px;
#   border-left: 5px solid #2980b9;
#   border-radius: 5px;
# }


# from flask import Flask, render_template
# from datetime import datetime

# app = Flask(__name__)

# @app.route('/')
# def index():
#     junction = "Main Junction"
#     current_time = datetime.now().strftime("%H:%M:%S")
#     diversions = [
#         {"destination": "Airport", "via": "East Blvd → Central Rd"},
#         {"destination": "Mall", "via": "Hill Rd → Park Lane"},
#         {"destination": "Tech Park", "via": "Loop St → North Ave"},
#     ]
#     return render_template("index.html", junction=junction, time=current_time, diversions=diversions)

# if __name__ == "__main__":
#     app.run(debug=True)
