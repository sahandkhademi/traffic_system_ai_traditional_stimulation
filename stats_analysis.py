import pandas as pd
import numpy as np
import os
import glob

# Find the most recent results directory
result_dirs = sorted(glob.glob("results/*"), key=os.path.getmtime, reverse=True)
if not result_dirs:
    print("No results directories found")
    exit(1)

latest_dir = result_dirs[0]
print(f"Analyzing results from: {latest_dir}")

# Load the data
csv_path = os.path.join(latest_dir, "traffic_simulation_data.csv")
if not os.path.exists(csv_path):
    print(f"No CSV file found in {latest_dir}")
    exit(1)

df = pd.read_csv(csv_path)

# Calculate statistics by control type
stats = df.groupby("control_type").agg({
    "wait_time": ["mean", "std", "min", "max"],
    "emergency_wait_time": ["mean", "std", "min", "max"],
    "fuel_consumption": ["mean", "std", "min", "max"],
    "emissions": ["mean", "std", "min", "max"],
    "cycle_time": ["mean", "std", "min", "max"]
})

# Print statistics
print("\n====== DETAILED STATISTICS ======")
print("\n--- WAIT TIME (seconds) ---")
wait_stats = stats["wait_time"]
for control_type in wait_stats.index:
    print(f"{control_type.capitalize()}:")
    print(f"  Mean ± Std: {wait_stats.loc[control_type, 'mean']:.2f} ± {wait_stats.loc[control_type, 'std']:.2f}")
    print(f"  Range: {wait_stats.loc[control_type, 'min']:.2f} - {wait_stats.loc[control_type, 'max']:.2f}")
    
print("\n--- EMERGENCY WAIT TIME (seconds) ---")
emerg_stats = stats["emergency_wait_time"]
for control_type in emerg_stats.index:
    print(f"{control_type.capitalize()}:")
    print(f"  Mean ± Std: {emerg_stats.loc[control_type, 'mean']:.2f} ± {emerg_stats.loc[control_type, 'std']:.2f}")
    print(f"  Range: {emerg_stats.loc[control_type, 'min']:.2f} - {emerg_stats.loc[control_type, 'max']:.2f}")

print("\n--- FUEL CONSUMPTION (liters) ---")
fuel_stats = stats["fuel_consumption"]
for control_type in fuel_stats.index:
    print(f"{control_type.capitalize()}:")
    print(f"  Mean ± Std: {fuel_stats.loc[control_type, 'mean']:.4f} ± {fuel_stats.loc[control_type, 'std']:.4f}")
    print(f"  Range: {fuel_stats.loc[control_type, 'min']:.4f} - {fuel_stats.loc[control_type, 'max']:.4f}")

print("\n--- EMISSIONS (kg CO2) ---")
emissions_stats = stats["emissions"]
for control_type in emissions_stats.index:
    print(f"{control_type.capitalize()}:")
    print(f"  Mean ± Std: {emissions_stats.loc[control_type, 'mean']:.4f} ± {emissions_stats.loc[control_type, 'std']:.4f}")
    print(f"  Range: {emissions_stats.loc[control_type, 'min']:.4f} - {emissions_stats.loc[control_type, 'max']:.4f}")

print("\n--- CYCLE TIME (seconds) ---")
cycle_stats = stats["cycle_time"]
for control_type in cycle_stats.index:
    print(f"{control_type.capitalize()}:")
    print(f"  Mean ± Std: {cycle_stats.loc[control_type, 'mean']:.2f} ± {cycle_stats.loc[control_type, 'std']:.2f}")
    print(f"  Range: {cycle_stats.loc[control_type, 'min']:.2f} - {cycle_stats.loc[control_type, 'max']:.2f}")

# Define peak hours
peak_hours = [7, 8, 9, 16, 17, 18, 19]
off_peak_hours = [h for h in range(24) if h not in peak_hours]

# Calculate peak/off-peak statistics
peak_df = df[df["hour"].isin(peak_hours)]
offpeak_df = df[df["hour"].isin(off_peak_hours)]

peak_stats = peak_df.groupby("control_type").agg({
    "wait_time": ["mean", "std"],
    "emergency_wait_time": ["mean", "std"]
})

offpeak_stats = offpeak_df.groupby("control_type").agg({
    "wait_time": ["mean", "std"],
    "emergency_wait_time": ["mean", "std"]
})

print("\n====== STATISTICS BY TIME PERIOD ======")
print("\n--- PEAK HOURS WAIT TIME (7-9, 16-19) ---")
for control_type in peak_stats.index:
    stat = peak_stats["wait_time"].loc[control_type]
    print(f"{control_type.capitalize()}: {stat['mean']:.2f} ± {stat['std']:.2f}")

print("\n--- OFF-PEAK HOURS WAIT TIME ---")
for control_type in offpeak_stats.index:
    stat = offpeak_stats["wait_time"].loc[control_type]
    print(f"{control_type.capitalize()}: {stat['mean']:.2f} ± {stat['std']:.2f}")

# Emergency vehicle statistics
if "emergency_preemption_count" in df.columns:
    print("\n====== EMERGENCY VEHICLE STATISTICS ======")
    emergency_stats = df.groupby("control_type").agg({
        "emergency_vehicle_count": ["sum", "mean", "std"],
        "emergency_preemption_count": ["sum", "mean", "std"] if "emergency_preemption_count" in df.columns else ["sum", "mean"]
    })
    
    for control_type in emergency_stats.index:
        print(f"{control_type.capitalize()}:")
        veh_count = emergency_stats["emergency_vehicle_count"].loc[control_type]
        print(f"  Total emergency vehicles: {veh_count['sum']:.0f}")
        print(f"  Average per hour: {veh_count['mean']:.2f} ± {veh_count['std']:.2f}")
        
        if control_type == "ai" and "emergency_preemption_count" in df.columns:
            preempt = emergency_stats["emergency_preemption_count"].loc[control_type]
            preempt_rate = (preempt["sum"] / veh_count["sum"]) * 100 if veh_count["sum"] > 0 else 0
            print(f"  Preemptions: {preempt['sum']:.0f} ({preempt_rate:.1f}%)")
            preempt_std = preempt.get("std", 0)
            print(f"  Average preemptions per hour: {preempt['mean']:.2f} ± {preempt_std:.2f}")

# Road type and congestion level
print("\n====== WAIT TIME BY ROAD TYPE AND CONGESTION LEVEL ======")
road_congestion_stats = df.groupby(["control_type", "road_type", "congestion_level"])["wait_time"].agg(["mean", "std"])

for index, row in road_congestion_stats.iterrows():
    control, road, congestion = index
    print(f"{control.capitalize()} - {road.capitalize()} - {congestion.capitalize()}: {row['mean']:.2f} ± {row['std']:.2f}") 