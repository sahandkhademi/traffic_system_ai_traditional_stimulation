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
print(f"Analyzing emergency hourly patterns from: {latest_dir}")

# Load the data
csv_path = os.path.join(latest_dir, "traffic_simulation_data.csv")
if not os.path.exists(csv_path):
    print(f"No CSV file found in {latest_dir}")
    exit(1)

df = pd.read_csv(csv_path)

print("\n====== EMERGENCY WAIT TIME DETAILED ANALYSIS ======")

# Filter data to only include rows with emergency vehicles
emergency_df = df[df['emergency_vehicle_count'] > 0].copy()

if len(emergency_df) == 0:
    print("No emergency vehicle data found in the dataset.")
    exit(0)

print(f"Total emergency vehicle incidents: {len(emergency_df)}")
print(f"  - Traditional system: {len(emergency_df[emergency_df['control_type'] == 'traditional'])}")
print(f"  - AI system: {len(emergency_df[emergency_df['control_type'] == 'ai'])}")

# Group by control type and hour
emergency_hourly = emergency_df.groupby(['control_type', 'hour']).agg({
    'emergency_wait_time': ['mean', 'std', 'count'],
    'emergency_vehicle_count': 'sum'
}).reset_index()

# Format the emergency_hourly dataframe for easier printing
hours_with_emergencies = sorted(emergency_df['hour'].unique())

print("\n====== EMERGENCY WAIT TIME BY HOUR ======")
print(f"{'Hour':<5} {'System':<12} {'Wait Time (sec)':<15} {'Count':<8} {'Std Dev':<10}")
print(f"{'-'*5:<5} {'-'*12:<12} {'-'*15:<15} {'-'*8:<8} {'-'*10:<10}")

for hour in hours_with_emergencies:
    for control_type in ['traditional', 'ai']:
        data = emergency_df[(emergency_df['hour'] == hour) & (emergency_df['control_type'] == control_type)]
        if len(data) > 0:
            mean_wait = data['emergency_wait_time'].mean()
            std_wait = data['emergency_wait_time'].std()
            if np.isnan(std_wait):
                std_wait = 0.0
            count = len(data)
            
            # Mark peak hours with asterisk
            time_mark = "*" if hour in [7, 8, 9, 16, 17, 18, 19] else " "
            
            print(f"{hour:<4}{time_mark} {control_type:<12} {mean_wait:.2f}s{'':<8} {count:<8} {std_wait:.2f}")

# Compare systems
trad_emergency = emergency_df[emergency_df['control_type'] == 'traditional']
ai_emergency = emergency_df[emergency_df['control_type'] == 'ai']

print("\n====== EMERGENCY VEHICLE WAIT TIME SUMMARY ======")
if len(trad_emergency) > 0:
    trad_mean = trad_emergency['emergency_wait_time'].mean()
    trad_std = trad_emergency['emergency_wait_time'].std()
    print(f"Traditional system: {trad_mean:.2f} ± {trad_std:.2f} seconds")

if len(ai_emergency) > 0:
    ai_mean = ai_emergency['emergency_wait_time'].mean()
    ai_std = ai_emergency['emergency_wait_time'].std()
    print(f"AI system: {ai_mean:.2f} ± {ai_std:.2f} seconds")

if len(trad_emergency) > 0 and len(ai_emergency) > 0:
    improvement = ((trad_mean - ai_mean) / trad_mean * 100) if trad_mean > 0 else 0
    print(f"Improvement: {improvement:.1f}%")

# Preemption data
if 'emergency_preemption_count' in df.columns:
    print("\n====== EMERGENCY PREEMPTION STATS ======")
    ai_with_preempt = ai_emergency[ai_emergency['emergency_preemption_count'] > 0]
    ai_without_preempt = ai_emergency[ai_emergency['emergency_preemption_count'] == 0]
    
    if len(ai_with_preempt) > 0:
        with_mean = ai_with_preempt['emergency_wait_time'].mean()
        with_std = ai_with_preempt['emergency_wait_time'].std()
        print(f"AI with preemption: {with_mean:.2f} ± {with_std:.2f} seconds ({len(ai_with_preempt)} incidents)")
    
    if len(ai_without_preempt) > 0:
        without_mean = ai_without_preempt['emergency_wait_time'].mean()
        without_std = ai_without_preempt['emergency_wait_time'].std()
        print(f"AI without preemption: {without_mean:.2f} ± {without_std:.2f} seconds ({len(ai_without_preempt)} incidents)")
    
    if len(ai_with_preempt) > 0 and len(ai_without_preempt) > 0:
        pre_improvement = ((without_mean - with_mean) / without_mean * 100) if without_mean > 0 else 0
        print(f"Preemption benefit: {pre_improvement:.1f}% reduction in wait time") 