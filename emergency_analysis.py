import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Find the most recent results directory
result_dirs = sorted(glob.glob("results/*"), key=os.path.getmtime, reverse=True)
if not result_dirs:
    print("No results directories found")
    exit(1)

latest_dir = result_dirs[0]
print(f"Analyzing emergency vehicle data from: {latest_dir}")

# Load the data
csv_path = os.path.join(latest_dir, "traffic_simulation_data.csv")
if not os.path.exists(csv_path):
    print(f"No CSV file found in {latest_dir}")
    exit(1)

df = pd.read_csv(csv_path)

# Create a detailed emergency vehicle analysis
print("\n====== EMERGENCY VEHICLE DETAILED ANALYSIS ======")

# 1. Filter emergency vehicle data
emergency_df = df[df['emergency_vehicle_count'] > 0].copy()
print(f"Total emergency vehicle incidents: {len(emergency_df)}")
print(f"  - Traditional system: {len(emergency_df[emergency_df['control_type'] == 'traditional'])}")
print(f"  - AI system: {len(emergency_df[emergency_df['control_type'] == 'ai'])}")

# 2. Calculate statistical significance
from scipy import stats

try:
    trad_wait = df[df['control_type'] == 'traditional']['emergency_wait_time']
    ai_wait = df[df['control_type'] == 'ai']['emergency_wait_time']
    
    # Mann-Whitney U test for statistical significance
    u_stat, p_value = stats.mannwhitneyu(
        trad_wait.replace(0, np.nan).dropna(),
        ai_wait.replace(0, np.nan).dropna()
    )
    
    print(f"\nStatistical significance of emergency wait time difference:")
    print(f"  - Mann-Whitney U: {u_stat}")
    print(f"  - p-value: {p_value:.6f}")
    print(f"  - Significant: {'Yes' if p_value < 0.05 else 'No'}")
except Exception as e:
    print(f"Could not perform statistical test: {e}")

# 3. Hourly analysis
if len(emergency_df) > 0:
    print("\n--- EMERGENCY WAIT TIME BY HOUR ---")
    hourly_emergency = emergency_df.groupby(['control_type', 'hour'])['emergency_wait_time'].mean().reset_index()
    
    # Print peak hour emergency wait times
    peak_hours = [7, 8, 9, 16, 17, 18, 19]
    peak_emergency = hourly_emergency[hourly_emergency['hour'].isin(peak_hours)]
    
    if len(peak_emergency) > 0:
        print("Peak Hours (7-9, 16-19):")
        for control_type in peak_emergency['control_type'].unique():
            mean_wait = peak_emergency[peak_emergency['control_type'] == control_type]['emergency_wait_time'].mean()
            print(f"  - {control_type.capitalize()}: {mean_wait:.2f} seconds")
    
    # Print off-peak emergency wait times
    offpeak_emergency = hourly_emergency[~hourly_emergency['hour'].isin(peak_hours)]
    
    if len(offpeak_emergency) > 0:
        print("\nOff-Peak Hours:")
        for control_type in offpeak_emergency['control_type'].unique():
            mean_wait = offpeak_emergency[offpeak_emergency['control_type'] == control_type]['emergency_wait_time'].mean()
            print(f"  - {control_type.capitalize()}: {mean_wait:.2f} seconds")

# 4. Preemption analysis
if 'emergency_preemption_count' in df.columns:
    print("\n--- EMERGENCY PREEMPTION ANALYSIS ---")
    
    ai_emergency = emergency_df[emergency_df['control_type'] == 'ai']
    
    # Overall preemption rate
    total_emergency = ai_emergency['emergency_vehicle_count'].sum()
    total_preemptions = ai_emergency['emergency_preemption_count'].sum()
    
    preemption_rate = (total_preemptions / total_emergency * 100) if total_emergency > 0 else 0
    print(f"Overall preemption rate: {preemption_rate:.1f}%")
    
    # Preemption by hour
    hourly_preemption = ai_emergency.groupby('hour').agg({
        'emergency_vehicle_count': 'sum',
        'emergency_preemption_count': 'sum'
    }).reset_index()
    
    hourly_preemption['preemption_rate'] = (
        hourly_preemption['emergency_preemption_count'] / 
        hourly_preemption['emergency_vehicle_count'] * 100
    ).fillna(0)
    
    print("\nPreemption rate by hour (where emergency vehicles present):")
    for _, row in hourly_preemption.iterrows():
        if row['emergency_vehicle_count'] > 0:
            print(f"  Hour {row['hour']}: {row['preemption_rate']:.1f}%")
    
    # Effect on wait times
    with_preemption = ai_emergency[ai_emergency['emergency_preemption_count'] > 0]
    without_preemption = ai_emergency[ai_emergency['emergency_preemption_count'] == 0]
    
    print("\nEmergency wait time comparison (AI system):")
    if len(with_preemption) > 0:
        preempt_wait = with_preemption['emergency_wait_time'].mean()
        print(f"  - With preemption: {preempt_wait:.2f} seconds")
    
    if len(without_preemption) > 0:
        no_preempt_wait = without_preemption['emergency_wait_time'].mean()
        print(f"  - Without preemption: {no_preempt_wait:.2f} seconds")
    
    if len(with_preemption) > 0 and len(without_preemption) > 0:
        improvement = ((no_preempt_wait - preempt_wait) / no_preempt_wait * 100) if no_preempt_wait > 0 else 0
        print(f"  - Improvement from preemption: {improvement:.1f}%")

# 5. Road type and congestion analysis
print("\n--- EMERGENCY RESPONSE BY INTERSECTION TYPE ---")

# The 'road_type' column doesn't exist, but 'intersection_type' combines road_type and congestion_level
intersection_emergency = emergency_df.groupby(['control_type', 'intersection_type'])['emergency_wait_time'].agg(['mean', 'std']).reset_index()

for _, row in intersection_emergency.iterrows():
    std_value = row['std'] if not np.isnan(row['std']) else 0
    print(f"{row['control_type'].capitalize()} - {row['intersection_type']}: {row['mean']:.2f} ± {std_value:.2f} seconds")

print("\n====== END OF EMERGENCY ANALYSIS ======") 