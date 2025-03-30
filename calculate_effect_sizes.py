import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('results/20250327_120413/traffic_simulation_data.csv')

# Calculate for each metric
metrics = ['wait_time', 'emergency_wait_time', 'fuel_consumption', 'emissions']

print("\nEffect Size Analysis (Cohen's d)")
print("-" * 80)
print(f"{'Metric':<20} {'Mean (Trad)':<12} {'Mean (AI)':<12} {'Cohen\'s d':<10} {'Effect Size':<10}")
print("-" * 80)

for metric in metrics:
    trad = df[df['control_type'] == 'traditional'][metric]
    ai = df[df['control_type'] == 'ai'][metric]
    
    # Calculate means and standard deviations
    mean_trad = trad.mean()
    mean_ai = ai.mean()
    std_trad = trad.std()
    std_ai = ai.std()
    
    # Calculate pooled standard deviation
    s_pooled = np.sqrt((std_trad**2 + std_ai**2) / 2)
    
    # Calculate Cohen's d
    d = (mean_trad - mean_ai) / s_pooled
    
    # Determine effect size
    if abs(d) >= 0.8:
        effect = "Large"
    elif abs(d) >= 0.5:
        effect = "Medium"
    else:
        effect = "Small"
    
    print(f"{metric:<20} {mean_trad:>11.2f} {mean_ai:>11.2f} {d:>10.2f} {effect:>10}")

print("\nPeak Hour Analysis (7 AM)")
print("-" * 80)
print(f"{'Metric':<20} {'Traditional':<15} {'AI':<15} {'Reduction %':<10}")
print("-" * 80)

# Get 7 AM data
peak_df = df[df['hour'] == 7]
for metric in ['emissions', 'fuel_consumption']:
    trad_peak = peak_df[peak_df['control_type'] == 'traditional'][metric].sum()
    ai_peak = peak_df[peak_df['control_type'] == 'ai'][metric].sum()
    reduction = ((trad_peak - ai_peak) / trad_peak) * 100
    
    print(f"{metric:<20} {trad_peak:>14.2f} {ai_peak:>14.2f} {reduction:>9.1f}%")

print("\nPer Vehicle Analysis")
print("-" * 80)
print(f"{'Metric':<20} {'Traditional':<15} {'AI':<15} {'Reduction %':<10}")
print("-" * 80)

# Calculate per-vehicle metrics
for metric in ['emissions', 'fuel_consumption']:
    trad_total = df[df['control_type'] == 'traditional'][metric].sum()
    ai_total = df[df['control_type'] == 'ai'][metric].sum()
    trad_vehicles = df[df['control_type'] == 'traditional']['vehicle_count'].sum()
    ai_vehicles = df[df['control_type'] == 'ai']['vehicle_count'].sum()
    
    trad_per_vehicle = trad_total / trad_vehicles
    ai_per_vehicle = ai_total / ai_vehicles
    reduction = ((trad_per_vehicle - ai_per_vehicle) / trad_per_vehicle) * 100
    
    print(f"{metric:<20} {trad_per_vehicle:>14.3f} {ai_per_vehicle:>14.3f} {reduction:>9.1f}%") 