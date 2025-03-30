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
print(f"Analyzing hourly patterns from: {latest_dir}")

# Load the data
csv_path = os.path.join(latest_dir, "traffic_simulation_data.csv")
if not os.path.exists(csv_path):
    print(f"No CSV file found in {latest_dir}")
    exit(1)

df = pd.read_csv(csv_path)

# Group by control type and hour
hourly_stats = df.groupby(['control_type', 'hour']).agg({
    'wait_time': ['mean', 'std', 'count'],
    'emergency_wait_time': ['mean', 'std', 'count'],
    'vehicle_count': ['sum', 'mean'],
    'fuel_consumption': ['mean', 'sum'],
    'emissions': ['mean', 'sum'],
    'cycle_time': ['mean', 'std']
}).reset_index()

# Format the hourly_stats dataframe
hours = sorted(df['hour'].unique())

print("\n====== HOURLY WAIT TIME PATTERNS ======")
print(f"{'Hour':<5} {'Traditional (s)':<20} {'AI (s)':<20} {'Difference':<15} {'Reduction %':<12}")
print(f"{'-'*5:<5} {'-'*20:<20} {'-'*20:<20} {'-'*15:<15} {'-'*12:<12}")

for hour in hours:
    try:
        trad_data = hourly_stats[(hourly_stats['hour'] == hour) & (hourly_stats['control_type'] == 'traditional')]
        ai_data = hourly_stats[(hourly_stats['hour'] == hour) & (hourly_stats['control_type'] == 'ai')]
        
        if len(trad_data) > 0 and len(ai_data) > 0:
            trad_mean = trad_data['wait_time']['mean'].values[0]
            trad_std = trad_data['wait_time']['std'].values[0]
            ai_mean = ai_data['wait_time']['mean'].values[0]
            ai_std = ai_data['wait_time']['std'].values[0]
            
            diff = trad_mean - ai_mean
            pct_reduction = (diff / trad_mean) * 100 if trad_mean > 0 else 0
            
            # Mark peak hours with asterisk
            time_mark = "*" if hour in [7, 8, 9, 16, 17, 18, 19] else " "
            
            print(f"{hour:<4}{time_mark} {trad_mean:.2f} ± {trad_std:.2f}       {ai_mean:.2f} ± {ai_std:.2f}       {diff:.2f}          {pct_reduction:.1f}%")
    except (KeyError, IndexError):
        pass

# Fuel consumption hourly patterns
print("\n====== HOURLY FUEL CONSUMPTION PATTERNS ======")
print(f"{'Hour':<5} {'Traditional (L)':<20} {'AI (L)':<20} {'Difference':<15} {'Reduction %':<12}")
print(f"{'-'*5:<5} {'-'*20:<20} {'-'*20:<20} {'-'*15:<15} {'-'*12:<12}")

for hour in hours:
    try:
        trad_data = hourly_stats[(hourly_stats['hour'] == hour) & (hourly_stats['control_type'] == 'traditional')]
        ai_data = hourly_stats[(hourly_stats['hour'] == hour) & (hourly_stats['control_type'] == 'ai')]
        
        if len(trad_data) > 0 and len(ai_data) > 0:
            trad_sum = trad_data['fuel_consumption']['sum'].values[0]
            ai_sum = ai_data['fuel_consumption']['sum'].values[0]
            
            diff = trad_sum - ai_sum
            pct_reduction = (diff / trad_sum) * 100 if trad_sum > 0 else 0
            
            # Mark peak hours with asterisk
            time_mark = "*" if hour in [7, 8, 9, 16, 17, 18, 19] else " "
            
            print(f"{hour:<4}{time_mark} {trad_sum:.2f}              {ai_sum:.2f}              {diff:.2f}          {pct_reduction:.1f}%")
    except (KeyError, IndexError):
        pass

# Emissions hourly patterns
print("\n====== HOURLY EMISSIONS PATTERNS ======")
print(f"{'Hour':<5} {'Traditional (kg)':<20} {'AI (kg)':<20} {'Difference':<15} {'Reduction %':<12}")
print(f"{'-'*5:<5} {'-'*20:<20} {'-'*20:<20} {'-'*15:<15} {'-'*12:<12}")

for hour in hours:
    try:
        trad_data = hourly_stats[(hourly_stats['hour'] == hour) & (hourly_stats['control_type'] == 'traditional')]
        ai_data = hourly_stats[(hourly_stats['hour'] == hour) & (hourly_stats['control_type'] == 'ai')]
        
        if len(trad_data) > 0 and len(ai_data) > 0:
            trad_sum = trad_data['emissions']['sum'].values[0]
            ai_sum = ai_data['emissions']['sum'].values[0]
            
            diff = trad_sum - ai_sum
            pct_reduction = (diff / trad_sum) * 100 if trad_sum > 0 else 0
            
            # Mark peak hours with asterisk
            time_mark = "*" if hour in [7, 8, 9, 16, 17, 18, 19] else " "
            
            print(f"{hour:<4}{time_mark} {trad_sum:.2f}             {ai_sum:.2f}             {diff:.2f}          {pct_reduction:.1f}%")
    except (KeyError, IndexError):
        pass

# Stats by intersection type
print("\n====== STATS BY INTERSECTION TYPE ======")
if 'intersection_type' in df.columns:
    intersection_stats = df.groupby(['control_type', 'intersection_type']).agg({
        'wait_time': ['mean', 'std'],
        'cycle_time': ['mean', 'std']
    }).reset_index()
    
    print(f"{'Type':<15} {'System':<12} {'Wait Time (s)':<20} {'Cycle Time (s)':<20}")
    print(f"{'-'*15:<15} {'-'*12:<12} {'-'*20:<20} {'-'*20:<20}")
    
    for int_type in sorted(df['intersection_type'].unique()):
        for control_type in ['traditional', 'ai']:
            try:
                data = intersection_stats[(intersection_stats['intersection_type'] == int_type) & 
                                         (intersection_stats['control_type'] == control_type)]
                if len(data) > 0:
                    wait_mean = data['wait_time']['mean'].values[0]
                    wait_std = data['wait_time']['std'].values[0]
                    cycle_mean = data['cycle_time']['mean'].values[0]
                    cycle_std = data['cycle_time']['std'].values[0]
                    
                    print(f"{int_type:<15} {control_type:<12} {wait_mean:.2f} ± {wait_std:.2f}        {cycle_mean:.2f} ± {cycle_std:.2f}")
            except (KeyError, IndexError):
                pass

# Day-to-day variability
print("\n====== DAY-TO-DAY VARIABILITY ======")
if 'day' in df.columns:
    day_stats = df.groupby(['control_type', 'day']).agg({
        'wait_time': ['mean', 'std']
    }).reset_index()
    
    print(f"{'Day':<5} {'Traditional (s)':<20} {'AI (s)':<20} {'Improvement %':<15}")
    print(f"{'-'*5:<5} {'-'*20:<20} {'-'*20:<20} {'-'*15:<15}")
    
    improvements = []
    
    for day in sorted(df['day'].unique()):
        trad_data = day_stats[(day_stats['day'] == day) & (day_stats['control_type'] == 'traditional')]
        ai_data = day_stats[(day_stats['day'] == day) & (day_stats['control_type'] == 'ai')]
        
        if len(trad_data) > 0 and len(ai_data) > 0:
            trad_mean = trad_data['wait_time']['mean'].values[0]
            trad_std = trad_data['wait_time']['std'].values[0]
            ai_mean = ai_data['wait_time']['mean'].values[0]
            ai_std = ai_data['wait_time']['std'].values[0]
            
            improvement = ((trad_mean - ai_mean) / trad_mean) * 100 if trad_mean > 0 else 0
            improvements.append(improvement)
            
            print(f"{day:<5} {trad_mean:.2f} ± {trad_std:.2f}       {ai_mean:.2f} ± {ai_std:.2f}       {improvement:.1f}%")
    
    if improvements:
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        print(f"\nOverall improvement: {avg_improvement:.1f}% ± {std_improvement:.1f}%")

# Peak vs. Off-peak analysis
print("\n====== PEAK VS. OFF-PEAK ANALYSIS ======")
df['peak_hour'] = df['hour'].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18, 19] else 0)
peak_stats = df.groupby(['control_type', 'peak_hour']).agg({
    'wait_time': ['mean', 'std', 'count'],
    'emergency_wait_time': ['mean', 'std', 'count'],
    'fuel_consumption': ['mean', 'sum'],
    'emissions': ['mean', 'sum']
}).reset_index()

print(f"{'Period':<10} {'System':<12} {'Wait Time (s)':<20} {'Fuel (L)':<15} {'Emissions (kg)':<15}")
print(f"{'-'*10:<10} {'-'*12:<12} {'-'*20:<20} {'-'*15:<15} {'-'*15:<15}")

for peak in [0, 1]:
    period = "Peak" if peak == 1 else "Off-peak"
    for control_type in ['traditional', 'ai']:
        try:
            data = peak_stats[(peak_stats['peak_hour'] == peak) & 
                             (peak_stats['control_type'] == control_type)]
            if len(data) > 0:
                wait_mean = data['wait_time']['mean'].values[0]
                wait_std = data['wait_time']['std'].values[0]
                fuel_mean = data['fuel_consumption']['mean'].values[0]
                emissions_mean = data['emissions']['mean'].values[0]
                
                print(f"{period:<10} {control_type:<12} {wait_mean:.2f} ± {wait_std:.2f}        {fuel_mean:.2f}         {emissions_mean:.2f}")
        except (KeyError, IndexError):
            pass

# Standard deviation analysis across all metrics
print("\n====== STANDARD DEVIATION ANALYSIS ======")
overall_stats = df.groupby(['control_type']).agg({
    'wait_time': ['mean', 'std', 'min', 'max'],
    'emergency_wait_time': ['mean', 'std', 'min', 'max'],
    'vehicle_count': ['mean', 'std', 'sum'],
    'fuel_consumption': ['mean', 'std', 'sum'],
    'emissions': ['mean', 'std', 'sum'],
    'cycle_time': ['mean', 'std', 'min', 'max']
}).reset_index()

print(f"{'Metric':<20} {'Traditional':<25} {'AI':<25} {'Reduction %':<12}")
print(f"{'-'*20:<20} {'-'*25:<25} {'-'*25:<25} {'-'*12:<12}")

metrics = [
    ('Wait Time (s)', 'wait_time'),
    ('Emergency Wait (s)', 'emergency_wait_time'),
    ('Fuel (L)', 'fuel_consumption'),
    ('Emissions (kg)', 'emissions'),
    ('Cycle Time (s)', 'cycle_time')
]

for metric_name, metric in metrics:
    try:
        trad_data = overall_stats[overall_stats['control_type'] == 'traditional']
        ai_data = overall_stats[overall_stats['control_type'] == 'ai']
        
        if len(trad_data) > 0 and len(ai_data) > 0:
            trad_mean = trad_data[metric]['mean'].values[0]
            trad_std = trad_data[metric]['std'].values[0]
            trad_min = trad_data[metric]['min'].values[0]
            trad_max = trad_data[metric]['max'].values[0]
            
            ai_mean = ai_data[metric]['mean'].values[0]
            ai_std = ai_data[metric]['std'].values[0]
            ai_min = ai_data[metric]['min'].values[0]
            ai_max = ai_data[metric]['max'].values[0]
            
            reduction = ((trad_mean - ai_mean) / trad_mean) * 100 if trad_mean > 0 else 0
            
            print(f"{metric_name:<20} {trad_mean:.2f} ± {trad_std:.2f} [{trad_min:.2f}-{trad_max:.2f}] {ai_mean:.2f} ± {ai_std:.2f} [{ai_min:.2f}-{ai_max:.2f}] {reduction:.1f}%")
    except (KeyError, IndexError):
        pass

print("\nNote: * indicates peak hours (7-9 AM and 4-7 PM)")
print("All statistics are calculated using the raw simulation data") 