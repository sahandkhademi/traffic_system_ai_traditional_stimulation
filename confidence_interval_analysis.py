import pandas as pd
import numpy as np
import os
import glob
import scipy.stats as stats
from scipy.stats import bootstrap

# Find the most recent results directory
result_dirs = sorted(glob.glob("results/*"), key=os.path.getmtime, reverse=True)
if not result_dirs:
    print("No results directories found")
    exit(1)

latest_dir = result_dirs[0]
print(f"Analyzing statistical significance from: {latest_dir}")

# Load the data
csv_path = os.path.join(latest_dir, "traffic_simulation_data.csv")
if not os.path.exists(csv_path):
    print(f"No CSV file found in {latest_dir}")
    exit(1)

df = pd.read_csv(csv_path)

print("\n====== STATISTICAL SIGNIFICANCE ANALYSIS ======")

# Function to calculate confidence intervals using bootstrap
def calculate_bootstrap_ci(data, confidence=0.95):
    if len(data) < 2:
        return (np.nan, np.nan)
    
    try:
        # Use scipy's bootstrap function
        data_array = np.array(data).reshape(-1, 1)
        bootstrap_result = bootstrap((data_array,), np.mean, confidence_level=confidence, 
                                    random_state=42, n_resamples=2000)
        
        return (bootstrap_result.confidence_interval.low[0], 
                bootstrap_result.confidence_interval.high[0])
    except Exception as e:
        print(f"Bootstrap error: {e}")
        # Fallback to normal approximation if bootstrap fails
        mean = np.mean(data)
        se = stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return (mean - h, mean + h)

# Statistical tests for key metrics
metrics = ['wait_time', 'emergency_wait_time', 'fuel_consumption', 'emissions']
print("\n====== MANN-WHITNEY U TEST RESULTS ======")
print(f"{'Metric':<20} {'U-statistic':<15} {'p-value':<15} {'Significant?':<15}")
print(f"{'-'*20:<20} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15}")

for metric in metrics:
    trad_data = df[df['control_type'] == 'traditional'][metric]
    ai_data = df[df['control_type'] == 'ai'][metric]
    
    # Mann-Whitney U Test
    u_stat, p_value = stats.mannwhitneyu(trad_data, ai_data, alternative='greater')
    is_significant = "Yes" if p_value < 0.05 else "No"
    
    print(f"{metric:<20} {u_stat:<15.1f} {p_value:<15.6f} {is_significant:<15}")

# Confidence intervals for wait time differences
print("\n====== BOOTSTRAP CONFIDENCE INTERVALS (95%) ======")
print(f"{'Metric':<20} {'Mean Difference':<20} {'Lower Bound':<15} {'Upper Bound':<15}")
print(f"{'-'*20:<20} {'-'*20:<20} {'-'*15:<15} {'-'*15:<15}")

for metric in metrics:
    trad_data = df[df['control_type'] == 'traditional'][metric]
    ai_data = df[df['control_type'] == 'ai'][metric]
    
    # Calculate differences for bootstrap
    differences = []
    for i in range(min(len(trad_data), len(ai_data))):
        differences.append(trad_data.iloc[i] - ai_data.iloc[i])
    
    mean_diff = np.mean(differences)
    lower, upper = calculate_bootstrap_ci(differences)
    
    print(f"{metric:<20} {mean_diff:<20.2f} {lower:<15.2f} {upper:<15.2f}")

# Hourly confidence intervals for wait times
print("\n====== HOURLY WAIT TIME CONFIDENCE INTERVALS ======")
print(f"{'Hour':<5} {'Traditional (95% CI)':<30} {'AI (95% CI)':<30} {'Significant?':<15}")
print(f"{'-'*5:<5} {'-'*30:<30} {'-'*30:<30} {'-'*15:<15}")

for hour in sorted(df['hour'].unique()):
    trad_hour = df[(df['control_type'] == 'traditional') & (df['hour'] == hour)]['wait_time']
    ai_hour = df[(df['control_type'] == 'ai') & (df['hour'] == hour)]['wait_time']
    
    if len(trad_hour) > 1 and len(ai_hour) > 1:
        trad_mean = trad_hour.mean()
        ai_mean = ai_hour.mean()
        
        trad_lower, trad_upper = calculate_bootstrap_ci(trad_hour)
        ai_lower, ai_upper = calculate_bootstrap_ci(ai_hour)
        
        # Test for significance
        _, p_value = stats.mannwhitneyu(trad_hour, ai_hour, alternative='greater')
        is_significant = "Yes" if p_value < 0.05 else "No"
        
        # Mark peak hours with asterisk
        time_mark = "*" if hour in [7, 8, 9, 16, 17, 18, 19] else " "
        
        print(f"{hour:<4}{time_mark} {trad_mean:.2f} [{trad_lower:.2f}-{trad_upper:.2f}]{'':<5} {ai_mean:.2f} [{ai_lower:.2f}-{ai_upper:.2f}]{'':<5} {is_significant:<15}")

# Day-to-day variability confidence intervals
print("\n====== DAY-TO-DAY IMPROVEMENT CONFIDENCE INTERVALS ======")
improvements_by_day = []

for day in sorted(df['day'].unique()):
    trad_day = df[(df['control_type'] == 'traditional') & (df['day'] == day)]['wait_time']
    ai_day = df[(df['control_type'] == 'ai') & (df['day'] == day)]['wait_time']
    
    if len(trad_day) > 0 and len(ai_day) > 0:
        trad_mean = trad_day.mean()
        ai_mean = ai_day.mean()
        
        improvement = ((trad_mean - ai_mean) / trad_mean) * 100
        improvements_by_day.append(improvement)

if improvements_by_day:
    mean_improvement = np.mean(improvements_by_day)
    lower, upper = calculate_bootstrap_ci(improvements_by_day)
    
    print(f"Overall improvement: {mean_improvement:.2f}% [{lower:.2f}%-{upper:.2f}%]")

# Effect sizes
print("\n====== EFFECT SIZE ANALYSIS ======")
print(f"{'Metric':<20} {'Cohen\'s d':<15} {'Effect Size':<15}")
print(f"{'-'*20:<20} {'-'*15:<15} {'-'*15:<15}")

for metric in metrics:
    trad_data = df[df['control_type'] == 'traditional'][metric]
    ai_data = df[df['control_type'] == 'ai'][metric]
    
    # Calculate Cohen's d
    trad_mean = trad_data.mean()
    ai_mean = ai_data.mean()
    trad_std = trad_data.std()
    ai_std = ai_data.std()
    
    # Pooled standard deviation
    n1, n2 = len(trad_data), len(ai_data)
    pooled_std = np.sqrt(((n1 - 1) * trad_std**2 + (n2 - 1) * ai_std**2) / (n1 + n2 - 2))
    
    cohens_d = (trad_mean - ai_mean) / pooled_std
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        interpretation = "Negligible"
    elif abs(cohens_d) < 0.5:
        interpretation = "Small"
    elif abs(cohens_d) < 0.8:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    
    print(f"{metric:<20} {cohens_d:<15.2f} {interpretation:<15}")

print("\nNote: * indicates peak hours (7-9 AM and 4-7 PM)")
print("Statistical significance is determined at p < 0.05 level")
print("All confidence intervals are calculated at 95% confidence level") 