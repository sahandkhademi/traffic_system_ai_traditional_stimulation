import pandas as pd
import numpy as np
from scipy import stats

# Read the data
df = pd.read_csv('results/20250327_120413/traffic_simulation_data.csv')

# Calculate per-vehicle metrics
df['emissions_per_vehicle'] = df['emissions'] / df['vehicle_count']
df['fuel_per_vehicle'] = df['fuel_consumption'] / df['vehicle_count']

metrics = ['emissions_per_vehicle', 'fuel_per_vehicle']

print("\nStatistical Analysis")
print("-" * 80)
print(f"{'Metric':<20} {'U-statistic':<12} {'p-value':<12} {'95% CI Lower':<12} {'95% CI Upper':<12}")
print("-" * 80)

for metric in metrics:
    # Get traditional and AI data
    trad = df[df['control_type'] == 'traditional'][metric]
    ai = df[df['control_type'] == 'ai'][metric]
    
    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(trad, ai, alternative='greater')
    
    # Calculate 95% confidence interval using bootstrap
    def calc_diff(x, y):
        return np.mean(x) - np.mean(y)
    
    boot = stats.bootstrap((trad, ai), calc_diff, n_resamples=10000)
    ci_lower, ci_upper = boot.confidence_interval
    
    print(f"{metric:<20} {u_stat:>11.0f} {p_value:>11.6f} {ci_lower:>11.3f} {ci_upper:>11.3f}")

# Also calculate percentage reductions and their confidence intervals
print("\nReduction Analysis")
print("-" * 80)
print(f"{'Metric':<20} {'Reduction %':<12} {'95% CI Lower %':<15} {'95% CI Upper %':<15}")
print("-" * 80)

for metric in metrics:
    trad_mean = df[df['control_type'] == 'traditional'][metric].mean()
    ai_mean = df[df['control_type'] == 'ai'][metric].mean()
    reduction = ((trad_mean - ai_mean) / trad_mean) * 100
    
    # Bootstrap for reduction percentage CI
    def calc_reduction(x, y):
        return ((np.mean(x) - np.mean(y)) / np.mean(x)) * 100
    
    boot = stats.bootstrap((trad, ai), calc_reduction, n_resamples=10000)
    red_ci_lower, red_ci_upper = boot.confidence_interval
    
    print(f"{metric:<20} {reduction:>11.1f}% {red_ci_lower:>14.1f}% {red_ci_upper:>14.1f}%") 