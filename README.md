# AI Traffic Control System Simulation

A comprehensive simulation framework comparing traditional fixed-time traffic light systems with AI-driven adaptive systems across multiple intersections and time periods. This project demonstrates significant improvements in traffic efficiency, emergency response times, and environmental metrics through AI-based traffic management.

## Features

- **Dual System Simulation**: Models both traditional fixed-time and AI-driven adaptive traffic control systems
- **Realistic Traffic Patterns**: 
  - Various road types (suburban, highway) and congestion levels (low, medium, high)
  - Time-of-day traffic variations with peak/off-peak hours
  - Day-to-day variability for realistic modeling
- **Emergency Vehicle Handling**: Implements preemptive signal control in AI mode
- **Comprehensive Metrics**: 
  - Vehicle wait times (overall and by hour)
  - Emergency vehicle response times
  - Fuel consumption and efficiency
  - CO₂ emissions and environmental impact
- **Statistical Analysis**: Detailed statistical comparisons including effect sizes and significance tests
- **Data Visualization**: Generates charts and graphs for all key metrics

## Key Findings

- **Wait Time Reduction**: AI systems reduced average vehicle wait times by 73.4%
- **Emergency Response**: Emergency vehicle wait times decreased by 62.0%
- **Fuel Efficiency**: Fuel consumption per vehicle reduced by 41.3%
- **Environmental Impact**: CO₂ emissions reduced by 59.2%
- **Emergency Vehicle Preemption**: AI systems detected and preemptively responded to 59.8% of emergency vehicles

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- tqdm

## Project Structure

```
C25D/
├── C25D.py                   # Main simulation script
├── hourly_analysis.py        # Hourly patterns analysis
├── emergency_analysis.py     # Emergency vehicle analysis
├── statistical_analysis.py   # Statistical significance testing
├── calculate_effect_sizes.py # Effect size calculations
├── generate_visualizations.py # Data visualization generator
├── run_full_analysis.sh      # Complete analysis pipeline
├── run_silent_analysis.sh    # Analysis with warnings suppressed
├── run_silent_analysis_with_vis.sh # Analysis with visualizations
├── results/                  # Simulation results and visualizations
└── LICENSE                   # MIT License
```

## Usage Instructions

### Running the Main Simulation

```bash
# Run the main simulation
python C25D.py
```

### Running Analysis Tools

```bash
# Full analysis with detailed output
./run_full_analysis.sh

# Analysis with warnings suppressed
./run_silent_analysis.sh

# Analysis with warnings suppressed and visualization generation
./run_silent_analysis_with_vis.sh
```

### Individual Analysis Scripts

```bash
# Analyze hourly patterns in wait times and other metrics
python hourly_analysis.py

# Analyze emergency vehicle data and preemption effects
python emergency_analysis.py

# Calculate statistical significance of improvements
python statistical_analysis.py

# Calculate effect sizes for various metrics
python calculate_effect_sizes.py

# Generate all visualizations with descriptive filenames
python generate_visualizations.py
```

### Viewing Results

The simulation results are stored in timestamped directories under `results/`, with each simulation run generating:

- CSV data files with raw simulation data
- Visualization images for key metrics
- Summary reports in markdown format

## How to Extend

The simulation can be extended by:

1. Modifying `C25D.py` to adjust traffic parameters or add new metrics
2. Creating additional analysis scripts to examine specific aspects
3. Enhancing visualization options in `generate_visualizations.py`

## License

[MIT License](LICENSE)