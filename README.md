# AI Traffic Control System Simulation

A comprehensive simulation framework for comparing traditional fixed-time traffic light systems with AI-driven adaptive systems across multiple intersections over multiple days.

## Features

- Simulates both traditional fixed-time and AI-driven adaptive traffic control systems
- Models various road types (suburban, highway) and congestion levels (low, medium, high)
- Includes time-of-day traffic variations and peak hours
- Handles emergency vehicles with preemptive signal control in AI mode
- Measures and analyzes key performance metrics:
  - Vehicle wait times
  - Emergency vehicle response times
  - Fuel consumption
  - CO₂ emissions
- Generates detailed visualizations and statistical analyses

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

## Usage

Run the main simulation:

```bash
python C25D.py
```

For detailed analysis:

```bash
python hourly_analysis.py
python emergency_analysis.py
python statistical_analysis.py
```

## License

[MIT License](LICENSE)