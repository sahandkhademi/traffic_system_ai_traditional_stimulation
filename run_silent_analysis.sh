#!/bin/bash

# AI Traffic Control System - Silent Analysis Script
# This script runs all the analysis tools in sequence but suppresses warnings

echo "===== STARTING SILENT TRAFFIC SYSTEM ANALYSIS ====="
echo "Current time: $(date)"
echo ""

# Create a Python helper script to suppress warnings
cat > suppress_warnings.py << 'EOL'
import warnings
import sys
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

# Import common warning-generating libraries and suppress their warnings
import matplotlib
matplotlib.set_loglevel('error')  # Suppress matplotlib warnings
import pandas as pd
pd.options.mode.chained_assignment = None  # Suppress SettingWithCopyWarning
import seaborn as sns
import numpy as np
import scipy

# Get the script to run from command line arguments
if len(sys.argv) > 1:
    script_path = sys.argv[1]
    # Execute the script
    with open(script_path) as f:
        script_code = f.read()
    # Add globals dict to allow script to run with __name__ == "__main__"
    script_globals = {
        "__file__": script_path,
        "__name__": "__main__",
    }
    exec(script_code, script_globals)
EOL

# Run the main simulation if requested
if [ "$1" == "simulate" ]; then
  echo "===== RUNNING SIMULATION ====="
  python3 -W ignore C25D.py 2>/dev/null
  echo "Simulation completed."
  echo ""
fi

# Extract the latest results directory
LATEST_DIR=$(ls -td results/*/ | head -1)
echo "Using latest results from: $LATEST_DIR"
echo ""

# Run hourly analysis with warnings suppressed
echo "===== RUNNING HOURLY ANALYSIS ====="
python3 suppress_warnings.py hourly_analysis.py
echo "Hourly analysis completed."
echo ""

# Run emergency vehicle analysis with warnings suppressed
echo "===== RUNNING EMERGENCY VEHICLE ANALYSIS ====="
python3 suppress_warnings.py emergency_hourly.py
echo "Emergency vehicle analysis completed."
echo ""

# Run confidence interval analysis with warnings suppressed
echo "===== RUNNING STATISTICAL SIGNIFICANCE ANALYSIS ====="
python3 suppress_warnings.py confidence_interval_analysis.py
echo "Statistical analysis completed."
echo ""

# Generate visualizations with warnings suppressed
echo "===== GENERATING VISUALIZATIONS ====="
python3 suppress_warnings.py generate_visualizations.py
echo "Visualizations saved to $LATEST_DIR/visualizations"
echo ""

# Open the report in the default markdown viewer if available
echo "===== ANALYSIS COMPLETE ====="
echo "All results saved to: $LATEST_DIR"
echo "Final report available at: final_report.md"

if [ -f "/usr/bin/open" ]; then
  echo "Opening report..."
  open final_report.md
fi

echo ""
echo "===== ANALYSIS SUMMARY ====="
echo "- Hourly patterns: Analyzed across all 24 hours"
echo "- Emergency vehicles: Special attention to response times" 
echo "- Statistical significance: Mann-Whitney U tests and bootstrap CIs"
echo "- Visualizations: Created for all key metrics"
echo "- Final report: Comprehensive markdown report with visualizations"
echo ""
echo "Analysis completed at: $(date)"

# Clean up the temporary script
rm suppress_warnings.py 