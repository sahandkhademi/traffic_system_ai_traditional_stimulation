#!/bin/bash

# AI Traffic Control System - Comprehensive Analysis Script
# This script runs all the analysis tools in sequence

echo "===== STARTING COMPREHENSIVE TRAFFIC SYSTEM ANALYSIS ====="
echo "Current time: $(date)"
echo ""

# Run the main simulation if requested
if [ "$1" == "simulate" ]; then
  echo "===== RUNNING SIMULATION ====="
  python3 C25D.py
  echo ""
fi

# Extract the latest results directory
LATEST_DIR=$(ls -td results/*/ | head -1)
echo "Using latest results from: $LATEST_DIR"
echo ""

# Run hourly analysis
echo "===== RUNNING HOURLY ANALYSIS ====="
python3 hourly_analysis.py
echo ""

# Run emergency vehicle analysis
echo "===== RUNNING EMERGENCY VEHICLE ANALYSIS ====="
python3 emergency_hourly.py
echo ""

# Run confidence interval analysis
echo "===== RUNNING STATISTICAL SIGNIFICANCE ANALYSIS ====="
python3 confidence_interval_analysis.py
echo ""

# Generate visualizations
echo "===== GENERATING VISUALIZATIONS ====="
python3 generate_visualizations.py
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