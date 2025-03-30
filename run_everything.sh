#!/bin/bash

# AI Traffic Control System - Complete Execution Script
# This script runs the entire workflow: simulation, RL training, and all analyses

echo "===== STARTING COMPLETE AI TRAFFIC SYSTEM WORKFLOW ====="
echo "Current time: $(date)"
echo ""

# Step 1: Run the main simulation
echo "===== RUNNING MAIN SIMULATION ====="
python3 C25D.py
echo ""

# Step 2: Train and evaluate the RL model
echo "===== TRAINING AND EVALUATING RL MODEL ====="
# Reduce episodes and days for faster execution
python3 train_rl_model.py --mode both --n_intersections 2 --n_days 3 --n_episodes 3
echo "RL training and evaluation complete!"
echo ""

# Extract the latest results directory
LATEST_DIR=$(ls -td results/*/ | head -1)
echo "Using latest results from: $LATEST_DIR"
echo ""

# Step 3: Run hourly analysis
echo "===== RUNNING HOURLY ANALYSIS ====="
python3 hourly_analysis.py
echo ""

# Step 4: Run emergency vehicle analysis
echo "===== RUNNING EMERGENCY VEHICLE ANALYSIS ====="
python3 emergency_analysis.py
echo ""
python3 emergency_hourly.py
echo ""

# Step 5: Run statistical analysis
echo "===== RUNNING STATISTICAL ANALYSIS ====="
python3 statistical_analysis.py
echo ""
python3 confidence_interval_analysis.py
echo ""
python3 calculate_effect_sizes.py
echo ""

# Step 6: Generate visualizations
echo "===== GENERATING VISUALIZATIONS ====="
python3 generate_visualizations.py
echo ""

# Step 7: Final report
echo "===== ANALYSIS COMPLETE ====="
echo "All results saved to: $LATEST_DIR"
echo "Final report available at: final_report.md"

if [ -f "/usr/bin/open" ]; then
  echo "Opening report..."
  open final_report.md
fi

echo ""
echo "===== WORKFLOW SUMMARY ====="
echo "- Main simulation: Complete"
echo "- RL training & evaluation: Complete"
echo "- Hourly patterns: Analyzed across all 24 hours"
echo "- Emergency vehicles: Special attention to response times"
echo "- Statistical analysis: Complete with significance tests"
echo "- Visualizations: Created for all key metrics"
echo "- Final report: Comprehensive markdown report with visualizations"
echo ""
echo "Workflow completed at: $(date)" 