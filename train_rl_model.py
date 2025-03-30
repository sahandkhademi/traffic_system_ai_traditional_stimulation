#!/usr/bin/env python3
"""
Reinforcement Learning Traffic Control Training Script
=====================================================

This script provides a command-line interface for training and evaluating 
the deep reinforcement learning model for traffic control.
"""

import os
import sys
import argparse
from datetime import datetime
import torch

from rl_integration import train_rl_model, evaluate_rl_model, setup_directories


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate reinforcement learning models for traffic control."
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "evaluate", "both"], 
        default="both",
        help="Operation mode: train, evaluate, or both"
    )
    
    parser.add_argument(
        "--n_intersections", 
        type=int, 
        default=5,
        help="Number of intersections to simulate"
    )
    
    parser.add_argument(
        "--n_days", 
        type=int, 
        default=30,
        help="Number of days to simulate per episode (training) or total (evaluation)"
    )
    
    parser.add_argument(
        "--n_episodes", 
        type=int, 
        default=10,
        help="Number of episodes for training"
    )
    
    parser.add_argument(
        "--cuda", 
        action="store_true",
        help="Use CUDA for training if available"
    )
    
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="rl_logs",
        help="Directory to store logs"
    )
    
    return parser.parse_args()


def print_system_info():
    """Print system information."""
    print("\n=== System Information ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print("=========================\n")


def setup_logging(log_dir):
    """Setup logging directory and redirect stdout/stderr."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # Redirect stdout and stderr to log file
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout
    
    print(f"Logging to: {log_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return log_file


def main():
    """Main function to run training and evaluation."""
    args = parse_arguments()
    
    # Setup logging
    log_file = setup_logging(args.log_dir)
    
    # Print system information
    print_system_info()
    
    # Use CUDA if requested and available
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for training.")
    else:
        device = torch.device("cpu")
        print("Using CPU for training.")
    
    # Create directories
    setup_directories()
    
    # Training
    if args.mode in ["train", "both"]:
        print("\n===== Starting Reinforcement Learning Training =====")
        print(f"Parameters:")
        print(f"  Intersections: {args.n_intersections}")
        print(f"  Days per episode: {args.n_days}")
        print(f"  Episodes: {args.n_episodes}")
        
        # Train model
        training_stats = train_rl_model(
            n_intersections=args.n_intersections,
            n_days=args.n_days,
            n_episodes=args.n_episodes
        )
        
        print("\n===== Training Complete =====")
        print("✅ Model trained successfully")
    
    # Evaluation
    if args.mode in ["evaluate", "both"]:
        print("\n===== Starting Reinforcement Learning Evaluation =====")
        print(f"Parameters:")
        print(f"  Intersections: {args.n_intersections}")
        print(f"  Days: {args.n_days}")
        
        # Evaluate model
        results, eval_metrics = evaluate_rl_model(
            n_intersections=args.n_intersections,
            n_days=args.n_days
        )
        
        # Print key metrics
        print("\nKey Evaluation Metrics:")
        for metric, improvement in eval_metrics['improvements'].items():
            print(f"  {metric.replace('_', ' ').title()}: {improvement:.2f}% improvement")
        
        print("\n===== Evaluation Complete =====")
        print("✅ Model evaluated successfully")
    
    # Close log file
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.close()
    
    # Restore stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
    print(f"Training and evaluation complete. Log saved to {log_file}")


if __name__ == "__main__":
    main() 