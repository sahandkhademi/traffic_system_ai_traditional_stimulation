#!/usr/bin/env python3
"""
Short test script for evaluating the RL integration
"""

import os
import torch
import numpy as np
from traffic_rl_agent import TrafficRLAgent

def create_mock_intersection():
    """Create a mock intersection object that mimics the interface of the real one"""
    class MockIntersection:
        def __init__(self, intersection_id, control_type, road_type, congestion_level):
            self.intersection_id = intersection_id
            self.control_type = control_type
            self.road_type = road_type
            self.congestion_level = congestion_level
            self.traffic_light = MockTrafficLight()
            
        def get_intersection_state(self):
            """Return mock intersection state"""
            return {
                'vehicle_count': 25,
                'wait_time': 30.0,
                'hour': 12,
                'is_peak_hour': False,
                'emergency_vehicles': 0,
                'pedestrian_count': 5,
                'bicycle_count': 2,
                'congestion_level': self.congestion_level,
                'road_type': self.road_type,
                'current_phase_duration': 45.0,
                'queue_length': 10,
                'avg_speed': 8.0
            }
            
        def simulate_hour(self, hour):
            """Simulate an hour of traffic"""
            # Return mock results
            return {
                'wait_time': 15.0 + np.random.rand() * 10,
                'emergency_wait_time': 5.0 + np.random.rand() * 3 if np.random.rand() > 0.7 else 0,
                'fuel_consumption': 100 + np.random.rand() * 50,
                'emissions': 200 + np.random.rand() * 100,
                'intersection_id': self.intersection_id
            }
    
    class MockTrafficLight:
        def __init__(self):
            self.green_time = 30.0
            
    return MockIntersection

def test_rl_adapter():
    """Test the RL adapter with a mock intersection"""
    from rl_integration import RLIntersectionAdapter, setup_directories
    
    # Setup directories
    setup_directories()
    print("Created necessary directories")
    
    # Create mock intersection class
    MockIntersection = create_mock_intersection()
    
    # Create mock intersection
    intersection = MockIntersection(
        intersection_id="test_rl_id",
        control_type='ai',
        road_type="highway",
        congestion_level="medium"
    )
    print(f"Created mock intersection with ID: {intersection.intersection_id}")
    
    # Create RL adapter
    adapter = RLIntersectionAdapter(intersection)
    print("Created RL adapter")
    
    # Reset adapter
    state = adapter.reset()
    print(f"Reset adapter. State shape: {state.shape}")
    
    # Run a few steps
    total_rewards = 0
    for hour in range(24):
        results = adapter.step(hour)
        reward = results.get('rl_reward', 0)
        total_rewards += reward
        wait_time = results.get('wait_time', 0)
        
        print(f"Hour {hour}: Green time={results.get('rl_green_time', 0):.1f}s, "
              f"Wait time={wait_time:.1f}s, Reward={reward:.1f}")
    
    # End episode
    stats = adapter.end_episode()
    print("\nEpisode stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return total_rewards > 0

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("-" * 50)
    
    success = test_rl_adapter()
    
    if success:
        print("\nSuccess! The RL integration is working correctly.")
    else:
        print("\nSome issues were encountered. Please check the output above for details.") 