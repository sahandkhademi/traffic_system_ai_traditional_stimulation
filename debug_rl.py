#!/usr/bin/env python3
"""
Debug script for reinforcement learning implementation
"""

import os
import sys
import torch
import numpy as np
from traffic_rl_agent import TrafficRLAgent

def test_rl_agent():
    print("Testing RL agent functionality...")
    
    # Create agent
    agent = TrafficRLAgent(state_dim=12, action_dim=5)
    print(f"Agent created with device: {agent.device}")
    
    # Test state creation
    test_state = np.random.rand(12)
    print(f"Created test state: {test_state}")
    
    # Test action selection
    action_idx, green_time = agent.select_action(test_state)
    print(f"Selected action: {action_idx}, green time: {green_time} seconds")
    
    # Test model saving/loading
    agent.save_model()
    print(f"Model saved to: {agent.model_path}")
    
    # Create new agent to test loading
    new_agent = TrafficRLAgent(state_dim=12, action_dim=5)
    print(f"New agent load successful: {os.path.exists(new_agent.model_path)}")
    
    # Test experience storage
    next_state = np.random.rand(12)
    agent.store_experience(test_state, action_idx, 10.0, next_state, False)
    print(f"Experience stored. Memory size: {len(agent.memory)}")
    
    # Test model update
    if len(agent.memory) > 0:
        print("Attempting model update...")
        loss = agent.update_model()
        print(f"Model update loss: {loss}")
    
    print("RL agent test completed successfully!")
    return True

def test_c25d_import():
    print("\nTesting C25D import...")
    try:
        # Try importing the Intersection class
        from C25D import Intersection
        print("Successfully imported Intersection from C25D")
        
        # Create a test intersection
        intersection = Intersection(
            intersection_id="test_id",
            control_type='traditional',
            road_type="highway",
            congestion_level="medium"
        )
        print(f"Created test intersection with ID: {intersection.intersection_id}")
        
        # Test get_intersection_state
        state = intersection.get_intersection_state()
        print(f"Intersection state keys: {list(state.keys())}")
        
        # Test simulate_hour
        try:
            results = intersection.simulate_hour(hour=12)
            print(f"Simulate hour results keys: {list(results.keys())}")
            print("C25D intersection test successful!")
            return True
        except Exception as e:
            print(f"Error simulating hour: {str(e)}")
            return False
        
    except ImportError as e:
        print(f"Error importing from C25D: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error testing C25D: {str(e)}")
        return False

def test_integration():
    print("\nTesting RL integration...")
    try:
        from rl_integration import RLIntersectionAdapter, setup_directories
        from C25D import Intersection
        
        # Setup directories
        setup_directories()
        print("Directories set up successfully")
        
        # Create test intersection
        intersection = Intersection(
            intersection_id="test_rl_id",
            control_type='ai',
            road_type="highway",
            congestion_level="medium"
        )
        
        # Create adapter
        adapter = RLIntersectionAdapter(intersection)
        print(f"Created RL adapter for intersection: {intersection.intersection_id}")
        
        # Reset adapter
        state = adapter.reset()
        print(f"Reset adapter. State shape: {state.shape}")
        
        # Test step
        try:
            results = adapter.step(hour=12)
            print(f"Step results keys: {list(results.keys())}")
            print(f"Reward from step: {results.get('rl_reward', 'N/A')}")
            
            # End episode
            stats = adapter.end_episode()
            print(f"Episode stats: {stats}")
            
            print("RL integration test successful!")
            return True
        except Exception as e:
            print(f"Error in step or end_episode: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"Error importing RL integration: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error testing integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("-" * 50)
    
    # Test components
    agent_ok = test_rl_agent()
    c25d_ok = test_c25d_import()
    integration_ok = test_integration() if c25d_ok else False
    
    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    print(f"RL Agent:      {'✓ PASS' if agent_ok else '✗ FAIL'}")
    print(f"C25D Import:   {'✓ PASS' if c25d_ok else '✗ FAIL'}")
    print(f"RL Integration: {'✓ PASS' if integration_ok else '✗ FAIL'}")
    print("=" * 50)
    
    if agent_ok and c25d_ok and integration_ok:
        print("\nAll tests passed! The RL implementation is working correctly.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please check the output above for details.")
        sys.exit(1) 