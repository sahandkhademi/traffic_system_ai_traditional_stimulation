"""
Deep Reinforcement Learning Agent for Traffic Control
=====================================================

This module implements a Deep Q-Network (DQN) for reinforcement learning-based
traffic light control. It includes:

1. Experience replay buffer to store and sample past experiences
2. Target network for stable learning
3. Prioritized experience replay for more efficient learning
4. Double DQN to reduce overestimation bias
5. Advanced state representation for traffic system
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import json
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class Experience:
    """Represents a single experience tuple (s, a, r, s', done)"""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    priority: float = 1.0


class ReplayBuffer:
    """Experience replay buffer with prioritized sampling"""
    
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment  # For annealing beta
        self.epsilon = 1e-6  # Small value to prevent zero priorities
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority on first insertion"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            priority=max_priority
        )
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample experiences based on priorities"""
        if len(self.buffer) < batch_size:
            return [], [], []
        
        # Anneal beta over time to reduce bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize to max weight
        weights = torch.FloatTensor(weights)
        
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


class TrafficDQN(nn.Module):
    """Deep Q-Network for traffic control"""
    
    def __init__(self, state_dim=12, hidden_dim=128, action_dim=5):
        """
        Initialize DQN.
        
        Args:
            state_dim: Dimension of state representation
            hidden_dim: Size of hidden layers
            action_dim: Number of possible actions (e.g., different green time durations)
        """
        super(TrafficDQN, self).__init__()
        
        # State encoding network - using LayerNorm instead of BatchNorm for better performance with batch size=1
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Replace BatchNorm with LayerNorm
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Replace BatchNorm with LayerNorm
        )
        
        # Value stream - estimates state value
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream - estimates advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Apply weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action
        """
        # Encode state
        features = self.state_encoder(x)
        
        # Dueling architecture: split into value and advantage streams
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class TrafficRLAgent:
    """Reinforcement learning agent for traffic control"""
    
    def __init__(
        self,
        state_dim=12,
        action_dim=5,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=10,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize agent parameters.
        
        Args:
            state_dim: State dimension
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
            target_update_freq: How often to update target network
            batch_size: Batch size for learning
            device: Device to run model on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device
        self.update_count = 0
        
        # Define action space (different green time durations)
        # Each action corresponds to different green light durations (in seconds)
        self.action_values = [20, 30, 45, 60, 90]  # 5 possible actions
        
        # Create policy and target networks
        self.policy_net = TrafficDQN(state_dim, 128, action_dim).to(device)
        self.target_net = TrafficDQN(state_dim, 128, action_dim).to(device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Setup optimizer (Adam with weight decay for regularization)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=50000)
        
        # Learning statistics
        self.training_stats = {
            'episode_rewards': [],
            'losses': [],
            'avg_q_values': [],
            'epsilons': []
        }
        
        # Load previously trained model if available
        self.model_path = 'traffic_rl_model.pth'
        self.stats_path = 'traffic_rl_stats.json'
        self.load_model()
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            
        Returns:
            action index and corresponding green time value
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action
            action_idx = random.randrange(self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Return action index and actual green time value
        return action_idx, self.action_values[action_idx]
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.add(
            torch.FloatTensor(state),
            action,
            reward,
            torch.FloatTensor(next_state),
            done
        )
    
    def update_model(self):
        """Update model weights using batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough experiences
        
        # Sample from replay buffer
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        if not experiences:
            return 0.0
        
        # Extract batches
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32).to(self.device)
        weights = weights.to(self.device)
        
        # Compute current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use policy network to select actions and target network to evaluate them
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Evaluate Q-values using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            # Calculate target Q values
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Calculate TD errors for prioritized replay
        td_errors = (q_values - target_q_values).detach().cpu().numpy()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors.flatten())
        
        # Calculate weighted loss (to account for prioritized sampling)
        loss = (weights * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Store training statistics
        self.training_stats['losses'].append(loss.item())
        with torch.no_grad():
            self.training_stats['avg_q_values'].append(self.policy_net(states).mean().item())
        self.training_stats['epsilons'].append(self.epsilon)
        
        return loss.item()
    
    def save_model(self):
        """Save model weights and training statistics"""
        torch.save(self.policy_net.state_dict(), self.model_path)
        with open(self.stats_path, 'w') as f:
            json.dump(self.training_stats, f)
    
    def load_model(self):
        """Load saved model if available"""
        try:
            if os.path.exists(self.model_path):
                self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
                if os.path.exists(self.stats_path):
                    with open(self.stats_path, 'r') as f:
                        self.training_stats = json.load(f)
                print(f"Loaded saved model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def prepare_state(self, intersection_state):
        """
        Convert intersection state dictionary to model input tensor.
        
        Args:
            intersection_state: Dictionary containing intersection state information
            
        Returns:
            Normalized state tensor
        """
        # Extract relevant features
        features = [
            intersection_state['vehicle_count'] / 50.0,  # Normalize by expected max
            intersection_state['wait_time'] / 100.0,     # Normalize by expected max
            intersection_state['hour'] / 24.0,           # Time of day (0-23)
            1.0 if intersection_state.get('is_peak_hour', False) else 0.0,
            intersection_state.get('emergency_vehicles', 0) / 3.0,
            intersection_state.get('pedestrian_count', 0) / 20.0,
            intersection_state.get('bicycle_count', 0) / 10.0,
            
            # Convert categorical features to numeric
            0.33 if intersection_state.get('congestion_level') == 'low' else
            0.67 if intersection_state.get('congestion_level') == 'medium' else 1.0,
            
            0.5 if intersection_state.get('road_type') == 'suburban' else 1.0,
            
            intersection_state.get('current_phase_duration', 0) / 120.0,
            intersection_state.get('queue_length', 0) / 30.0,
            intersection_state.get('avg_speed', 0) / 15.0
        ]
        
        return np.array(features, dtype=np.float32)
    
    def plot_training_stats(self, save_path='rl_training_progress.png'):
        """
        Plot training statistics.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.training_stats['losses']:
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Smooth values for better visualization
        def smooth(data, window_size=10):
            if len(data) < window_size:
                return data
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window_size // 2)
                end = min(len(data), i + window_size // 2 + 1)
                smoothed.append(sum(data[start:end]) / (end - start))
            return smoothed
        
        # Plot losses
        smoothed_losses = smooth(self.training_stats['losses'])
        axes[0].plot(smoothed_losses)
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True)
        
        # Plot average Q-values
        smoothed_q_values = smooth(self.training_stats['avg_q_values'])
        axes[1].plot(smoothed_q_values)
        axes[1].set_ylabel('Avg Q-value')
        axes[1].set_title('Average Q-values')
        axes[1].grid(True)
        
        # Plot epsilon value
        axes[2].plot(self.training_stats['epsilons'])
        axes[2].set_ylabel('Epsilon')
        axes[2].set_xlabel('Training step')
        axes[2].set_title('Exploration Rate (Epsilon)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def test_agent():
    """Simple function to test agent creation and basic functionality"""
    agent = TrafficRLAgent(state_dim=12, action_dim=5)
    
    # Create random state
    state = np.random.rand(12)
    
    # Test action selection
    action_idx, green_time = agent.select_action(state)
    print(f"Selected action: {action_idx}, green time: {green_time} seconds")
    
    # Test model saving
    agent.save_model()
    print("Agent test completed successfully!")


if __name__ == "__main__":
    test_agent() 