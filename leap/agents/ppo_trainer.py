"""PPO trainer implementation for LEAP agents."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional
import numpy as np
import logging


class PPOTrainer:
    """Proximal Policy Optimization trainer."""
    
    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_function_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs: int = 4,
        batch_size: int = 64,
        device: torch.device = None
    ):
        self.policy = policy
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_function_coef = value_function_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        
        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Episode buffer
        self.episode_buffer = []
        
        self.logger = logging.getLogger(__name__)
    
    def add_episode_data(self, episode_data: Dict[str, torch.Tensor]):
        """Add episode data to buffer."""
        self.episode_buffer.append(episode_data)
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update using collected episodes."""
        
        if len(self.episode_buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        
        # Concatenate all episode data
        all_states = torch.cat([ep['states'] for ep in self.episode_buffer], dim=0)
        all_actions = torch.cat([ep['actions'] for ep in self.episode_buffer], dim=0)
        all_log_probs = torch.cat([ep['log_probs'] for ep in self.episode_buffer], dim=0)
        all_advantages = torch.cat([ep['advantages'] for ep in self.episode_buffer], dim=0)
        all_returns = torch.cat([ep['returns'] for ep in self.episode_buffer], dim=0)
        all_values = torch.cat([ep['values'] for ep in self.episode_buffer], dim=0)
        
        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            all_states, all_actions, all_log_probs, 
            all_advantages, all_returns, all_values
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # PPO epochs
        for epoch in range(self.epochs):
            for batch in dataloader:
                states, actions, old_log_probs, advantages, returns, old_values = batch
                
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    states, actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = old_values + torch.clamp(
                    new_values.squeeze() - old_values,
                    -self.clip_range,
                    self.clip_range
                )
                value_loss1 = (new_values.squeeze() - returns) ** 2
                value_loss2 = (value_pred_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.value_function_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Clear buffer
        self.episode_buffer = []
        
        # Return training statistics
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(policy_losses) + 
                         self.value_function_coef * np.mean(value_losses) +
                         self.entropy_coef * np.mean(entropy_losses)
        }
    
    def clear_buffer(self):
        """Clear episode buffer."""
        self.episode_buffer = []
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.episode_buffer)

