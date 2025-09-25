"""Pruning Agent implementation using Meta-RL for expert subset selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from ..config import PruningConfig
from .ppo_trainer import PPOTrainer


@dataclass
class PruningState:
    """State representation for pruning agent."""
    
    selected_experts: torch.Tensor  # Binary mask of selected experts
    cumulative_contribution: torch.Tensor  # Cumulative validation accuracy
    budget_used: float  # Fraction of budget used
    episode_step: int  # Current step in episode
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert state to tensor representation."""
        state_vector = torch.cat([
            self.selected_experts.float(),
            self.cumulative_contribution.unsqueeze(0),
            torch.tensor([self.budget_used, self.episode_step / 128.0], device=device)
        ])
        return state_vector


class PruningPolicy(nn.Module):
    """Policy network for expert selection."""
    
    def __init__(self, num_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.num_experts = num_experts
        
        # Input: [selected_experts(128), cumulative_contrib(1), budget_used(1), step(1)]
        input_dim = num_experts + 3
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action head: probability of selecting each expert
        self.action_head = nn.Linear(hidden_dim, num_experts)
        
        # Value head: state value estimation
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network."""
        features = self.encoder(state)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value
    
    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """Get action probabilities and value, optionally evaluate specific action."""
        action_logits, value = self.forward(state)
        
        # Mask already selected experts to prevent re-selection
        selected_mask = state[:, :self.num_experts].bool()
        action_logits = action_logits.masked_fill(selected_mask, -float('inf'))
        
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value


class PruningAgent:
    """Meta-RL agent for expert subset selection."""
    
    def __init__(
        self,
        num_experts: int,
        config: PruningConfig,
        device: torch.device = torch.device("cpu")
    ):
        self.num_experts = num_experts
        self.config = config
        self.device = device
        
        # Initialize policy network
        self.policy = PruningPolicy(num_experts).to(device)
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            learning_rate=config.learning_rate,
            clip_range=config.clip_range,
            entropy_coef=config.entropy_coef,
            value_function_coef=config.value_function_coef,
            max_grad_norm=config.max_grad_norm
        )
        
        # Episode tracking
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
        self.logger = logging.getLogger(__name__)
    
    def reset_episode(self) -> PruningState:
        """Reset for new episode."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
        # Initial state: no experts selected
        initial_state = PruningState(
            selected_experts=torch.zeros(self.num_experts, device=self.device),
            cumulative_contribution=torch.tensor(0.0, device=self.device),
            budget_used=0.0,
            episode_step=0
        )
        
        return initial_state
    
    def select_action(self, state: PruningState) -> Tuple[int, float, float]:
        """Select next expert to include in subset."""
        state_tensor = state.to_tensor(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action_and_value(state_tensor)
        
        # Store for training
        self.episode_states.append(state_tensor)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_values.append(value)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_reward(
        self,
        state: PruningState,
        action: int,
        model_performance: float,
        baseline_performance: float
    ) -> float:
        """Compute reward for state-action pair."""
        
        # Performance reward: normalized improvement
        perf_reward = (model_performance - baseline_performance) / baseline_performance
        perf_reward = max(0.0, perf_reward)  # Only positive improvements
        
        # Efficiency reward: budget constraint satisfaction
        budget_penalty = 0.0
        if state.budget_used > self.config.budget_constraint:
            budget_penalty = -10.0 * (state.budget_used - self.config.budget_constraint)
        
        # Sparsity bonus: encourage using fewer experts
        sparsity_bonus = 0.1 * (1.0 - state.budget_used)
        
        # Combined reward
        reward = (
            self.config.performance_weight * perf_reward +
            self.config.efficiency_weight * sparsity_bonus +
            budget_penalty
        )
        
        return reward
    
    def update_state(
        self,
        state: PruningState,
        action: int,
        performance: float
    ) -> PruningState:
        """Update state after taking action."""
        
        new_selected = state.selected_experts.clone()
        new_selected[action] = 1.0
        
        new_budget = new_selected.sum().item() / self.num_experts
        
        new_state = PruningState(
            selected_experts=new_selected,
            cumulative_contribution=torch.tensor(performance, device=self.device),
            budget_used=new_budget,
            episode_step=state.episode_step + 1
        )
        
        return new_state
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.episode_rewards.append(reward)
    
    def finish_episode(self) -> Dict[str, float]:
        """Finish episode and compute returns."""
        
        if len(self.episode_rewards) == 0:
            return {"episode_length": 0, "total_reward": 0.0}
        
        # Convert to tensors
        states = torch.cat(self.episode_states, dim=0)
        actions = torch.stack(self.episode_actions)
        rewards = torch.tensor(self.episode_rewards, device=self.device)
        values = torch.cat(self.episode_values)
        log_probs = torch.stack(self.episode_log_probs)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values)
        returns = advantages + values
        
        # Store episode data for PPO update
        episode_data = {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns,
            'values': values
        }
        
        # Add to PPO buffer
        self.ppo_trainer.add_episode_data(episode_data)
        
        episode_stats = {
            "episode_length": len(self.episode_rewards),
            "total_reward": rewards.sum().item(),
            "mean_reward": rewards.mean().item(),
            "final_value": values[-1].item() if len(values) > 0 else 0.0
        }
        
        return episode_stats
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_ * last_gae
        
        return advantages
    
    def update_policy(self) -> Dict[str, float]:
        """Update policy using PPO."""
        return self.ppo_trainer.update()
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved pruning agent checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Loaded pruning agent checkpoint from {path}")
    
    def get_expert_subset(self, model, val_dataloader, max_experts: Optional[int] = None) -> List[int]:
        """Run full episode to get optimal expert subset."""
        
        if max_experts is None:
            max_experts = self.config.target_experts
        
        state = self.reset_episode()
        selected_experts = []
        
        # Get baseline performance (all experts)
        baseline_perf = self._evaluate_model(model, val_dataloader, list(range(self.num_experts)))
        
        for step in range(max_experts):
            # Select next expert
            action, log_prob, value = self.select_action(state)
            selected_experts.append(action)
            
            # Evaluate model with current subset
            current_perf = self._evaluate_model(model, val_dataloader, selected_experts)
            
            # Compute reward
            reward = self.compute_reward(state, action, current_perf, baseline_perf)
            self.store_reward(reward)
            
            # Update state
            state = self.update_state(state, action, current_perf)
            
            # Early stopping if budget exceeded
            if state.budget_used > self.config.budget_constraint:
                break
        
        return selected_experts
    
    def _evaluate_model(self, model, dataloader, expert_indices: List[int]) -> float:
        """Evaluate model performance with given expert subset."""
        # This would be implemented based on the specific model and task
        # For now, return a placeholder
        return np.random.random()  # Placeholder

