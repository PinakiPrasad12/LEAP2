"""Routing Agent implementation with RL-based routing adaptation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from ..config import RoutingConfig
from .ppo_trainer import PPOTrainer


class RoutingPolicy(nn.Module):
    """Policy network for dynamic routing adaptation."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_pruned_experts: int,
        top_k: int = 2,
        temperature: float = 1.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_pruned_experts = num_pruned_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_pruned_experts)
        )
        
        # Value network for RL training
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through routing network."""
        # hidden_states: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = hidden_states.shape
        
        # Flatten for processing
        flat_hidden = hidden_states.view(-1, self.hidden_dim)
        
        # Get routing scores
        router_logits = self.router(flat_hidden)  # [batch_size * seq_len, num_experts]
        router_logits = router_logits / self.temperature
        
        # Get value estimates
        values = self.value_head(flat_hidden)  # [batch_size * seq_len, 1]
        
        # Reshape back
        router_logits = router_logits.view(batch_size, seq_len, self.num_pruned_experts)
        values = values.view(batch_size, seq_len, 1)
        
        return router_logits, values
    
    def get_routing_weights(
        self, 
        hidden_states: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get routing weights for experts."""
        router_logits, values = self.forward(hidden_states)
        
        if training:
            # Use Gumbel-TopK for differentiable top-k sampling
            routing_weights, selected_experts = self._gumbel_topk(router_logits, self.top_k)
        else:
            # Deterministic top-k for inference
            routing_weights, selected_experts = self._deterministic_topk(router_logits, self.top_k)
        
        return routing_weights, selected_experts, values
    
    def _gumbel_topk(
        self, 
        logits: torch.Tensor, 
        k: int, 
        tau: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gumbel-TopK sampling for differentiable expert selection."""
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy_logits = (logits + gumbel_noise) / tau
        
        # Get top-k
        top_k_values, top_k_indices = torch.topk(noisy_logits, k, dim=-1)
        
        # Create routing weights using softmax
        routing_weights = F.softmax(top_k_values, dim=-1)
        
        return routing_weights, top_k_indices
    
    def _deterministic_topk(
        self, 
        logits: torch.Tensor, 
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic top-k selection."""
        top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
        routing_weights = F.softmax(top_k_values, dim=-1)
        return routing_weights, top_k_indices


class RoutingAgent:
    """RL-based routing agent for dynamic expert selection."""
    
    def __init__(
        self,
        hidden_dim: int,
        pruned_expert_indices: List[int],
        config: RoutingConfig,
        device: torch.device = torch.device("cpu")
    ):
        self.hidden_dim = hidden_dim
        self.pruned_expert_indices = pruned_expert_indices
        self.num_pruned_experts = len(pruned_expert_indices)
        self.config = config
        self.device = device
        
        # Create mapping from pruned indices to dense indices
        self.expert_index_map = {idx: i for i, idx in enumerate(pruned_expert_indices)}
        
        # Initialize routing policy
        self.policy = RoutingPolicy(
            hidden_dim=hidden_dim,
            num_pruned_experts=self.num_pruned_experts,
            top_k=config.top_k,
            temperature=config.temperature
        ).to(device)
        
        # Initialize PPO trainer for policy updates
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            learning_rate=config.router_lr,
            clip_range=config.clip_range,
            epochs=config.ppo_epochs
        )
        
        # Episode tracking for RL training
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
        self.logger = logging.getLogger(__name__)
    
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
        return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Route tokens to appropriate experts."""
        
        routing_weights, selected_experts, values = self.policy.get_routing_weights(
            hidden_states, training=training
        )
        
        # Store for RL training if in training mode
        if training:
            self.episode_states.append(hidden_states.detach())
            self.episode_values.append(values.detach())
            
            # Compute log probabilities for selected experts
            router_logits, _ = self.policy.forward(hidden_states)
            log_probs = F.log_softmax(router_logits, dim=-1)
            
            # Get log probabilities of selected actions
            selected_log_probs = torch.gather(
                log_probs, -1, selected_experts
            ).sum(dim=-1)  # Sum over top-k selections
            
            self.episode_log_probs.append(selected_log_probs.detach())
            self.episode_actions.append(selected_experts.detach())
        
        routing_info = None
        if return_routing_info:
            routing_info = {
                'routing_weights': routing_weights,
                'selected_experts': selected_experts,
                'values': values,
                'load_balancing_loss': self._compute_load_balancing_loss(routing_weights)
            }
        
        return routing_weights, routing_info
    
    def _compute_load_balancing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        # Compute expert utilization
        expert_counts = routing_weights.sum(dim=(0, 1))  # Sum over batch and sequence
        total_tokens = routing_weights.shape[0] * routing_weights.shape[1]
        
        # Compute load balancing loss (encourage uniform distribution)
        uniform_distribution = total_tokens / self.num_pruned_experts
        load_balancing_loss = torch.sum(
            (expert_counts - uniform_distribution) ** 2
        ) / self.num_pruned_experts
        
        return load_balancing_loss
    
    def compute_routing_reward(
        self,
        routing_weights: torch.Tensor,
        task_loss: torch.Tensor,
        latency: float,
        baseline_loss: float,
        baseline_latency: float
    ) -> torch.Tensor:
        """Compute reward for routing decisions."""
        
        # Performance reward (lower task loss is better)
        perf_reward = (baseline_loss - task_loss) / baseline_loss
        perf_reward = torch.clamp(perf_reward, -1.0, 1.0)
        
        # Efficiency reward (lower latency is better)
        efficiency_reward = (baseline_latency - latency) / baseline_latency
        efficiency_reward = torch.clamp(torch.tensor(efficiency_reward), -1.0, 1.0)
        
        # Load balancing penalty
        load_penalty = -self._compute_load_balancing_loss(routing_weights) * 0.01
        
        # Combined reward
        total_reward = perf_reward + 0.1 * efficiency_reward + load_penalty
        
        return total_reward
    
    def store_reward(self, reward: torch.Tensor):
        """Store reward for current step."""
        self.episode_rewards.append(reward.detach())
    
    def finish_episode(self) -> Dict[str, float]:
        """Finish episode and prepare data for PPO update."""
        
        if len(self.episode_rewards) == 0:
            return {"episode_length": 0, "total_reward": 0.0}
        
        # Convert to tensors
        states = torch.cat([s.flatten(0, 1) for s in self.episode_states], dim=0)
        actions = torch.cat([a.flatten(0, 1) for a in self.episode_actions], dim=0)
        rewards = torch.cat(self.episode_rewards)
        values = torch.cat([v.flatten(0, 1) for v in self.episode_values], dim=0).squeeze(-1)
        log_probs = torch.cat([lp.flatten() for lp in self.episode_log_probs])
        
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
        
        self.ppo_trainer.add_episode_data(episode_data)
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
        episode_stats = {
            "episode_length": len(rewards),
            "total_reward": rewards.sum().item(),
            "mean_reward": rewards.mean().item(),
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
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_ * last_gae
        
        return advantages
    
    def update_policy(self) -> Dict[str, float]:
        """Update routing policy using PPO."""
        return self.ppo_trainer.update()
    
    def active_learning_selection(
        self,
        hidden_states: torch.Tensor,
        num_samples: int
    ) -> List[int]:
        """Select samples for active learning based on routing uncertainty."""
        
        with torch.no_grad():
            router_logits, _ = self.policy.forward(hidden_states)
            
            # Compute entropy as uncertainty measure
            probs = F.softmax(router_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            
            # Average entropy across sequence length
            avg_entropy = entropy.mean(dim=1)  # [batch_size]
            
            # Select samples with highest uncertainty
            _, selected_indices = torch.topk(avg_entropy, num_samples)
        
        return selected_indices.tolist()
    
    def save_checkpoint(self, path: str):
        """Save routing agent checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'pruned_expert_indices': self.pruned_expert_indices,
            'config': self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved routing agent checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load routing agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.pruned_expert_indices = checkpoint['pruned_expert_indices']
        self.logger.info(f"Loaded routing agent checkpoint from {path}")
    
    def set_temperature(self, temperature: float):
        """Set routing temperature for exploration control."""
        self.policy.temperature = temperature
        self.config.temperature = temperature

