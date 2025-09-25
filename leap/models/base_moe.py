"""Base MoE model implementation for LEAP framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging
from abc import ABC, abstractmethod

from ..config import ModelConfig


class Expert(nn.Module):
    """Individual expert network in MoE."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_fn: str = "silu",
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Feed-forward layers
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Activation function
        if activation_fn == "silu":
            self.act_fn = F.silu
        elif activation_fn == "gelu":
            self.act_fn = F.gelu
        elif activation_fn == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        # SwiGLU activation: gate * up * act_fn(up)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = self.act_fn(gate) * up
        
        if self.dropout is not None:
            intermediate = self.dropout(intermediate)
        
        output = self.down_proj(intermediate)
        return output


class Router(nn.Module):
    """Router network for expert selection."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router projection
        self.gate = nn.Linear(hidden_size, num_experts, bias=router_bias)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through router."""
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Get router logits
        router_logits = self.gate(hidden_states_flat)  # [batch_size * seq_len, num_experts]
        
        # Apply softmax to get probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        
        # Renormalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Reshape back
        routing_weights = routing_weights.view(batch_size, seq_len, self.top_k)
        selected_experts = selected_experts.view(batch_size, seq_len, self.top_k)
        
        return routing_weights, selected_experts


class MoELayer(nn.Module):
    """Mixture of Experts layer."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        activation_fn: str = "silu",
        router_aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size, activation_fn, dropout)
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = Router(hidden_size, num_experts, top_k)
        
        # For pruned routing
        self.pruned_expert_indices: Optional[List[int]] = None
        self.is_pruned = False
        
    def set_pruned_experts(self, expert_indices: List[int]):
        """Set which experts to use after pruning."""
        self.pruned_expert_indices = expert_indices
        self.is_pruned = True
        
        # Update router to only route to pruned experts
        if hasattr(self.router, 'set_pruned_experts'):
            self.router.set_pruned_experts(expert_indices)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
        selected_experts: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through MoE layer."""
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions if not provided
        if routing_weights is None or selected_experts is None:
            routing_weights, selected_experts = self.router(hidden_states)
        
        # Flatten inputs
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # Process each token
        for i in range(batch_size * seq_len):
            token_hidden = hidden_states_flat[i:i+1]  # [1, hidden_size]
            token_output = torch.zeros_like(token_hidden)
            
            # Get routing info for this token
            token_routing_weights = routing_weights.view(-1, self.top_k)[i]  # [top_k]
            token_selected_experts = selected_experts.view(-1, self.top_k)[i]  # [top_k]
            
            # Route to selected experts
            for k in range(self.top_k):
                expert_idx = token_selected_experts[k].item()
                expert_weight = token_routing_weights[k]
                
                # Skip if using pruned experts and this expert is not selected
                if self.is_pruned and expert_idx not in self.pruned_expert_indices:
                    continue
                
                # Get expert output
                expert_output = self.experts[expert_idx](token_hidden)
                token_output += expert_weight * expert_output
            
            final_hidden_states[i] = token_output
        
        # Reshape back
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
        
        # Compute auxiliary losses
        aux_losses = self._compute_auxiliary_losses(
            routing_weights.view(-1, self.top_k),
            selected_experts.view(-1, self.top_k)
        )
        
        return final_hidden_states, aux_losses
    
    def _compute_auxiliary_losses(
        self,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training."""
        
        # Load balancing loss (aux loss)
        expert_counts = torch.zeros(self.num_experts, device=routing_weights.device)
        for i in range(self.num_experts):
            expert_mask = (selected_experts == i).float()
            expert_counts[i] = expert_mask.sum()
        
        # Normalize by number of tokens
        num_tokens = routing_weights.shape[0]
        expert_probs = expert_counts / num_tokens
        
        # Compute load balancing loss
        uniform_prob = 1.0 / self.num_experts
        aux_loss = torch.sum((expert_probs - uniform_prob) ** 2)
        
        # Router z-loss (encourage sparsity)
        router_z_loss = torch.sum(routing_weights ** 2)
        
        return {
            "aux_loss": self.router_aux_loss_coef * aux_loss,
            "z_loss": self.router_z_loss_coef * router_z_loss,
            "expert_counts": expert_counts,
            "expert_probs": expert_probs
        }


class BaseMoE(nn.Module, ABC):
    """Base class for MoE models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model dimensions
        self.hidden_size = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        
        # MoE configuration
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        pass
    
    @abstractmethod
    def get_moe_layers(self) -> List[MoELayer]:
        """Get all MoE layers in the model."""
        pass
    
    def prune_experts(self, expert_indices: List[int], layer_idx: Optional[int] = None):
        """Prune experts to specified subset."""
        moe_layers = self.get_moe_layers()
        
        if layer_idx is not None:
            # Prune specific layer
            moe_layers[layer_idx].set_pruned_experts(expert_indices)
            self.logger.info(f"Pruned layer {layer_idx} to experts: {expert_indices}")
        else:
            # Prune all layers
            for i, layer in enumerate(moe_layers):
                layer.set_pruned_experts(expert_indices)
            self.logger.info(f"Pruned all layers to experts: {expert_indices}")
    
    def get_expert_utilization(self) -> Dict[str, torch.Tensor]:
        """Get expert utilization statistics."""
        moe_layers = self.get_moe_layers()
        utilization_stats = {}
        
        for i, layer in enumerate(moe_layers):
            # This would be populated during forward pass
            if hasattr(layer, 'last_expert_counts'):
                utilization_stats[f'layer_{i}'] = layer.last_expert_counts
        
        return utilization_stats
    
    def compute_flops(self, input_length: int) -> Dict[str, float]:
        """Compute FLOPs for current model configuration."""
        # Base computation for non-MoE components
        base_flops = self._compute_base_flops(input_length)
        
        # MoE computation
        moe_flops = 0
        moe_layers = self.get_moe_layers()
        
        for layer in moe_layers:
            if layer.is_pruned:
                # Only count active experts
                active_experts = len(layer.pruned_expert_indices)
            else:
                active_experts = self.top_k  # Only top-k experts are active
            
            # FLOPs per expert per token
            expert_flops = 2 * self.hidden_size * self.config.intermediate_size
            layer_flops = expert_flops * active_experts * input_length
            moe_flops += layer_flops
        
        total_flops = base_flops + moe_flops
        
        return {
            "base_flops": base_flops,
            "moe_flops": moe_flops,
            "total_flops": total_flops,
            "flops_per_token": total_flops / input_length
        }
    
    @abstractmethod
    def _compute_base_flops(self, input_length: int) -> float:
        """Compute FLOPs for non-MoE components."""
        pass
    
    def save_pruned_model(self, path: str, expert_indices: List[int]):
        """Save model with only pruned experts."""
        # Create new model config for pruned version
        pruned_config = self.config
        pruned_config.num_experts = len(expert_indices)
        
        # Save model state and pruning info
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': pruned_config,
            'pruned_expert_indices': expert_indices,
            'original_num_experts': self.num_experts
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved pruned model to {path}")
    
    @classmethod
    def load_pruned_model(cls, path: str, device: torch.device = None):
        """Load a pruned model."""
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with pruned configuration
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply pruning
        expert_indices = checkpoint['pruned_expert_indices']
        model.prune_experts(expert_indices)
        
        return model

