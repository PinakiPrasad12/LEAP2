"""Model utility functions for LEAP."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in bytes and MB."""
    
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_size = param_size + buffer_size
    
    return {
        "total_bytes": total_size,
        "total_mb": total_size / (1024**2),
        "param_bytes": param_size,
        "param_mb": param_size / (1024**2),
        "buffer_bytes": buffer_size,
        "buffer_mb": buffer_size / (1024**2)
    }


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Estimate FLOPs for model forward pass."""
    
    if device is None:
        device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Count FLOPs using hooks
    flop_counts = {}
    handles = []
    
    def flop_count_hook(module, input, output):
        """Hook to count FLOPs for different layer types."""
        
        module_name = module.__class__.__name__
        
        if isinstance(module, nn.Linear):
            # FLOPs = input_features * output_features * batch_size
            input_numel = input[0].numel()
            output_numel = output.numel() 
            flops = input_numel * module.out_features // input[0].shape[0]
            flop_counts[f"{module_name}_{id(module)}"] = flops
            
        elif isinstance(module, nn.Conv1d):
            # FLOPs = kernel_size * in_channels * out_channels * output_length * batch_size
            kernel_size = module.kernel_size[0]
            in_channels = module.in_channels
            out_channels = module.out_channels
            output_length = output.shape[-1]
            batch_size = output.shape[0]
            flops = kernel_size * in_channels * out_channels * output_length * batch_size
            flop_counts[f"{module_name}_{id(module)}"] = flops
            
        elif isinstance(module, nn.MultiheadAttention):
            # Attention FLOPs: Q*K^T + Attn*V
            seq_len = input[0].shape[1] if len(input[0].shape) > 2 else input[0].shape[0]
            hidden_dim = input[0].shape[-1]
            batch_size = input[0].shape[0] if len(input[0].shape) > 2 else 1
            
            # Q*K^T: batch_size * seq_len * seq_len * hidden_dim
            # Attn*V: batch_size * seq_len * hidden_dim * hidden_dim
            qk_flops = batch_size * seq_len * seq_len * hidden_dim
            av_flops = batch_size * seq_len * hidden_dim * hidden_dim
            flops = qk_flops + av_flops
            flop_counts[f"{module_name}_{id(module)}"] = flops
    
    # Register hooks
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.MultiheadAttention)):
            handle = module.register_forward_hook(flop_count_hook)
            handles.append(handle)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Calculate total FLOPs
    total_flops = sum(flop_counts.values())
    
    return {
        "total_flops": total_flops,
        "flops_per_layer": flop_counts,
        "gflops": total_flops / 1e9,
        "mflops": total_flops / 1e6
    }


def get_layer_info(model: nn.Module) -> List[Dict[str, Any]]:
    """Get information about model layers."""
    
    layer_info = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            info = {
                "name": name,
                "type": module.__class__.__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            
            # Add layer-specific information
            if isinstance(module, nn.Linear):
                info.update({
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None
                })
            elif isinstance(module, nn.Embedding):
                info.update({
                    "num_embeddings": module.num_embeddings,
                    "embedding_dim": module.embedding_dim,
                    "padding_idx": module.padding_idx
                })
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                info.update({
                    "normalized_shape": getattr(module, "normalized_shape", None),
                    "eps": getattr(module, "eps", None)
                })
            
            layer_info.append(info)
    
    return layer_info


def analyze_gradient_flow(model: nn.Module) -> Dict[str, Any]:
    """Analyze gradient flow through model."""
    
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "max": grad.max().item(),
                "min": grad.min().item(),
                "norm": grad.norm().item(),
                "shape": list(grad.shape)
            }
        else:
            grad_stats[name] = {
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
                "norm": 0.0,
                "shape": list(param.shape)
            }
    
    return grad_stats


def get_activation_stats(
    model: nn.Module,
    input_tensor: torch.Tensor
) -> Dict[str, Dict[str, float]]:
    """Get activation statistics for model layers."""
    
    activation_stats = {}
    
    def activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_stats[name] = {
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                    "max": output.max().item(),
                    "min": output.min().item(),
                    "shape": list(output.shape)
                }
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            handle = module.register_forward_hook(activation_hook(name))
            handles.append(handle)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return activation_stats


def compare_models(
    model1: nn.Module,
    model2: nn.Module,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2"
) -> Dict[str, Any]:
    """Compare two models."""
    
    # Parameter comparison
    params1 = count_parameters(model1)
    params2 = count_parameters(model2)
    
    # Size comparison
    size1 = get_model_size(model1)
    size2 = get_model_size(model2)
    
    comparison = {
        "parameter_comparison": {
            model1_name: params1,
            model2_name: params2,
            "difference": {
                "total": params2["total"] - params1["total"],
                "trainable": params2["trainable"] - params1["trainable"]
            },
            "ratio": {
                "total": params2["total"] / params1["total"] if params1["total"] > 0 else float('inf'),
                "trainable": params2["trainable"] / params1["trainable"] if params1["trainable"] > 0 else float('inf')
            }
        },
        "size_comparison": {
            model1_name: size1,
            model2_name: size2,
            "difference_mb": size2["total_mb"] - size1["total_mb"],
            "ratio": size2["total_mb"] / size1["total_mb"] if size1["total_mb"] > 0 else float('inf')
        }
    }
    
    return comparison


def freeze_parameters(model: nn.Module, patterns: List[str]):
    """Freeze parameters matching given patterns."""
    
    frozen_count = 0
    
    for name, param in model.named_parameters():
        for pattern in patterns:
            if pattern in name:
                param.requires_grad = False
                frozen_count += 1
                break
    
    logging.info(f"Frozen {frozen_count} parameters matching patterns: {patterns}")


def unfreeze_parameters(model: nn.Module, patterns: List[str]):
    """Unfreeze parameters matching given patterns."""
    
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        for pattern in patterns:
            if pattern in name:
                param.requires_grad = True
                unfrozen_count += 1
                break
    
    logging.info(f"Unfrozen {unfrozen_count} parameters matching patterns: {patterns}")


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_patterns: List[str] = None
) -> List[Dict[str, Any]]:
    """Get parameter groups for optimizer with different weight decay."""
    
    if no_decay_patterns is None:
        no_decay_patterns = ["bias", "LayerNorm", "RMSNorm", "layernorm", "norm"]
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should have weight decay
        should_decay = True
        for pattern in no_decay_patterns:
            if pattern in name:
                should_decay = False
                break
        
        if should_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    logging.info(f"Created parameter groups: {len(decay_params)} with decay, {len(no_decay_params)} without decay")
    
    return param_groups


def initialize_weights(model: nn.Module, init_type: str = "normal"):
    """Initialize model weights."""
    
    def init_func(module):
        if isinstance(module, nn.Linear):
            if init_type == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_type == "xavier":
                torch.nn.init.xavier_uniform_(module.weight)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    model.apply(init_func)
    logging.info(f"Initialized model weights with {init_type} initialization")
