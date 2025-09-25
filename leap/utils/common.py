"""Common utility functions for LEAP."""

import torch
import numpy as np
import random
import os
import logging
import yaml
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    loss: float,
    filepath: str,
    **kwargs
):
    """Save model checkpoint."""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'torch_version': torch.__version__,
        **kwargs
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f"Checkpoint loaded from {filepath}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'torch_version': checkpoint.get('torch_version', 'unknown')
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
):
    """Setup logging configuration."""
    
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device."""
    
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


def format_time(seconds: float) -> str:
    """Format time duration in human readable format."""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def format_number(num: float, precision: int = 2) -> str:
    """Format large numbers in human readable format."""
    
    if num >= 1e12:
        return f"{num/1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def create_output_dir(base_dir: str, experiment_name: str) -> str:
    """Create output directory for experiment."""
    
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    
    import psutil
    
    # System memory
    system_memory = psutil.virtual_memory()
    
    stats = {
        "system_memory_total_gb": system_memory.total / (1024**3),
        "system_memory_used_gb": system_memory.used / (1024**3),
        "system_memory_percent": system_memory.percent,
    }
    
    # GPU memory if available
    if torch.cuda.is_available():
        stats.update({
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        })
    
    return stats


def cleanup_memory():
    """Clean up GPU memory."""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_model_summary(model: torch.nn.Module, input_shape: Optional[tuple] = None):
    """Print model summary."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {format_number(total_params)}")
    print(f"Trainable parameters: {format_number(trainable_params)}")
    print(f"Non-trainable parameters: {format_number(total_params - trainable_params)}")
    
    if input_shape:
        print(f"Input shape: {input_shape}")
    
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = (param_size + buffer_size) / (1024**2)
    print(f"Model size: {model_size:.2f} MB")
