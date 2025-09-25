"""Configuration classes for LEAP framework."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class ModelConfig:
    """Configuration for MoE model architecture."""
    
    model_type: str = "llama_maverick"  # llama_maverick or qwen3_235b
    num_experts: int = 128
    expert_size: int = 17_000_000_000  # 17B parameters per expert for Llama Maverick
    hidden_dim: int = 8192
    num_layers: int = 80
    vocab_size: int = 128256
    intermediate_size: int = 28672
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: float = 500000.0
    max_position_embeddings: int = 32768
    
    # MoE specific
    top_k: int = 2
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001


@dataclass 
class PruningConfig:
    """Configuration for pruning agent."""
    
    target_experts: int = 16
    budget_constraint: float = 0.125  # Keep 12.5% of experts
    meta_episodes: int = 500
    ppo_epochs: int = 4
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_function_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Reward function weights
    performance_weight: float = 1.0
    efficiency_weight: float = 0.1


@dataclass
class RoutingConfig:
    """Configuration for routing agent."""
    
    top_k: int = 2
    temperature: float = 1.0
    exploration_noise: float = 0.1
    active_learning_samples: int = 1000
    uncertainty_threshold: float = 0.1
    
    # Training parameters
    router_lr: float = 1e-4
    warmup_steps: int = 100
    ppo_epochs: int = 4
    clip_range: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    max_length: int = 2048
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    total_steps: int = 5000
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"


@dataclass
class LEAPConfig:
    """Main LEAP framework configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Task-specific settings
    task: str = "code_generation"
    dataset_name: str = "humaneval"
    metric: str = "pass@1"
    
    # Experiment settings
    seed: int = 42
    experiment_name: str = "leap_experiment"
    wandb_project: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "LEAPConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")


def get_model_config(model_type: str) -> ModelConfig:
    """Get default model configuration for specific model type."""
    
    if model_type == "llama_maverick":
        return ModelConfig(
            model_type="llama_maverick",
            num_experts=128,
            expert_size=17_000_000_000,  # 17B per expert
            hidden_dim=8192,
            num_layers=80,
            vocab_size=128256,
            intermediate_size=28672,
            num_attention_heads=64,
            num_key_value_heads=8,
        )
    elif model_type == "qwen3_235b":
        return ModelConfig(
            model_type="qwen3_235b", 
            num_experts=128,
            expert_size=1_800_000_000,  # ~1.8B per expert (235B total / 128 experts)
            hidden_dim=12288,
            num_layers=96,
            vocab_size=152064,
            intermediate_size=32768,
            num_attention_heads=96,
            num_key_value_heads=96,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_task_config(task: str) -> Dict[str, Any]:
    """Get task-specific configuration."""
    
    task_configs = {
        "code_generation": {
            "dataset_name": "humaneval",
            "metric": "pass@1",
            "max_length": 2048,
            "target_experts": 16,
        },
        "reasoning": {
            "dataset_name": "gsm8k", 
            "metric": "accuracy",
            "max_length": 1024,
            "target_experts": 12,
        },
        "summarization": {
            "dataset_name": "xsum",
            "metric": "rouge_l",
            "max_length": 512,
            "target_experts": 8,
        }
    }
    
    if task not in task_configs:
        raise ValueError(f"Unknown task: {task}")
    
    return task_configs[task]
