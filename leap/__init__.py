"""LEAP: Learning Expert Adaptation & Pruning for Task-Specialized MoE Language Models.

This package implements the LEAP framework for optimizing Mixture-of-Experts (MoE) 
language models through expert pruning and routing adaptation.
"""

from .framework import LEAPFramework
from .config import ModelConfig, TrainingConfig, PruningConfig, RoutingConfig
from .models import LlamaMaverickMoE, QwenMoE, BaseMoE

__version__ = "0.1.0"
__author__ = "LEAP Research Team"

__all__ = [
    "LEAPFramework",
    "ModelConfig",
    "TrainingConfig", 
    "PruningConfig",
    "RoutingConfig",
    "LlamaMaverickMoE",
    "QwenMoE",
    "BaseMoE",
]