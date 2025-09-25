"""Main LEAP framework implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import json
from tqdm import tqdm
import numpy as np

from .config import LEAPConfig, ModelConfig, get_model_config
from .models import LlamaMaverickMoE, QwenMoE, BaseMoE
from .agents import PruningAgent, RoutingAgent
from .training import ActiveLearningTrainer, JointTrainer
from .evaluation import LEAPEvaluator
from .utils import set_seed, save_checkpoint, load_checkpoint


class LEAPFramework:
    """Main LEAP framework for expert pruning and routing adaptation."""
    
    def __init__(
        self,
        config: LEAPConfig,
        device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        set_seed(config.seed)
        
        # Initialize model
        self.model: Optional[BaseMoE] = None
        self.pruning_agent: Optional[PruningAgent] = None
        self.routing_agent: Optional[RoutingAgent] = None
        self.joint_trainer: Optional[JointTrainer] = None
        self.evaluator: Optional[LEAPEvaluator] = None
        
        # Training state
        self.pruned_expert_indices: Optional[List[int]] = None
        self.is_pruned = False
        self.is_routing_adapted = False
        
        self.logger.info(f"Initialized LEAP framework with config: {config.task}")
        self.logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None) -> BaseMoE:
        """Load or create MoE model."""
        
        if model_path and os.path.exists(model_path):
            # Load existing model
            self.logger.info(f"Loading model from {model_path}")
            if self.config.model.model_type == "llama_maverick":
                self.model = LlamaMaverickMoE.load_pruned_model(model_path, self.device)
            elif self.config.model.model_type == "qwen3_235b":
                self.model = QwenMoE.load_pruned_model(model_path, self.device)
            else:
                raise ValueError(f"Unknown model type: {self.config.model.model_type}")
        else:
            # Create new model
            self.logger.info(f"Creating new {self.config.model.model_type} model")
            if self.config.model.model_type == "llama_maverick":
                self.model = LlamaMaverickMoE(self.config.model).to(self.device)
            elif self.config.model.model_type == "qwen3_235b":
                self.model = QwenMoE(self.config.model).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {self.config.model.model_type}")
        
        # Initialize evaluator
        self.evaluator = LEAPEvaluator(self.config, self.device)
        
        return self.model
    
    def initialize_agents(self):
        """Initialize pruning and routing agents."""
        
        if self.model is None:
            raise ValueError("Model must be loaded before initializing agents")
        
        # Initialize pruning agent
        self.pruning_agent = PruningAgent(
            num_experts=self.config.model.num_experts,
            config=self.config.pruning,
            device=self.device
        )
        
        self.logger.info("Initialized pruning agent")
    
    def run_pruning_phase(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        save_path: Optional[str] = None
    ) -> List[int]:
        """Run the pruning phase to select optimal expert subset."""
        
        if self.pruning_agent is None:
            self.initialize_agents()
        
        self.logger.info("Starting pruning phase...")
        self.logger.info(f"Target experts: {self.config.pruning.target_experts}")
        self.logger.info(f"Budget constraint: {self.config.pruning.budget_constraint}")
        
        # Training loop for pruning agent
        best_expert_subset = []
        best_performance = 0.0
        
        for episode in tqdm(range(self.config.pruning.meta_episodes), desc="Pruning Episodes"):
            # Reset episode
            state = self.pruning_agent.reset_episode()
            selected_experts = []
            
            # Get baseline performance (all experts)
            baseline_perf = self.evaluator.evaluate_model_performance(
                self.model, val_dataloader, expert_indices=None
            )
            
            # Episode loop
            for step in range(self.config.pruning.target_experts):
                # Select next expert
                action, log_prob, value = self.pruning_agent.select_action(state)
                selected_experts.append(action)
                
                # Evaluate model with current subset
                current_perf = self.evaluator.evaluate_model_performance(
                    self.model, val_dataloader, expert_indices=selected_experts
                )
                
                # Compute reward
                reward = self.pruning_agent.compute_reward(
                    state, action, current_perf, baseline_perf
                )
                self.pruning_agent.store_reward(reward)
                
                # Update state
                state = self.pruning_agent.update_state(state, action, current_perf)
                
                # Early stopping if budget exceeded
                if state.budget_used > self.config.pruning.budget_constraint:
                    break
            
            # Finish episode
            episode_stats = self.pruning_agent.finish_episode()
            
            # Track best subset
            if current_perf > best_performance:
                best_performance = current_perf
                best_expert_subset = selected_experts.copy()
            
            # Update policy every few episodes
            if (episode + 1) % self.config.pruning.ppo_epochs == 0:
                policy_stats = self.pruning_agent.update_policy()
                
                self.logger.info(
                    f"Episode {episode + 1}: Performance={current_perf:.3f}, "
                    f"Best={best_performance:.3f}, Policy Loss={policy_stats.get('policy_loss', 0):.4f}"
                )
        
        # Store pruned expert indices
        self.pruned_expert_indices = best_expert_subset
        self.is_pruned = True
        
        # Apply pruning to model
        self.model.prune_experts(self.pruned_expert_indices)
        
        self.logger.info(f"Pruning complete. Selected experts: {self.pruned_expert_indices}")
        self.logger.info(f"Final performance: {best_performance:.3f}")
        
        # Save pruned model
        if save_path:
            self.model.save_pruned_model(save_path, self.pruned_expert_indices)
            self.logger.info(f"Saved pruned model to {save_path}")
        
        return self.pruned_expert_indices
    
    def run_routing_phase(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """Run the routing adaptation phase."""
        
        if not self.is_pruned or self.pruned_expert_indices is None:
            raise ValueError("Must run pruning phase before routing phase")
        
        self.logger.info("Starting routing phase...")
        
        # Initialize routing agent
        self.routing_agent = RoutingAgent(
            hidden_dim=self.config.model.hidden_dim,
            pruned_expert_indices=self.pruned_expert_indices,
            config=self.config.routing,
            device=self.device
        )
        
        # Initialize joint trainer
        self.joint_trainer = JointTrainer(
            model=self.model,
            routing_agent=self.routing_agent,
            config=self.config.training,
            device=self.device
        )
        
        # Router warm-up phase
        self.logger.info("Router warm-up phase...")
        warmup_stats = self.joint_trainer.router_warmup(
            train_dataloader, 
            val_dataloader,
            steps=self.config.routing.warmup_steps
        )
        
        # Joint fine-tuning phase
        self.logger.info("Joint fine-tuning phase...")
        training_stats = self.joint_trainer.joint_training(
            train_dataloader,
            val_dataloader,
            steps=self.config.training.total_steps
        )
        
        self.is_routing_adapted = True
        
        # Final evaluation
        final_performance = self.evaluator.evaluate_model_performance(
            self.model, val_dataloader, expert_indices=self.pruned_expert_indices
        )
        
        results = {
            "final_performance": final_performance,
            "warmup_stats": warmup_stats,
            "training_stats": training_stats,
            "pruned_experts": self.pruned_expert_indices,
            "num_experts": len(self.pruned_expert_indices),
            "compression_ratio": len(self.pruned_expert_indices) / self.config.model.num_experts
        }
        
        self.logger.info(f"Routing adaptation complete. Final performance: {final_performance:.3f}")
        
        # Save adapted model
        if save_path:
            self.save_full_checkpoint(save_path)
        
        return results
    
    def optimize(
        self,
        model: Optional[BaseMoE] = None,
        train_data: Optional[DataLoader] = None,
        val_data: Optional[DataLoader] = None,
        model_path: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete LEAP optimization pipeline."""
        
        self.logger.info("Starting LEAP optimization...")
        
        # Load model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.load_model(model_path)
        
        # Validate data loaders
        if train_data is None or val_data is None:
            raise ValueError("Training and validation data loaders are required")
        
        # Phase 1: Expert Pruning
        self.logger.info("Phase 1: Expert Pruning")
        pruned_experts = self.run_pruning_phase(
            train_data, 
            val_data, 
            save_path=f"{save_path}_pruned.pt" if save_path else None
        )
        
        # Phase 2: Routing Adaptation
        self.logger.info("Phase 2: Routing Adaptation")
        routing_results = self.run_routing_phase(
            train_data,
            val_data,
            save_path=f"{save_path}_final.pt" if save_path else None
        )
        
        # Comprehensive evaluation
        self.logger.info("Final evaluation...")
        evaluation_results = self.evaluator.comprehensive_evaluation(
            self.model, val_data, self.pruned_expert_indices
        )
        
        # Compile final results
        final_results = {
            "pruned_experts": pruned_experts,
            "routing_results": routing_results,
            "evaluation_results": evaluation_results,
            "config": self.config,
            "model_info": {
                "model_type": self.config.model.model_type,
                "original_experts": self.config.model.num_experts,
                "pruned_experts": len(pruned_experts),
                "compression_ratio": len(pruned_experts) / self.config.model.num_experts,
            }
        }
        
        self.logger.info("LEAP optimization complete!")
        self.logger.info(f"Compression ratio: {final_results['model_info']['compression_ratio']:.2%}")
        self.logger.info(f"Final performance: {evaluation_results.get('accuracy', 0):.3f}")
        
        return final_results
    
    def evaluate(
        self,
        model: BaseMoE,
        test_data: DataLoader,
        expert_indices: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        
        if self.evaluator is None:
            self.evaluator = LEAPEvaluator(self.config, self.device)
        
        return self.evaluator.comprehensive_evaluation(model, test_data, expert_indices)
    
    def save_full_checkpoint(self, path: str):
        """Save complete LEAP checkpoint."""
        
        checkpoint = {
            "config": self.config,
            "model_state_dict": self.model.state_dict() if self.model else None,
            "pruned_expert_indices": self.pruned_expert_indices,
            "is_pruned": self.is_pruned,
            "is_routing_adapted": self.is_routing_adapted,
        }
        
        # Add agent states if available
        if self.pruning_agent:
            checkpoint["pruning_agent_state"] = {
                "policy_state_dict": self.pruning_agent.policy.state_dict(),
                "optimizer_state_dict": self.pruning_agent.ppo_trainer.optimizer.state_dict(),
            }
        
        if self.routing_agent:
            checkpoint["routing_agent_state"] = {
                "policy_state_dict": self.routing_agent.policy.state_dict(),
                "optimizer_state_dict": self.routing_agent.ppo_trainer.optimizer.state_dict(),
            }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved full LEAP checkpoint to {path}")
    
    def load_full_checkpoint(self, path: str):
        """Load complete LEAP checkpoint."""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load configuration
        self.config = checkpoint["config"]
        
        # Load model
        if checkpoint["model_state_dict"]:
            self.load_model()
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load pruning state
        self.pruned_expert_indices = checkpoint.get("pruned_expert_indices")
        self.is_pruned = checkpoint.get("is_pruned", False)
        self.is_routing_adapted = checkpoint.get("is_routing_adapted", False)
        
        # Apply pruning if needed
        if self.is_pruned and self.pruned_expert_indices:
            self.model.prune_experts(self.pruned_expert_indices)
        
        # Load agent states if available
        if "pruning_agent_state" in checkpoint:
            self.initialize_agents()
            agent_state = checkpoint["pruning_agent_state"]
            self.pruning_agent.policy.load_state_dict(agent_state["policy_state_dict"])
            self.pruning_agent.ppo_trainer.optimizer.load_state_dict(agent_state["optimizer_state_dict"])
        
        if "routing_agent_state" in checkpoint and self.is_pruned:
            self.routing_agent = RoutingAgent(
                hidden_dim=self.config.model.hidden_dim,
                pruned_expert_indices=self.pruned_expert_indices,
                config=self.config.routing,
                device=self.device
            )
            agent_state = checkpoint["routing_agent_state"]
            self.routing_agent.policy.load_state_dict(agent_state["policy_state_dict"])
            self.routing_agent.ppo_trainer.optimizer.load_state_dict(agent_state["optimizer_state_dict"])
        
        self.logger.info(f"Loaded full LEAP checkpoint from {path}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and performance metrics."""
        
        if self.model is None:
            return {"error": "No model loaded"}
        
        stats = {
            "model_type": self.config.model.model_type,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "original_experts": self.config.model.num_experts,
            "is_pruned": self.is_pruned,
        }
        
        if self.is_pruned and self.pruned_expert_indices:
            stats.update({
                "pruned_experts": self.pruned_expert_indices,
                "num_active_experts": len(self.pruned_expert_indices),
                "compression_ratio": len(self.pruned_expert_indices) / self.config.model.num_experts,
                "parameter_reduction": 1 - (len(self.pruned_expert_indices) / self.config.model.num_experts),
            })
        
        # Add FLOP estimates
        sample_length = 512
        flop_stats = self.model.compute_flops(sample_length)
        stats.update({
            "estimated_flops": flop_stats,
            "flops_per_token": flop_stats["flops_per_token"],
        })
        
        return stats

