"""Joint training implementation with PPO and Active Learning."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from tqdm import tqdm
import numpy as np

from ..config import TrainingConfig
from ..models import BaseMoE
from ..agents import RoutingAgent
from .active_learning import ActiveLearningTrainer


class JointTrainer:
    """Joint trainer for router and experts with PPO and Active Learning."""
    
    def __init__(
        self,
        model: BaseMoE,
        routing_agent: RoutingAgent,
        config: TrainingConfig,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.routing_agent = routing_agent
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize optimizers
        self.model_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize active learning trainer
        self.active_learning_trainer = ActiveLearningTrainer(
            model=model,
            routing_agent=routing_agent,
            config=config,
            device=device,
            logger=logger
        )
        
        # Training state
        self.global_step = 0
        self.best_performance = 0.0
        
        # Metrics tracking
        self.training_metrics = {
            "losses": [],
            "routing_rewards": [],
            "expert_utilization": [],
            "performance_scores": []
        }
    
    def router_warmup(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        steps: int
    ) -> Dict[str, float]:
        """Warm-up phase: train router with frozen experts."""
        
        self.logger.info(f"Starting router warm-up for {steps} steps...")
        
        # Freeze expert parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Only train routing agent
        warmup_losses = []
        warmup_rewards = []
        
        self.model.train()
        
        for step in tqdm(range(steps), desc="Router Warm-up"):
            batch_losses = []
            batch_rewards = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= steps:
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device) if "labels" in batch else input_ids
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass with routing agent
                with torch.no_grad():
                    # Get hidden states from model (without routing)
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    hidden_states = outputs["hidden_states"][-1]  # Last layer
                
                # Get routing decisions
                routing_weights, routing_info = self.routing_agent.route_tokens(
                    hidden_states, training=True, return_routing_info=True
                )
                
                # Compute task loss (language modeling)
                logits = outputs["logits"]
                task_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Compute routing reward
                baseline_loss = task_loss.detach()  # Use current loss as baseline
                baseline_latency = 1.0  # Placeholder
                current_latency = 0.8  # Placeholder (routing should be faster)
                
                routing_reward = self.routing_agent.compute_routing_reward(
                    routing_weights, task_loss, current_latency, baseline_loss, baseline_latency
                )
                
                # Store reward for routing agent
                self.routing_agent.store_reward(routing_reward)
                
                batch_losses.append(task_loss.item())
                batch_rewards.append(routing_reward.mean().item())
            
            # Update routing policy
            if len(self.routing_agent.episode_states) > 0:
                routing_stats = self.routing_agent.finish_episode()
                policy_stats = self.routing_agent.update_policy()
                
                warmup_losses.extend(batch_losses)
                warmup_rewards.extend(batch_rewards)
            
            # Logging
            if step % 10 == 0:
                avg_loss = np.mean(batch_losses) if batch_losses else 0.0
                avg_reward = np.mean(batch_rewards) if batch_rewards else 0.0
                self.logger.info(f"Warmup Step {step}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")
        
        # Unfreeze expert parameters for joint training
        for param in self.model.parameters():
            param.requires_grad = True
        
        warmup_stats = {
            "avg_loss": np.mean(warmup_losses) if warmup_losses else 0.0,
            "avg_reward": np.mean(warmup_rewards) if warmup_rewards else 0.0,
            "total_steps": steps
        }
        
        self.logger.info(f"Router warm-up complete. Avg Loss: {warmup_stats['avg_loss']:.4f}")
        return warmup_stats
    
    def joint_training(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        steps: int
    ) -> Dict[str, Any]:
        """Joint training phase: train router and experts together."""
        
        self.logger.info(f"Starting joint training for {steps} steps...")
        
        self.model.train()
        
        # Training metrics
        training_losses = []
        routing_rewards = []
        aux_losses = []
        
        step_count = 0
        
        for epoch in range(100):  # Large number, will break when steps reached
            for batch in tqdm(train_dataloader, desc=f"Joint Training Epoch {epoch}"):
                if step_count >= steps:
                    break
                
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device) if "labels" in batch else input_ids
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get routing decisions from routing agent
                hidden_states = outputs["hidden_states"][-1]
                routing_weights, routing_info = self.routing_agent.route_tokens(
                    hidden_states, training=True, return_routing_info=True
                )
                
                # Compute losses
                logits = outputs["logits"]
                task_loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # MoE auxiliary losses
                aux_loss = outputs.get("aux_loss", torch.tensor(0.0))
                z_loss = outputs.get("z_loss", torch.tensor(0.0))
                
                # Load balancing loss from routing
                load_balancing_loss = routing_info.get("load_balancing_loss", torch.tensor(0.0))
                
                # Total loss
                total_loss = task_loss + aux_loss + z_loss + 0.01 * load_balancing_loss
                
                # Backward pass
                self.model_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Update model
                self.model_optimizer.step()
                
                # Compute routing reward
                baseline_loss = task_loss.detach()
                routing_reward = self.routing_agent.compute_routing_reward(
                    routing_weights, task_loss, 0.8, baseline_loss, 1.0
                )
                self.routing_agent.store_reward(routing_reward)
                
                # Update routing agent
                if step_count % 10 == 0:  # Update routing policy periodically
                    routing_episode_stats = self.routing_agent.finish_episode()
                    routing_policy_stats = self.routing_agent.update_policy()
                
                # Store metrics
                training_losses.append(total_loss.item())
                routing_rewards.append(routing_reward.mean().item())
                aux_losses.append(aux_loss.item())
                
                step_count += 1
                self.global_step += 1
                
                # Logging
                if step_count % self.config.logging_steps == 0:
                    avg_loss = np.mean(training_losses[-self.config.logging_steps:])
                    avg_reward = np.mean(routing_rewards[-self.config.logging_steps:])
                    avg_aux = np.mean(aux_losses[-self.config.logging_steps:])
                    
                    self.logger.info(
                        f"Step {step_count}: Loss={avg_loss:.4f}, "
                        f"Reward={avg_reward:.4f}, Aux={avg_aux:.4f}"
                    )
                
                # Validation
                if step_count % self.config.eval_steps == 0:
                    val_performance = self.evaluate_step(val_dataloader)
                    
                    if val_performance > self.best_performance:
                        self.best_performance = val_performance
                        self.logger.info(f"New best performance: {val_performance:.4f}")
                
                # Active learning sample selection
                if step_count % 100 == 0:  # Periodic active learning
                    self.active_learning_step(train_dataloader)
                
                if step_count >= steps:
                    break
            
            if step_count >= steps:
                break
        
        # Final statistics
        training_stats = {
            "total_steps": step_count,
            "avg_loss": np.mean(training_losses),
            "avg_reward": np.mean(routing_rewards),
            "avg_aux_loss": np.mean(aux_losses),
            "best_performance": self.best_performance,
            "final_performance": self.evaluate_step(val_dataloader)
        }
        
        self.logger.info(f"Joint training complete. Final performance: {training_stats['final_performance']:.4f}")
        return training_stats
    
    def evaluate_step(self, val_dataloader: DataLoader) -> float:
        """Evaluate model performance on validation set."""
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device) if "labels" in batch else input_ids
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"]
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_samples += (labels != -100).sum().item()
        
        self.model.train()
        
        # Return perplexity (lower is better, so we return negative for maximization)
        perplexity = torch.exp(torch.tensor(total_loss / total_samples))
        return -perplexity.item()
    
    def active_learning_step(self, train_dataloader: DataLoader):
        """Perform active learning sample selection."""
        
        self.logger.info("Performing active learning sample selection...")
        
        # Get a batch for uncertainty estimation
        batch = next(iter(train_dataloader))
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs["hidden_states"][-1]
        
        # Select high-uncertainty samples
        selected_indices = self.routing_agent.active_learning_selection(
            hidden_states, num_samples=min(32, input_ids.size(0))
        )
        
        self.logger.info(f"Selected {len(selected_indices)} samples for active learning")
        
        # In a full implementation, you would use these samples for additional training
        # For now, we just log the selection
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        
        checkpoint = {
            "global_step": self.global_step,
            "best_performance": self.best_performance,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model_optimizer.state_dict(),
            "routing_agent_state": {
                "policy_state_dict": self.routing_agent.policy.state_dict(),
                "optimizer_state_dict": self.routing_agent.ppo_trainer.optimizer.state_dict(),
            },
            "training_metrics": self.training_metrics,
            "config": self.config
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved training checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint["global_step"]
        self.best_performance = checkpoint["best_performance"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load routing agent state
        routing_state = checkpoint["routing_agent_state"]
        self.routing_agent.policy.load_state_dict(routing_state["policy_state_dict"])
        self.routing_agent.ppo_trainer.optimizer.load_state_dict(routing_state["optimizer_state_dict"])
        
        self.training_metrics = checkpoint.get("training_metrics", self.training_metrics)
        
        self.logger.info(f"Loaded training checkpoint from {path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        
        return {
            "global_step": self.global_step,
            "best_performance": self.best_performance,
            "training_metrics": self.training_metrics,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

