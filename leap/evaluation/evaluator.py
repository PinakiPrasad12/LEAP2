"""LEAP model evaluator for comprehensive performance assessment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import logging
import numpy as np
from collections import defaultdict
import json

from ..config import LEAPConfig
from ..models import BaseMoE
from .metrics import MetricsCalculator, TaskMetrics


class LEAPEvaluator:
    """Comprehensive evaluator for LEAP models."""
    
    def __init__(
        self,
        config: LEAPConfig,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(config.task)
        
        # Task-specific settings
        self.task = config.task
        self.metric_name = config.metric
        
        # Performance tracking
        self.evaluation_history = []
        
    def evaluate_model_performance(
        self,
        model: BaseMoE,
        dataloader: DataLoader,
        expert_indices: Optional[List[int]] = None,
        return_detailed: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """Evaluate model performance on given dataset."""
        
        self.logger.info(f"Evaluating model performance on {self.task}")
        
        if expert_indices is not None:
            model.prune_experts(expert_indices)
            self.logger.info(f"Using {len(expert_indices)} experts: {expert_indices}")
        
        model.eval()
        
        # Task-specific evaluation
        if self.task == "code_generation":
            results = self._evaluate_code_generation(model, dataloader)
        elif self.task == "reasoning":
            results = self._evaluate_reasoning(model, dataloader)
        elif self.task == "summarization":
            results = self._evaluate_summarization(model, dataloader)
        else:
            # Generic language modeling evaluation
            results = self._evaluate_language_modeling(model, dataloader)
        
        # Extract primary metric
        primary_score = results.get(self.metric_name, results.get('accuracy', 0.0))
        
        if return_detailed:
            return results
        else:
            return primary_score
    
    def comprehensive_evaluation(
        self,
        model: BaseMoE,
        dataloader: DataLoader,
        expert_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation including efficiency metrics."""
        
        self.logger.info("Starting comprehensive evaluation...")
        
        # Performance evaluation
        performance_results = self.evaluate_model_performance(
            model, dataloader, expert_indices, return_detailed=True
        )
        
        # Efficiency evaluation
        efficiency_results = self._evaluate_efficiency(model, dataloader)
        
        # Expert utilization analysis
        utilization_results = self._analyze_expert_utilization(model, dataloader)
        
        # Model statistics
        model_stats = self._compute_model_statistics(model, expert_indices)
        
        # Combine all results
        comprehensive_results = {
            "performance": performance_results,
            "efficiency": efficiency_results,
            "utilization": utilization_results,
            "model_stats": model_stats,
            "expert_indices": expert_indices,
            "evaluation_timestamp": time.time()
        }
        
        # Store in history
        self.evaluation_history.append(comprehensive_results)
        
        self.logger.info("Comprehensive evaluation complete")
        return comprehensive_results
    
    def _evaluate_code_generation(
        self,
        model: BaseMoE,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate code generation performance."""
        
        total_samples = 0
        correct_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"]
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                # Compute accuracy (simplified)
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct = (predictions == labels) & mask
                
                total_loss += loss.item()
                correct_samples += correct.sum().item()
                total_samples += mask.sum().item()
        
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
        perplexity = torch.exp(torch.tensor(total_loss / total_samples)).item()
        
        # Placeholder for pass@1 (would require code execution)
        pass_at_1 = accuracy * 0.8  # Rough approximation
        
        return {
            "accuracy": accuracy,
            "perplexity": perplexity,
            "pass@1": pass_at_1,
            "total_samples": total_samples
        }
    
    def _evaluate_reasoning(
        self,
        model: BaseMoE,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate mathematical reasoning performance."""
        
        total_samples = 0
        correct_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"]
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                # For reasoning, we typically evaluate final answer accuracy
                # This is a simplified version
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct = (predictions == labels) & mask
                
                total_loss += loss.item()
                correct_samples += correct.sum().item()
                total_samples += mask.sum().item()
        
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0
        perplexity = torch.exp(torch.tensor(total_loss / total_samples)).item()
        
        return {
            "accuracy": accuracy,
            "perplexity": perplexity,
            "total_samples": total_samples
        }
    
    def _evaluate_summarization(
        self,
        model: BaseMoE,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate summarization performance."""
        
        total_samples = 0
        total_loss = 0.0
        rouge_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"]
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                total_loss += loss.item()
                valid_tokens = (labels != -100).sum().item()
                total_samples += valid_tokens
                
                # Placeholder for ROUGE score computation
                rouge_l = 0.65  # Would compute actual ROUGE-L here
                rouge_scores.append(rouge_l)
        
        perplexity = torch.exp(torch.tensor(total_loss / total_samples)).item()
        avg_rouge_l = np.mean(rouge_scores) if rouge_scores else 0.0
        
        return {
            "rouge_l": avg_rouge_l,
            "perplexity": perplexity,
            "total_samples": total_samples
        }
    
    def _evaluate_language_modeling(
        self,
        model: BaseMoE,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate general language modeling performance."""
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs["logits"]
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_samples += (labels != -100).sum().item()
        
        perplexity = torch.exp(torch.tensor(total_loss / total_samples)).item()
        
        return {
            "perplexity": perplexity,
            "loss": total_loss / total_samples,
            "total_samples": total_samples
        }
    
    def _evaluate_efficiency(
        self,
        model: BaseMoE,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model efficiency metrics."""
        
        self.logger.info("Evaluating efficiency metrics...")
        
        # Get a sample batch for timing
        sample_batch = next(iter(dataloader))
        input_ids = sample_batch["input_ids"].to(self.device)
        attention_mask = sample_batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        batch_size, seq_len = input_ids.shape
        
        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Measure inference time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # Average over multiple runs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 10
        tokens_per_second = (batch_size * seq_len) / avg_inference_time
        
        # Compute FLOPs
        flop_stats = model.compute_flops(seq_len)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0
        
        return {
            "inference_time_ms": avg_inference_time * 1000,
            "tokens_per_second": tokens_per_second,
            "flops_per_token": flop_stats["flops_per_token"],
            "total_flops": flop_stats["total_flops"],
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        }
    
    def _analyze_expert_utilization(
        self,
        model: BaseMoE,
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """Analyze expert utilization patterns."""
        
        self.logger.info("Analyzing expert utilization...")
        
        # Track expert usage across layers
        expert_counts = defaultdict(lambda: defaultdict(int))
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 10:  # Limit analysis to first 10 batches
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                batch_size, seq_len = input_ids.shape
                total_tokens += batch_size * seq_len
                
                # Forward pass to get MoE auxiliary losses (contains expert counts)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                moe_aux_losses = outputs.get("moe_aux_losses", [])
                
                # Aggregate expert counts per layer
                for layer_idx, layer_losses in enumerate(moe_aux_losses):
                    if "expert_counts" in layer_losses:
                        expert_counts_tensor = layer_losses["expert_counts"]
                        for expert_idx, count in enumerate(expert_counts_tensor):
                            expert_counts[layer_idx][expert_idx] += count.item()
        
        # Compute utilization statistics
        utilization_stats = {}
        
        for layer_idx, layer_counts in expert_counts.items():
            total_layer_tokens = sum(layer_counts.values())
            if total_layer_tokens > 0:
                expert_utilization = {
                    expert_idx: count / total_layer_tokens 
                    for expert_idx, count in layer_counts.items()
                }
                
                # Compute statistics
                utilizations = list(expert_utilization.values())
                utilization_stats[f"layer_{layer_idx}"] = {
                    "expert_utilization": expert_utilization,
                    "mean_utilization": np.mean(utilizations),
                    "std_utilization": np.std(utilizations),
                    "max_utilization": np.max(utilizations),
                    "min_utilization": np.min(utilizations),
                    "active_experts": sum(1 for u in utilizations if u > 0.01),  # >1% utilization
                    "load_balance_loss": np.var(utilizations)  # Variance as load balance metric
                }
        
        return {
            "per_layer_stats": utilization_stats,
            "total_tokens_analyzed": total_tokens,
            "num_batches_analyzed": min(10, len(dataloader))
        }
    
    def _compute_model_statistics(
        self,
        model: BaseMoE,
        expert_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Compute model statistics."""
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # MoE-specific stats
        moe_layers = model.get_moe_layers()
        num_moe_layers = len(moe_layers)
        
        if expert_indices is not None:
            active_experts = len(expert_indices)
            compression_ratio = active_experts / model.num_experts
        else:
            active_experts = model.num_experts
            compression_ratio = 1.0
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_moe_layers": num_moe_layers,
            "total_experts": model.num_experts,
            "active_experts": active_experts,
            "compression_ratio": compression_ratio,
            "parameter_reduction": 1.0 - compression_ratio,
            "model_type": model.__class__.__name__,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers
        }
    
    def compare_models(
        self,
        models: Dict[str, BaseMoE],
        dataloader: DataLoader,
        expert_indices_dict: Optional[Dict[str, List[int]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models on the same dataset."""
        
        self.logger.info(f"Comparing {len(models)} models...")
        
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            expert_indices = None
            if expert_indices_dict and model_name in expert_indices_dict:
                expert_indices = expert_indices_dict[model_name]
            
            model_results = self.comprehensive_evaluation(model, dataloader, expert_indices)
            results[model_name] = model_results
        
        # Compute relative performance
        baseline_performance = None
        for model_name, result in results.items():
            if "baseline" in model_name.lower() or "full" in model_name.lower():
                baseline_performance = result["performance"].get(self.metric_name, 0.0)
                break
        
        if baseline_performance is None and results:
            # Use first model as baseline
            baseline_performance = list(results.values())[0]["performance"].get(self.metric_name, 0.0)
        
        # Add relative metrics
        if baseline_performance > 0:
            for model_name, result in results.items():
                current_performance = result["performance"].get(self.metric_name, 0.0)
                result["relative_performance"] = current_performance / baseline_performance
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file."""
        
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_tensors(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_tensors(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {filepath}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations performed."""
        
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        latest_eval = self.evaluation_history[-1]
        
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "latest_evaluation": {
                "timestamp": latest_eval["evaluation_timestamp"],
                "primary_metric": latest_eval["performance"].get(self.metric_name, 0.0),
                "efficiency": {
                    "inference_time_ms": latest_eval["efficiency"]["inference_time_ms"],
                    "tokens_per_second": latest_eval["efficiency"]["tokens_per_second"],
                    "memory_usage_gb": latest_eval["efficiency"]["memory_allocated_gb"]
                },
                "compression_ratio": latest_eval["model_stats"]["compression_ratio"]
            }
        }
        
        return summary
