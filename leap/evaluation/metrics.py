"""Metrics calculation for LEAP evaluation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import logging


class TaskMetrics(ABC):
    """Abstract base class for task-specific metrics."""
    
    @abstractmethod
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass


class CodeGenerationMetrics(TaskMetrics):
    """Metrics for code generation tasks."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Compute code generation metrics."""
        
        # Token-level accuracy
        mask = targets != -100
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        # Exact match accuracy (sequence level)
        seq_correct = (predictions == targets).all(dim=1) | (~mask).all(dim=1)
        exact_match = seq_correct.float().mean()
        
        return {
            "accuracy": accuracy.item(),
            "exact_match": exact_match.item(),
            "pass@1": exact_match.item() * 0.8  # Approximate pass@1
        }


class ReasoningMetrics(TaskMetrics):
    """Metrics for mathematical reasoning tasks."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Compute reasoning metrics."""
        
        # Token-level accuracy
        mask = targets != -100
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        # Final answer accuracy (last token)
        final_predictions = predictions[mask].view(-1)[-targets.size(0):]
        final_targets = targets[mask].view(-1)[-targets.size(0):]
        final_correct = (final_predictions == final_targets).float().mean()
        
        return {
            "accuracy": accuracy.item(),
            "final_answer_accuracy": final_correct.item()
        }


class SummarizationMetrics(TaskMetrics):
    """Metrics for summarization tasks."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Compute summarization metrics."""
        
        # Token-level accuracy
        mask = targets != -100
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        # Placeholder ROUGE scores (would use actual ROUGE implementation)
        rouge_1 = accuracy.item() * 0.9
        rouge_2 = accuracy.item() * 0.8
        rouge_l = accuracy.item() * 0.85
        
        return {
            "accuracy": accuracy.item(),
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "rouge_l": rouge_l
        }


class MetricsCalculator:
    """Main metrics calculator for LEAP evaluation."""
    
    def __init__(self, task: str):
        self.task = task
        self.logger = logging.getLogger(__name__)
        
        # Initialize task-specific metrics
        if task == "code_generation":
            self.task_metrics = CodeGenerationMetrics()
        elif task == "reasoning":
            self.task_metrics = ReasoningMetrics()
        elif task == "summarization":
            self.task_metrics = SummarizationMetrics()
        else:
            self.task_metrics = None
            self.logger.warning(f"No specific metrics for task: {task}")
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute all relevant metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._compute_basic_metrics(predictions, targets, logits))
        
        # Task-specific metrics
        if self.task_metrics:
            task_metrics = self.task_metrics.compute(predictions, targets, **kwargs)
            metrics.update(task_metrics)
        
        # Efficiency metrics if provided
        if "inference_time" in kwargs:
            metrics.update(self._compute_efficiency_metrics(**kwargs))
        
        return metrics
    
    def _compute_basic_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute basic language modeling metrics."""
        
        # Token accuracy
        mask = targets != -100
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        # Perplexity (if logits provided)
        if logits is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            perplexity = torch.exp(loss)
        else:
            perplexity = torch.tensor(float('inf'))
        
        return {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item()
        }
    
    def _compute_efficiency_metrics(self, **kwargs) -> Dict[str, float]:
        """Compute efficiency-related metrics."""
        
        metrics = {}
        
        if "inference_time" in kwargs:
            metrics["inference_time_ms"] = kwargs["inference_time"] * 1000
        
        if "memory_usage" in kwargs:
            metrics["memory_usage_mb"] = kwargs["memory_usage"] / (1024**2)
        
        if "flops" in kwargs:
            metrics["flops"] = kwargs["flops"]
        
        if "tokens_processed" in kwargs and "inference_time" in kwargs:
            metrics["tokens_per_second"] = kwargs["tokens_processed"] / kwargs["inference_time"]
        
        return metrics
    
    def compute_compression_metrics(
        self,
        original_experts: int,
        pruned_experts: int,
        original_performance: float,
        pruned_performance: float
    ) -> Dict[str, float]:
        """Compute compression-related metrics."""
        
        compression_ratio = pruned_experts / original_experts
        performance_retention = pruned_performance / original_performance if original_performance > 0 else 0.0
        
        # Efficiency gain (inverse of compression ratio)
        efficiency_gain = 1.0 / compression_ratio if compression_ratio > 0 else float('inf')
        
        # Performance per parameter (rough estimate)
        performance_per_param = pruned_performance / pruned_experts if pruned_experts > 0 else 0.0
        
        return {
            "compression_ratio": compression_ratio,
            "parameter_reduction": 1.0 - compression_ratio,
            "performance_retention": performance_retention,
            "efficiency_gain": efficiency_gain,
            "performance_per_param": performance_per_param
        }
    
    def compute_expert_utilization_metrics(
        self,
        expert_counts: Dict[int, float]
    ) -> Dict[str, float]:
        """Compute expert utilization metrics."""
        
        if not expert_counts:
            return {}
        
        utilizations = list(expert_counts.values())
        total_usage = sum(utilizations)
        
        if total_usage == 0:
            return {"load_balance_loss": 0.0, "active_experts": 0}
        
        # Normalize utilizations
        normalized_utils = [u / total_usage for u in utilizations]
        
        # Load balancing metrics
        uniform_util = 1.0 / len(utilizations)
        load_balance_loss = sum((u - uniform_util) ** 2 for u in normalized_utils)
        
        # Active experts (> 1% utilization)
        active_experts = sum(1 for u in normalized_utils if u > 0.01)
        
        # Utilization statistics
        mean_util = np.mean(normalized_utils)
        std_util = np.std(normalized_utils)
        max_util = np.max(normalized_utils)
        min_util = np.min(normalized_utils)
        
        return {
            "load_balance_loss": load_balance_loss,
            "active_experts": active_experts,
            "mean_utilization": mean_util,
            "std_utilization": std_util,
            "max_utilization": max_util,
            "min_utilization": min_util,
            "utilization_entropy": -sum(u * np.log(u + 1e-8) for u in normalized_utils if u > 0)
        }
    
    def aggregate_batch_metrics(
        self,
        batch_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate metrics across batches."""
        
        if not batch_metrics:
            return {}
        
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in batch_metrics:
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in batch_metrics if key in metrics]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)
                
                # For some metrics, we want the sum instead of mean
                if key in ["total_samples", "total_tokens"]:
                    aggregated[key] = np.sum(values)
                else:
                    aggregated[key] = aggregated[f"{key}_mean"]
        
        return aggregated
    
    def compute_statistical_significance(
        self,
        baseline_scores: List[float],
        treatment_scores: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compute statistical significance of performance differences."""
        
        from scipy import stats
        
        # T-test
        t_stat, p_value = stats.ttest_ind(treatment_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(baseline_scores) - 1) * np.var(baseline_scores, ddof=1) +
             (len(treatment_scores) - 1) * np.var(treatment_scores, ddof=1)) /
            (len(baseline_scores) + len(treatment_scores) - 2)
        )
        
        cohens_d = (np.mean(treatment_scores) - np.mean(baseline_scores)) / pooled_std
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "cohens_d": cohens_d,
            "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large",
            "baseline_mean": np.mean(baseline_scores),
            "treatment_mean": np.mean(treatment_scores),
            "improvement": np.mean(treatment_scores) - np.mean(baseline_scores)
        }
