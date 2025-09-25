"""Active Learning implementation for LEAP framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ..config import TrainingConfig, RoutingConfig
from ..models import BaseMoE
from ..agents import RoutingAgent


class ActiveLearningDataset(Dataset):
    """Dataset wrapper for active learning."""
    
    def __init__(self, original_dataset: Dataset, selected_indices: List[int]):
        self.original_dataset = original_dataset
        self.selected_indices = selected_indices
    
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        return self.original_dataset[original_idx]


class UncertaintyEstimator:
    """Uncertainty estimation methods for active learning."""
    
    @staticmethod
    def entropy_uncertainty(logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy-based uncertainty."""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy.mean(dim=-1)  # Average over sequence length
    
    @staticmethod
    def max_probability_uncertainty(logits: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty as 1 - max probability."""
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        uncertainty = 1.0 - max_probs.mean(dim=-1)  # Average over sequence length
        return uncertainty
    
    @staticmethod
    def variance_uncertainty(logits: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Compute uncertainty using Monte Carlo dropout."""
        # This would require multiple forward passes with dropout
        # For simplicity, we'll use entropy as a proxy
        return UncertaintyEstimator.entropy_uncertainty(logits)
    
    @staticmethod
    def routing_uncertainty(routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty in routing decisions."""
        # Compute entropy of routing distribution
        routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1)
        return routing_entropy.mean(dim=-1)  # Average over sequence length


class DiversitySelector:
    """Methods for selecting diverse samples."""
    
    @staticmethod
    def cosine_diversity(
        embeddings: torch.Tensor,
        uncertainty_scores: torch.Tensor,
        num_samples: int,
        diversity_weight: float = 0.5
    ) -> List[int]:
        """Select samples based on uncertainty and diversity."""
        
        # Convert to numpy for sklearn
        embeddings_np = embeddings.cpu().numpy()
        uncertainty_np = uncertainty_scores.cpu().numpy()
        
        # Normalize uncertainty scores
        uncertainty_normalized = (uncertainty_np - uncertainty_np.min()) / (uncertainty_np.max() - uncertainty_np.min() + 1e-8)
        
        selected_indices = []
        remaining_indices = list(range(len(embeddings_np)))
        
        # Select first sample with highest uncertainty
        first_idx = np.argmax(uncertainty_normalized)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining samples
        for _ in range(num_samples - 1):
            if not remaining_indices:
                break
            
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Uncertainty component
                uncertainty_score = uncertainty_normalized[idx]
                
                # Diversity component (minimum cosine similarity to selected samples)
                if selected_indices:
                    selected_embeddings = embeddings_np[selected_indices]
                    current_embedding = embeddings_np[idx:idx+1]
                    
                    similarities = cosine_similarity(current_embedding, selected_embeddings)[0]
                    min_similarity = np.min(similarities)
                    diversity_score = 1.0 - min_similarity
                else:
                    diversity_score = 1.0
                
                # Combined score
                combined_score = (1 - diversity_weight) * uncertainty_score + diversity_weight * diversity_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return selected_indices
    
    @staticmethod
    def kmeans_diversity(
        embeddings: torch.Tensor,
        uncertainty_scores: torch.Tensor,
        num_samples: int
    ) -> List[int]:
        """Select samples using K-means clustering for diversity."""
        
        embeddings_np = embeddings.cpu().numpy()
        uncertainty_np = uncertainty_scores.cpu().numpy()
        
        # Perform K-means clustering
        n_clusters = min(num_samples, len(embeddings_np))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        # Select one sample from each cluster (highest uncertainty)
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # Select sample with highest uncertainty in this cluster
                cluster_uncertainties = uncertainty_np[cluster_indices]
                best_local_idx = np.argmax(cluster_uncertainties)
                best_global_idx = cluster_indices[best_local_idx]
                selected_indices.append(best_global_idx)
        
        return selected_indices


class ActiveLearningTrainer:
    """Active learning trainer for LEAP framework."""
    
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
        
        # Active learning parameters
        self.uncertainty_threshold = getattr(config, 'uncertainty_threshold', 0.1)
        self.diversity_weight = getattr(config, 'diversity_weight', 0.5)
        self.selection_method = getattr(config, 'selection_method', 'entropy_diversity')
        
        # Sample pools
        self.labeled_pool: List[int] = []
        self.unlabeled_pool: List[int] = []
        
        self.logger.info(f"Initialized Active Learning Trainer with method: {self.selection_method}")
    
    def initialize_pools(self, dataset_size: int, initial_labeled_ratio: float = 0.1):
        """Initialize labeled and unlabeled pools."""
        
        initial_labeled_size = int(dataset_size * initial_labeled_ratio)
        
        # Random initial selection
        all_indices = list(range(dataset_size))
        np.random.shuffle(all_indices)
        
        self.labeled_pool = all_indices[:initial_labeled_size]
        self.unlabeled_pool = all_indices[initial_labeled_size:]
        
        self.logger.info(f"Initialized pools: {len(self.labeled_pool)} labeled, {len(self.unlabeled_pool)} unlabeled")
    
    def select_samples(
        self,
        dataloader: DataLoader,
        num_samples: int,
        method: Optional[str] = None
    ) -> List[int]:
        """Select samples for active learning."""
        
        method = method or self.selection_method
        
        self.logger.info(f"Selecting {num_samples} samples using method: {method}")
        
        # Get uncertainty scores and embeddings for unlabeled samples
        uncertainty_scores, embeddings = self._compute_uncertainty_and_embeddings(dataloader)
        
        # Select samples based on method
        if method == 'entropy':
            selected_indices = self._select_by_entropy(uncertainty_scores, num_samples)
        elif method == 'max_prob':
            selected_indices = self._select_by_max_prob(uncertainty_scores, num_samples)
        elif method == 'entropy_diversity':
            selected_indices = DiversitySelector.cosine_diversity(
                embeddings, uncertainty_scores, num_samples, self.diversity_weight
            )
        elif method == 'kmeans_diversity':
            selected_indices = DiversitySelector.kmeans_diversity(
                embeddings, uncertainty_scores, num_samples
            )
        elif method == 'routing_uncertainty':
            routing_uncertainty = self._compute_routing_uncertainty(dataloader)
            selected_indices = self._select_by_entropy(routing_uncertainty, num_samples)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Convert to global indices (if using unlabeled pool)
        if self.unlabeled_pool:
            selected_indices = [self.unlabeled_pool[i] for i in selected_indices if i < len(self.unlabeled_pool)]
        
        self.logger.info(f"Selected {len(selected_indices)} samples for active learning")
        return selected_indices
    
    def _compute_uncertainty_and_embeddings(
        self,
        dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute uncertainty scores and embeddings for samples."""
        
        self.model.eval()
        
        uncertainty_scores = []
        embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get logits and hidden states
                logits = outputs["logits"]
                hidden_states = outputs["hidden_states"][-1]  # Last layer
                
                # Compute uncertainty
                batch_uncertainty = UncertaintyEstimator.entropy_uncertainty(logits)
                uncertainty_scores.append(batch_uncertainty)
                
                # Use mean pooled hidden states as embeddings
                if attention_mask is not None:
                    # Mask out padding tokens
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    hidden_states = hidden_states * mask_expanded
                    seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                    pooled_embeddings = hidden_states.sum(dim=1) / seq_lengths
                else:
                    pooled_embeddings = hidden_states.mean(dim=1)
                
                embeddings.append(pooled_embeddings)
        
        self.model.train()
        
        uncertainty_scores = torch.cat(uncertainty_scores, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        
        return uncertainty_scores, embeddings
    
    def _compute_routing_uncertainty(self, dataloader: DataLoader) -> torch.Tensor:
        """Compute uncertainty in routing decisions."""
        
        self.model.eval()
        routing_uncertainties = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Get hidden states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs["hidden_states"][-1]
                
                # Get routing weights
                routing_weights, _ = self.routing_agent.route_tokens(
                    hidden_states, training=False, return_routing_info=False
                )
                
                # Compute routing uncertainty
                batch_uncertainty = UncertaintyEstimator.routing_uncertainty(routing_weights)
                routing_uncertainties.append(batch_uncertainty)
        
        self.model.train()
        return torch.cat(routing_uncertainties, dim=0)
    
    def _select_by_entropy(self, uncertainty_scores: torch.Tensor, num_samples: int) -> List[int]:
        """Select samples with highest entropy."""
        _, indices = torch.topk(uncertainty_scores, min(num_samples, len(uncertainty_scores)))
        return indices.tolist()
    
    def _select_by_max_prob(self, uncertainty_scores: torch.Tensor, num_samples: int) -> List[int]:
        """Select samples with lowest max probability."""
        _, indices = torch.topk(uncertainty_scores, min(num_samples, len(uncertainty_scores)))
        return indices.tolist()
    
    def update_pools(self, selected_indices: List[int]):
        """Update labeled and unlabeled pools after sample selection."""
        
        # Move selected samples from unlabeled to labeled pool
        for idx in selected_indices:
            if idx in self.unlabeled_pool:
                self.unlabeled_pool.remove(idx)
                self.labeled_pool.append(idx)
        
        self.logger.info(f"Updated pools: {len(self.labeled_pool)} labeled, {len(self.unlabeled_pool)} unlabeled")
    
    def create_active_dataset(
        self,
        original_dataset: Dataset,
        selected_indices: List[int]
    ) -> ActiveLearningDataset:
        """Create dataset with actively selected samples."""
        return ActiveLearningDataset(original_dataset, selected_indices)
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics about current sample pools."""
        
        total_samples = len(self.labeled_pool) + len(self.unlabeled_pool)
        
        return {
            "total_samples": total_samples,
            "labeled_samples": len(self.labeled_pool),
            "unlabeled_samples": len(self.unlabeled_pool),
            "labeled_ratio": len(self.labeled_pool) / total_samples if total_samples > 0 else 0.0,
            "selection_method": self.selection_method,
            "uncertainty_threshold": self.uncertainty_threshold,
        }
    
    def adaptive_selection(
        self,
        dataloader: DataLoader,
        performance_history: List[float],
        num_samples: int
    ) -> List[int]:
        """Adaptive sample selection based on performance history."""
        
        # Analyze performance trend
        if len(performance_history) >= 3:
            recent_trend = np.mean(np.diff(performance_history[-3:]))
            
            if recent_trend > 0.01:  # Good improvement
                # Use more diverse selection
                method = 'kmeans_diversity'
                self.diversity_weight = min(0.8, self.diversity_weight + 0.1)
            elif recent_trend < -0.01:  # Performance declining
                # Focus more on uncertainty
                method = 'entropy'
                self.diversity_weight = max(0.2, self.diversity_weight - 0.1)
            else:  # Stable performance
                # Balanced approach
                method = 'entropy_diversity'
                self.diversity_weight = 0.5
        else:
            method = 'entropy_diversity'
        
        self.logger.info(f"Adaptive selection using method: {method}, diversity_weight: {self.diversity_weight:.2f}")
        
        return self.select_samples(dataloader, num_samples, method)

