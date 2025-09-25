#!/usr/bin/env python3
"""Example of adapting LEAP for a custom task."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any
import json

from leap import LEAPFramework
from leap.config import LEAPConfig, ModelConfig
from leap.utils.common import set_seed, setup_logging, get_device
from leap.evaluation.metrics import TaskMetrics


class CustomTaskDataset(Dataset):
    """Custom dataset for demonstration."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load custom data
        with open(data_file, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Example: sentiment classification
        text = item["text"]
        label = item["label"]  # 0: negative, 1: positive
        
        # Tokenize text
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class CustomTaskMetrics(TaskMetrics):
    """Custom metrics for the new task."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, float]:
        """Compute custom task metrics."""
        
        # Binary classification accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        # Precision and recall for positive class
        true_positives = ((predictions == 1) & (targets == 1)).sum().float()
        predicted_positives = (predictions == 1).sum().float()
        actual_positives = (targets == 1).sum().float()
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1_score": f1_score.item()
        }


class CustomTaskModel(nn.Module):
    """Custom model head for the new task."""
    
    def __init__(self, base_model, num_classes: int = 2):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state for classification
        hidden_states = outputs["hidden_states"][-1]  # [batch, seq_len, hidden_size]
        
        # Pool the sequence (mean pooling over non-padding tokens)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = (hidden_states * mask_expanded).sum(1)
            seq_lengths = attention_mask.sum(1, keepdim=True).float()
            pooled_output = sum_hidden / seq_lengths
        else:
            pooled_output = hidden_states.mean(1)
        
        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs["hidden_states"],
            "aux_loss": outputs.get("aux_loss", torch.tensor(0.0)),
            "z_loss": outputs.get("z_loss", torch.tensor(0.0))
        }


def create_custom_config():
    """Create configuration for custom task."""
    
    config = LEAPConfig()
    
    # Model configuration (using Llama Maverick as base)
    config.model = ModelConfig(
        model_type="llama_maverick",
        num_experts=128,
        expert_size=17_000_000_000,
        hidden_dim=8192,
        num_layers=80,
        vocab_size=128256,
        intermediate_size=28672,
        num_attention_heads=64,
        num_key_value_heads=8,
    )
    
    # Custom task settings
    config.task = "sentiment_classification"
    config.metric = "f1_score"
    
    # Pruning configuration (aggressive for this task)
    config.pruning.target_experts = 8
    config.pruning.budget_constraint = 0.0625  # 6.25%
    config.pruning.meta_episodes = 200
    
    # Training configuration
    config.training.batch_size = 16
    config.training.max_length = 512
    config.training.learning_rate = 2e-5
    config.training.total_steps = 2000
    
    # Experiment settings
    config.experiment_name = "custom_sentiment_task"
    config.seed = 42
    
    return config


def create_sample_data():
    """Create sample data for demonstration."""
    
    sample_data = [
        {"text": "I love this product! It's amazing and works perfectly.", "label": 1},
        {"text": "This is the worst thing I've ever bought. Terrible quality.", "label": 0},
        {"text": "Pretty good, but could be better. Worth the price though.", "label": 1},
        {"text": "Disappointed with this purchase. Not what I expected.", "label": 0},
        {"text": "Excellent service and fast delivery. Highly recommended!", "label": 1},
        {"text": "Poor customer service and defective product. Avoid!", "label": 0},
        {"text": "Good value for money. Does what it's supposed to do.", "label": 1},
        {"text": "Broke after one day. Complete waste of money.", "label": 0},
    ] * 50  # Repeat to have more samples
    
    return sample_data


def main():
    """Run custom task example."""
    
    print("🎯 LEAP Custom Task Example: Sentiment Classification")
    print("=" * 60)
    
    # Setup
    setup_logging(log_level="INFO")
    set_seed(42)
    device = get_device()
    
    # Create configuration
    config = create_custom_config()
    print(f"📋 Task: {config.task}")
    print(f"🎯 Target experts: {config.pruning.target_experts}/{config.model.num_experts}")
    
    # Create sample data files
    train_data = create_sample_data()
    val_data = train_data[:50]  # Use subset for validation
    
    # Save sample data
    with open("custom_train.json", "w") as f:
        json.dump(train_data, f)
    with open("custom_val.json", "w") as f:
        json.dump(val_data, f)
    
    print("📊 Created sample dataset")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    try:
        # Create datasets and dataloaders
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        train_dataset = CustomTaskDataset("custom_train.json", tokenizer, config.training.max_length)
        val_dataset = CustomTaskDataset("custom_val.json", tokenizer, config.training.max_length)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.training.batch_size, 
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config.training.batch_size, 
            shuffle=False
        )
        
        print("✅ Data loaders created")
        
        # Initialize LEAP framework
        print("\n🧠 Initializing LEAP framework...")
        leap = LEAPFramework(config, device)
        
        # Load base model
        base_model = leap.load_model()
        print(f"🏗️  Base model loaded: {base_model.__class__.__name__}")
        
        # Create custom model with classification head
        custom_model = CustomTaskModel(base_model, num_classes=2)
        custom_model = custom_model.to(device)
        
        print(f"🎯 Custom model created for {config.task}")
        print(f"   Total parameters: {sum(p.numel() for p in custom_model.parameters()):,}")
        
        # Replace the model in LEAP framework
        leap.model = custom_model
        
        # Register custom metrics
        leap.evaluator.metrics_calculator.task_metrics = CustomTaskMetrics()
        
        print("\n🔥 Starting LEAP optimization for custom task...")
        
        # For demonstration, we'll simulate the process
        print("Phase 1: Expert Pruning (Simulated)")
        print("   - Analyzing expert importance for sentiment classification")
        print("   - Selecting top 8 experts based on task performance")
        
        # Simulate expert selection
        selected_experts = [0, 5, 12, 23, 45, 67, 89, 112]  # Example selection
        custom_model.base_model.prune_experts(selected_experts)
        
        print(f"   ✅ Selected experts: {selected_experts}")
        print(f"   📉 Model size reduced by {(1 - len(selected_experts)/128)*100:.1f}%")
        
        print("\nPhase 2: Routing Adaptation (Simulated)")
        print("   - Fine-tuning routing for sentiment-specific patterns")
        print("   - Optimizing expert utilization for classification")
        
        # Simulate training
        print("   ✅ Routing adaptation completed")
        
        print("\n📊 Evaluation Results (Simulated):")
        print("   Accuracy: 0.892")
        print("   Precision: 0.885")
        print("   Recall: 0.898")
        print("   F1-Score: 0.891")
        print("   Inference speedup: 7.2x")
        print("   Memory reduction: 85%")
        
        print("\n✅ Custom task adaptation completed successfully!")
        
        # Cleanup sample files
        import os
        os.remove("custom_train.json")
        os.remove("custom_val.json")
        
    except Exception as e:
        print(f"\n❌ Error during custom task adaptation: {e}")
        print("\n💡 This is a demonstration of how to adapt LEAP for custom tasks.")
        print("   In practice, you would:")
        print("   1. Implement your custom dataset class")
        print("   2. Create appropriate model heads for your task")
        print("   3. Define task-specific metrics")
        print("   4. Configure LEAP for your specific requirements")
    
    print("\n🎉 Custom task example completed!")


if __name__ == "__main__":
    main()
