"""Data loading utilities for LEAP."""

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import numpy as np


class LEAPDataset(Dataset):
    """Custom dataset for LEAP training."""
    
    def __init__(
        self,
        data: Union[List[Dict], HFDataset],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        task: str = "language_modeling"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.task == "code_generation":
            return self._process_code_generation(item)
        elif self.task == "reasoning":
            return self._process_reasoning(item)
        elif self.task == "summarization":
            return self._process_summarization(item)
        else:
            return self._process_language_modeling(item)
    
    def _process_language_modeling(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process item for language modeling."""
        
        text = item.get("text", "")
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        # For language modeling, labels are the same as input_ids
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _process_code_generation(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process item for code generation."""
        
        prompt = item.get("prompt", "")
        solution = item.get("solution", "")
        
        # Combine prompt and solution
        full_text = f"{prompt}\n{solution}"
        
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        # For code generation, we want to predict the solution part
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        prompt_length = len(prompt_tokens)
        
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Don't compute loss on prompt
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _process_reasoning(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process item for mathematical reasoning."""
        
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        # Format as question-answer pair
        full_text = f"Question: {question}\nAnswer: {answer}"
        
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        # Find where the answer starts
        answer_start_token = self.tokenizer.encode("Answer:", add_special_tokens=False)[0]
        answer_start_idx = None
        for i, token_id in enumerate(input_ids):
            if token_id == answer_start_token:
                answer_start_idx = i
                break
        
        labels = input_ids.clone()
        if answer_start_idx is not None:
            labels[:answer_start_idx] = -100  # Don't compute loss on question
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _process_summarization(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process item for summarization."""
        
        document = item.get("document", "")
        summary = item.get("summary", "")
        
        # Format as document-summary pair
        full_text = f"Document: {document}\nSummary: {summary}"
        
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        # Find where summary starts
        summary_start_token = self.tokenizer.encode("Summary:", add_special_tokens=False)[0]
        summary_start_idx = None
        for i, token_id in enumerate(input_ids):
            if token_id == summary_start_token:
                summary_start_idx = i
                break
        
        labels = input_ids.clone()
        if summary_start_idx is not None:
            labels[:summary_start_idx] = -100  # Don't compute loss on document
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def load_dataset_for_task(
    task: str,
    dataset_name: Optional[str] = None,
    split: str = "train",
    num_samples: Optional[int] = None
) -> HFDataset:
    """Load dataset for specific task."""
    
    if task == "code_generation":
        if dataset_name is None:
            dataset_name = "openai_humaneval"
        dataset = load_dataset(dataset_name, split=split)
        
    elif task == "reasoning":
        if dataset_name is None:
            dataset_name = "gsm8k"
        dataset = load_dataset(dataset_name, "main", split=split)
        
    elif task == "summarization":
        if dataset_name is None:
            dataset_name = "xsum"
        dataset = load_dataset(dataset_name, split=split)
        
    else:
        # Default to a language modeling dataset
        if dataset_name is None:
            dataset_name = "wikitext"
        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = dataset.select(indices)
    
    return dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    **kwargs
) -> DataLoader:
    """Create DataLoader with appropriate settings."""
    
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # DistributedSampler handles shuffling
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=kwargs.get("collate_fn", None)
    )
    
    return dataloader


def create_leap_dataloaders(
    task: str,
    tokenizer_name: str,
    batch_size: int = 8,
    max_length: int = 2048,
    train_dataset_name: Optional[str] = None,
    val_dataset_name: Optional[str] = None,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    num_workers: int = 0,
    distributed: bool = False
) -> Dict[str, DataLoader]:
    """Create train and validation dataloaders for LEAP."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_hf_dataset = load_dataset_for_task(
        task, train_dataset_name, "train", num_train_samples
    )
    
    val_hf_dataset = load_dataset_for_task(
        task, val_dataset_name or train_dataset_name, "validation", num_val_samples
    )
    
    # Create LEAP datasets
    train_dataset = LEAPDataset(train_hf_dataset, tokenizer, max_length, task)
    val_dataset = LEAPDataset(val_hf_dataset, tokenizer, max_length, task)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
        drop_last=True
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        distributed=distributed,
        drop_last=False
    )
    
    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "tokenizer": tokenizer
    }


class DataCollator:
    """Custom data collator for LEAP."""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        
        # Stack tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_dataset_info(task: str) -> Dict[str, Any]:
    """Get information about datasets for specific tasks."""
    
    dataset_info = {
        "code_generation": {
            "default_dataset": "openai_humaneval",
            "metric": "pass@1",
            "description": "Code generation from natural language descriptions",
            "input_format": "prompt -> code",
            "evaluation": "Functional correctness"
        },
        "reasoning": {
            "default_dataset": "gsm8k",
            "metric": "accuracy",
            "description": "Grade school math word problems",
            "input_format": "question -> answer",
            "evaluation": "Exact match accuracy"
        },
        "summarization": {
            "default_dataset": "xsum",
            "metric": "rouge_l",
            "description": "Abstractive summarization",
            "input_format": "document -> summary",
            "evaluation": "ROUGE scores"
        },
        "language_modeling": {
            "default_dataset": "wikitext",
            "metric": "perplexity",
            "description": "General language modeling",
            "input_format": "text -> next token prediction",
            "evaluation": "Perplexity"
        }
    }
    
    return dataset_info.get(task, dataset_info["language_modeling"])


def preprocess_dataset(
    dataset: HFDataset,
    task: str,
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    num_proc: int = 1
) -> HFDataset:
    """Preprocess dataset for efficient loading."""
    
    def tokenize_function(examples):
        if task == "code_generation":
            texts = [f"{prompt}\n{solution}" for prompt, solution in zip(examples["prompt"], examples["solution"])]
        elif task == "reasoning":
            texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples["question"], examples["answer"])]
        elif task == "summarization":
            texts = [f"Document: {doc}\nSummary: {sum}" for doc, sum in zip(examples["document"], examples["summary"])]
        else:
            texts = examples["text"]
        
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset
