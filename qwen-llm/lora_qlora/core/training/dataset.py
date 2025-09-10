"""
Dataset classes for LoRA and QLoRA training.

This module provides dataset classes for loading and preprocessing data,
following the original fine-tuning approach.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
from datasets import load_dataset


class LoRADataset(Dataset):
    """
    📚 LORA DATASET CLASS
    
    Custom dataset for LoRA fine-tuning on IMDB sentiment analysis.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        """
        Initialize LoRA dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    @classmethod
    def from_imdb(cls, tokenizer: AutoTokenizer, max_length: int = 256, num_samples: int = 1000):
        """
        Create LoRA dataset from IMDB dataset.
        
        Args:
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            num_samples: Number of samples to use
            
        Returns:
            LoRADataset instance
        """
        print("📊 Loading IMDB dataset for LoRA...")
        
        # Load dataset
        dataset = load_dataset("imdb")
        
        # Take subset for training
        train_texts = dataset['train']['text'][:num_samples]
        train_labels = dataset['train']['label'][:num_samples]
        
        print(f"✅ IMDB dataset loaded: {len(train_texts)} samples")
        
        return cls(train_texts, train_labels, tokenizer, max_length)
    
    @classmethod
    def from_file(cls, file_path: str, tokenizer: AutoTokenizer, max_length: int = 256):
        """
        Create dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            
        Returns:
            LoRADataset instance
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        return cls(texts, labels, tokenizer, max_length)


class QLoRADataset(Dataset):
    """
    📚 QLORA DATASET CLASS
    
    Custom dataset for QLoRA fine-tuning on IMDB sentiment analysis.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 256):
        """
        Initialize QLoRA dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    @classmethod
    def from_imdb(cls, tokenizer: AutoTokenizer, max_length: int = 256, num_samples: int = 1000):
        """
        Create QLoRA dataset from IMDB dataset.
        
        Args:
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            num_samples: Number of samples to use
            
        Returns:
            QLoRADataset instance
        """
        print("📊 Loading IMDB dataset for QLoRA...")
        
        # Load dataset
        dataset = load_dataset("imdb")
        
        # Take subset for training
        train_texts = dataset['train']['text'][:num_samples]
        train_labels = dataset['train']['label'][:num_samples]
        
        print(f"✅ IMDB dataset loaded: {len(train_texts)} samples")
        
        return cls(train_texts, train_labels, tokenizer, max_length)
    
    @classmethod
    def from_file(cls, file_path: str, tokenizer: AutoTokenizer, max_length: int = 256):
        """
        Create dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            
        Returns:
            QLoRADataset instance
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        return cls(texts, labels, tokenizer, max_length)


def load_data_for_lora(tokenizer: AutoTokenizer, max_length: int = 256, num_samples: int = 1000):
    """
    📊 LOAD DATA FOR LORA FINE-TUNING
    
    Loads IMDB dataset for LoRA fine-tuning.
    
    Args:
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length
        num_samples: Number of samples to use
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    print("📊 Loading data for LoRA fine-tuning...")
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Take subset for training and testing
    train_texts = dataset['train']['text'][:num_samples]
    train_labels = dataset['train']['label'][:num_samples]
    test_texts = dataset['test']['text'][:num_samples // 5]
    test_labels = dataset['test']['label'][:num_samples // 5]
    
    # Create datasets
    train_dataset = LoRADataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = LoRADataset(test_texts, test_labels, tokenizer, max_length)
    
    print(f"✅ Data loaded successfully")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def load_data_for_qlora(tokenizer: AutoTokenizer, max_length: int = 256, num_samples: int = 1000):
    """
    📊 LOAD DATA FOR QLORA FINE-TUNING
    
    Loads IMDB dataset for QLoRA fine-tuning.
    
    Args:
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length
        num_samples: Number of samples to use
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    print("📊 Loading data for QLoRA fine-tuning...")
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Take subset for training and testing
    train_texts = dataset['train']['text'][:num_samples]
    train_labels = dataset['train']['label'][:num_samples]
    test_texts = dataset['test']['text'][:num_samples // 5]
    test_labels = dataset['test']['label'][:num_samples // 5]
    
    # Create datasets
    train_dataset = QLoRADataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = QLoRADataset(test_texts, test_labels, tokenizer, max_length)
    
    print(f"✅ Data loaded successfully")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset