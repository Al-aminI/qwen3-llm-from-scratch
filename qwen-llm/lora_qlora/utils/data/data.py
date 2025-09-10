"""
Data processing utilities.

This module provides utilities for loading, preprocessing, and splitting data.
"""

import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import random


def load_data(data_path: str, format: str = 'auto') -> List[Dict[str, Any]]:
    """
    🎯 LOAD DATA
    
    Load data from file.
    
    Args:
        data_path: Path to data file
        format: Data format ('json', 'csv', 'auto')
        
    Returns:
        List of data samples
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if format == 'auto':
        format = data_path.suffix.lower().lstrip('.')
    
    if format == 'json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif format == 'csv':
        df = pd.read_csv(data_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported data format: {format}")
    
    return data


def preprocess_data(data: List[Dict[str, Any]], text_field: str = 'text', label_field: str = 'label') -> List[Dict[str, Any]]:
    """
    🎯 PREPROCESS DATA
    
    Preprocess data for training.
    
    Args:
        data: Raw data
        text_field: Name of text field
        label_field: Name of label field
        
    Returns:
        Preprocessed data
    """
    processed_data = []
    
    for sample in data:
        # Extract text and label
        text = sample.get(text_field, '')
        label = sample.get(label_field, 0)
        
        # Basic preprocessing
        if isinstance(text, str):
            text = text.strip()
            if text:  # Only include non-empty text
                processed_data.append({
                    'text': text,
                    'label': label
                })
    
    return processed_data


def split_data(data: List[Dict[str, Any]], train_split: float = 0.8, val_split: float = 0.1, test_split: float = 0.1, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    🎯 SPLIT DATA
    
    Split data into train, validation, and test sets.
    
    Args:
        data: Data to split
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    
    # Set random seed
    random.seed(random_seed)
    
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split sizes
    total_size = len(shuffled_data)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data


def balance_data(data: List[Dict[str, Any]], label_field: str = 'label', method: str = 'undersample') -> List[Dict[str, Any]]:
    """
    🎯 BALANCE DATA
    
    Balance data by class distribution.
    
    Args:
        data: Data to balance
        label_field: Name of label field
        method: Balancing method ('undersample', 'oversample')
        
    Returns:
        Balanced data
    """
    # Get label distribution
    label_counts = {}
    for sample in data:
        label = sample[label_field]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    if method == 'undersample':
        # Undersample majority class
        min_count = min(label_counts.values())
        balanced_data = []
        
        for label, count in label_counts.items():
            label_samples = [s for s in data if s[label_field] == label]
            balanced_data.extend(label_samples[:min_count])
        
        return balanced_data
    
    elif method == 'oversample':
        # Oversample minority class
        max_count = max(label_counts.values())
        balanced_data = []
        
        for label, count in label_counts.items():
            label_samples = [s for s in data if s[label_field] == label]
            # Repeat samples to reach max_count
            while len(label_samples) < max_count:
                label_samples.extend(label_samples[:max_count - len(label_samples)])
            balanced_data.extend(label_samples[:max_count])
        
        return balanced_data
    
    else:
        raise ValueError(f"Unsupported balancing method: {method}")


def filter_data(data: List[Dict[str, Any]], min_length: int = 10, max_length: int = 1000) -> List[Dict[str, Any]]:
    """
    🎯 FILTER DATA
    
    Filter data by text length.
    
    Args:
        data: Data to filter
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        Filtered data
    """
    filtered_data = []
    
    for sample in data:
        text = sample.get('text', '')
        if min_length <= len(text) <= max_length:
            filtered_data.append(sample)
    
    return filtered_data


def get_data_stats(data: List[Dict[str, Any]], label_field: str = 'label') -> Dict[str, Any]:
    """
    🎯 GET DATA STATISTICS
    
    Get statistics about the data.
    
    Args:
        data: Data to analyze
        label_field: Name of label field
        
    Returns:
        Data statistics
    """
    if not data:
        return {}
    
    # Text length statistics
    text_lengths = [len(sample.get('text', '')) for sample in data]
    
    # Label distribution
    label_counts = {}
    for sample in data:
        label = sample[label_field]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    return {
        'total_samples': len(data),
        'text_length': {
            'min': min(text_lengths),
            'max': max(text_lengths),
            'mean': sum(text_lengths) / len(text_lengths),
            'median': sorted(text_lengths)[len(text_lengths) // 2]
        },
        'label_distribution': label_counts,
        'num_classes': len(label_counts)
    }
