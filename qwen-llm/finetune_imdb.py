#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 IMDB SENTIMENT ANALYSIS FINE-TUNING SCRIPT

This script fine-tunes our pre-trained Qwen3 model on the IMDB sentiment analysis dataset.
It adapts the language model for binary classification (positive/negative sentiment).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import time
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Import our custom components
from config.qwen3_small_config import SmallModelConfig, set_seed
from qwen3_complete_model import MinimalLLM

@dataclass
class FineTuneConfig:
    """
    🎯 FINE-TUNING CONFIGURATION
    
    Configuration for fine-tuning the pre-trained model on IMDB sentiment analysis.
    """
    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    vocab_size: int = 50000
    
    # Fine-tuning parameters
    num_classes: int = 2  # Binary classification (positive/negative)
    max_seq_len: int = 512  # Longer sequences for reviews
    batch_size: int = 16
    learning_rate: float = 1e-4  # Lower learning rate for fine-tuning
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Technical
    use_amp: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class IMDBDataset(Dataset):
    """
    📚 IMDB DATASET CLASS
    
    Custom dataset class for IMDB sentiment analysis.
    Handles tokenization and padding of movie reviews.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
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

class SentimentClassifier(nn.Module):
    """
    🎯 SENTIMENT CLASSIFICATION MODEL
    
    This model adds a classification head on top of our pre-trained Qwen3 model.
    It freezes the pre-trained weights and only trains the classification layer.
    """
    
    def __init__(self, pretrained_model: MinimalLLM, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pretrained_model = pretrained_model
        
        # Freeze pre-trained model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pretrained_model.config.d_model, pretrained_model.config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pretrained_model.config.d_model // 2, num_classes)
        )
        
        # Initialize classification head
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the classification head"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits for classification (batch_size, num_classes)
        """
        # Get features from pre-trained model
        with torch.no_grad():
            # Get the last hidden states from the transformer blocks
            x = self.pretrained_model.token_embedding(input_ids) * math.sqrt(self.pretrained_model.config.d_model)
            x = self.pretrained_model.position_dropout(x)
            
            # Pass through transformer blocks
            for block in self.pretrained_model.transformer_blocks:
                x = block(x)
            
            # Apply final normalization
            x = self.pretrained_model.norm(x)
            
            # Use the first token representation (like [CLS] token)
            cls_representation = x[:, 0, :]  # (batch_size, d_model)
        
        # Pass through classification head
        logits = self.classifier(cls_representation)
        return logits

def load_imdb_data(config: FineTuneConfig):
    """
    📊 LOAD IMDB DATASET
    
    Loads and prepares the IMDB sentiment analysis dataset.
    """
    print("📊 Loading IMDB dataset...")
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Load tokenizer (same as used in pre-training)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, config.max_seq_len)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, config.max_seq_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"✅ Dataset loaded successfully")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    return train_loader, test_loader, tokenizer

def evaluate_model(model: SentimentClassifier, test_loader: DataLoader, config: FineTuneConfig):
    """
    📊 EVALUATE MODEL PERFORMANCE
    
    Evaluates the model on the test set and returns accuracy and loss.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    model.train()
    return accuracy, avg_loss

def fine_tune_model(config: FineTuneConfig, pretrained_model_path: str):
    """
    🎯 FINE-TUNE MODEL ON IMDB SENTIMENT ANALYSIS
    
    Fine-tunes the pre-trained model on IMDB sentiment analysis task.
    """
    print("🎯 Starting IMDB Sentiment Analysis Fine-tuning")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load data
    train_loader, test_loader, tokenizer = load_imdb_data(config)
    
    # Load pre-trained model
    print(f"\n📦 Loading pre-trained model from {pretrained_model_path}")
    checkpoint = torch.load(pretrained_model_path, map_location='cpu')
    
    # Create model config from checkpoint
    model_config = SmallModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create pre-trained model
    pretrained_model = MinimalLLM(model_config)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classification model
    model = SentimentClassifier(pretrained_model, config.num_classes, config.dropout)
    model = model.to(config.device)
    
    print(f"✅ Model loaded successfully")
    print(f"   Pre-trained parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_steps / total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    print(f"\n🚀 Starting fine-tuning for {config.num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if config.use_amp:
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
            
            scheduler.step()
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{epoch_correct/epoch_total:.3f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Epoch evaluation
        epoch_accuracy = epoch_correct / epoch_total
        epoch_avg_loss = epoch_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {epoch_avg_loss:.4f}")
        print(f"  Train Accuracy: {epoch_accuracy:.4f}")
        
        # Test evaluation
        test_accuracy, test_loss = evaluate_model(model, test_loader, config)
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    training_time = time.time() - start_time
    print(f"\n🎉 Fine-tuning completed in {training_time/60:.1f} minutes")
    
    # Final evaluation
    final_accuracy, final_loss = evaluate_model(model, test_loader, config)
    print(f"\n🏆 Final Results:")
    print(f"  Test Accuracy: {final_accuracy:.4f}")
    print(f"  Test Loss: {final_loss:.4f}")
    
    # Save fine-tuned model
    save_path = "models/imdb_sentiment_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer,
        'final_accuracy': final_accuracy,
        'final_loss': final_loss
    }, save_path)
    print(f"💾 Fine-tuned model saved to {save_path}")
    
    return model, final_accuracy

def test_sentiment_classifier(model_path: str = "models/imdb_sentiment_classifier.pt"):
    """
    🧪 TEST SENTIMENT CLASSIFIER
    
    Tests the fine-tuned sentiment classifier on sample texts.
    """
    print("🧪 Testing Sentiment Classifier")
    print("=" * 40)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    tokenizer = checkpoint['tokenizer']
    
    # Recreate model
    model_config = SmallModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    
    pretrained_model = MinimalLLM(model_config)
    model = SentimentClassifier(pretrained_model, config.num_classes, config.dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test samples
    test_samples = [
        "I absolutely loved this movie! The acting was fantastic and the plot was engaging.",
        "This was the worst film I've ever seen. Terrible acting and boring story.",
        "The movie was okay, nothing special but not bad either.",
        "Amazing cinematography and brilliant performances from all actors.",
        "Waste of time and money. I regret watching this."
    ]
    
    print(f"Model Accuracy: {checkpoint['final_accuracy']:.4f}")
    print("\nSample Predictions:")
    print("-" * 50)
    
    with torch.no_grad():
        for i, text in enumerate(test_samples, 1):
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=config.max_seq_len,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # Predict
            logits = model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            prediction = logits.argmax(dim=-1).item()
            confidence = probabilities[0][prediction].item()
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            
            print(f"{i}. Text: {text[:60]}...")
            print(f"   Sentiment: {sentiment} (confidence: {confidence:.3f})")
            print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3 on IMDB sentiment analysis')
    parser.add_argument('--pretrained', default='models/final_model1.pt', help='Path to pre-trained model')
    parser.add_argument('--test', action='store_true', help='Test the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.test:
        test_sentiment_classifier()
    else:
        # Create config
        config = FineTuneConfig()
        config.num_epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        
        # Fine-tune model
        fine_tune_model(config, args.pretrained)
