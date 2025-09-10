#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 QLORA FINE-TUNING SCRIPT

This script demonstrates how to fine-tune a pre-trained model using QLoRA
for maximum memory efficiency with 4-bit quantization + LoRA adaptation.
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
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import our custom components
from config.qwen3_small_config import SmallModelConfig, set_seed
from qwen3_complete_model import MinimalLLM
from quantization_tutorial import QLoRALayer, QLoRALinear, QLoRAManager, QuantizationConfig

@dataclass
class QLoRATrainingConfig:
    """
    🎯 QLORA TRAINING CONFIGURATION
    """
    # Model parameters
    pretrained_model_path: str = "models/final_model1.pt"
    
    # QLoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    qlora_bits: int = 4  # 4-bit quantization
    target_modules: List[str] = None
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 2e-4  # Higher LR for QLoRA
    num_epochs: int = 3
    max_seq_len: int = 256
    
    # Data parameters
    dataset_name: str = "imdb"
    num_samples: int = 1000  # For demo, use subset
    
    # Technical
    use_amp: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"]

class QLoRADataset(Dataset):
    """
    📚 QLORA DATASET CLASS
    
    Custom dataset for QLoRA fine-tuning on IMDB sentiment analysis.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
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

class QLoRAClassifier(nn.Module):
    """
    🎯 QLORA CLASSIFIER
    
    Combines pre-trained model with QLoRA adaptation and classification head.
    """
    
    def __init__(self, pretrained_model: MinimalLLM, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pretrained_model = pretrained_model
        
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
        # Get features from pre-trained model (with QLoRA)
        with torch.no_grad():
            # Get the last hidden states from the transformer blocks
            x = self.pretrained_model.token_embedding(input_ids) * math.sqrt(self.pretrained_model.config.d_model)
            x = self.pretrained_model.position_dropout(x)
            
            # Pass through transformer blocks (with QLoRA)
            for block in self.pretrained_model.transformer_blocks:
                x = block(x)
            
            # Apply final normalization
            x = self.pretrained_model.norm(x)
            
            # Use the first token representation (like [CLS] token)
            cls_representation = x[:, 0, :]  # (batch_size, d_model)
        
        # Pass through classification head
        logits = self.classifier(cls_representation)
        return logits

def load_data_for_qlora(config: QLoRATrainingConfig):
    """
    📊 LOAD DATA FOR QLORA FINE-TUNING
    """
    print("📊 Loading data for QLoRA fine-tuning...")
    
    # Load dataset
    if config.dataset_name == "imdb":
        dataset = load_dataset("imdb")
        train_texts = dataset['train']['text'][:config.num_samples]
        train_labels = dataset['train']['label'][:config.num_samples]
        test_texts = dataset['test']['text'][:config.num_samples // 5]
        test_labels = dataset['test']['label'][:config.num_samples // 5]
    else:
        raise ValueError(f"Dataset {config.dataset_name} not supported")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = QLoRADataset(train_texts, train_labels, tokenizer, config.max_seq_len)
    test_dataset = QLoRADataset(test_texts, test_labels, tokenizer, config.max_seq_len)
    
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
    
    print(f"✅ Data loaded successfully")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    return train_loader, test_loader, tokenizer

def setup_qlora_model(config: QLoRATrainingConfig, tokenizer):
    """
    🏗️ SETUP QLORA MODEL
    
    Loads pre-trained model and applies QLoRA adaptation.
    """
    print(f"🏗️ Setting up QLoRA model...")
    
    # Load pre-trained model
    print(f"📦 Loading pre-trained model from {config.pretrained_model_path}")
    checkpoint = torch.load(config.pretrained_model_path, map_location='cpu')
    
    # Create model config
    model_config = SmallModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create pre-trained model
    pretrained_model = MinimalLLM(model_config)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create QLoRA config
    qlora_config = QuantizationConfig()
    qlora_config.lora_rank = config.lora_rank
    qlora_config.lora_alpha = config.lora_alpha
    qlora_config.lora_dropout = config.lora_dropout
    qlora_config.qlora_bits = config.qlora_bits
    qlora_config.target_modules = config.target_modules
    
    # Apply QLoRA
    qlora_manager = QLoRAManager(pretrained_model, qlora_config)
    qlora_manager.apply_qlora()
    
    # Create classifier
    classifier = QLoRAClassifier(pretrained_model, num_classes=2, dropout=0.1)
    classifier = classifier.to(config.device)
    
    # Analyze memory usage
    memory_usage = qlora_manager.get_memory_usage()
    
    print(f"✅ QLoRA model setup complete")
    print(f"   Original Memory: {memory_usage['original_memory_mb']:.2f} MB")
    print(f"   Quantized Memory: {memory_usage['quantized_memory_mb']:.2f} MB")
    print(f"   LoRA Memory: {memory_usage['lora_memory_mb']:.2f} MB")
    print(f"   Total Memory: {memory_usage['total_memory_mb']:.2f} MB")
    print(f"   Memory Reduction: {memory_usage['memory_reduction']:.1f}x")
    print(f"   Trainable Parameters: {memory_usage['trainable_params']:,}")
    print(f"   Total Parameters: {memory_usage['total_params']:,}")
    
    return classifier, qlora_manager

def train_qlora_model(classifier: QLoRAClassifier, qlora_manager: QLoRAManager, 
                      train_loader: DataLoader, test_loader: DataLoader, config: QLoRATrainingConfig):
    """
    🎯 TRAIN QLORA MODEL
    
    Fine-tunes the model using QLoRA adaptation.
    """
    print(f"🎯 Starting QLoRA fine-tuning...")
    
    # Setup optimizer (only for trainable parameters)
    trainable_params = qlora_manager.get_trainable_parameters()
    trainable_params.extend([p for p in classifier.classifier.parameters()])
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        classifier.train()
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
                    logits = classifier(input_ids, attention_mask)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = classifier(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
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
        test_accuracy, test_loss = evaluate_qlora_model(classifier, test_loader, config)
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    training_time = time.time() - start_time
    print(f"\n🎉 QLoRA fine-tuning completed in {training_time/60:.1f} minutes")
    
    return classifier

def evaluate_qlora_model(classifier: QLoRAClassifier, test_loader: DataLoader, config: QLoRATrainingConfig):
    """
    📊 EVALUATE QLORA MODEL
    """
    classifier.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Forward pass
            logits = classifier(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    
    classifier.train()
    return accuracy, avg_loss

def save_qlora_model(classifier: QLoRAClassifier, qlora_manager: QLoRAManager, 
                     config: QLoRATrainingConfig, final_accuracy: float):
    """
    💾 SAVE QLORA MODEL
    
    Saves the QLoRA-adapted model and QLoRA weights separately.
    """
    print(f"💾 Saving QLoRA model...")
    
    # Save QLoRA weights
    qlora_weights = {}
    for name, qlora_layer in qlora_manager.qlora_layers.items():
        qlora_weights[name] = {
            'lora_A': qlora_layer.qlora.lora_A.data.clone(),
            'lora_B': qlora_layer.qlora.lora_B.data.clone(),
            'quantized_weights': qlora_layer.qlora.quantized_weights.clone(),
            'scale': qlora_layer.qlora.scale.clone(),
            'zero_point': qlora_layer.qlora.zero_point.clone()
        }
    
    # Save complete model
    save_path = "models/qlora_sentiment_classifier.pt"
    torch.save({
        'classifier_state_dict': classifier.state_dict(),
        'qlora_weights': qlora_weights,
        'config': config,
        'final_accuracy': final_accuracy,
        'memory_usage': qlora_manager.get_memory_usage()
    }, save_path)
    
    print(f"✅ QLoRA model saved to {save_path}")
    
    # Save QLoRA weights separately for easy loading
    qlora_save_path = "models/qlora_weights.pt"
    torch.save(qlora_weights, qlora_save_path)
    print(f"✅ QLoRA weights saved to {qlora_save_path}")

def test_qlora_model(model_path: str = "models/qlora_sentiment_classifier.pt"):
    """
    🧪 TEST QLORA MODEL
    
    Tests the QLoRA fine-tuned model on sample texts.
    """
    print("🧪 Testing QLoRA Model")
    print("=" * 40)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    memory_usage = checkpoint['memory_usage']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Recreate model
    model_config = SmallModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    
    pretrained_model = MinimalLLM(model_config)
    
    # Apply QLoRA to the model first
    qlora_config = QuantizationConfig()
    qlora_config.lora_rank = config.lora_rank
    qlora_config.lora_alpha = config.lora_alpha
    qlora_config.lora_dropout = config.lora_dropout
    qlora_config.qlora_bits = config.qlora_bits
    
    qlora_manager = QLoRAManager(pretrained_model, qlora_config)
    qlora_manager.apply_qlora()
    
    # Load QLoRA weights
    if 'qlora_weights' in checkpoint:
        for name, qlora_layer in qlora_manager.qlora_layers.items():
            if name in checkpoint['qlora_weights']:
                qlora_layer.qlora.lora_A.data = checkpoint['qlora_weights'][name]['lora_A']
                qlora_layer.qlora.lora_B.data = checkpoint['qlora_weights'][name]['lora_B']
                qlora_layer.qlora.register_buffer('quantized_weights', checkpoint['qlora_weights'][name]['quantized_weights'])
                qlora_layer.qlora.register_buffer('scale', checkpoint['qlora_weights'][name]['scale'])
                qlora_layer.qlora.register_buffer('zero_point', checkpoint['qlora_weights'][name]['zero_point'])
    
    classifier = QLoRAClassifier(pretrained_model, num_classes=2, dropout=0.1)
    
    # Load only the classifier weights (not the QLoRA weights)
    classifier_state_dict = checkpoint['classifier_state_dict']
    classifier_only_state_dict = {k: v for k, v in classifier_state_dict.items() if 'classifier' in k}
    classifier.load_state_dict(classifier_only_state_dict, strict=False)
    classifier.eval()
    
    # Test samples
    test_samples = [
        "I absolutely loved this movie! The acting was fantastic and the plot was engaging.",
        "This was the worst film I've ever seen. Terrible acting and boring story.",
        "The movie was okay, nothing special but not bad either.",
        "Amazing cinematography and brilliant performances from all actors.",
        "Waste of time and money. I regret watching this."
    ]
    
    print(f"Model Accuracy: {checkpoint['final_accuracy']:.4f}")
    print(f"Memory Reduction: {memory_usage['memory_reduction']:.1f}x")
    print(f"Original Memory: {memory_usage['original_memory_mb']:.2f} MB")
    print(f"Total Memory: {memory_usage['total_memory_mb']:.2f} MB")
    print(f"Trainable Parameters: {memory_usage['trainable_params']:,}")
    print(f"Total Parameters: {memory_usage['total_params']:,}")
    
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
            logits = classifier(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            prediction = logits.argmax(dim=-1).item()
            confidence = probabilities[0][prediction].item()
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            
            print(f"{i}. Text: {text[:60]}...")
            print(f"   Sentiment: {sentiment} (confidence: {confidence:.3f})")
            print()

def main():
    """
    🎯 MAIN QLORA FINE-TUNING FUNCTION
    """
    print("🎯 QLORA FINE-TUNING FOR SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Set seed
    set_seed(42)
    
    # Create config
    config = QLoRATrainingConfig()
    
    print(f"📋 Configuration:")
    print(f"   QLoRA Bits: {config.qlora_bits}")
    print(f"   LoRA Rank: {config.lora_rank}")
    print(f"   LoRA Alpha: {config.lora_alpha}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Target Modules: {config.target_modules}")
    
    # Load data
    train_loader, test_loader, tokenizer = load_data_for_qlora(config)
    
    # Setup model
    classifier, qlora_manager = setup_qlora_model(config, tokenizer)
    
    # Train model
    classifier = train_qlora_model(classifier, qlora_manager, train_loader, test_loader, config)
    
    # Final evaluation
    final_accuracy, final_loss = evaluate_qlora_model(classifier, test_loader, config)
    print(f"\n🏆 Final Results:")
    print(f"  Test Accuracy: {final_accuracy:.4f}")
    print(f"  Test Loss: {final_loss:.4f}")
    
    # Save model
    save_qlora_model(classifier, qlora_manager, config, final_accuracy)
    
    # Test model
    test_qlora_model()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QLoRA fine-tuning for sentiment analysis')
    parser.add_argument('--pretrained', default='models/final_model1.pt', help='Path to pre-trained model')
    parser.add_argument('--test', action='store_true', help='Test the QLoRA model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--qlora_bits', type=int, default=4, help='QLoRA quantization bits')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples')
    
    args = parser.parse_args()
    
    if args.test:
        test_qlora_model()
    else:
        # Update config with command line arguments
        config = QLoRATrainingConfig()
        config.pretrained_model_path = args.pretrained
        config.num_epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.lora_rank = args.lora_rank
        config.qlora_bits = args.qlora_bits
        config.num_samples = args.samples
        
        main()
