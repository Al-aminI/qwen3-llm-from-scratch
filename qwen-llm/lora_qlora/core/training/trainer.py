"""
Trainer classes for LoRA and QLoRA.

This module provides trainer classes for fine-tuning with LoRA and QLoRA,
following the original fine-tuning approach without HuggingFace dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import math
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import sys
import os

# Add parent directories to path to import your custom model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from config.qwen3_small_config import SmallModelConfig, set_seed
from qwen3_complete_model import MinimalLLM
from transformers import AutoTokenizer

from .config import LoRATrainingConfig, QLoRATrainingConfig
from .dataset import LoRADataset, QLoRADataset
from ..lora.lora_manager import LoRAManager
from ..qlora.qlora_manager import QLoRAManager
from ..quantization.quantization_expert import QuantizationConfig


class LoRAClassifier(nn.Module):
    """
    🎯 LORA CLASSIFIER
    
    Combines pre-trained model with LoRA adaptation and classification head.
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
        # Get features from pre-trained model (with LoRA)
        with torch.no_grad():
            # Get the last hidden states from the transformer blocks
            x = self.pretrained_model.token_embedding(input_ids) * math.sqrt(self.pretrained_model.config.d_model)
            x = self.pretrained_model.position_dropout(x)
            
            # Pass through transformer blocks (with LoRA)
            for block in self.pretrained_model.transformer_blocks:
                x = block(x)
            
            # Apply final normalization
            x = self.pretrained_model.norm(x)
            
            # Use the first token representation (like [CLS] token)
            cls_representation = x[:, 0, :]  # (batch_size, d_model)
        
        # Pass through classification head
        logits = self.classifier(cls_representation)
        return logits


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


class LoRATrainer:
    """
    🎯 LORA TRAINER
    
    Trainer class for LoRA fine-tuning following the original approach.
    """
    
    def __init__(self, config: LoRATrainingConfig):
        """
        Initialize LoRA trainer.
        
        Args:
            config: LoRA training configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.lora_manager = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup_model(self):
        """Setup model and tokenizer."""
        print("🎯 Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model config
        model_config = SmallModelConfig()
        model_config.vocab_size = self.tokenizer.vocab_size
        
        # Load your custom model
        if self.config.model_path and os.path.exists(self.config.model_path):
            print(f"📦 Loading pre-trained model from {self.config.model_path}")
            checkpoint = torch.load(self.config.model_path, map_location='cpu')
            self.base_model = MinimalLLM(model_config)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("🏗️ Creating new model from scratch")
            self.base_model = MinimalLLM(model_config)
        
        # Setup LoRA
        lora_config = QuantizationConfig(
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )
        
        self.lora_manager = LoRAManager(self.base_model, lora_config)
        self.lora_manager.apply_lora()
        
        # Create classifier
        self.model = LoRAClassifier(self.base_model, num_classes=2, dropout=0.1)
        self.model = self.model.to(self.config.device)
        
        # Analyze parameters
        param_counts = self.lora_manager.get_parameter_count()
        
        print(f"✅ LoRA model setup complete")
        print(f"   Total parameters: {param_counts['total']:,}")
        print(f"   Trainable parameters: {param_counts['trainable']:,}")
        print(f"   Frozen parameters: {param_counts['frozen']:,}")
        print(f"   Trainable percentage: {param_counts['trainable_percentage']:.2f}%")
    
    def load_data(self):
        """Load and prepare data."""
        print("🎯 Loading data...")
        
        # Load full dataset
        full_dataset = LoRADataset.from_file(
            self.config.data_path,
            self.tokenizer,
            self.config.max_length
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        test_size = total_size - train_size - val_size
        
        # Create splits
        train_data = full_dataset.data[:train_size]
        val_data = full_dataset.data[train_size:train_size + val_size]
        test_data = full_dataset.data[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = LoRADataset(train_data, self.tokenizer, self.config.max_length)
        self.val_dataset = LoRADataset(val_data, self.tokenizer, self.config.max_length)
        self.test_dataset = LoRADataset(test_data, self.tokenizer, self.config.max_length)
        
        print(f"✅ Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
    
    def train(self):
        """Train the model using the original approach."""
        print("🎯 Starting LoRA fine-tuning...")
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = self.lora_manager.get_trainable_parameters()
        trainable_params.extend([p for p in self.model.classifier.parameters()])
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_dataset) // self.config.batch_size * self.config.num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_steps / total_steps
        )
        
        # Mixed precision scaler
        scaler = GradScaler() if self.config.mixed_precision else None
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.dataloader_num_workers
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.dataloader_num_workers
        )
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.dataloader_num_workers
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss = F.cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
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
            
            # Validation evaluation
            val_accuracy, val_loss = self._evaluate_model(val_loader)
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
        
        training_time = time.time() - start_time
        print(f"\n🎉 LoRA fine-tuning completed in {training_time/60:.1f} minutes")
        
        # Final test evaluation
        test_accuracy, test_loss = self._evaluate_model(test_loader)
        print(f"\n🏆 Final Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    def _evaluate_model(self, data_loader: DataLoader):
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                
                # Calculate accuracy
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        
        self.model.train()
        return accuracy, avg_loss
    
    def save_model(self, path: str):
        """Save the trained model."""
        print(f"🎯 Saving model to {path}...")
        
        # Save LoRA weights
        lora_weights = {}
        for name, lora_layer in self.lora_manager.lora_layers.items():
            lora_weights[name] = {
                'lora_A': lora_layer.lora.lora_A.data.clone(),
                'lora_B': lora_layer.lora.lora_B.data.clone()
            }
        
        # Save complete model
        torch.save({
            'classifier_state_dict': self.model.state_dict(),
            'lora_weights': lora_weights,
            'config': self.config,
            'parameter_counts': self.lora_manager.get_parameter_count()
        }, f"{path}/lora_sentiment_classifier.pt")
        
        # Save LoRA weights separately
        torch.save(lora_weights, f"{path}/lora_weights.pt")
        
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        print(f"🎯 Loading model from {path}...")
        
        # Load LoRA weights
        lora_weights = torch.load(f"{path}/lora_weights.pt")
        self.lora_manager.load_lora_weights(lora_weights)
        
        print("✅ Model loaded")


class QLoRATrainer:
    """
    🎯 QLORA TRAINER
    
    Trainer class for QLoRA fine-tuning following the original approach.
    """
    
    def __init__(self, config: QLoRATrainingConfig):
        """
        Initialize QLoRA trainer.
        
        Args:
            config: QLoRA training configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.qlora_manager = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup_model(self):
        """Setup model and tokenizer."""
        print("🎯 Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model config
        model_config = SmallModelConfig()
        model_config.vocab_size = self.tokenizer.vocab_size
        
        # Load your custom model
        if self.config.model_path and os.path.exists(self.config.model_path):
            print(f"📦 Loading pre-trained model from {self.config.model_path}")
            checkpoint = torch.load(self.config.model_path, map_location='cpu')
            self.base_model = MinimalLLM(model_config)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("🏗️ Creating new model from scratch")
            self.base_model = MinimalLLM(model_config)
        
        # Setup QLoRA
        qlora_config = QuantizationConfig(
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )
        
        self.qlora_manager = QLoRAManager(self.base_model, qlora_config)
        self.qlora_manager.apply_qlora()
        
        # Create classifier
        self.model = QLoRAClassifier(self.base_model, num_classes=2, dropout=0.1)
        self.model = self.model.to(self.config.device)
        
        # Analyze memory usage
        memory_usage = self.qlora_manager.get_memory_usage()
        
        print(f"✅ QLoRA model setup complete")
        print(f"   Original Memory: {memory_usage['original_memory_mb']:.2f} MB")
        print(f"   Quantized Memory: {memory_usage['quantized_memory_mb']:.2f} MB")
        print(f"   LoRA Memory: {memory_usage['lora_memory_mb']:.2f} MB")
        print(f"   Total Memory: {memory_usage['total_memory_mb']:.2f} MB")
        print(f"   Memory Reduction: {memory_usage['memory_reduction']:.1f}x")
        print(f"   Trainable Parameters: {memory_usage['trainable_params']:,}")
        print(f"   Total Parameters: {memory_usage['total_params']:,}")
    
    def load_data(self):
        """Load and prepare data."""
        print("🎯 Loading data...")
        
        # Load full dataset
        full_dataset = QLoRADataset.from_file(
            self.config.data_path,
            self.tokenizer,
            self.config.max_length
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * self.config.train_split)
        val_size = int(total_size * self.config.val_split)
        test_size = total_size - train_size - val_size
        
        # Create splits
        train_data = full_dataset.data[:train_size]
        val_data = full_dataset.data[train_size:train_size + val_size]
        test_data = full_dataset.data[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = QLoRADataset(train_data, self.tokenizer, self.config.max_length)
        self.val_dataset = QLoRADataset(val_data, self.tokenizer, self.config.max_length)
        self.test_dataset = QLoRADataset(test_data, self.tokenizer, self.config.max_length)
        
        print(f"✅ Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
    
    def train(self):
        """Train the model using the original approach."""
        print("🎯 Starting QLoRA fine-tuning...")
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = self.qlora_manager.get_trainable_parameters()
        trainable_params.extend([p for p in self.model.classifier.parameters()])
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_dataset) // self.config.batch_size * self.config.num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_steps / total_steps
        )
        
        # Mixed precision scaler
        scaler = GradScaler() if self.config.mixed_precision else None
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.dataloader_num_workers
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.dataloader_num_workers
        )
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.dataloader_num_workers
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss = F.cross_entropy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
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
            
            # Validation evaluation
            val_accuracy, val_loss = self._evaluate_model(val_loader)
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
        
        training_time = time.time() - start_time
        print(f"\n🎉 QLoRA fine-tuning completed in {training_time/60:.1f} minutes")
        
        # Final test evaluation
        test_accuracy, test_loss = self._evaluate_model(test_loader)
        print(f"\n🏆 Final Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    def _evaluate_model(self, data_loader: DataLoader):
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                
                # Calculate accuracy
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        
        self.model.train()
        return accuracy, avg_loss
    
    def save_model(self, path: str):
        """Save the trained model."""
        print(f"🎯 Saving model to {path}...")
        
        # Save QLoRA weights
        qlora_weights = {}
        for name, qlora_layer in self.qlora_manager.qlora_layers.items():
            qlora_weights[name] = {
                'lora_A': qlora_layer.qlora.lora_A.data.clone(),
                'lora_B': qlora_layer.qlora.lora_B.data.clone(),
                'quantized_weights': qlora_layer.qlora.quantized_weights.clone(),
                'scale': qlora_layer.qlora.scale.clone(),
                'zero_point': qlora_layer.qlora.zero_point.clone()
            }
        
        # Save complete model
        torch.save({
            'classifier_state_dict': self.model.state_dict(),
            'qlora_weights': qlora_weights,
            'config': self.config,
            'memory_usage': self.qlora_manager.get_memory_usage()
        }, f"{path}/qlora_sentiment_classifier.pt")
        
        # Save QLoRA weights separately
        torch.save(qlora_weights, f"{path}/qlora_weights.pt")
        
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        print(f"🎯 Loading model from {path}...")
        
        # Load QLoRA weights
        qlora_weights = torch.load(f"{path}/qlora_weights.pt")
        self.qlora_manager.load_qlora_weights(qlora_weights)
        
        print("✅ Model loaded")