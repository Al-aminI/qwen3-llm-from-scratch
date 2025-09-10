"""
Universal trainer classes for LoRA and QLoRA.

This module provides trainer classes that can work with any base model,
not just the custom MinimalLLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import math
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Union
import sys
import os

# Add parent directories to path to import your custom model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from config.qwen3_small_config import SmallModelConfig, set_seed
from qwen3_complete_model import MinimalLLM
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from .config import LoRATrainingConfig, QLoRATrainingConfig
from .dataset import LoRADataset, QLoRADataset
from ..lora.lora_manager import LoRAManager
from ..qlora.qlora_manager import QLoRAManager
from ..quantization.quantization_expert import QuantizationConfig


class UniversalLoRAClassifier(nn.Module):
    """
    🎯 UNIVERSAL LORA CLASSIFIER
    
    Works with any base model (HuggingFace or custom).
    """
    
    def __init__(self, base_model: nn.Module, num_classes: int = 2, dropout: float = 0.1, 
                 model_type: str = "custom", hidden_size: Optional[int] = None):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        self.num_classes = num_classes
        
        # Determine hidden size
        if hidden_size is None:
            if hasattr(base_model, 'config'):
                if hasattr(base_model.config, 'd_model'):
                    hidden_size = base_model.config.d_model
                elif hasattr(base_model.config, 'hidden_size'):
                    hidden_size = base_model.config.hidden_size
                else:
                    hidden_size = 768  # Default
            else:
                hidden_size = 768  # Default
        
        self.hidden_size = hidden_size
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
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
        if self.model_type == "huggingface":
            # For HuggingFace models
            with torch.no_grad():
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                # Use the last hidden state of the first token (like [CLS])
                cls_representation = outputs.last_hidden_state[:, 0, :]
        else:
            # For custom models (like your MinimalLLM)
            with torch.no_grad():
                # Get embeddings
                x = self.base_model.token_embedding(input_ids) * math.sqrt(self.base_model.config.d_model)
                x = self.base_model.position_dropout(x)
                
                # Pass through transformer blocks
                for block in self.base_model.transformer_blocks:
                    x = block(x)
                
                # Apply final normalization
                x = self.base_model.norm(x)
                
                # Use the first token representation
                cls_representation = x[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(cls_representation)
        return logits


class UniversalQLoRAClassifier(nn.Module):
    """
    🎯 UNIVERSAL QLORA CLASSIFIER
    
    Works with any base model (HuggingFace or custom).
    """
    
    def __init__(self, base_model: nn.Module, num_classes: int = 2, dropout: float = 0.1, 
                 model_type: str = "custom", hidden_size: Optional[int] = None):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        self.num_classes = num_classes
        
        # Determine hidden size
        if hidden_size is None:
            if hasattr(base_model, 'config'):
                if hasattr(base_model.config, 'd_model'):
                    hidden_size = base_model.config.d_model
                elif hasattr(base_model.config, 'hidden_size'):
                    hidden_size = base_model.config.hidden_size
                else:
                    hidden_size = 768  # Default
            else:
                hidden_size = 768  # Default
        
        self.hidden_size = hidden_size
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
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
        if self.model_type == "huggingface":
            # For HuggingFace models
            with torch.no_grad():
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                # Use the last hidden state of the first token (like [CLS])
                cls_representation = outputs.last_hidden_state[:, 0, :]
        else:
            # For custom models (like your MinimalLLM)
            with torch.no_grad():
                # Get embeddings
                x = self.base_model.token_embedding(input_ids) * math.sqrt(self.base_model.config.d_model)
                x = self.base_model.position_dropout(x)
                
                # Pass through transformer blocks
                for block in self.base_model.transformer_blocks:
                    x = block(x)
                
                # Apply final normalization
                x = self.base_model.norm(x)
                
                # Use the first token representation
                cls_representation = x[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(cls_representation)
        return logits


class UniversalLoRATrainer:
    """
    🎯 UNIVERSAL LORA TRAINER
    
    Can work with any base model (HuggingFace or custom).
    """
    
    def __init__(self, config: LoRATrainingConfig):
        """
        Initialize Universal LoRA trainer.
        
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
        self.model_type = "custom"  # Default to custom
    
    def setup_model(self, model_name_or_path: Optional[str] = None, model_type: str = "custom"):
        """
        Setup model and tokenizer.
        
        Args:
            model_name_or_path: HuggingFace model name or path to custom model
            model_type: "huggingface" or "custom"
        """
        print("🎯 Setting up model and tokenizer...")
        
        self.model_type = model_type
        
        # Load tokenizer
        if model_name_or_path and model_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        if model_type == "huggingface" and model_name_or_path:
            print(f"📦 Loading HuggingFace model: {model_name_or_path}")
            self.base_model = AutoModel.from_pretrained(model_name_or_path)
            hidden_size = self.base_model.config.hidden_size
        else:
            # Load your custom model
            if self.config.pretrained_model_path and os.path.exists(self.config.pretrained_model_path):
                print(f"📦 Loading pre-trained model from {self.config.pretrained_model_path}")
                checkpoint = torch.load(self.config.pretrained_model_path, map_location='cpu')
                model_config = SmallModelConfig()
                model_config.vocab_size = self.tokenizer.vocab_size
                self.base_model = MinimalLLM(model_config)
                self.base_model.load_state_dict(checkpoint['model_state_dict'])
                hidden_size = self.base_model.config.d_model
            else:
                print("🏗️ Creating new model from scratch")
                model_config = SmallModelConfig()
                model_config.vocab_size = self.tokenizer.vocab_size
                self.base_model = MinimalLLM(model_config)
                hidden_size = self.base_model.config.d_model
        
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
        self.model = UniversalLoRAClassifier(
            self.base_model, 
            num_classes=2, 
            dropout=0.1,
            model_type=model_type,
            hidden_size=hidden_size
        )
        self.model = self.model.to(self.config.device)
        
        # Analyze parameters
        param_counts = self.lora_manager.get_parameter_count()
        
        print(f"✅ Universal LoRA model setup complete")
        print(f"   Model type: {model_type}")
        print(f"   Total parameters: {param_counts['total']:,}")
        print(f"   Trainable parameters: {param_counts['trainable']:,}")
        print(f"   Frozen parameters: {param_counts['frozen']:,}")
        print(f"   Trainable percentage: {param_counts['trainable_percentage']:.2f}%")
    
    def load_data(self):
        """Load and prepare data."""
        print("🎯 Loading data...")
        
        # Load full dataset
        full_dataset = LoRADataset.from_imdb(
            self.tokenizer,
            self.config.max_seq_len,
            self.config.num_samples
        )
        
        # Split dataset (80% train, 10% val, 10% test)
        total_size = len(full_dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size
        
        # Create splits
        train_texts = full_dataset.texts[:train_size]
        train_labels = full_dataset.labels[:train_size]
        val_texts = full_dataset.texts[train_size:train_size + val_size]
        val_labels = full_dataset.labels[train_size:train_size + val_size]
        test_texts = full_dataset.texts[train_size + val_size:]
        test_labels = full_dataset.labels[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = LoRADataset(train_texts, train_labels, self.tokenizer, self.config.max_seq_len)
        self.val_dataset = LoRADataset(val_texts, val_labels, self.tokenizer, self.config.max_seq_len)
        self.test_dataset = LoRADataset(test_texts, test_labels, self.tokenizer, self.config.max_seq_len)
        
        print(f"✅ Data loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
    
    def train(self):
        """Train the model using the original approach."""
        print("🎯 Starting Universal LoRA fine-tuning...")
        
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
        print(f"\n🎉 Universal LoRA fine-tuning completed in {training_time/60:.1f} minutes")
        
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
            'model_type': self.model_type,
            'parameter_counts': self.lora_manager.get_parameter_count()
        }, f"{path}/universal_lora_classifier.pt")
        
        # Save LoRA weights separately
        torch.save(lora_weights, f"{path}/lora_weights.pt")
        
        print(f"✅ Model saved to {path}")


# Example usage functions
def fine_tune_huggingface_model(model_name: str, config: LoRATrainingConfig):
    """
    Fine-tune a HuggingFace model with LoRA.
    
    Args:
        model_name: HuggingFace model name (e.g., "bert-base-uncased")
        config: LoRA training configuration
    """
    trainer = UniversalLoRATrainer(config)
    trainer.setup_model(model_name_or_path=model_name, model_type="huggingface")
    trainer.load_data()
    results = trainer.train()
    trainer.save_model("outputs/huggingface_lora")
    return results


def fine_tune_custom_model(model_path: str, config: LoRATrainingConfig):
    """
    Fine-tune your custom model with LoRA.
    
    Args:
        model_path: Path to your custom model
        config: LoRA training configuration
    """
    config.pretrained_model_path = model_path
    trainer = UniversalLoRATrainer(config)
    trainer.setup_model(model_type="custom")
    trainer.load_data()
    results = trainer.train()
    trainer.save_model("outputs/custom_lora")
    return results
