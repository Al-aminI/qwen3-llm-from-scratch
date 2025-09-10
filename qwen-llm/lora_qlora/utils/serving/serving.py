"""
Model serving utilities.

This module provides utilities for serving trained LoRA and QLoRA models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directories to path to import your custom model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from quantization_tutorial import LoRAManager, QLoRAManager, QuantizationConfig
from config.qwen3_small_config import SmallModelConfig
from qwen3_complete_model import MinimalLLM


class ModelServer:
    """
    🎯 MODEL SERVER
    
    Server for serving trained LoRA and QLoRA models using your custom MinimalLLM.
    
    This class provides:
    - Model loading and initialization
    - Inference capabilities
    - Batch processing
    - Performance monitoring
    """
    
    def __init__(self, model_path: str, model_type: str = 'lora', base_model_path: str = None):
        """
        Initialize model server.
        
        Args:
            model_path: Path to trained LoRA/QLoRA model
            model_type: Type of model ('lora' or 'qlora')
            base_model_path: Path to base pretrained model (optional)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.base_model_path = base_model_path
        self.model = None
        self.tokenizer = None
        self.manager = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """Load the trained model."""
        print(f"🎯 Loading {self.model_type.upper()} model from {self.model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        config = checkpoint.get('config', None)
        
        # Load tokenizer
        if config and hasattr(config, 'tokenizer_path'):
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model config
        model_config = SmallModelConfig()
        model_config.vocab_size = self.tokenizer.vocab_size
        
        # Load base model
        if self.base_model_path and os.path.exists(self.base_model_path):
            print(f"📦 Loading base model from {self.base_model_path}")
            base_checkpoint = torch.load(self.base_model_path, map_location='cpu')
            self.base_model = MinimalLLM(model_config)
            self.base_model.load_state_dict(base_checkpoint['model_state_dict'])
        else:
            print("🏗️ Creating new base model")
            self.base_model = MinimalLLM(model_config)
        
        # Load LoRA/QLoRA weights and apply
        if self.model_type == 'lora':
            lora_weights = checkpoint.get('lora_weights', {})
            lora_config = QuantizationConfig(
                lora_rank=config.lora_rank if config else 16,
                lora_alpha=config.lora_alpha if config else 32.0,
                lora_dropout=config.lora_dropout if config else 0.1,
                target_modules=config.target_modules if config else ["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"]
            )
            self.manager = LoRAManager(self.base_model, lora_config)
            self.manager.apply_lora()
            
            # Load LoRA weights
            for name, lora_layer in self.manager.lora_layers.items():
                if name in lora_weights:
                    lora_layer.lora.lora_A.data = lora_weights[name]['lora_A']
                    lora_layer.lora.lora_B.data = lora_weights[name]['lora_B']
        
        elif self.model_type == 'qlora':
            qlora_weights = checkpoint.get('qlora_weights', {})
            qlora_config = QuantizationConfig(
                lora_rank=config.lora_rank if config else 16,
                lora_alpha=config.lora_alpha if config else 32.0,
                lora_dropout=config.lora_dropout if config else 0.1,
                qlora_bits=config.qlora_bits if config else 4,
                target_modules=config.target_modules if config else ["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"]
            )
            self.manager = QLoRAManager(self.base_model, qlora_config)
            self.manager.apply_qlora()
            
            # Load QLoRA weights
            for name, qlora_layer in self.manager.qlora_layers.items():
                if name in qlora_weights:
                    qlora_layer.qlora.lora_A.data = qlora_weights[name]['lora_A']
                    qlora_layer.qlora.lora_B.data = qlora_weights[name]['lora_B']
                    qlora_layer.qlora.register_buffer('quantized_weights', qlora_weights[name]['quantized_weights'])
                    qlora_layer.qlora.register_buffer('scale', qlora_weights[name]['scale'])
                    qlora_layer.qlora.register_buffer('zero_point', qlora_weights[name]['zero_point'])
        
        # Create classifier (similar to your training setup)
        self.model = self._create_classifier(self.base_model, num_classes=2, dropout=0.1)
        
        # Load classifier weights
        if 'classifier_state_dict' in checkpoint:
            classifier_state_dict = checkpoint['classifier_state_dict']
            classifier_only_state_dict = {k: v for k, v in classifier_state_dict.items() if 'classifier' in k}
            self.model.load_state_dict(classifier_only_state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def _create_classifier(self, base_model, num_classes=2, dropout=0.1):
        """Create classification head similar to your training setup."""
        class CustomClassificationModel(nn.Module):
            def __init__(self, base_model, num_classes=2, dropout=0.1):
                super().__init__()
                self.base_model = base_model
                self.num_labels = num_classes
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(base_model.config.d_model, base_model.config.d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(base_model.config.d_model // 2, num_classes)
                )
                self.apply(self._init_weights)
            
            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                import math
                with torch.no_grad():
                    x = self.base_model.token_embedding(input_ids) * torch.sqrt(torch.tensor(self.base_model.config.d_model, dtype=torch.float32))
                    x = self.base_model.position_dropout(x)
                    for block in self.base_model.transformer_blocks:
                        x = block(x)
                    x = self.base_model.norm(x)
                    cls_representation = x[:, 0, :]
                logits = self.classifier(cls_representation)
                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                return {
                    'loss': loss,
                    'logits': logits
                }
        
        return CustomClassificationModel(base_model, num_classes, dropout)
    
    def predict(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Predict on a single text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1)
        
        return {
            'text': text,
            'predicted_class': predicted_class.item(),
            'probabilities': probabilities.cpu().numpy().tolist()[0],
            'confidence': torch.max(probabilities).item()
        }
    
    def predict_batch(self, texts: List[str], max_length: int = 512, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Predict on a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(logits, dim=-1)
            
            # Process results
            for j, text in enumerate(batch_texts):
                results.append({
                    'text': text,
                    'predicted_class': predicted_classes[j].item(),
                    'probabilities': probabilities[j].cpu().numpy().tolist(),
                    'confidence': torch.max(probabilities[j]).item()
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        info = {
            'model_type': self.model_type,
            'model_path': str(self.model_path),
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        if self.manager is not None:
            if self.model_type == 'lora':
                param_counts = self.manager.get_parameter_count()
                info.update(param_counts)
            elif self.model_type == 'qlora':
                memory_usage = self.manager.get_memory_usage()
                info.update(memory_usage)
        
        return info


class InferenceEngine:
    """
    🎯 INFERENCE ENGINE
    
    High-performance inference engine for LoRA and QLoRA models.
    
    This class provides:
    - Optimized inference
    - Caching mechanisms
    - Performance monitoring
    - Batch processing
    """
    
    def __init__(self, model_server: ModelServer):
        """
        Initialize inference engine.
        
        Args:
            model_server: Model server instance
        """
        self.model_server = model_server
        self.cache = {}
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0
        }
    
    def predict(self, text: str, use_cache: bool = True, max_length: int = 512) -> Dict[str, Any]:
        """
        Predict with caching support.
        
        Args:
            text: Input text
            use_cache: Whether to use caching
            max_length: Maximum sequence length
            
        Returns:
            Prediction results
        """
        import time
        start_time = time.time()
        
        # Check cache
        if use_cache and text in self.cache:
            self.stats['cache_hits'] += 1
            result = self.cache[text].copy()
            result['cached'] = True
        else:
            self.stats['cache_misses'] += 1
            result = self.model_server.predict(text, max_length)
            result['cached'] = False
            
            # Cache result
            if use_cache:
                self.cache[text] = result.copy()
        
        # Update stats
        self.stats['total_requests'] += 1
        self.stats['total_time'] += time.time() - start_time
        
        return result
    
    def predict_batch(self, texts: List[str], use_cache: bool = True, max_length: int = 512, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Predict batch with caching support.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use caching
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        import time
        start_time = time.time()
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if use_cache and text in self.cache:
                self.stats['cache_hits'] += 1
                result = self.cache[text].copy()
                result['cached'] = True
                results.append(result)
            else:
                self.stats['cache_misses'] += 1
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)  # Placeholder
        
        # Process uncached texts
        if uncached_texts:
            uncached_results = self.model_server.predict_batch(uncached_texts, max_length, batch_size)
            
            # Update results and cache
            for i, result in enumerate(uncached_results):
                result['cached'] = False
                results[uncached_indices[i]] = result
                
                if use_cache:
                    self.cache[uncached_texts[i]] = result.copy()
        
        # Update stats
        self.stats['total_requests'] += len(texts)
        self.stats['total_time'] += time.time() - start_time
        
        return results
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self.cache.clear()
        print("✅ Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'total_requests': self.stats['total_requests'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / max(self.stats['total_requests'], 1),
            'average_time': self.stats['total_time'] / max(self.stats['total_requests'], 1)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_requests': self.stats['total_requests'],
            'total_time': self.stats['total_time'],
            'average_time': self.stats['total_time'] / max(self.stats['total_requests'], 1),
            'requests_per_second': self.stats['total_requests'] / max(self.stats['total_time'], 1e-6)
        }
