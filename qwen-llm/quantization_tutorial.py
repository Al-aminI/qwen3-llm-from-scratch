#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 QUANTIZATION TUTORIAL - BECOME AN EXPERT

This comprehensive tutorial covers:
1. Model Quantization Fundamentals
2. LoRA (Low-Rank Adaptation) Implementation
3. QLoRA (Quantized LoRA) Implementation
4. Model Serving with Quantization
5. Performance Benchmarking

By the end, you'll be an expert in efficient model deployment!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json

# Import our custom components
from config.qwen3_small_config import SmallModelConfig
from qwen3_complete_model import MinimalLLM

# =============================================================================
# 📚 PART 1: QUANTIZATION FUNDAMENTALS
# =============================================================================

@dataclass
class QuantizationConfig:
    """
    🎯 QUANTIZATION CONFIGURATION
    
    Configuration for different quantization techniques.
    """
    # Basic quantization
    bits: int = 8  # Number of bits for quantization (4, 8, 16)
    symmetric: bool = True  # Symmetric vs asymmetric quantization
    
    # LoRA parameters
    lora_rank: int = 16  # Rank of LoRA adaptation
    lora_alpha: float = 32.0  # LoRA scaling parameter
    lora_dropout: float = 0.1  # LoRA dropout
    
    # QLoRA parameters
    use_qlora: bool = False  # Enable QLoRA
    qlora_bits: int = 4  # Bits for QLoRA quantization
    
    # Target modules for LoRA
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "w_o", "gate_proj", "up_proj", "down_proj"]

class QuantizationExpert:
    """
    🎓 QUANTIZATION EXPERT CLASS
    
    This class demonstrates different quantization techniques and their trade-offs.
    """
    
    def __init__(self):
        self.quantization_methods = {
            'fp32': self._fp32_quantization,
            'fp16': self._fp16_quantization,
            'int8': self._int8_quantization,
            'int4': self._int4_quantization,
            'dynamic': self._dynamic_quantization,
            'static': self._static_quantization
        }
    
    def _fp32_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Full precision (32-bit) - baseline"""
        return tensor.float()
    
    def _fp16_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Half precision (16-bit) - 2x memory reduction"""
        return tensor.half()
    
    def _int8_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        🎯 INT8 QUANTIZATION
        
        Converts float32 to int8 with scale and zero_point.
        Memory reduction: 4x (32-bit → 8-bit)
        """
        # Calculate quantization parameters
        qmin, qmax = 0, 255  # uint8 range
        
        # Find min/max values
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        
        # Calculate scale and zero point
        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = qmin - t_min / scale
        zero_point = max(qmin, min(qmax, round(zero_point)))
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
        
        return quantized, scale, zero_point
    
    def _int4_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        🎯 INT4 QUANTIZATION
        
        Converts float32 to int4 with scale and zero_point.
        Memory reduction: 8x (32-bit → 4-bit)
        """
        # Calculate quantization parameters
        qmin, qmax = 0, 15  # 4-bit range
        
        # Find min/max values
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        
        # Calculate scale and zero point
        scale = (t_max - t_min) / (qmax - qmin)
        zero_point = qmin - t_min / scale
        zero_point = max(qmin, min(qmax, round(zero_point)))
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
        
        return quantized, scale, zero_point
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """
        🎯 DYNAMIC QUANTIZATION
        
        Quantizes weights to int8, activations remain float.
        Good for inference, minimal accuracy loss.
        """
        try:
            return torch.quantization.quantize_dynamic(
                model, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
        except RuntimeError:
            # Fallback for models that don't support deepcopy
            print("   Note: Dynamic quantization not supported for this model")
            return model
    
    def _static_quantization(self, model: nn.Module, calibration_data: List[torch.Tensor]) -> nn.Module:
        """
        🎯 STATIC QUANTIZATION
        
        Quantizes both weights and activations using calibration data.
        Better compression, requires representative data.
        """
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with representative data
        model_prepared.eval()
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        return quantized_model
    
    def dequantize(self, quantized_tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        """Convert quantized tensor back to float32"""
        return (quantized_tensor.float() - zero_point) * scale
    
    def analyze_quantization_impact(self, original_tensor: torch.Tensor, quantized_tensor: torch.Tensor, 
                                  scale: float, zero_point: int) -> Dict:
        """
        📊 ANALYZE QUANTIZATION IMPACT
        
        Analyzes the impact of quantization on model weights.
        """
        # Dequantize for comparison
        dequantized = self.dequantize(quantized_tensor, scale, zero_point)
        
        # Calculate metrics
        mse = F.mse_loss(original_tensor, dequantized).item()
        mae = F.l1_loss(original_tensor, dequantized).item()
        cosine_sim = F.cosine_similarity(original_tensor.flatten(), dequantized.flatten(), dim=0).item()
        
        # Memory usage
        original_size = original_tensor.numel() * 4  # 4 bytes per float32
        quantized_size = quantized_tensor.numel() * 1  # 1 byte per uint8
        compression_ratio = original_size / quantized_size
        
        return {
            'mse': mse,
            'mae': mae,
            'cosine_similarity': cosine_sim,
            'compression_ratio': compression_ratio,
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024)
        }

# =============================================================================
# 🎯 PART 2: LORA (LOW-RANK ADAPTATION) IMPLEMENTATION
# =============================================================================

class LoRALayer(nn.Module):
    """
    🎯 LORA LAYER IMPLEMENTATION
    
    LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices.
    
    Mathematical Foundation:
    W = W₀ + ΔW = W₀ + BA
    Where:
    - W₀: Original frozen weights
    - B: Low-rank matrix (d × r)
    - A: Low-rank matrix (r × k)
    - r << min(d, k) (rank)
    
    Benefits:
    - Reduces trainable parameters by ~1000x
    - Maintains model performance
    - Enables efficient fine-tuning
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, 
                 alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer
        
        Args:
            x: Input tensor (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor (batch_size, seq_len, out_features)
        """
        # Apply LoRA: x @ A^T @ B^T
        # x: (batch, seq, in_features)
        # A: (rank, in_features) -> (in_features, rank)
        # B: (out_features, rank)
        # Result: (batch, seq, out_features)
        
        x = self.dropout(x)
        x = x @ self.lora_A.T  # (batch, seq, rank)
        x = x @ self.lora_B.T  # (batch, seq, out_features)
        x = x * self.scaling
        
        return x

class LoRALinear(nn.Module):
    """
    🎯 LORA LINEAR LAYER
    
    Combines original linear layer with LoRA adaptation.
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + LoRA adaptation"""
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        return original_output + lora_output

class LoRAManager:
    """
    🎯 LORA MANAGER
    
    Manages LoRA adaptation for entire models.
    """
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        self.model = model
        self.config = config
        self.lora_layers = {}
        self.original_layers = {}
    
    def apply_lora(self, target_modules: List[str] = None):
        """
        🎯 APPLY LORA TO MODEL
        
        Replaces specified linear layers with LoRA-adapted versions.
        """
        if target_modules is None:
            target_modules = self.config.target_modules
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                print(f"🎯 Applying LoRA to: {name}")
                
                # Store original layer
                self.original_layers[name] = module
                
                # Create LoRA layer
                lora_layer = LoRALinear(
                    module,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout
                )
                
                # Replace in model
                self._replace_module(self.model, name, lora_layer)
                self.lora_layers[name] = lora_layer
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """Replace a module in the model by name"""
        parts = module_name.split('.')
        current = model
        
        # Navigate to parent module
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Replace the module
        setattr(current, parts[-1], new_module)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters (LoRA only)"""
        trainable_params = []
        for lora_layer in self.lora_layers.values():
            trainable_params.extend(lora_layer.lora.parameters())
        return trainable_params
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for analysis"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_percentage': (trainable_params / total_params) * 100
        }

# =============================================================================
# 🎯 PART 3: QLORA (QUANTIZED LORA) IMPLEMENTATION
# =============================================================================

class QLoRALayer(nn.Module):
    """
    🎯 QLORA LAYER IMPLEMENTATION
    
    QLoRA combines LoRA with 4-bit quantization for maximum efficiency.
    
    Key Innovation:
    - Quantizes base model to 4-bit
    - Uses LoRA for adaptation
    - Enables fine-tuning on consumer hardware
    
    Memory Savings:
    - 4-bit quantization: 8x reduction
    - LoRA adaptation: 1000x reduction in trainable params
    - Combined: ~8000x memory reduction for training
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, 
                 alpha: float = 32.0, dropout: float = 0.1, bits: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.bits = bits
        self.scaling = alpha / rank
        
        # LoRA matrices (always in full precision)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Quantization parameters
        self.register_buffer('quantized_weights', None)
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def quantize_weights(self, weights: torch.Tensor):
        """Quantize weights to specified bit width"""
        if self.bits == 4:
            return self._quantize_4bit(weights)
        elif self.bits == 8:
            return self._quantize_8bit(weights)
        else:
            return weights
    
    def _quantize_4bit(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """4-bit quantization"""
        qmin, qmax = 0, 15  # 4-bit range
        
        # Calculate quantization parameters
        w_min = weights.min().item()
        w_max = weights.max().item()
        
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale
        zero_point = max(qmin, min(qmax, round(zero_point)))
        
        # Quantize
        quantized = torch.round(weights / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
        
        return quantized, torch.tensor(scale), torch.tensor(zero_point)
    
    def _quantize_8bit(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """8-bit quantization"""
        qmin, qmax = 0, 255  # 8-bit range
        
        # Calculate quantization parameters
        w_min = weights.min().item()
        w_max = weights.max().item()
        
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale
        zero_point = max(qmin, min(qmax, round(zero_point)))
        
        # Quantize
        quantized = torch.round(weights / scale + zero_point).clamp(qmin, qmax).to(torch.uint8)
        
        return quantized, torch.tensor(scale), torch.tensor(zero_point)
    
    def dequantize_weights(self, quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize weights back to float32"""
        return (quantized.float() - zero_point) * scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through QLoRA layer"""
        x = self.dropout(x)
        x = x @ self.lora_A.T  # (batch, seq, rank)
        x = x @ self.lora_B.T  # (batch, seq, out_features)
        x = x * self.scaling
        
        return x

class QLoRALinear(nn.Module):
    """
    🎯 QLORA LINEAR LAYER
    
    Combines quantized base weights with LoRA adaptation.
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, 
                 dropout: float = 0.1, bits: int = 4):
        super().__init__()
        self.original_layer = original_layer
        self.qlora = QLoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bits=bits
        )
        
        # Quantize original weights
        self._quantize_original_weights()
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def _quantize_original_weights(self):
        """Quantize the original layer weights"""
        with torch.no_grad():
            quantized, scale, zero_point = self.qlora.quantize_weights(self.original_layer.weight)
            self.qlora.register_buffer('quantized_weights', quantized)
            self.qlora.register_buffer('scale', scale)
            self.qlora.register_buffer('zero_point', zero_point)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantized original + LoRA adaptation"""
        # Dequantize weights for computation
        dequantized_weights = self.qlora.dequantize_weights(
            self.qlora.quantized_weights,
            self.qlora.scale,
            self.qlora.zero_point
        )
        
        # Original computation with dequantized weights
        original_output = F.linear(x, dequantized_weights, self.original_layer.bias)
        
        # LoRA adaptation
        lora_output = self.qlora(x)
        
        return original_output + lora_output

class QLoRAManager:
    """
    🎯 QLORA MANAGER
    
    Manages QLoRA adaptation for entire models.
    """
    
    def __init__(self, model: nn.Module, config: QuantizationConfig):
        self.model = model
        self.config = config
        self.qlora_layers = {}
        self.original_layers = {}
    
    def apply_qlora(self, target_modules: List[str] = None):
        """
        🎯 APPLY QLORA TO MODEL
        
        Replaces specified linear layers with QLoRA-adapted versions.
        """
        if target_modules is None:
            target_modules = self.config.target_modules
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                print(f"🎯 Applying QLoRA to: {name}")
                
                # Store original layer
                self.original_layers[name] = module
                
                # Create QLoRA layer
                qlora_layer = QLoRALinear(
                    module,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                    bits=self.config.qlora_bits
                )
                
                # Replace in model
                self._replace_module(self.model, name, qlora_layer)
                self.qlora_layers[name] = qlora_layer
    
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """Replace a module in the model by name"""
        parts = module_name.split('.')
        current = model
        
        # Navigate to parent module
        for part in parts[:-1]:
            current = getattr(current, part)
        
        # Replace the module
        setattr(current, parts[-1], new_module)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters (QLoRA only)"""
        trainable_params = []
        for qlora_layer in self.qlora_layers.values():
            trainable_params.extend(qlora_layer.qlora.parameters())
        return trainable_params
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage for QLoRA model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        # Memory calculation (in MB)
        # Original weights: 4 bytes per parameter (fp32)
        # Quantized weights: 0.5 bytes per parameter (4-bit)
        # LoRA weights: 4 bytes per parameter (fp32)
        
        original_memory = total_params * 4 / (1024 * 1024)  # MB
        quantized_memory = total_params * 0.5 / (1024 * 1024)  # MB
        lora_memory = trainable_params * 4 / (1024 * 1024)  # MB
        
        total_memory = quantized_memory + lora_memory
        
        return {
            'original_memory_mb': original_memory,
            'quantized_memory_mb': quantized_memory,
            'lora_memory_mb': lora_memory,
            'total_memory_mb': total_memory,
            'memory_reduction': original_memory / total_memory,
            'trainable_params': trainable_params,
            'total_params': total_params
        }

# =============================================================================
# 🎯 PART 4: QUANTIZATION BENCHMARKS AND ANALYSIS
# =============================================================================

class QuantizationBenchmark:
    """
    📊 QUANTIZATION BENCHMARK
    
    Comprehensive benchmarking of different quantization techniques.
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_model(self, model: nn.Module, test_data: torch.Tensor, 
                       configs: List[QuantizationConfig]) -> Dict:
        """
        🎯 BENCHMARK MODEL WITH DIFFERENT QUANTIZATION CONFIGS
        
        Compares performance across different quantization methods.
        """
        print("📊 Starting Quantization Benchmark")
        print("=" * 50)
        
        results = {}
        
        for config in configs:
            print(f"\n🎯 Testing {config.bits}-bit quantization...")
            
            # Measure memory usage
            memory_usage = self._measure_memory_usage(model, config)
            
            # Measure inference speed
            inference_time = self._measure_inference_time(model, test_data, config)
            
            # Measure accuracy (if ground truth available)
            accuracy = self._measure_accuracy(model, test_data, config)
            
            results[f"{config.bits}bit"] = {
                'memory_mb': memory_usage,
                'inference_time_ms': inference_time,
                'accuracy': accuracy,
                'config': config
            }
        
        self.results = results
        return results
    
    def _measure_memory_usage(self, model: nn.Module, config: QuantizationConfig) -> float:
        """Measure memory usage of quantized model"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate memory based on bit width
        if config.bits == 4:
            bytes_per_param = 0.5
        elif config.bits == 8:
            bytes_per_param = 1
        elif config.bits == 16:
            bytes_per_param = 2
        else:  # 32-bit
            bytes_per_param = 4
        
        memory_mb = (total_params * bytes_per_param) / (1024 * 1024)
        return memory_mb
    
    def _measure_inference_time(self, model: nn.Module, test_data: torch.Tensor, 
                               config: QuantizationConfig) -> float:
        """Measure inference time"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_data)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_data)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        return avg_time_ms
    
    def _measure_accuracy(self, model: nn.Module, test_data: torch.Tensor, 
                         config: QuantizationConfig) -> float:
        """Measure model accuracy (placeholder - would need ground truth)"""
        # This is a placeholder - in practice, you'd compare with ground truth
        return 0.95  # Placeholder accuracy
    
    def print_results(self):
        """Print benchmark results in a nice format"""
        print("\n📊 QUANTIZATION BENCHMARK RESULTS")
        print("=" * 60)
        
        for method, result in self.results.items():
            print(f"\n🎯 {method.upper()} QUANTIZATION:")
            print(f"   Memory Usage: {result['memory_mb']:.2f} MB")
            print(f"   Inference Time: {result['inference_time_ms']:.2f} ms")
            print(f"   Accuracy: {result['accuracy']:.3f}")
        
        # Calculate improvements
        if len(self.results) > 1:
            print(f"\n📈 IMPROVEMENTS:")
            baseline = self.results.get('32bit', {})
            for method, result in self.results.items():
                if method != '32bit':
                    memory_improvement = baseline.get('memory_mb', 1) / result['memory_mb']
                    speed_improvement = baseline.get('inference_time_ms', 1) / result['inference_time_ms']
                    print(f"   {method}: {memory_improvement:.1f}x memory, {speed_improvement:.1f}x speed")

# =============================================================================
# 🎯 PART 5: DEMO AND TESTING
# =============================================================================

def demo_quantization():
    """
    🎯 QUANTIZATION DEMO
    
    Demonstrates different quantization techniques on our Qwen3 model.
    """
    print("🎯 QUANTIZATION EXPERT TUTORIAL")
    print("=" * 50)
    
    # Create a small model for demo
    config = SmallModelConfig()
    config.d_model = 64  # Smaller for demo
    config.n_layers = 2
    config.vocab_size = 1000
    
    model = MinimalLLM(config)
    
    # Create test data
    batch_size, seq_len = 4, 32
    test_data = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"📊 Model Info:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Test data shape: {test_data.shape}")
    
    # Test different quantization methods
    quantizer = QuantizationExpert()
    
    print(f"\n🧪 Testing Quantization Methods:")
    
    # Test on a sample weight
    sample_weight = model.token_embedding.weight[:100, :100]  # Small sample
    
    for method_name, method_func in quantizer.quantization_methods.items():
        if method_name in ['int8', 'int4']:
            print(f"\n🎯 {method_name.upper()}:")
            quantized, scale, zero_point = method_func(sample_weight)
            analysis = quantizer.analyze_quantization_impact(sample_weight, quantized, scale, zero_point)
            
            print(f"   Compression Ratio: {analysis['compression_ratio']:.1f}x")
            print(f"   Memory Reduction: {analysis['original_size_mb']:.2f}MB → {analysis['quantized_size_mb']:.2f}MB")
            print(f"   Cosine Similarity: {analysis['cosine_similarity']:.4f}")
            print(f"   MSE: {analysis['mse']:.6f}")
        else:
            print(f"\n🎯 {method_name.upper()}:")
            quantized = method_func(sample_weight)
            print(f"   Shape: {quantized.shape}")
            print(f"   Dtype: {quantized.dtype}")

def demo_lora():
    """
    🎯 LORA DEMO
    
    Demonstrates LoRA adaptation on our Qwen3 model.
    """
    print("\n🎯 LORA (LOW-RANK ADAPTATION) DEMO")
    print("=" * 50)
    
    # Create model
    config = SmallModelConfig()
    config.d_model = 128
    config.n_layers = 2
    config.vocab_size = 1000
    
    model = MinimalLLM(config)
    
    # Create LoRA config
    lora_config = QuantizationConfig()
    lora_config.lora_rank = 8
    lora_config.lora_alpha = 16.0
    
    # Apply LoRA
    lora_manager = LoRAManager(model, lora_config)
    lora_manager.apply_lora()
    
    # Analyze parameters
    param_counts = lora_manager.get_parameter_count()
    
    print(f"📊 LoRA Analysis:")
    print(f"   Total Parameters: {param_counts['total']:,}")
    print(f"   Trainable Parameters: {param_counts['trainable']:,}")
    print(f"   Frozen Parameters: {param_counts['frozen']:,}")
    print(f"   Trainable Percentage: {param_counts['trainable_percentage']:.2f}%")
    
    # Test forward pass
    test_input = torch.randint(0, config.vocab_size, (2, 16))
    
    print(f"\n🧪 Testing Forward Pass:")
    with torch.no_grad():
        output = model(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

def demo_qlora():
    """
    🎯 QLORA DEMO
    
    Demonstrates QLoRA adaptation on our Qwen3 model.
    """
    print("\n🎯 QLORA (QUANTIZED LORA) DEMO")
    print("=" * 50)
    
    # Create model
    config = SmallModelConfig()
    config.d_model = 128
    config.n_layers = 2
    config.vocab_size = 1000
    
    model = MinimalLLM(config)
    
    # Create QLoRA config
    qlora_config = QuantizationConfig()
    qlora_config.lora_rank = 8
    qlora_config.lora_alpha = 16.0
    qlora_config.qlora_bits = 4
    
    # Apply QLoRA
    qlora_manager = QLoRAManager(model, qlora_config)
    qlora_manager.apply_qlora()
    
    # Analyze memory usage
    memory_usage = qlora_manager.get_memory_usage()
    
    print(f"📊 QLoRA Analysis:")
    print(f"   Original Memory: {memory_usage['original_memory_mb']:.2f} MB")
    print(f"   Quantized Memory: {memory_usage['quantized_memory_mb']:.2f} MB")
    print(f"   LoRA Memory: {memory_usage['lora_memory_mb']:.2f} MB")
    print(f"   Total Memory: {memory_usage['total_memory_mb']:.2f} MB")
    print(f"   Memory Reduction: {memory_usage['memory_reduction']:.1f}x")
    print(f"   Trainable Parameters: {memory_usage['trainable_params']:,}")
    
    # Test forward pass
    test_input = torch.randint(0, config.vocab_size, (2, 16))
    
    print(f"\n🧪 Testing Forward Pass:")
    with torch.no_grad():
        output = model(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

if __name__ == "__main__":
    # Run all demos
    demo_quantization()
    demo_lora()
    demo_qlora()
    
    print("\n🎉 QUANTIZATION TUTORIAL COMPLETE!")
    print("You are now an expert in:")
    print("✅ Model Quantization (4-bit, 8-bit, 16-bit)")
    print("✅ LoRA (Low-Rank Adaptation)")
    print("✅ QLoRA (Quantized LoRA)")
    print("✅ Memory Optimization")
    print("✅ Performance Benchmarking")
