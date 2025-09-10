#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 QUANTIZATION BENCHMARK SCRIPT

This script benchmarks different quantization methods (FP32, FP16, INT8, INT4, LoRA, QLoRA)
and compares their performance, memory usage, and accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# Import our custom components
from config.qwen3_small_config import SmallModelConfig
from qwen3_complete_model import MinimalLLM
from quantization_tutorial import (
    QuantizationExpert, LoRAManager, QLoRAManager, 
    QuantizationConfig, QuantizationBenchmark
)

@dataclass
class BenchmarkConfig:
    """
    🎯 BENCHMARK CONFIGURATION
    """
    # Model parameters
    d_model: int = 128
    n_layers: int = 3
    vocab_size: int = 1000
    
    # Benchmark parameters
    num_samples: int = 100
    sequence_length: int = 64
    batch_size: int = 8
    
    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    
    # QLoRA parameters
    qlora_bits: int = 4

class QuantizationBenchmarker:
    """
    📊 QUANTIZATION BENCHMARKER
    
    Comprehensive benchmarking of different quantization techniques.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create test data
        self.test_data = torch.randint(0, config.vocab_size, (config.batch_size, config.sequence_length))
        
        # Create models
        self.models = {}
        self._create_models()
    
    def _create_models(self):
        """Create different quantized models for benchmarking"""
        print("🏗️ Creating models for benchmarking...")
        
        # Base model config
        model_config = SmallModelConfig()
        model_config.d_model = self.config.d_model
        model_config.n_layers = self.config.n_layers
        model_config.vocab_size = self.config.vocab_size
        
        # 1. FP32 Model (baseline)
        self.models['fp32'] = MinimalLLM(model_config).to(self.device)
        
        # 2. FP16 Model
        self.models['fp16'] = MinimalLLM(model_config).half().to(self.device)
        
        # 3. LoRA Model
        lora_model = MinimalLLM(model_config).to(self.device)
        lora_config = QuantizationConfig()
        lora_config.lora_rank = self.config.lora_rank
        lora_config.lora_alpha = self.config.lora_alpha
        
        lora_manager = LoRAManager(lora_model, lora_config)
        lora_manager.apply_lora()
        self.models['lora'] = lora_model
        self.lora_manager = lora_manager
        
        # 4. QLoRA Model
        qlora_model = MinimalLLM(model_config).to(self.device)
        qlora_config = QuantizationConfig()
        qlora_config.lora_rank = self.config.lora_rank
        qlora_config.lora_alpha = self.config.lora_alpha
        qlora_config.qlora_bits = self.config.qlora_bits
        
        qlora_manager = QLoRAManager(qlora_model, qlora_config)
        qlora_manager.apply_qlora()
        self.models['qlora'] = qlora_model
        self.qlora_manager = qlora_manager
        
        print(f"✅ Created {len(self.models)} models for benchmarking")
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """
        📊 BENCHMARK MEMORY USAGE
        
        Measures memory usage for each model.
        """
        print("📊 Benchmarking memory usage...")
        
        memory_results = {}
        
        for name, model in self.models.items():
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load model
            model = model.to(self.device)
            
            # Measure memory after
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                process = psutil.Process(os.getpid())
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_usage = memory_after - memory_before
            memory_results[name] = memory_usage
            
            print(f"   {name.upper()}: {memory_usage:.2f} MB")
        
        return memory_results
    
    def benchmark_inference_speed(self) -> Dict[str, float]:
        """
        ⚡ BENCHMARK INFERENCE SPEED
        
        Measures inference speed for each model.
        """
        print("⚡ Benchmarking inference speed...")
        
        speed_results = {}
        
        for name, model in self.models.items():
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(self.test_data)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(self.config.num_samples):
                    _ = model(self.test_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / self.config.num_samples
            speed_results[name] = avg_time
            
            print(f"   {name.upper()}: {avg_time*1000:.2f} ms per inference")
        
        return speed_results
    
    def benchmark_parameter_count(self) -> Dict[str, Dict[str, int]]:
        """
        📊 BENCHMARK PARAMETER COUNT
        
        Counts parameters for each model.
        """
        print("📊 Benchmarking parameter count...")
        
        param_results = {}
        
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            param_results[name] = {
                'total': total_params,
                'trainable': trainable_params,
                'frozen': frozen_params,
                'trainable_percentage': (trainable_params / total_params) * 100
            }
            
            print(f"   {name.upper()}:")
            print(f"     Total: {total_params:,}")
            print(f"     Trainable: {trainable_params:,}")
            print(f"     Frozen: {frozen_params:,}")
            print(f"     Trainable %: {param_results[name]['trainable_percentage']:.2f}%")
        
        return param_results
    
    def benchmark_accuracy(self) -> Dict[str, float]:
        """
        🎯 BENCHMARK ACCURACY
        
        Measures accuracy by comparing outputs with FP32 baseline.
        """
        print("🎯 Benchmarking accuracy...")
        
        accuracy_results = {}
        
        # Get FP32 baseline outputs
        self.models['fp32'].eval()
        with torch.no_grad():
            baseline_outputs = self.models['fp32'](self.test_data)
        
        for name, model in self.models.items():
            if name == 'fp32':
                accuracy_results[name] = 1.0  # Perfect accuracy for baseline
                continue
            
            model.eval()
            with torch.no_grad():
                outputs = model(self.test_data)
            
            # Calculate cosine similarity
            cosine_sim = F.cosine_similarity(
                baseline_outputs.flatten(), 
                outputs.flatten(), 
                dim=0
            ).item()
            
            # Calculate MSE
            mse = F.mse_loss(baseline_outputs, outputs).item()
            
            # Calculate relative error
            relative_error = torch.mean(torch.abs(baseline_outputs - outputs) / (torch.abs(baseline_outputs) + 1e-8)).item()
            
            accuracy_results[name] = {
                'cosine_similarity': cosine_sim,
                'mse': mse,
                'relative_error': relative_error
            }
            
            print(f"   {name.upper()}:")
            print(f"     Cosine Similarity: {cosine_sim:.4f}")
            print(f"     MSE: {mse:.6f}")
            print(f"     Relative Error: {relative_error:.4f}")
        
        return accuracy_results
    
    def benchmark_training_speed(self) -> Dict[str, float]:
        """
        🚀 BENCHMARK TRAINING SPEED
        
        Measures training speed for trainable models.
        """
        print("🚀 Benchmarking training speed...")
        
        training_results = {}
        
        for name, model in self.models.items():
            if name == 'fp32':
                # Full model training
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            elif name == 'fp16':
                # FP16 training
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            elif name == 'lora':
                # LoRA training
                trainable_params = self.lora_manager.get_trainable_parameters()
                optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
            elif name == 'qlora':
                # QLoRA training
                trainable_params = self.qlora_manager.get_trainable_parameters()
                optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
            else:
                continue
            
            model.train()
            
            # Warmup
            for _ in range(5):
                optimizer.zero_grad()
                outputs = model(self.test_data)
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
            
            # Benchmark
            start_time = time.time()
            for _ in range(self.config.num_samples // 10):  # Fewer iterations for training
                optimizer.zero_grad()
                outputs = model(self.test_data)
                loss = outputs.mean()
                loss.backward()
                optimizer.step()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / (self.config.num_samples // 10)
            training_results[name] = avg_time
            
            print(f"   {name.upper()}: {avg_time*1000:.2f} ms per training step")
        
        return training_results
    
    def run_comprehensive_benchmark(self) -> Dict:
        """
        🎯 RUN COMPREHENSIVE BENCHMARK
        
        Runs all benchmarks and returns comprehensive results.
        """
        print("🎯 Starting Comprehensive Quantization Benchmark")
        print("=" * 60)
        
        # Run all benchmarks
        memory_results = self.benchmark_memory_usage()
        speed_results = self.benchmark_inference_speed()
        param_results = self.benchmark_parameter_count()
        accuracy_results = self.benchmark_accuracy()
        training_results = self.benchmark_training_speed()
        
        # Compile results
        comprehensive_results = {
            'memory_usage': memory_results,
            'inference_speed': speed_results,
            'parameter_count': param_results,
            'accuracy': accuracy_results,
            'training_speed': training_results,
            'config': self.config
        }
        
        self.results = comprehensive_results
        return comprehensive_results
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return
        
        print("\n📊 QUANTIZATION BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Memory usage comparison
        print("\n💾 MEMORY USAGE:")
        baseline_memory = self.results['memory_usage']['fp32']
        for name, memory in self.results['memory_usage'].items():
            reduction = baseline_memory / memory if memory > 0 else float('inf')
            print(f"   {name.upper()}: {memory:.2f} MB ({reduction:.1f}x reduction)")
        
        # Inference speed comparison
        print("\n⚡ INFERENCE SPEED:")
        baseline_speed = self.results['inference_speed']['fp32']
        for name, speed in self.results['inference_speed'].items():
            speedup = baseline_speed / speed if speed > 0 else float('inf')
            print(f"   {name.upper()}: {speed*1000:.2f} ms ({speedup:.1f}x speedup)")
        
        # Parameter count comparison
        print("\n📊 PARAMETER COUNT:")
        baseline_params = self.results['parameter_count']['fp32']['total']
        for name, params in self.results['parameter_count'].items():
            reduction = baseline_params / params['total'] if params['total'] > 0 else float('inf')
            print(f"   {name.upper()}: {params['total']:,} total, {params['trainable']:,} trainable ({reduction:.1f}x reduction)")
        
        # Accuracy comparison
        print("\n🎯 ACCURACY:")
        for name, accuracy in self.results['accuracy'].items():
            if isinstance(accuracy, dict):
                print(f"   {name.upper()}: Cosine Sim: {accuracy['cosine_similarity']:.4f}, MSE: {accuracy['mse']:.6f}")
            else:
                print(f"   {name.upper()}: {accuracy:.4f}")
        
        # Training speed comparison
        if 'training_speed' in self.results:
            print("\n🚀 TRAINING SPEED:")
            baseline_training = self.results['training_speed']['fp32']
            for name, speed in self.results['training_speed'].items():
                speedup = baseline_training / speed if speed > 0 else float('inf')
                print(f"   {name.upper()}: {speed*1000:.2f} ms per step ({speedup:.1f}x speedup)")
    
    def save_results(self, filename: str = "quantization_benchmark_results.json"):
        """Save benchmark results to file"""
        if not self.results:
            print("No benchmark results to save.")
            return
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            if key == 'config':
                json_results[key] = value.__dict__
            else:
                json_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Benchmark results saved to {filename}")
    
    def plot_results(self, save_path: str = "quantization_benchmark.png"):
        """Plot benchmark results"""
        if not self.results:
            print("No benchmark results to plot.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Quantization Benchmark Results', fontsize=16)
            
            # Memory usage
            names = list(self.results['memory_usage'].keys())
            memory_values = list(self.results['memory_usage'].values())
            axes[0, 0].bar(names, memory_values)
            axes[0, 0].set_title('Memory Usage (MB)')
            axes[0, 0].set_ylabel('Memory (MB)')
            
            # Inference speed
            speed_values = [v * 1000 for v in self.results['inference_speed'].values()]  # Convert to ms
            axes[0, 1].bar(names, speed_values)
            axes[0, 1].set_title('Inference Speed (ms)')
            axes[0, 1].set_ylabel('Time (ms)')
            
            # Parameter count
            param_values = [self.results['parameter_count'][name]['total'] for name in names]
            axes[1, 0].bar(names, param_values)
            axes[1, 0].set_title('Total Parameters')
            axes[1, 0].set_ylabel('Parameter Count')
            axes[1, 0].set_yscale('log')
            
            # Accuracy (cosine similarity)
            accuracy_values = []
            for name in names:
                if isinstance(self.results['accuracy'][name], dict):
                    accuracy_values.append(self.results['accuracy'][name]['cosine_similarity'])
                else:
                    accuracy_values.append(self.results['accuracy'][name])
            axes[1, 1].bar(names, accuracy_values)
            axes[1, 1].set_title('Accuracy (Cosine Similarity)')
            axes[1, 1].set_ylabel('Cosine Similarity')
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Benchmark plots saved to {save_path}")
            
        except ImportError:
            print("Matplotlib not available. Skipping plotting.")

def main():
    """Main benchmark function"""
    print("🎯 QUANTIZATION BENCHMARK")
    print("=" * 40)
    
    # Create benchmark config
    config = BenchmarkConfig()
    config.d_model = 128
    config.n_layers = 3
    config.vocab_size = 1000
    config.num_samples = 100
    config.sequence_length = 64
    config.batch_size = 8
    
    print(f"📋 Benchmark Configuration:")
    print(f"   Model Size: {config.d_model}D, {config.n_layers} layers")
    print(f"   Test Samples: {config.num_samples}")
    print(f"   Sequence Length: {config.sequence_length}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   LoRA Rank: {config.lora_rank}")
    print(f"   QLoRA Bits: {config.qlora_bits}")
    
    # Create benchmarker
    benchmarker = QuantizationBenchmarker(config)
    
    # Run comprehensive benchmark
    results = benchmarker.run_comprehensive_benchmark()
    
    # Print summary
    benchmarker.print_summary()
    
    # Save results
    benchmarker.save_results()
    
    # Plot results
    benchmarker.plot_results()
    
    print("\n🎉 Benchmark completed successfully!")

if __name__ == "__main__":
    main()
