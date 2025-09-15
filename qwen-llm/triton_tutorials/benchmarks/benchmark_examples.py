"""
📊 Benchmark Suite for Examples

This module contains benchmarks for the example implementations.
"""

import torch
import triton
import triton.language as tl
import time
import sys
import os

# Add the parent directory to the path to import examples
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from examples.llm_inference_optimization import optimized_attention, transformer_matmul, layer_norm

class BenchmarkExamples:
    """
    📊 BENCHMARK SUITE FOR EXAMPLES
    
    Benchmarks for the example implementations.
    """
    
    def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 100):
        """
        Initialize the benchmark suite.
        
        Args:
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        
        print(f"🚀 Initializing Examples Benchmark Suite")
        print(f"   Device: {self.device}")
        print(f"   Warmup runs: {self.warmup_runs}")
        print(f"   Benchmark runs: {self.benchmark_runs}")
    
    def benchmark_transformer_matmul(self):
        """Benchmark transformer matrix multiplication."""
        print("\n📊 Benchmarking Transformer Matrix Multiplication:")
        print("=" * 50)
        
        sizes = [
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ]
        
        for M, K, N in sizes:
            print(f"\n📈 Size: {M}x{K} @ {K}x{N} = {M}x{N}")
            
            # Create test data
            a = torch.randn(M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = transformer_matmul(a, b)
                _ = torch.matmul(a, b)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = transformer_matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = torch.matmul(a, b)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            triton_result = transformer_matmul(a, b)
            pytorch_result = torch.matmul(a, b)
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_layer_norm(self):
        """Benchmark layer normalization."""
        print("\n📊 Benchmarking Layer Normalization:")
        print("=" * 50)
        
        sizes = [
            (128, 512),
            (256, 1024),
            (512, 2048),
            (1024, 4096),
        ]
        
        for seq_len, hidden_dim in sizes:
            print(f"\n📈 Size: {seq_len}x{hidden_dim}")
            
            # Create test data
            input_tensor = torch.randn(seq_len, hidden_dim, device=self.device, dtype=self.dtype)
            weight = torch.ones(hidden_dim, device=self.device, dtype=self.dtype)
            bias = torch.zeros(hidden_dim, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = layer_norm(input_tensor, weight, bias)
                _ = torch.nn.functional.layer_norm(input_tensor, [hidden_dim], weight, bias)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = layer_norm(input_tensor, weight, bias)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = torch.nn.functional.layer_norm(input_tensor, [hidden_dim], weight, bias)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            triton_result = layer_norm(input_tensor, weight, bias)
            pytorch_result = torch.nn.functional.layer_norm(input_tensor, [hidden_dim], weight, bias)
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def benchmark_attention_mechanism(self):
        """Benchmark attention mechanism."""
        print("\n📊 Benchmarking Attention Mechanism:")
        print("=" * 50)
        
        # Test different attention configurations
        configs = [
            (2, 8, 128, 64),   # batch_size, num_heads, seq_len, head_dim
            (4, 8, 256, 64),
            (8, 16, 512, 64),
        ]
        
        for batch_size, num_heads, seq_len, head_dim in configs:
            print(f"\n📈 Config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
            
            # Create test data
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                           device=self.device, dtype=self.dtype)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                           device=self.device, dtype=self.dtype)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                           device=self.device, dtype=self.dtype)
            
            try:
                # Warmup
                for _ in range(self.warmup_runs):
                    _ = optimized_attention(q, k, v)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark Triton
                start_time = time.time()
                for _ in range(self.benchmark_runs):
                    _ = optimized_attention(q, k, v)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
                
                print(f"  Triton: {triton_time:.3f} ms")
                print(f"  Status: ✅ Working")
                
            except Exception as e:
                print(f"  Status: ❌ Error - {str(e)}")
    
    def benchmark_llm_pipeline(self):
        """Benchmark complete LLM pipeline."""
        print("\n📊 Benchmarking LLM Pipeline:")
        print("=" * 50)
        
        # Test different pipeline configurations
        configs = [
            (2, 128, 512),   # batch_size, seq_len, hidden_dim
            (4, 256, 1024),
            (8, 512, 2048),
        ]
        
        for batch_size, seq_len, hidden_dim in configs:
            print(f"\n📈 Config: batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
            
            # Create test data
            input_embeddings = torch.randn(batch_size, seq_len, hidden_dim, 
                                          device=self.device, dtype=self.dtype)
            
            # Attention weights
            q_weight = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype)
            k_weight = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype)
            v_weight = torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=self.dtype)
            
            # Feed-forward weights
            ff_weight1 = torch.randn(hidden_dim, hidden_dim * 4, device=self.device, dtype=self.dtype)
            ff_weight2 = torch.randn(hidden_dim * 4, hidden_dim, device=self.device, dtype=self.dtype)
            
            # Layer normalization weights
            norm_weight = torch.ones(hidden_dim, device=self.device, dtype=self.dtype)
            norm_bias = torch.zeros(hidden_dim, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                # Attention
                q = torch.matmul(input_embeddings, q_weight)
                k = torch.matmul(input_embeddings, k_weight)
                v = torch.matmul(input_embeddings, v_weight)
                
                # Feed-forward
                ff_output = transformer_matmul(input_embeddings, ff_weight1)
                ff_output = torch.relu(ff_output)
                ff_output = transformer_matmul(ff_output, ff_weight2)
                
                # Layer normalization
                norm_output = layer_norm(input_embeddings, norm_weight, norm_bias)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark complete pipeline
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                # Attention
                q = torch.matmul(input_embeddings, q_weight)
                k = torch.matmul(input_embeddings, k_weight)
                v = torch.matmul(input_embeddings, v_weight)
                
                # Feed-forward
                ff_output = transformer_matmul(input_embeddings, ff_weight1)
                ff_output = torch.relu(ff_output)
                ff_output = transformer_matmul(ff_output, ff_weight2)
                
                # Layer normalization
                norm_output = layer_norm(input_embeddings, norm_weight, norm_bias)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            pipeline_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            print(f"  Pipeline Time: {pipeline_time:.3f} ms")
            print(f"  Status: ✅ Working")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        print("\n📊 Benchmarking Memory Usage:")
        print("=" * 50)
        
        if self.device != 'cuda':
            print("  Skipping memory usage benchmarks (CUDA not available)")
            return
        
        # Test with different tensor sizes
        sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]
        
        for M, K, N in sizes:
            print(f"\n📈 Size: {M}x{K} @ {K}x{N} = {M}x{N}")
            
            # Create test data
            a = torch.randn(M, K, device=self.device, dtype=self.dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            # Measure memory before
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
            
            # Perform operation
            result = transformer_matmul(a, b)
            
            # Measure memory after
            memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_used = memory_after - memory_before
            
            print(f"  Memory Used: {memory_used:.2f} GB")
            print(f"  Status: ✅ Working")
            
            # Clean up
            del a, b, result
            torch.cuda.empty_cache()
    
    def benchmark_different_dtypes(self):
        """Benchmark different data types."""
        print("\n📊 Benchmarking Different Data Types:")
        print("=" * 50)
        
        if self.device != 'cuda':
            print("  Skipping data type benchmarks (CUDA not available)")
            return
        
        # Test different data types
        dtypes = [
            (torch.float32, "float32"),
            (torch.float16, "float16"),
            (torch.bfloat16, "bfloat16"),
        ]
        
        M, K, N = 1024, 512, 768
        
        for dtype, name in dtypes:
            print(f"\n📈 {name}:")
            
            # Create test data
            a = torch.randn(M, K, device=self.device, dtype=dtype)
            b = torch.randn(K, N, device=self.device, dtype=self.dtype)
            
            # Warmup
            for _ in range(self.warmup_runs):
                _ = transformer_matmul(a, b)
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Benchmark Triton
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = transformer_matmul(a, b)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            # Benchmark PyTorch
            start_time = time.time()
            for _ in range(self.benchmark_runs):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / self.benchmark_runs * 1000
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            print(f"  Triton:  {triton_time:.3f} ms")
            print(f"  PyTorch: {pytorch_time:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Verify correctness
            triton_result = transformer_matmul(a, b)
            pytorch_result = torch.matmul(a, b)
            is_correct = torch.allclose(triton_result, pytorch_result, rtol=1e-3, atol=1e-3)
            print(f"  Correct: {'✅' if is_correct else '❌'}")
    
    def run_all_benchmarks(self):
        """Run all example benchmarks."""
        print("🚀 Running All Example Benchmarks")
        print("=" * 70)
        
        self.benchmark_transformer_matmul()
        self.benchmark_layer_norm()
        self.benchmark_attention_mechanism()
        self.benchmark_llm_pipeline()
        self.benchmark_memory_usage()
        self.benchmark_different_dtypes()
        
        print("\n🎉 All Example Benchmarks Complete!")

def main():
    """Main function to run example benchmarks."""
    benchmark_suite = BenchmarkExamples()
    benchmark_suite.run_all_benchmarks()

if __name__ == "__main__":
    main()
