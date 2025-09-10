"""
Benchmarking utilities for fast inference.

This module provides utilities for measuring and comparing inference performance
across different methods and configurations.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """
    Results from a benchmark run.
    
    Attributes:
        method_name: Name of the inference method
        total_time: Total time in seconds
        total_tokens: Total number of tokens generated
        total_requests: Total number of requests processed
        throughput_tokens_per_sec: Tokens per second
        throughput_requests_per_sec: Requests per second
        memory_usage_mb: Peak memory usage in MB
        avg_time_per_request: Average time per request
        avg_tokens_per_request: Average tokens per request
    """
    method_name: str
    total_time: float
    total_tokens: int
    total_requests: int
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    memory_usage_mb: float
    avg_time_per_request: float
    avg_tokens_per_request: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method_name': self.method_name,
            'total_time': self.total_time,
            'total_tokens': self.total_tokens,
            'total_requests': self.total_requests,
            'throughput_tokens_per_sec': self.throughput_tokens_per_sec,
            'throughput_requests_per_sec': self.throughput_requests_per_sec,
            'memory_usage_mb': self.memory_usage_mb,
            'avg_time_per_request': self.avg_time_per_request,
            'avg_tokens_per_request': self.avg_tokens_per_request
        }


class BenchmarkRunner:
    """
    Benchmark runner for inference methods.
    
    This class provides utilities for running benchmarks and collecting
    performance metrics across different inference methods.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize benchmark runner.
        
        Args:
            device: Device to run benchmarks on ("auto", "cpu", "cuda")
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.results = []
    
    def run_benchmark(self, 
                     method_name: str,
                     inference_func: Callable,
                     test_prompts: List[str],
                     max_new_tokens: int = 50,
                     warmup_runs: int = 3,
                     num_runs: int = 1) -> BenchmarkResult:
        """
        Run a benchmark for a specific inference method.
        
        Args:
            method_name: Name of the inference method
            inference_func: Function that takes prompts and returns results
            test_prompts: List of test prompts
            max_new_tokens: Maximum tokens to generate per prompt
            warmup_runs: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            BenchmarkResult with performance metrics
        """
        print(f"🚀 Benchmarking {method_name}...")
        
        # Warmup runs
        if warmup_runs > 0:
            print(f"   Warming up with {warmup_runs} runs...")
            for _ in range(warmup_runs):
                _ = inference_func(test_prompts[:2], max_new_tokens)
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # Run benchmark
        total_times = []
        total_tokens = 0
        
        for run in range(num_runs):
            print(f"   Run {run + 1}/{num_runs}...")
            
            start_time = time.time()
            results = inference_func(test_prompts, max_new_tokens)
            end_time = time.time()
            
            run_time = end_time - start_time
            total_times.append(run_time)
            
            # Count tokens in results
            run_tokens = sum(len(str(result).split()) for result in results)
            total_tokens += run_tokens
        
        # Calculate metrics
        avg_time = np.mean(total_times)
        total_time = sum(total_times)
        total_requests = len(test_prompts) * num_runs
        
        throughput_tokens_per_sec = total_tokens / total_time
        throughput_requests_per_sec = total_requests / total_time
        avg_time_per_request = avg_time / len(test_prompts)
        avg_tokens_per_request = total_tokens / total_requests
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_usage_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_usage_mb = 0.0
        
        result = BenchmarkResult(
            method_name=method_name,
            total_time=total_time,
            total_tokens=total_tokens,
            total_requests=total_requests,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            throughput_requests_per_sec=throughput_requests_per_sec,
            memory_usage_mb=memory_usage_mb,
            avg_time_per_request=avg_time_per_request,
            avg_tokens_per_request=avg_tokens_per_request
        )
        
        self.results.append(result)
        return result
    
    def compare_methods(self) -> Dict[str, Any]:
        """
        Compare all benchmarked methods.
        
        Returns:
            Dictionary with comparison results
        """
        if len(self.results) < 2:
            return {"error": "Need at least 2 methods to compare"}
        
        # Sort by throughput
        sorted_results = sorted(self.results, key=lambda x: x.throughput_tokens_per_sec, reverse=True)
        
        comparison = {
            "best_method": sorted_results[0].method_name,
            "best_throughput": sorted_results[0].throughput_tokens_per_sec,
            "speedup_vs_worst": sorted_results[0].throughput_tokens_per_sec / sorted_results[-1].throughput_tokens_per_sec,
            "methods": [result.to_dict() for result in sorted_results]
        }
        
        return comparison
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        data = {
            "device": self.device,
            "results": [result.to_dict() for result in self.results],
            "comparison": self.compare_methods()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Results saved to {filepath}")
    
    def print_summary(self):
        """Print a summary of all benchmark results."""
        print("\n📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            print(f"\n🔹 {result.method_name}:")
            print(f"   Throughput: {result.throughput_tokens_per_sec:.1f} tokens/s")
            print(f"   Requests/s: {result.throughput_requests_per_sec:.2f}")
            print(f"   Memory: {result.memory_usage_mb:.1f} MB")
            print(f"   Avg time/request: {result.avg_time_per_request:.3f}s")
        
        if len(self.results) > 1:
            comparison = self.compare_methods()
            print(f"\n🏆 Best method: {comparison['best_method']}")
            print(f"   Speedup vs worst: {comparison['speedup_vs_worst']:.1f}x")


def benchmark_inference(engine, test_prompts: List[str], max_new_tokens: int = 50,
                       method_name: str = "Fast Inference") -> BenchmarkResult:
    """
    Quick benchmark function for inference engines.
    
    Args:
        engine: Inference engine with generate_batch method
        test_prompts: List of test prompts
        max_new_tokens: Maximum tokens to generate
        method_name: Name for the benchmark
        
    Returns:
        BenchmarkResult
    """
    runner = BenchmarkRunner()
    
    def inference_func(prompts, max_tokens):
        return engine.generate_batch(prompts, max_new_tokens=max_tokens)
    
    return runner.run_benchmark(method_name, inference_func, test_prompts, max_new_tokens)


def compare_methods(methods: Dict[str, Callable], test_prompts: List[str], 
                   max_new_tokens: int = 50) -> Dict[str, Any]:
    """
    Compare multiple inference methods.
    
    Args:
        methods: Dictionary mapping method names to inference functions
        test_prompts: List of test prompts
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Comparison results
    """
    runner = BenchmarkRunner()
    
    for method_name, inference_func in methods.items():
        runner.run_benchmark(method_name, inference_func, test_prompts, max_new_tokens)
    
    return runner.compare_methods()


def generate_test_prompts(num_prompts: int = 10, 
                         min_length: int = 20, 
                         max_length: int = 100) -> List[str]:
    """
    Generate test prompts for benchmarking.
    
    Args:
        num_prompts: Number of prompts to generate
        min_length: Minimum prompt length
        max_length: Maximum prompt length
        
    Returns:
        List of test prompts
    """
    base_prompts = [
        "Write a short story about",
        "Explain the concept of",
        "Tell me a joke about",
        "Describe the process of",
        "What is the meaning of",
        "How does one",
        "In a world where",
        "The ancient art of",
        "Once upon a time",
        "The future of technology"
    ]
    
    prompts = []
    for i in range(num_prompts):
        base = base_prompts[i % len(base_prompts)]
        length = np.random.randint(min_length, max_length)
        prompt = f"{base} {length} words: "
        prompts.append(prompt)
    
    return prompts


def create_performance_report(results: List[BenchmarkResult], 
                            output_file: Optional[str] = None) -> str:
    """
    Create a detailed performance report.
    
    Args:
        results: List of benchmark results
        output_file: Optional file to save report
        
    Returns:
        Report as string
    """
    report = []
    report.append("# 🚀 Fast Inference Performance Report")
    report.append("=" * 50)
    report.append("")
    
    # Summary table
    report.append("## 📊 Performance Summary")
    report.append("")
    report.append("| Method | Tokens/s | Requests/s | Memory (MB) | Avg Time/Req |")
    report.append("|--------|----------|------------|-------------|--------------|")
    
    for result in results:
        report.append(f"| {result.method_name} | "
                     f"{result.throughput_tokens_per_sec:.1f} | "
                     f"{result.throughput_requests_per_sec:.2f} | "
                     f"{result.memory_usage_mb:.1f} | "
                     f"{result.avg_time_per_request:.3f}s |")
    
    report.append("")
    
    # Detailed results
    report.append("## 📈 Detailed Results")
    report.append("")
    
    for result in results:
        report.append(f"### {result.method_name}")
        report.append("")
        report.append(f"- **Total Time**: {result.total_time:.2f}s")
        report.append(f"- **Total Tokens**: {result.total_tokens:,}")
        report.append(f"- **Total Requests**: {result.total_requests}")
        report.append(f"- **Throughput**: {result.throughput_tokens_per_sec:.1f} tokens/s")
        report.append(f"- **Request Rate**: {result.throughput_requests_per_sec:.2f} requests/s")
        report.append(f"- **Memory Usage**: {result.memory_usage_mb:.1f} MB")
        report.append(f"- **Avg Time/Request**: {result.avg_time_per_request:.3f}s")
        report.append(f"- **Avg Tokens/Request**: {result.avg_tokens_per_request:.1f}")
        report.append("")
    
    # Speedup analysis
    if len(results) > 1:
        report.append("## ⚡ Speedup Analysis")
        report.append("")
        
        sorted_results = sorted(results, key=lambda x: x.throughput_tokens_per_sec, reverse=True)
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        speedup = best.throughput_tokens_per_sec / worst.throughput_tokens_per_sec
        report.append(f"- **Best Method**: {best.method_name}")
        report.append(f"- **Worst Method**: {worst.method_name}")
        report.append(f"- **Speedup**: {speedup:.1f}x faster")
        report.append("")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"📄 Report saved to {output_file}")
    
    return report_text
