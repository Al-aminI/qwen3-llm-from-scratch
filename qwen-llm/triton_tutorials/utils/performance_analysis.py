"""
📈 Performance Analysis Utilities

This module provides utilities for analyzing performance characteristics
of Triton kernels and comparing them with reference implementations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time

@dataclass
class PerformanceMetrics:
    """Performance metrics for a kernel."""
    name: str
    execution_time: float
    memory_bandwidth: float
    arithmetic_intensity: float
    gpu_utilization: float
    cache_hit_rate: float
    throughput: float
    efficiency: float

class PerformanceAnalyzer:
    """
    📈 PERFORMANCE ANALYZER
    
    A comprehensive performance analyzer for Triton kernels.
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.metrics = []
        self.gpu_available = torch.cuda.is_available()
    
    def analyze_kernel(self,
                      kernel_func: callable,
                      name: str,
                      input_size: int,
                      dtype: torch.dtype = torch.float32,
                      num_runs: int = 100) -> PerformanceMetrics:
        """
        Analyze the performance of a kernel.
        
        Args:
            kernel_func: Kernel function to analyze
            name: Name of the kernel
            input_size: Size of input data
            dtype: Data type
            num_runs: Number of runs for averaging
            
        Returns:
            PerformanceMetrics object
        """
        try:
            # Create test data
            x = torch.randn(input_size, device='cuda' if self.gpu_available else 'cpu', dtype=dtype)
            output = torch.empty_like(x)
            
            # Warmup
            for _ in range(10):
                _ = kernel_func(x, output)
            
            if self.gpu_available:
                torch.cuda.synchronize()
            
            # Measure execution time
            start_time = time.time()
            for _ in range(num_runs):
                _ = kernel_func(x, output)
            if self.gpu_available:
                torch.cuda.synchronize()
            end_time = time.time()
            
            execution_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
            
            # Calculate memory bandwidth
            bytes_transferred = input_size * dtype.itemsize * 2  # read + write
            memory_bandwidth = (bytes_transferred / (execution_time / 1000)) / (1024**3)  # GB/s
            
            # Calculate arithmetic intensity (simplified)
            arithmetic_intensity = 1.0  # This would need to be calculated based on the kernel
            
            # Calculate throughput
            throughput = input_size / (execution_time / 1000)  # elements per second
            
            # Calculate efficiency (simplified)
            efficiency = min(memory_bandwidth / 1000, 1.0)  # Assume 1000 GB/s peak bandwidth
            
            metrics = PerformanceMetrics(
                name=name,
                execution_time=execution_time,
                memory_bandwidth=memory_bandwidth,
                arithmetic_intensity=arithmetic_intensity,
                gpu_utilization=0.0,  # Would need GPU profiling
                cache_hit_rate=0.0,   # Would need cache profiling
                throughput=throughput,
                efficiency=efficiency
            )
            
            self.metrics.append(metrics)
            return metrics
            
        except Exception as e:
            print(f"Error analyzing kernel {name}: {e}")
            return None
    
    def compare_kernels(self,
                       triton_func: callable,
                       pytorch_func: callable,
                       name: str,
                       input_size: int,
                       dtype: torch.dtype = torch.float32,
                       num_runs: int = 100) -> Tuple[PerformanceMetrics, PerformanceMetrics]:
        """
        Compare performance between Triton and PyTorch implementations.
        
        Args:
            triton_func: Triton implementation
            pytorch_func: PyTorch implementation
            name: Name of the comparison
            input_size: Size of input data
            dtype: Data type
            num_runs: Number of runs for averaging
            
        Returns:
            Tuple of (Triton metrics, PyTorch metrics)
        """
        triton_metrics = self.analyze_kernel(triton_func, f"{name}_triton", input_size, dtype, num_runs)
        pytorch_metrics = self.analyze_kernel(pytorch_func, f"{name}_pytorch", input_size, dtype, num_runs)
        
        return triton_metrics, pytorch_metrics
    
    def analyze_scaling(self,
                       kernel_func: callable,
                       name: str,
                       sizes: List[int],
                       dtype: torch.dtype = torch.float32,
                       num_runs: int = 100) -> List[PerformanceMetrics]:
        """
        Analyze how kernel performance scales with input size.
        
        Args:
            kernel_func: Kernel function to analyze
            name: Name of the kernel
            sizes: List of input sizes to test
            dtype: Data type
            num_runs: Number of runs for averaging
            
        Returns:
            List of PerformanceMetrics objects
        """
        scaling_metrics = []
        
        for size in sizes:
            metrics = self.analyze_kernel(kernel_func, f"{name}_size_{size}", size, dtype, num_runs)
            if metrics:
                scaling_metrics.append(metrics)
        
        return scaling_metrics
    
    def plot_performance(self, metrics_list: List[PerformanceMetrics], 
                        metric: str = 'execution_time', 
                        title: str = 'Performance Analysis'):
        """
        Plot performance metrics.
        
        Args:
            metrics_list: List of performance metrics
            metric: Metric to plot
            title: Plot title
        """
        if not metrics_list:
            print("No metrics to plot")
            return
        
        # Extract data
        names = [m.name for m in metrics_list]
        values = [getattr(m, metric) for m in metrics_list]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.bar(names, values)
        plt.title(title)
        plt.xlabel('Kernel')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_scaling(self, scaling_metrics: List[PerformanceMetrics], 
                    metric: str = 'execution_time',
                    title: str = 'Scaling Analysis'):
        """
        Plot scaling analysis.
        
        Args:
            scaling_metrics: List of performance metrics for different sizes
            metric: Metric to plot
            title: Plot title
        """
        if not scaling_metrics:
            print("No scaling metrics to plot")
            return
        
        # Extract data
        sizes = []
        values = []
        
        for metrics in scaling_metrics:
            # Extract size from name
            size_str = metrics.name.split('_size_')[-1]
            size = int(size_str)
            sizes.append(size)
            values.append(getattr(metrics, metric))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, values, 'o-')
        plt.title(title)
        plt.xlabel('Input Size')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, triton_metrics: PerformanceMetrics, 
                       pytorch_metrics: PerformanceMetrics,
                       title: str = 'Triton vs PyTorch Comparison'):
        """
        Plot comparison between Triton and PyTorch implementations.
        
        Args:
            triton_metrics: Triton performance metrics
            pytorch_metrics: PyTorch performance metrics
            title: Plot title
        """
        if not triton_metrics or not pytorch_metrics:
            print("Missing metrics for comparison")
            return
        
        # Extract metrics
        metrics = ['execution_time', 'memory_bandwidth', 'throughput', 'efficiency']
        triton_values = [getattr(triton_metrics, m) for m in metrics]
        pytorch_values = [getattr(pytorch_metrics, m) for m in metrics]
        
        # Create plot
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, triton_values, width, label='Triton', alpha=0.8)
        plt.bar(x + width/2, pytorch_values, width, label='PyTorch', alpha=0.8)
        
        plt.title(title)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics])
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, filename: str = 'performance_report.json'):
        """
        Generate a performance report.
        
        Args:
            filename: Output filename
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_available': self.gpu_available,
            'metrics': []
        }
        
        for metrics in self.metrics:
            report['metrics'].append({
                'name': metrics.name,
                'execution_time': metrics.execution_time,
                'memory_bandwidth': metrics.memory_bandwidth,
                'arithmetic_intensity': metrics.arithmetic_intensity,
                'gpu_utilization': metrics.gpu_utilization,
                'cache_hit_rate': metrics.cache_hit_rate,
                'throughput': metrics.throughput,
                'efficiency': metrics.efficiency
            })
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to {filename}")
    
    def print_summary(self):
        """Print a summary of performance analysis."""
        if not self.metrics:
            print("No performance metrics available")
            return
        
        print("\n📈 Performance Analysis Summary:")
        print("=" * 80)
        print(f"{'Name':<30} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15} {'Throughput':<15} {'Efficiency':<12}")
        print("-" * 80)
        
        for metrics in self.metrics:
            print(f"{metrics.name:<30} {metrics.execution_time:<12.3f} {metrics.memory_bandwidth:<15.1f} {metrics.throughput:<15.0f} {metrics.efficiency:<12.2f}")
    
    def clear_metrics(self):
        """Clear all performance metrics."""
        self.metrics = []

def analyze_vector_operations():
    """Analyze performance of vector operations."""
    print("📈 Analyzing Vector Operations:")
    print("=" * 50)
    
    analyzer = PerformanceAnalyzer()
    
    # Test different sizes
    sizes = [1024, 4096, 16384, 65536, 262144]
    
    for size in sizes:
        print(f"\n📊 Size: {size:,} elements")
        
        # Create test data
        a = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        b = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        
        # Analyze addition
        def vector_add(x, y):
            return x + y
        
        analyzer.analyze_kernel(vector_add, f"Vector Addition ({size:,})", size)
        
        # Analyze multiplication
        def vector_mul(x, y):
            return x * y
        
        analyzer.analyze_kernel(vector_mul, f"Vector Multiplication ({size:,})", size)
    
    analyzer.print_summary()
    return analyzer

def analyze_matrix_operations():
    """Analyze performance of matrix operations."""
    print("\n📈 Analyzing Matrix Operations:")
    print("=" * 50)
    
    analyzer = PerformanceAnalyzer()
    
    # Test different matrix sizes
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    for M, K, N in sizes:
        print(f"\n📊 Size: {M}x{K} @ {K}x{N} = {M}x{N}")
        
        # Create test data
        a = torch.randn(M, K, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        
        # Analyze matrix multiplication
        def matrix_mul(x, y):
            return torch.matmul(x, y)
        
        analyzer.analyze_kernel(matrix_mul, f"Matrix Multiplication ({M}x{K}x{N})", M * K)
    
    analyzer.print_summary()
    return analyzer

def analyze_scaling():
    """Analyze scaling performance."""
    print("\n📈 Analyzing Scaling Performance:")
    print("=" * 50)
    
    analyzer = PerformanceAnalyzer()
    
    # Test scaling with vector addition
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]
    
    def vector_add(x, y):
        return x + y
    
    scaling_metrics = analyzer.analyze_scaling(vector_add, "Vector Addition", sizes)
    
    # Plot scaling
    analyzer.plot_scaling(scaling_metrics, 'execution_time', 'Vector Addition Scaling')
    
    return analyzer

if __name__ == "__main__":
    # Analyze vector operations
    vector_analyzer = analyze_vector_operations()
    
    # Analyze matrix operations
    matrix_analyzer = analyze_matrix_operations()
    
    # Analyze scaling
    scaling_analyzer = analyze_scaling()
    
    # Generate reports
    vector_analyzer.generate_report("vector_performance_report.json")
    matrix_analyzer.generate_report("matrix_performance_report.json")
    scaling_analyzer.generate_report("scaling_performance_report.json")
    
    print("\n✅ Performance analysis complete!")
