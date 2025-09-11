"""
🔍 Profiling Utilities

This module provides utilities for profiling Triton kernels and analyzing
performance characteristics.
"""

import torch
import time
import psutil
import GPUtil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

@dataclass
class ProfileResult:
    """Results from a profiling run."""
    name: str
    execution_time: float
    memory_usage: float
    gpu_memory_usage: float
    gpu_utilization: float
    cpu_utilization: float
    throughput: Optional[float] = None
    error: Optional[str] = None

class PerformanceProfiler:
    """
    🔍 PERFORMANCE PROFILER
    
    A comprehensive profiler for Triton kernels and PyTorch operations.
    """
    
    def __init__(self):
        """Initialize the performance profiler."""
        self.results = []
        self.gpu_available = torch.cuda.is_available()
    
    def profile_function(self, 
                        func: Callable,
                        name: str,
                        *args, **kwargs) -> ProfileResult:
        """
        Profile a function's performance.
        
        Args:
            func: Function to profile
            name: Name of the function
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            ProfileResult object
        """
        try:
            # Get initial system state
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().used / 1024**3  # GB
            
            if self.gpu_available:
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                initial_gpu_util = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
            else:
                initial_gpu_memory = 0
                initial_gpu_util = 0
            
            # Warmup
            for _ in range(5):
                _ = func(*args, **kwargs)
            
            if self.gpu_available:
                torch.cuda.synchronize()
            
            # Profile execution
            start_time = time.time()
            result = func(*args, **kwargs)
            if self.gpu_available:
                torch.cuda.synchronize()
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Get final system state
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().used / 1024**3  # GB
            
            if self.gpu_available:
                final_gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                final_gpu_util = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0
            else:
                final_gpu_memory = 0
                final_gpu_util = 0
            
            # Calculate deltas
            memory_usage = final_memory - initial_memory
            gpu_memory_usage = final_gpu_memory - initial_gpu_memory
            cpu_utilization = (initial_cpu + final_cpu) / 2
            gpu_utilization = (initial_gpu_util + final_gpu_util) / 2
            
            profile_result = ProfileResult(
                name=name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_memory_usage=gpu_memory_usage,
                gpu_utilization=gpu_utilization,
                cpu_utilization=cpu_utilization
            )
            
            self.results.append(profile_result)
            return profile_result
            
        except Exception as e:
            error_result = ProfileResult(
                name=name,
                execution_time=0.0,
                memory_usage=0.0,
                gpu_memory_usage=0.0,
                gpu_utilization=0.0,
                cpu_utilization=0.0,
                error=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def profile_memory_bandwidth(self,
                                func: Callable,
                                name: str,
                                input_size: int,
                                dtype: torch.dtype = torch.float32) -> ProfileResult:
        """
        Profile memory bandwidth utilization.
        
        Args:
            func: Function to profile
            name: Name of the function
            input_size: Size of input tensor
            dtype: Data type of input tensor
            
        Returns:
            ProfileResult with bandwidth information
        """
        try:
            # Create test data
            x = torch.randn(input_size, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=dtype)
            output = torch.empty_like(x)
            
            # Warmup
            for _ in range(5):
                _ = func(x, output)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Profile execution
            start_time = time.time()
            _ = func(x, output)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Calculate bandwidth
            bytes_transferred = input_size * dtype.itemsize * 2  # read + write
            throughput = (bytes_transferred / (execution_time / 1000)) / (1024**3)  # GB/s
            
            profile_result = ProfileResult(
                name=name,
                execution_time=execution_time,
                memory_usage=0.0,
                gpu_memory_usage=0.0,
                gpu_utilization=0.0,
                cpu_utilization=0.0,
                throughput=throughput
            )
            
            self.results.append(profile_result)
            return profile_result
            
        except Exception as e:
            error_result = ProfileResult(
                name=name,
                execution_time=0.0,
                memory_usage=0.0,
                gpu_memory_usage=0.0,
                gpu_utilization=0.0,
                cpu_utilization=0.0,
                error=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def profile_kernel_launch_overhead(self,
                                     kernel_func: Callable,
                                     name: str,
                                     num_launches: int = 1000,
                                     *args, **kwargs) -> ProfileResult:
        """
        Profile kernel launch overhead.
        
        Args:
            kernel_func: Kernel function to profile
            name: Name of the kernel
            num_launches: Number of kernel launches
            *args: Arguments to pass to the kernel
            **kwargs: Keyword arguments to pass to the kernel
            
        Returns:
            ProfileResult with launch overhead information
        """
        try:
            # Warmup
            for _ in range(10):
                _ = kernel_func(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Profile multiple launches
            start_time = time.time()
            for _ in range(num_launches):
                _ = kernel_func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = (end_time - start_time) * 1000  # Convert to ms
            avg_launch_time = total_time / num_launches
            
            profile_result = ProfileResult(
                name=f"{name}_launch_overhead",
                execution_time=avg_launch_time,
                memory_usage=0.0,
                gpu_memory_usage=0.0,
                gpu_utilization=0.0,
                cpu_utilization=0.0,
                throughput=num_launches / (total_time / 1000)  # launches per second
            )
            
            self.results.append(profile_result)
            return profile_result
            
        except Exception as e:
            error_result = ProfileResult(
                name=f"{name}_launch_overhead",
                execution_time=0.0,
                memory_usage=0.0,
                gpu_memory_usage=0.0,
                gpu_utilization=0.0,
                cpu_utilization=0.0,
                error=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def print_results(self):
        """Print profiling results in a formatted table."""
        if not self.results:
            print("No profiling results available.")
            return
        
        print("\n🔍 Profiling Results:")
        print("=" * 100)
        print(f"{'Name':<30} {'Time (ms)':<12} {'Memory (GB)':<12} {'GPU Mem (GB)':<12} {'GPU Util %':<12} {'CPU Util %':<12}")
        print("-" * 100)
        
        for result in self.results:
            if result.error:
                print(f"{result.name:<30} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
            else:
                print(f"{result.name:<30} {result.execution_time:<12.3f} {result.memory_usage:<12.3f} {result.gpu_memory_usage:<12.3f} {result.gpu_utilization:<12.1f} {result.cpu_utilization:<12.1f}")
    
    def save_results(self, filename: str):
        """Save profiling results to a JSON file."""
        results_dict = []
        for result in self.results:
            results_dict.append({
                'name': result.name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'gpu_memory_usage': result.gpu_memory_usage,
                'gpu_utilization': result.gpu_utilization,
                'cpu_utilization': result.cpu_utilization,
                'throughput': result.throughput,
                'error': result.error
            })
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Profiling results saved to {filename}")
    
    def clear_results(self):
        """Clear all profiling results."""
        self.results = []

def profile_system_info():
    """Profile system information."""
    print("🔍 System Information:")
    print("=" * 50)
    
    # CPU information
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    
    # Memory information
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available")
    print(f"Memory Usage: {memory.percent}%")
    
    # GPU information
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA Available: No")
    
    # GPU utilization (if available)
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i} Utilization: {gpu.load * 100:.1f}%")
                print(f"  GPU {i} Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
    except:
        pass

def profile_kernel_performance():
    """Profile kernel performance."""
    print("\n🔍 Profiling Kernel Performance:")
    print("=" * 50)
    
    profiler = PerformanceProfiler()
    
    # Test vector operations
    size = 1024 * 1024  # 1M elements
    a = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
    b = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
    
    # Profile vector addition
    profiler.profile_function(
        lambda x, y: x + y,
        "Vector Addition",
        a, b
    )
    
    # Profile vector multiplication
    profiler.profile_function(
        lambda x, y: x * y,
        "Vector Multiplication",
        a, b
    )
    
    # Profile memory bandwidth
    def memory_copy(x, output):
        output.copy_(x)
    
    profiler.profile_memory_bandwidth(
        memory_copy,
        "Memory Copy",
        size
    )
    
    profiler.print_results()
    return profiler

if __name__ == "__main__":
    # Profile system information
    profile_system_info()
    
    # Profile kernel performance
    profiler = profile_kernel_performance()
    
    # Save results
    profiler.save_results("profiling_results.json")
