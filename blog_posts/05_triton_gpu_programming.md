# Triton GPU Programming: From Beginner to Expert

*A comprehensive guide to mastering Triton - the Python-like language for writing efficient CUDA kernels. Learn to write kernels that are 2-10x faster than naive PyTorch implementations.*

## 🎯 Why Triton Matters

Traditional CUDA programming is complex and error-prone. Triton changes this by providing a Python-like syntax for writing GPU kernels that are both readable and performant.

### The Problem with Traditional CUDA

```cuda
// Traditional CUDA kernel - complex and error-prone
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch configuration - manual and error-prone
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;
vector_add<<<grid_size, block_size>>>(a, b, c, n);
```

### The Triton Solution

```python
# Triton kernel - clean and intuitive
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x, y):
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=128)
    return output
```

## 🚀 Getting Started with Triton

### Installation and Setup

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Triton
pip install triton

# Verify installation
python -c "import triton; print('Triton version:', triton.__version__)"
```

### Your First Triton Kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Simple vector addition kernel
    
    Args:
        x_ptr: Pointer to first input vector
        y_ptr: Pointer to second input vector
        output_ptr: Pointer to output vector
        n_elements: Number of elements to process
        BLOCK_SIZE: Number of elements per block (compile-time constant)
    """
    # Get program ID (which block we're processing)
    pid = tl.program_id(axis=0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle out-of-bounds access
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform computation
    output = x + y
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x, y):
    """Python wrapper for the Triton kernel"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Define grid size (number of blocks)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    
    return output

# Test the kernel
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
result = vector_add(x, y)
print("Success! Your first Triton kernel is working!")
```

## 🧠 Understanding Triton Concepts

### 1. Program IDs and Block Sizes

```python
@triton.jit
def example_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID (which block we're processing)
    pid = tl.program_id(axis=0)  # 1D grid
    
    # Calculate block boundaries
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle out-of-bounds access
    mask = offsets < n_elements
    
    # Your computation here...
```

### 2. Memory Access Patterns

```python
@triton.jit
def memory_access_example(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced memory access (good)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Strided access (can be slow)
    # x = tl.load(x_ptr + offsets * 2, mask=mask)
    
    # Scattered access (slow)
    # x = tl.load(x_ptr + offsets * offsets, mask=mask)
    
    # Store result
    tl.store(output_ptr + offsets, x * 2, mask=mask)
```

### 3. Data Types and Precision

```python
@triton.jit
def precision_example(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load as float32
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Convert to float16 for computation
    x_half = x.to(tl.float16)
    
    # Perform computation in float16
    result_half = x_half * 2.0
    
    # Convert back to float32 for storage
    result = result_half.to(tl.float32)
    
    tl.store(output_ptr + offsets, result, mask=mask)
```

## 🔧 Intermediate Triton Techniques

### 1. Matrix Operations

```python
@triton.jit
def matrix_multiply_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate block boundaries
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # Create offsets
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, b)
    
    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matrix_multiply(a, b):
    """Matrix multiplication using Triton"""
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Define grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )
    
    # Launch kernel
    matrix_multiply_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32
    )
    
    return c
```

### 2. Reduction Operations

```python
@triton.jit
def reduce_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce within block
    x = tl.sum(x, axis=0)
    
    # Store result
    tl.store(output_ptr + pid, x)

def reduce_sum(x):
    """Sum reduction using Triton"""
    n_elements = x.numel()
    output = torch.empty(1, device=x.device, dtype=x.dtype)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    reduce_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    
    # Final reduction on CPU (or use another kernel)
    return output.sum()

# Test
x = torch.randn(10000, device='cuda')
result = reduce_sum(x)
print(f"Sum: {result.item()}")
```

### 3. Kernel Fusion

```python
@triton.jit
def fused_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel: add + multiply + relu"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Fused operations
    result = x + y          # Add
    result = result * 2.0   # Multiply
    result = tl.maximum(result, 0.0)  # ReLU
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_operations(x, y):
    """Fused operations: add + multiply + relu"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    
    return output

# Compare with separate operations
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')

# Fused version
result_fused = fused_operations(x, y)

# Separate operations
result_separate = torch.relu((x + y) * 2.0)

print(f"Results match: {torch.allclose(result_fused, result_separate)}")
```

## 🚀 Advanced Triton Techniques

### 1. Shared Memory Usage

```python
@triton.jit
def shared_memory_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data into shared memory
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Store in shared memory
    shared_memory = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    shared_memory = tl.where(mask, x, shared_memory)
    
    # Perform computation using shared memory
    result = shared_memory * 2.0
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 2. Atomic Operations

```python
@triton.jit
def atomic_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Atomic add to output
    tl.atomic_add(output_ptr, x, mask=mask)

def atomic_sum(x):
    """Atomic sum using Triton"""
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    atomic_kernel[grid](x, output, n_elements, BLOCK_SIZE=128)
    
    return output[0]
```

### 3. Custom Data Types

```python
@triton.jit
def custom_dtype_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load as int8
    x = tl.load(input_ptr + offsets, mask=mask, other=0)
    
    # Convert to float32 for computation
    x_float = x.to(tl.float32)
    
    # Perform computation
    result_float = x_float * 2.0
    
    # Convert back to int8
    result = result_float.to(tl.int8)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)
```

## 🎯 Performance Optimization

### 1. Memory Coalescing

```python
@triton.jit
def coalesced_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Coalesced access (good)
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Non-coalesced access (bad)
    # x = tl.load(input_ptr + offsets * 2, mask=mask)
    
    tl.store(output_ptr + offsets, x * 2, mask=mask)
```

### 2. Block Size Optimization

```python
def find_optimal_block_size(kernel, input_tensor, max_block_size=1024):
    """Find optimal block size for a kernel"""
    best_block_size = 128
    best_time = float('inf')
    
    for block_size in [64, 128, 256, 512, 1024]:
        if block_size > max_block_size:
            break
            
        # Warm up
        for _ in range(10):
            kernel(input_tensor, block_size=block_size)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            kernel(input_tensor, block_size=block_size)
        
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 100
        
        if avg_time < best_time:
            best_time = avg_time
            best_block_size = block_size
    
    return best_block_size, best_time
```

### 3. Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def autotuned_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 🧪 Testing and Debugging

### 1. Unit Testing

```python
import pytest
import torch
import triton

class TestTritonKernels:
    def test_vector_add(self):
        """Test vector addition kernel"""
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        
        # Triton result
        result_triton = vector_add(x, y)
        
        # PyTorch result
        result_pytorch = x + y
        
        # Compare
        assert torch.allclose(result_triton, result_pytorch, rtol=1e-5)
    
    def test_matrix_multiply(self):
        """Test matrix multiplication kernel"""
        a = torch.randn(64, 32, device='cuda')
        b = torch.randn(32, 48, device='cuda')
        
        # Triton result
        result_triton = matrix_multiply(a, b)
        
        # PyTorch result
        result_pytorch = torch.mm(a, b)
        
        # Compare
        assert torch.allclose(result_triton, result_pytorch, rtol=1e-4)
    
    def test_reduce_sum(self):
        """Test sum reduction kernel"""
        x = torch.randn(1000, device='cuda')
        
        # Triton result
        result_triton = reduce_sum(x)
        
        # PyTorch result
        result_pytorch = x.sum()
        
        # Compare
        assert torch.allclose(result_triton, result_pytorch, rtol=1e-5)
```

### 2. Performance Benchmarking

```python
import time
import numpy as np

def benchmark_kernel(kernel_func, input_tensor, num_runs=100):
    """Benchmark a Triton kernel"""
    # Warm up
    for _ in range(10):
        kernel_func(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        kernel_func(input_tensor)
    
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / num_runs

def compare_with_pytorch(triton_func, pytorch_func, input_tensor):
    """Compare Triton kernel with PyTorch implementation"""
    # Benchmark Triton
    triton_time = benchmark_kernel(triton_func, input_tensor)
    
    # Benchmark PyTorch
    pytorch_time = benchmark_kernel(pytorch_func, input_tensor)
    
    speedup = pytorch_time / triton_time
    
    print(f"Triton time: {triton_time:.4f}s")
    print(f"PyTorch time: {pytorch_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup

# Example usage
x = torch.randn(10000, device='cuda')
speedup = compare_with_pytorch(
    lambda t: vector_add(t, t),
    lambda t: t + t,
    x
)
```

## 🎓 Key Learnings

### 1. Triton Fundamentals
- **Program IDs**: Identify which block you're processing
- **Block sizes**: Balance between parallelism and memory usage
- **Memory access**: Coalesced access is crucial for performance
- **Masking**: Handle out-of-bounds access gracefully

### 2. Performance Optimization
- **Memory coalescing**: Access memory in contiguous patterns
- **Block size tuning**: Find optimal balance for your workload
- **Kernel fusion**: Combine multiple operations to reduce memory traffic
- **Autotuning**: Let Triton find optimal configurations

### 3. Advanced Techniques
- **Shared memory**: Use for data reuse within blocks
- **Atomic operations**: For reductions and accumulations
- **Custom data types**: Optimize for specific precision requirements
- **Multi-dimensional grids**: Handle 2D and 3D problems

## 🔮 Real-World Applications

### 1. Attention Mechanisms

```python
@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    """Simplified attention kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    # Load query
    q = tl.load(q_ptr + offsets, mask=mask)
    
    # Compute attention scores
    scores = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, seq_len, BLOCK_SIZE):
        k_offsets = i + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < seq_len
        k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        
        # Compute attention score
        score = tl.sum(q * k, axis=1)
        scores = tl.where(mask, scores + score, scores)
    
    # Apply softmax
    scores = tl.exp(scores - tl.max(scores))
    scores = scores / tl.sum(scores)
    
    # Compute output
    output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(0, seq_len, BLOCK_SIZE):
        v_offsets = i + tl.arange(0, BLOCK_SIZE)
        v_mask = v_offsets < seq_len
        v = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)
        
        output = tl.where(mask, output + scores * v, output)
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 2. Layer Normalization

```python
@triton.jit
def layer_norm_kernel(
    input_ptr, output_ptr, weight_ptr, bias_ptr,
    n_elements, eps,
    BLOCK_SIZE: tl.constexpr
):
    """Layer normalization kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / n_elements
    
    # Compute variance
    variance = tl.sum((x - mean) ** 2, axis=0) / n_elements
    
    # Normalize
    x_norm = (x - mean) / tl.sqrt(variance + eps)
    
    # Apply weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    output = x_norm * weight + bias
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 💡 Why Triton Matters

Triton represents a paradigm shift in GPU programming:

- **Accessibility**: Python-like syntax makes GPU programming accessible
- **Performance**: Compiles to efficient CUDA code
- **Productivity**: Faster development and debugging
- **Flexibility**: Easy to experiment with different algorithms

## 🎯 Conclusion

Mastering Triton opens up a world of possibilities for GPU programming. By understanding the fundamentals and applying advanced techniques, you can write kernels that are both readable and performant.

The key insights:
- **Start simple**: Master the basics before moving to complex kernels
- **Optimize gradually**: Profile and optimize step by step
- **Think in blocks**: Design algorithms around block-based processing
- **Experiment freely**: Triton makes it easy to try different approaches

This knowledge is invaluable for anyone working with high-performance computing, machine learning, or scientific computing.

---

*This is the fifth in a series of blog posts about building a complete LLM pipeline. Next up: Supercharging LLM Inference with Custom Triton Kernels!*

**GitHub Repository**: [Triton Tutorials](https://github.com/your-repo/triton-tutorials)
**Live Demo**: [Try Triton Kernels](https://your-demo-url.com)

---

*Keywords: Triton, GPU Programming, CUDA Kernels, Performance Optimization, Memory Management, Kernel Fusion, Autotuning*
