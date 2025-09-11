# 🚀 Quick Start Guide

Get up and running with Triton Tutorials in minutes!

## 📋 Prerequisites

### **Required:**
- Python 3.8+
- NVIDIA GPU with compute capability 7.0+
- CUDA 11.8+

### **Recommended:**
- 8GB+ GPU memory
- Ubuntu 20.04+ or similar Linux distribution

## ⚡ Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/qwen3/triton-tutorials.git
cd triton-tutorials
```

### **2. Install Dependencies**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Triton
pip install triton

# Install tutorial dependencies
pip install -r requirements.txt
```

### **3. Verify Installation**
```bash
python -c "import torch; import triton; print('✅ Installation successful!')"
```

## 🎯 Your First Kernel

### **Run Lesson 1: GPU Fundamentals**
```bash
# Using the CLI
triton-tutorial run beginner 1

# Or directly
python lessons/beginner/lesson_01_gpu_fundamentals.py
```

### **Expected Output:**
```
🎯 LESSON 1: GPU FUNDAMENTALS & TRITON BASICS
======================================================================
Welcome to your first Triton tutorial!

🧠 GPU Architecture Overview:
==================================================
GPU: NVIDIA GeForce RTX 4090
Compute Capability: 8.9
Multiprocessors: 128
CUDA Cores: 8192
Memory: 24.0 GB
Memory Bandwidth: ~2400 GB/s

🧪 Testing Vector Addition Kernel:
==================================================
📊 Test: Power of 2 size (size=1024)
✅ PASS: Results match PyTorch reference

📊 Benchmarking Vector Addition:
==================================================
📈 Size: 1,024 elements
  Triton:  0.045 ms
  PyTorch: 0.052 ms
  Speedup: 1.16x
```

## 📚 Learning Path

### **Phase 1: Foundations (Beginner)**
```bash
# Lesson 1: GPU Fundamentals & Triton Basics
triton-tutorial run beginner 1

# Lesson 2: Memory Management & Data Types
triton-tutorial run beginner 2

# Lesson 3: Basic Operations & Kernels
triton-tutorial run beginner 3
```

### **Phase 2: Optimization (Intermediate)**
```bash
# Lesson 4: Matrix Operations & Tiling Strategies
triton-tutorial run intermediate 4

# Lesson 5: Advanced Memory Patterns & Optimization
triton-tutorial run intermediate 5

# Lesson 6: Kernel Fusion & Performance Tuning
triton-tutorial run intermediate 6
```

### **Phase 3: Specialization (Advanced)**
```bash
# Lesson 7: Attention Mechanisms & FlashAttention
triton-tutorial run advanced 7

# Lesson 8: Transformer Components & Optimization
triton-tutorial run advanced 8

# Lesson 9: MoE (Mixture of Experts) Kernels
triton-tutorial run advanced 9
```

### **Phase 4: Mastery (Expert)**
```bash
# Lesson 10: Autotuning & Advanced Optimization
triton-tutorial run expert 10

# Lesson 11: Custom Data Types & Quantization
triton-tutorial run expert 11

# Lesson 12: Production Systems & Deployment
triton-tutorial run expert 12
```

## 🛠️ CLI Commands

### **List All Lessons**
```bash
triton-tutorial list
```

### **Run Specific Lesson**
```bash
triton-tutorial run <level> <lesson_number>
# Example: triton-tutorial run beginner 1
```

### **Run Benchmarks**
```bash
# Benchmark specific lesson
triton-tutorial benchmark --lesson 1

# Run all benchmarks
triton-tutorial benchmark --all
```

### **System Information**
```bash
triton-tutorial info
```

## 🧪 Testing Your Setup

### **Quick Test**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def hello_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2.0
    tl.store(output_ptr + offsets, output, mask=mask)

# Test the kernel
x = torch.randn(1024, device='cuda')
output = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
hello_kernel[grid](x, output, 1024, BLOCK_SIZE=128)

print("✅ Your first Triton kernel works!")
print(f"Input: {x[:5]}")
print(f"Output: {output[:5]}")
```

## 🚨 Troubleshooting

### **Common Issues:**

#### **1. CUDA Not Available**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **2. Triton Import Error**
```bash
# Install Triton
pip install triton

# Or install from source
pip install git+https://github.com/openai/triton.git
```

#### **3. Memory Issues**
```bash
# Reduce batch sizes in tutorials
# Or use smaller test cases
```

#### **4. Performance Issues**
```bash
# Check GPU utilization
nvidia-smi

# Ensure proper CUDA version
python -c "import torch; print(torch.version.cuda)"
```

## 📖 Next Steps

1. **Complete the beginner lessons** to understand fundamentals
2. **Experiment with different parameters** (block sizes, data types)
3. **Try the intermediate lessons** for optimization techniques
4. **Build your own kernels** using the patterns you learn
5. **Contribute to the tutorials** with improvements or new examples

## 🤝 Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Discussions**: Share ideas and get help from the community
- **Documentation**: Check the full README.md for detailed information

## 🎉 Success!

You're now ready to start learning Triton! Begin with Lesson 1 and work your way through the tutorials. Each lesson builds on the previous ones, so take your time to understand the concepts.

**Happy coding!** 🚀
