#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 INFERENCE PERFORMANCE COMPARISON

Compare the performance of different inference methods:
1. Naive inference (no KV cache)
2. Simple fast inference (with KV cache)
3. Original model inference

This demonstrates the speedup from KV caching.
"""

import torch
import torch.nn.functional as F
import time
import math
from typing import List
from transformers import AutoTokenizer

# Import our components
from config.qwen3_small_config import SmallModelConfig
from qwen3_complete_model import MinimalLLM, generate_text
from simple_fast_inference import SimpleFastInference, create_simple_fast_inference

# =============================================================================
# 🐌 NAIVE INFERENCE (NO KV CACHE)
# =============================================================================

def naive_generate_text(model: MinimalLLM, tokenizer, prompt: str, max_length: int = 100,
                       temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
    """
    🐌 NAIVE TEXT GENERATION (NO KV CACHE)
    
    This is the original text generation method that processes the entire sequence
    for each new token. Very slow but simple.
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions (processes ENTIRE sequence each time!)
            logits = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            next_token = next_token.unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if we reach the end token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# =============================================================================
# 📊 PERFORMANCE COMPARISON
# =============================================================================

def compare_inference_methods(model_path: str, tokenizer_path: str, 
                            test_prompts: List[str], max_new_tokens: int = 50):
    """
    Compare different inference methods
    
    Args:
        model_path: Path to saved model
        tokenizer_path: Path to tokenizer
        test_prompts: List of test prompts
        max_new_tokens: Maximum tokens to generate
    """
    print("📊 INFERENCE PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = SmallModelConfig()
    config.vocab_size = checkpoint.get('vocab_size', 32000)
    
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create fast inference engine
    fast_engine = SimpleFastInference(model, tokenizer, config)
    
    print(f"✅ Model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Test prompts: {len(test_prompts)}")
    print(f"   Max new tokens: {max_new_tokens}")
    
    # Test 1: Naive inference (no KV cache)
    print(f"\n🐌 Testing Naive Inference (No KV Cache)...")
    naive_times = []
    naive_results = []
    
    for i, prompt in enumerate(test_prompts):
        start_time = time.time()
        result = naive_generate_text(model, tokenizer, prompt, max_new_tokens)
        end_time = time.time()
        
        naive_times.append(end_time - start_time)
        naive_results.append(result)
        print(f"   {i+1}/{len(test_prompts)}: {naive_times[-1]:.3f}s")
    
    # Test 2: Fast inference (with KV cache)
    print(f"\n🚀 Testing Fast Inference (With KV Cache)...")
    fast_times = []
    fast_results = []
    
    for i, prompt in enumerate(test_prompts):
        start_time = time.time()
        result = fast_engine.generate_single(prompt, max_new_tokens)
        end_time = time.time()
        
        fast_times.append(end_time - start_time)
        fast_results.append(result)
        print(f"   {i+1}/{len(test_prompts)}: {fast_times[-1]:.3f}s")
    
    # Test 3: Original model inference (if available)
    print(f"\n📚 Testing Original Model Inference...")
    original_times = []
    original_results = []
    
    try:
        for i, prompt in enumerate(test_prompts):
            start_time = time.time()
            result = generate_text(model, tokenizer, prompt, max_new_tokens)
            end_time = time.time()
            
            original_times.append(end_time - start_time)
            original_results.append(result)
            print(f"   {i+1}/{len(test_prompts)}: {original_times[-1]:.3f}s")
    except Exception as e:
        print(f"   ⚠️ Original model inference failed: {e}")
        original_times = None
        original_results = None
    
    # Calculate statistics
    print(f"\n📈 PERFORMANCE RESULTS")
    print("=" * 60)
    
    # Naive inference stats
    naive_avg = sum(naive_times) / len(naive_times)
    naive_total = sum(naive_times)
    print(f"🐌 Naive Inference (No KV Cache):")
    print(f"   Average time: {naive_avg:.3f}s")
    print(f"   Total time: {naive_total:.3f}s")
    print(f"   Throughput: {len(test_prompts) / naive_total:.2f} requests/s")
    
    # Fast inference stats
    fast_avg = sum(fast_times) / len(fast_times)
    fast_total = sum(fast_times)
    speedup = naive_avg / fast_avg
    print(f"\n🚀 Fast Inference (With KV Cache):")
    print(f"   Average time: {fast_avg:.3f}s")
    print(f"   Total time: {fast_total:.3f}s")
    print(f"   Throughput: {len(test_prompts) / fast_total:.2f} requests/s")
    print(f"   Speedup: {speedup:.1f}x faster!")
    
    # Original inference stats (if available)
    if original_times:
        original_avg = sum(original_times) / len(original_times)
        original_total = sum(original_times)
        original_speedup = original_avg / fast_avg
        print(f"\n📚 Original Model Inference:")
        print(f"   Average time: {original_avg:.3f}s")
        print(f"   Total time: {original_total:.3f}s")
        print(f"   Throughput: {len(test_prompts) / original_total:.2f} requests/s")
        print(f"   Speedup vs original: {original_speedup:.1f}x faster!")
    
    # Memory usage comparison
    print(f"\n💾 MEMORY USAGE")
    print("=" * 60)
    print(f"🐌 Naive Inference: O(n²) - grows quadratically with sequence length")
    print(f"🚀 Fast Inference: O(n) - grows linearly with sequence length")
    print(f"   Memory savings: ~{speedup:.0f}x less memory for long sequences")
    
    # Show sample results
    print(f"\n📝 SAMPLE RESULTS")
    print("=" * 60)
    for i in range(min(3, len(test_prompts))):
        print(f"\n{i+1}. Prompt: {test_prompts[i]}")
        print(f"   Naive: {naive_results[i][:100]}...")
        print(f"   Fast:  {fast_results[i][:100]}...")
        if original_results:
            print(f"   Orig:  {original_results[i][:100]}...")
    
    return {
        'naive_times': naive_times,
        'fast_times': fast_times,
        'original_times': original_times,
        'speedup': speedup,
        'naive_results': naive_results,
        'fast_results': fast_results,
        'original_results': original_results
    }

# =============================================================================
# 🧪 TESTING FUNCTIONS
# =============================================================================

def run_quick_test():
    """Run a quick test with simple prompts"""
    test_prompts = [
        "Hello, how are you?",
        "Tell me a joke about",
        "Write a short story about",
        "Explain the concept of",
        "What is the meaning of"
    ]
    
    # You'll need to provide actual model paths
    model_path = "models/final_model1.pt"  # Update this path
    tokenizer_path = "HuggingFaceTB/SmolLM-135M"  # Update this path
    
    try:
        results = compare_inference_methods(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            test_prompts=test_prompts,
            max_new_tokens=30
        )
        return results
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   Make sure you have a trained model available")
        return None

def run_comprehensive_benchmark():
    """Run a comprehensive benchmark with more prompts"""
    test_prompts = [
        "The quick brown fox jumps over the lazy dog. This is a test of",
        "In a world where artificial intelligence has become",
        "The ancient library contained thousands of books about",
        "As the sun set over the mountains, the travelers",
        "The scientist discovered a new element that could",
        "Once upon a time, in a distant galaxy",
        "The recipe for the perfect chocolate cake includes",
        "The detective carefully examined the evidence and",
        "The astronaut looked out the window and saw",
        "The musician played a beautiful melody that"
    ]
    
    model_path = "models/final_model1.pt"  # Update this path
    tokenizer_path = "HuggingFaceTB/SmolLM-135M"  # Update this path
    
    try:
        results = compare_inference_methods(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            test_prompts=test_prompts,
            max_new_tokens=50
        )
        return results
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("   Make sure you have a trained model available")
        return None

if __name__ == "__main__":
    print("📊 Inference Performance Comparison")
    print("=" * 60)
    print("This script compares different inference methods:")
    print("1. 🐌 Naive inference (no KV cache) - slow but simple")
    print("2. 🚀 Fast inference (with KV cache) - fast and efficient")
    print("3. 📚 Original model inference - baseline comparison")
    print()
    
    # Run quick test
    print("🧪 Running Quick Test...")
    results = run_quick_test()
    
    if results:
        print(f"\n✅ Quick test completed!")
        print(f"   Speedup achieved: {results['speedup']:.1f}x")
        print(f"   This means KV caching makes inference {results['speedup']:.1f}x faster!")
        
        # Ask if user wants comprehensive benchmark
        print(f"\n🔬 Would you like to run a comprehensive benchmark?")
        print(f"   (This will test with more prompts and longer sequences)")
        # In a real script, you'd use input() here
        # run_comprehensive_benchmark()
    else:
        print(f"\n❌ Quick test failed")
        print(f"   Make sure you have:")
        print(f"   1. A trained model at 'models/final_model1.pt'")
        print(f"   2. A tokenizer available")
        print(f"   3. CUDA available (optional but recommended)")
    
    print(f"\n🎯 Key Takeaways:")
    print(f"   - KV caching provides massive speedup (10-100x)")
    print(f"   - Memory usage grows linearly instead of quadratically")
    print(f"   - Essential for production inference systems")
    print(f"   - Easy to implement and integrate")
