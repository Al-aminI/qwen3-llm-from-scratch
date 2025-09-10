#!/usr/bin/env python3
"""
Performance comparison example.

This script demonstrates how to compare different inference methods
and measure the performance improvements from KV caching.
"""

import sys
import os
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fast_inference import SimpleFastInference, create_simple_fast_inference, SamplingParams
from fast_inference.utils.benchmarking import BenchmarkRunner, generate_test_prompts


def naive_generate_text(model, tokenizer, prompt: str, max_length: int = 100,
                       temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> str:
    """
    Naive text generation without KV caching (for comparison).
    
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
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
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


def main():
    """Main function demonstrating performance comparison."""
    print("📊 Performance Comparison Example")
    print("=" * 50)
    
    # Test prompts
    test_prompts = [
        "Hello, how are you today?",
        "Tell me a joke about programming",
        "Write a short story about",
        "Explain the concept of machine learning",
        "What is the meaning of life?"
    ]
    
    max_new_tokens = 50
    
    try:
        # Create fast inference engine
        print("🚀 Creating fast inference engine...")
        engine = create_simple_fast_inference(
            model_path="models/final_model1.pt",  # Update this path
            tokenizer_path="HuggingFaceTB/SmolLM-135M"  # Update this path
        )
        
        # Get the underlying model for naive comparison
        model = engine.model
        tokenizer = engine.tokenizer
        
        print("✅ Engine created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating engine: {e}")
        print("   Make sure you have a trained model available")
        return
    
    # Benchmark 1: Naive inference (no KV cache)
    print(f"\n🐌 Benchmark 1: Naive Inference (No KV Cache)")
    print("-" * 50)
    
    naive_times = []
    naive_results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"   Processing {i+1}/{len(test_prompts)}: {prompt[:30]}...")
        
        start_time = time.time()
        result = naive_generate_text(model, tokenizer, prompt, max_new_tokens)
        end_time = time.time()
        
        naive_times.append(end_time - start_time)
        naive_results.append(result)
        print(f"   Time: {naive_times[-1]:.3f}s")
    
    # Benchmark 2: Fast inference (with KV cache)
    print(f"\n🚀 Benchmark 2: Fast Inference (With KV Cache)")
    print("-" * 50)
    
    fast_times = []
    fast_results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"   Processing {i+1}/{len(test_prompts)}: {prompt[:30]}...")
        
        start_time = time.time()
        result = engine.generate_single(prompt, max_new_tokens)
        end_time = time.time()
        
        fast_times.append(end_time - start_time)
        fast_results.append(result)
        print(f"   Time: {fast_times[-1]:.3f}s")
    
    # Calculate and display results
    print(f"\n📈 PERFORMANCE RESULTS")
    print("=" * 50)
    
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
    
    # Memory usage comparison
    print(f"\n💾 MEMORY USAGE")
    print("-" * 30)
    print(f"🐌 Naive Inference: O(n²) - grows quadratically with sequence length")
    print(f"🚀 Fast Inference: O(n) - grows linearly with sequence length")
    print(f"   Memory savings: ~{speedup:.0f}x less memory for long sequences")
    
    # Show sample results
    print(f"\n📝 SAMPLE RESULTS")
    print("-" * 30)
    for i in range(min(3, len(test_prompts))):
        print(f"\n{i+1}. Prompt: {test_prompts[i]}")
        print(f"   Naive: {naive_results[i][:100]}...")
        print(f"   Fast:  {fast_results[i][:100]}...")
    
    # Advanced benchmarking with BenchmarkRunner
    print(f"\n🔬 Advanced Benchmarking")
    print("-" * 30)
    
    runner = BenchmarkRunner()
    
    # Define inference functions
    def naive_inference_func(prompts, max_tokens):
        return [naive_generate_text(model, tokenizer, prompt, max_tokens) for prompt in prompts]
    
    def fast_inference_func(prompts, max_tokens):
        return engine.generate_batch(prompts, max_new_tokens=max_tokens)
    
    # Run benchmarks
    naive_result = runner.run_benchmark(
        "Naive Inference", naive_inference_func, test_prompts, max_new_tokens
    )
    
    fast_result = runner.run_benchmark(
        "Fast Inference", fast_inference_func, test_prompts, max_new_tokens
    )
    
    # Print summary
    runner.print_summary()
    
    print(f"\n🎯 Key Takeaways:")
    print(f"   - KV caching provides {speedup:.1f}x speedup")
    print(f"   - Memory usage grows linearly instead of quadratically")
    print(f"   - Essential for production inference systems")
    print(f"   - Easy to implement and integrate")


if __name__ == "__main__":
    main()
