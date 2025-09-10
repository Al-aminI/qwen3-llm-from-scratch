#!/usr/bin/env python3
"""
Quick start example for fast inference.

This script demonstrates the basic usage of the fast inference engine
with a simple text generation example.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fast_inference import SimpleFastInference, create_simple_fast_inference, SamplingParams


def main():
    """Main function demonstrating basic usage."""
    print("🚀 Fast Inference Quick Start Example")
    print("=" * 50)
    
    # Example 1: Single text generation
    print("\n📝 Example 1: Single Text Generation")
    print("-" * 40)
    
    try:
        # Create engine (you'll need to provide actual model paths)
        engine = create_simple_fast_inference(
            model_path="models/final_model1.pt",  # Update this path
            tokenizer_path="HuggingFaceTB/SmolLM-135M"  # Update this path
        )
        
        # Generate text
        prompt = "Hello, how are you today?"
        result = engine.generate_single(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8
        )
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have a trained model available")
        return
    
    # Example 2: Batch generation
    print("\n📚 Example 2: Batch Generation")
    print("-" * 40)
    
    prompts = [
        "Tell me a joke about",
        "Write a haiku about",
        "Explain the concept of"
    ]
    
    results = engine.generate_batch(
        prompts=prompts,
        max_new_tokens=30,
        temperature=0.7
    )
    
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result}")
    
    # Example 3: Different sampling strategies
    print("\n🎯 Example 3: Different Sampling Strategies")
    print("-" * 40)
    
    prompt = "The future of artificial intelligence is"
    
    # High creativity
    creative_result = engine.generate_single(
        prompt=prompt,
        max_new_tokens=40,
        temperature=1.2,  # Higher temperature = more creative
        top_k=100,        # Consider top 100 tokens
        top_p=0.95        # Use 95% of probability mass
    )
    
    # Low creativity (more focused)
    focused_result = engine.generate_single(
        prompt=prompt,
        max_new_tokens=40,
        temperature=0.3,  # Lower temperature = more focused
        top_k=20,         # Consider top 20 tokens
        top_p=0.8         # Use 80% of probability mass
    )
    
    print(f"Prompt: {prompt}")
    print(f"\nCreative (temp=1.2): {creative_result}")
    print(f"Focused (temp=0.3): {focused_result}")
    
    # Example 4: Using SamplingParams
    print("\n⚙️ Example 4: Using SamplingParams")
    print("-" * 40)
    
    # Create custom sampling parameters
    sampling_params = SamplingParams(
        max_new_tokens=60,
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    result = engine.generate_single(
        prompt="Write a short story about a robot",
        max_new_tokens=sampling_params.max_new_tokens,
        temperature=sampling_params.temperature,
        top_k=sampling_params.top_k,
        top_p=sampling_params.top_p
    )
    
    print(f"Custom sampling result: {result}")
    
    print("\n✅ Quick start example completed!")
    print("\n🎯 Key Takeaways:")
    print("   - Simple API for fast text generation")
    print("   - 10-100x speedup over naive inference")
    print("   - Flexible sampling parameters")
    print("   - Easy batch processing")


if __name__ == "__main__":
    main()
