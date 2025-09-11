"""
🌐 Universal Fast Inference Example

This example demonstrates how to use the universal fast inference engine
with different types of models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fast_inference.core.engine.universal_engine import create_universal_fast_inference

def example_minimal_llm():
    """
    🎯 EXAMPLE: Your Custom MinimalLLM Model
    """
    print("🎯 Example 1: Your Custom MinimalLLM Model")
    print("=" * 50)
    
    # Load your trained model
    model_path = "models/final_model1.pt"
    tokenizer_path = "HuggingFaceTB/SmolLM-135M"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("💡 Please train a model first with the pretraining package")
        return
    
    # Create universal engine
    engine = create_universal_fast_inference(
        model=model_path,
        tokenizer=tokenizer_path,
        max_seq_len=1024,
        model_type="minimal_llm"
    )
    
    # Get model info
    info = engine.get_model_info()
    print(f"📊 Model Info:")
    print(f"   Type: {info['model_type']}")
    print(f"   Parameters: {info['parameters']:,}")
    print(f"   Architecture: {info['architecture']['model_type']}")
    
    # Generate text
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important thing to remember is"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🎯 Prompt {i}: '{prompt}'")
        print("-" * 30)
        
        result = engine.generate_single(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        print(f"📝 Generated: {result}")
    
    print(f"\n✅ MinimalLLM example completed!")

def example_huggingface_models():
    """
    🎯 EXAMPLE: HuggingFace Models
    """
    print("\n🎯 Example 2: HuggingFace Models")
    print("=" * 50)
    
    # List of HuggingFace models to try
    models_to_try = [
        "microsoft/DialoGPT-small",
        "gpt2",
        "distilgpt2"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"\n🔄 Trying model: {model_name}")
            
            # Create universal engine
            engine = create_universal_fast_inference(
                model=model_name,
                tokenizer=model_name,
                max_seq_len=512,
                model_type="huggingface"
            )
            
            # Get model info
            info = engine.get_model_info()
            print(f"📊 Model Info:")
            print(f"   Type: {info['model_type']}")
            print(f"   Parameters: {info['parameters']:,}")
            print(f"   Architecture: {info['architecture']['model_type']}")
            
            # Generate text
            prompt = "The future of technology"
            result = engine.generate_single(
                prompt=prompt,
                max_new_tokens=30,
                temperature=0.8
            )
            
            print(f"📝 Generated: {result}")
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue
    
    print(f"\n✅ HuggingFace models example completed!")

def example_batch_generation():
    """
    🎯 EXAMPLE: Batch Generation
    """
    print("\n🎯 Example 3: Batch Generation")
    print("=" * 50)
    
    # Use a small model for batch generation
    model_name = "distilgpt2"
    
    try:
        # Create universal engine
        engine = create_universal_fast_inference(
            model=model_name,
            tokenizer=model_name,
            max_seq_len=256,
            model_type="huggingface"
        )
        
        # Batch prompts
        prompts = [
            "Tell me a joke about",
            "Write a haiku about",
            "Explain the concept of",
            "The best way to learn",
            "In the year 2050"
        ]
        
        print(f"🔄 Generating text for {len(prompts)} prompts...")
        
        # Generate batch
        results = engine.generate_batch(
            prompts=prompts,
            max_new_tokens=20,
            temperature=0.8
        )
        
        # Display results
        for prompt, result in zip(prompts, results):
            print(f"\n📝 {prompt} {result}")
        
    except Exception as e:
        print(f"❌ Error in batch generation: {e}")
    
    print(f"\n✅ Batch generation example completed!")

def example_performance_comparison():
    """
    🎯 EXAMPLE: Performance Comparison
    """
    print("\n🎯 Example 4: Performance Comparison")
    print("=" * 50)
    
    import time
    
    # Test with a small model
    model_name = "distilgpt2"
    
    try:
        # Create universal engine
        engine = create_universal_fast_inference(
            model=model_name,
            tokenizer=model_name,
            max_seq_len=256,
            model_type="huggingface"
        )
        
        # Test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "Once upon a time in a distant galaxy",
            "The most important thing to remember is"
        ]
        
        # Time generation
        start_time = time.time()
        
        for prompt in test_prompts:
            result = engine.generate_single(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.8
            )
            print(f"📝 Generated: {result[:50]}...")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️ Performance:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average per prompt: {total_time/len(test_prompts):.2f} seconds")
        print(f"   Prompts processed: {len(test_prompts)}")
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
    
    print(f"\n✅ Performance comparison completed!")

def main():
    """
    🎯 MAIN FUNCTION
    
    Run all examples to demonstrate universal fast inference.
    """
    print("🌐 UNIVERSAL FAST INFERENCE EXAMPLES")
    print("=" * 60)
    print("This example demonstrates how to use the universal fast inference")
    print("engine with different types of models.")
    print()
    
    # Run examples
    example_minimal_llm()
    example_huggingface_models()
    example_batch_generation()
    example_performance_comparison()
    
    print("\n🎉 All examples completed!")
    print("\n💡 Key Takeaways:")
    print("1. ✅ Universal engine works with your MinimalLLM")
    print("2. ✅ Universal engine works with HuggingFace models")
    print("3. ✅ Automatic model detection and optimization")
    print("4. ✅ Easy batch generation")
    print("5. ✅ Performance monitoring")

if __name__ == "__main__":
    main()
