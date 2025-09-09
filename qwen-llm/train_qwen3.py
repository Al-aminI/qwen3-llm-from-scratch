#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 MAIN TRAINING SCRIPT FOR QWEN3 FROM SCRATCH

This script puts everything together and trains the Qwen3 model.
Run this to start training your own language model!
"""

import torch
import torch.utils.data
import time
import os
from transformers import AutoTokenizer

# Import our custom components
from config.qwen3_small_config import SmallModelConfig, set_seed, load_and_cache_data, TextTokenDataset
from qwen3_complete_model import MinimalLLM, train_model, resume_training, generate_text

def main(resume_from: str = None):
    """
    🎯 MAIN TRAINING FUNCTION
    
    This function orchestrates the entire training process:
    1. System check and configuration
    2. Data loading and preparation
    3. Model training (from scratch or resume)
    4. Results reporting
    
    Args:
        resume_from: Path to checkpoint to resume training from
    """
    
    print("🚀 QWEN3 FROM SCRATCH - TRAINING SCRIPT")
    print("=" * 50)
    
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Running on CPU - this will be slower but still educational!")

    # Set seed for reproducibility
    set_seed(42)

    # Create configuration for small model
    config = SmallModelConfig()
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   GQA: {config.n_heads}Q heads, {config.n_kv_heads}KV heads")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")
    print(f"   Documents: {config.num_documents} documents")

    # Load and prepare data
    print(f"\n📊 Loading and preparing data...")
    texts, tokenizer, tokens = load_and_cache_data(config)
    
    # Create dataset
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/validation split
    val_size = len(dataset) // 10  # 10% for validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2
    )

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train the model
    start_time = time.time()
    if resume_from:
        print(f"\n🔄 Resuming training from {resume_from}...")
        model, final_metrics = resume_training(config, train_loader, val_loader, resume_from)
    else:
        print(f"\n🚀 Starting training...")
        model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    # Report results
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")

    # Test inference
    print(f"\n🔮 Testing text generation...")
    test_prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "The most important thing to remember is"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generate_text(
            model, tokenizer, prompt,
            max_length=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        print(f"Generated: {generated}")

    print(f"\n✅ Training complete! Model saved as 'models/final_model1.pt'")
    print(f"💡 You can now use the model for inference or continue training!")

def demo_inference(model_path: str = "models/final_model1.pt"):
    """
    🎭 DEMO INFERENCE FUNCTION
    
    This function demonstrates the trained model's capabilities
    """
    print("🎭 Running inference demo")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("💡 Please run training first with: python train_qwen3.py")
        return
    
    print(f"📦 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")
    
    # Demo prompts
    demo_prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important thing to remember is",
        "In the year 2050, technology will",
        "The best way to learn programming is"
    ]
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n🎯 Demo {i}: '{prompt}'")
        print("-" * 50)
        
        generated_text = generate_text(
            model, tokenizer, prompt,
            max_length=100,
            temperature=0.7,
            top_k=40,
            top_p=0.85
        )
        
        print(f"📝 {generated_text}")
        print()

def interactive_inference(model_path: str = "models/final_model1.pt"):
    """
    🤖 INTERACTIVE INFERENCE SESSION
    
    This function allows you to interact with the trained model
    """
    print("🤖 Starting interactive inference session")
    print("Type 'quit' to exit")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("💡 Please run training first with: python train_qwen3.py")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    while True:
        try:
            prompt = input("\n💬 Enter your prompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt.strip():
                continue
            
            print("🔄 Generating...")
            generated_text = generate_text(
                model, tokenizer, prompt,
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            print(f"\n🤖 Generated text:")
            print(f"📝 {generated_text}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Qwen3 model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training from')
    parser.add_argument('--demo', action='store_true', help='Run inference demo')
    parser.add_argument('--interactive', action='store_true', help='Start interactive session')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_inference()
    elif args.interactive:
        interactive_inference()
    else:
        main(resume_from=args.resume)
