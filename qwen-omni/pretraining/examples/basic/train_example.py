"""
🎯 BASIC PRETRAINING EXAMPLE

This example demonstrates how to use the pretraining package to train a model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
import torch.utils.data
from pretraining import (
    PretrainingConfig, 
    set_seed, 
    load_and_cache_data, 
    TextTokenDataset,
    PretrainingTrainer
)

def main():
    """
    🎯 MAIN TRAINING FUNCTION
    
    This function demonstrates the complete pretraining pipeline.
    """
    print("🚀 PRETRAINING EXAMPLE")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create configuration
    config = PretrainingConfig()
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
    trainer = PretrainingTrainer(config)
    model, final_metrics = trainer.train(train_loader, val_loader)
    
    # Report results
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    print(f"\n✅ Training complete! Model saved as 'models/final_model1.pt'")

if __name__ == "__main__":
    main()
