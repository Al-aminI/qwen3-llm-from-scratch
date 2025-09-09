#!/usr/bin/env python3
"""
🎯 DEMO: IMDB SENTIMENT ANALYSIS FINE-TUNING

This script demonstrates the fine-tuning process with a small subset of data
for quick testing and demonstration purposes.
"""

import torch
from finetune_imdb import FineTuneConfig, load_imdb_data, SentimentClassifier
from config.qwen3_small_config import SmallModelConfig
from qwen3_complete_model import MinimalLLM
from transformers import AutoTokenizer
import time

def demo_finetune():
    """
    🎯 DEMO FINE-TUNING PROCESS
    
    Demonstrates the fine-tuning process with a small subset of data.
    """
    print("🎯 IMDB Sentiment Analysis Fine-tuning Demo")
    print("=" * 50)
    
    # Create config for demo (smaller dataset, fewer epochs)
    config = FineTuneConfig()
    config.batch_size = 8
    config.num_epochs = 1
    config.learning_rate = 2e-4
    config.max_seq_len = 256  # Shorter sequences for demo
    
    print(f"📋 Demo Configuration:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Max sequence length: {config.max_seq_len}")
    
    # Load a small subset of data for demo
    print(f"\n📊 Loading IMDB dataset (demo subset)...")
    from datasets import load_dataset
    
    # Load full dataset
    dataset = load_dataset("imdb")
    
    # Take only a small subset for demo
    train_texts = dataset['train']['text'][:1000]  # Only 1000 samples
    train_labels = dataset['train']['label'][:1000]
    test_texts = dataset['test']['text'][:200]     # Only 200 samples
    test_labels = dataset['test']['label'][:200]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Demo dataset loaded")
    print(f"   Train samples: {len(train_texts)}")
    print(f"   Test samples: {len(test_texts)}")
    
    # Show sample data
    print(f"\n📝 Sample Data:")
    print(f"   Text: {train_texts[0][:100]}...")
    print(f"   Label: {'Positive' if train_labels[0] == 1 else 'Negative'}")
    
    # Create a dummy pre-trained model for demo
    print(f"\n🏗️ Creating demo model...")
    model_config = SmallModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    
    # Create pre-trained model (randomly initialized for demo)
    pretrained_model = MinimalLLM(model_config)
    
    # Create classification model
    model = SentimentClassifier(pretrained_model, config.num_classes, config.dropout)
    
    print(f"✅ Demo model created")
    print(f"   Pre-trained parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    sample_text = "This movie is absolutely amazing!"
    encoding = tokenizer(
        sample_text,
        truncation=True,
        padding='max_length',
        max_length=config.max_seq_len,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1).item()
        confidence = probabilities[0][prediction].item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"   Sample text: '{sample_text}'")
    print(f"   Prediction: {sentiment} (confidence: {confidence:.3f})")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"💡 To run full fine-tuning, use: python finetune_imdb.py")

if __name__ == "__main__":
    demo_finetune()
