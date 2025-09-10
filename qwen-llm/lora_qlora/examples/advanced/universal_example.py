"""
Universal LoRA/QLoRA fine-tuning example.

This script demonstrates how to fine-tune ANY model (HuggingFace or custom)
using the universal trainer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lora_qlora.core.training.config import LoRATrainingConfig, QLoRATrainingConfig
from lora_qlora.core.training.universal_trainer import UniversalLoRATrainer, fine_tune_huggingface_model, fine_tune_custom_model


def example_huggingface_models():
    """
    🎯 EXAMPLE: Fine-tune HuggingFace models with LoRA
    """
    print("🎯 HuggingFace Models with LoRA")
    print("=" * 50)
    
    # Example 1: BERT
    print("\n1. Fine-tuning BERT with LoRA:")
    config = LoRATrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        lora_rank=16,
        num_samples=200
    )
    
    try:
        results = fine_tune_huggingface_model("bert-base-uncased", config)
        print(f"✅ BERT Results: {results}")
    except Exception as e:
        print(f"❌ BERT failed: {e}")
    
    # Example 2: DistilBERT
    print("\n2. Fine-tuning DistilBERT with LoRA:")
    config.lora_rank = 8  # Smaller rank for smaller model
    
    try:
        results = fine_tune_huggingface_model("distilbert-base-uncased", config)
        print(f"✅ DistilBERT Results: {results}")
    except Exception as e:
        print(f"❌ DistilBERT failed: {e}")
    
    # Example 3: RoBERTa
    print("\n3. Fine-tuning RoBERTa with LoRA:")
    config.lora_rank = 16
    
    try:
        results = fine_tune_huggingface_model("roberta-base", config)
        print(f"✅ RoBERTa Results: {results}")
    except Exception as e:
        print(f"❌ RoBERTa failed: {e}")


def example_custom_model():
    """
    🎯 EXAMPLE: Fine-tune your custom model with LoRA
    """
    print("\n🎯 Custom Model with LoRA")
    print("=" * 50)
    
    config = LoRATrainingConfig(
        pretrained_model_path="models/final_model1.pt",
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        lora_rank=16,
        num_samples=200
    )
    
    try:
        results = fine_tune_custom_model("models/final_model1.pt", config)
        print(f"✅ Custom Model Results: {results}")
    except Exception as e:
        print(f"❌ Custom Model failed: {e}")


def example_manual_setup():
    """
    🎯 EXAMPLE: Manual setup for more control
    """
    print("\n🎯 Manual Setup Example")
    print("=" * 50)
    
    # Create config
    config = LoRATrainingConfig(
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-4,
        lora_rank=16,
        num_samples=200
    )
    
    # Initialize trainer
    trainer = UniversalLoRATrainer(config)
    
    # Setup with HuggingFace model
    print("Setting up with HuggingFace model...")
    trainer.setup_model(model_name_or_path="bert-base-uncased", model_type="huggingface")
    
    # Load data
    trainer.load_data()
    
    # Train
    results = trainer.train()
    print(f"Results: {results}")
    
    # Save
    trainer.save_model("outputs/manual_example")


def main():
    """
    🎯 MAIN FUNCTION
    
    Demonstrates different ways to use the universal trainer.
    """
    print("🎯 Universal LoRA Fine-tuning Examples")
    print("=" * 60)
    
    # Example 1: HuggingFace models
    example_huggingface_models()
    
    # Example 2: Custom model
    example_custom_model()
    
    # Example 3: Manual setup
    example_manual_setup()
    
    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
