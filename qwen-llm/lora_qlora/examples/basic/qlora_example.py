"""
Basic QLoRA fine-tuning example.

This script demonstrates how to use the QLoRA package for fine-tuning
following the original approach.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lora_qlora.core.training.config import QLoRATrainingConfig
from lora_qlora.core.training.trainer import QLoRATrainer


def main():
    """Main function for QLoRA example."""
    print("🎯 QLoRA Fine-tuning Example")
    print("=" * 50)
    
    # Configuration following original approach
    config = QLoRATrainingConfig(
        pretrained_model_path="models/final_model1.pt",
        num_epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        lora_rank=16,
        lora_alpha=32.0,
        qlora_bits=4,
        num_samples=500  # Small subset for demo
    )
    
    print(f"Configuration:")
    print(f"  Pre-trained Model: {config.pretrained_model_path}")
    print(f"  QLoRA Bits: {config.qlora_bits}")
    print(f"  LoRA Rank: {config.lora_rank}")
    print(f"  LoRA Alpha: {config.lora_alpha}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Samples: {config.num_samples}")
    print()
    
    # Initialize trainer
    trainer = QLoRATrainer(config)
    
    # Setup model
    trainer.setup_model()
    
    # Load data
    trainer.load_data()
    
    # Train
    results = trainer.train()
    print(f"Final Results: {results}")
    
    # Save model
    trainer.save_model("outputs/qlora_example")
    
    print("✅ QLoRA fine-tuning complete!")


if __name__ == "__main__":
    main()