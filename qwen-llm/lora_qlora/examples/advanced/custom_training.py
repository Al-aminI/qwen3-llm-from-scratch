"""
Advanced custom training example.

This script demonstrates advanced usage of LoRA and QLoRA with custom configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lora_qlora.core.training.config import LoRATrainingConfig, QLoRATrainingConfig
from lora_qlora.core.training.trainer import LoRATrainer, QLoRATrainer
from lora_qlora.utils.data import load_data, preprocess_data, split_data, balance_data
from lora_qlora.utils.config import load_config, save_config


def compare_lora_qlora():
    """Compare LoRA and QLoRA performance."""
    print("🎯 LoRA vs QLoRA Comparison")
    print("=" * 50)
    
    # Common configuration
    common_config = {
        'model_name': "Qwen/Qwen2.5-0.5B",
        'data_path': "data/classification_data.json",
        'num_epochs': 2,
        'batch_size': 8,
        'learning_rate': 2e-4,
        'lora_rank': 32,
        'lora_alpha': 64.0,
        'lora_dropout': 0.1,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
    
    # LoRA configuration
    lora_config = LoRATrainingConfig(
        **common_config,
        output_dir="outputs/lora_comparison"
    )
    
    # QLoRA configuration
    qlora_config = QLoRATrainingConfig(
        **common_config,
        output_dir="outputs/qlora_comparison",
        quantization_bits=4
    )
    
    # Train LoRA
    print("Training LoRA model...")
    lora_trainer = LoRATrainer(lora_config)
    lora_trainer.setup_model()
    lora_trainer.load_data()
    lora_trainer.setup_trainer()
    lora_trainer.train()
    lora_results = lora_trainer.evaluate()
    
    # Train QLoRA
    print("Training QLoRA model...")
    qlora_trainer = QLoRATrainer(qlora_config)
    qlora_trainer.setup_model()
    qlora_trainer.load_data()
    qlora_trainer.setup_trainer()
    qlora_trainer.train()
    qlora_results = qlora_trainer.evaluate()
    
    # Compare results
    print("\n📊 Comparison Results:")
    print(f"LoRA Test Accuracy: {lora_results['test']['eval_accuracy']:.4f}")
    print(f"QLoRA Test Accuracy: {qlora_results['test']['eval_accuracy']:.4f}")
    print(f"LoRA Test F1: {lora_results['test']['eval_f1']:.4f}")
    print(f"QLoRA Test F1: {qlora_results['test']['eval_f1']:.4f}")
    
    # Save comparison results
    comparison_results = {
        'lora': lora_results,
        'qlora': qlora_results
    }
    save_config(comparison_results, "outputs/comparison_results.json")


def hyperparameter_search():
    """Perform hyperparameter search."""
    print("🎯 Hyperparameter Search")
    print("=" * 50)
    
    # Define search space
    lora_ranks = [8, 16, 32, 64]
    lora_alphas = [16.0, 32.0, 64.0, 128.0]
    learning_rates = [1e-4, 2e-4, 5e-4, 1e-3]
    
    best_config = None
    best_score = 0.0
    results = []
    
    for rank in lora_ranks:
        for alpha in lora_alphas:
            for lr in learning_rates:
                print(f"Testing: rank={rank}, alpha={alpha}, lr={lr}")
                
                # Create config
                config = LoRATrainingConfig(
                    model_name="Qwen/Qwen2.5-0.5B",
                    data_path="data/classification_data.json",
                    output_dir=f"outputs/hp_search/rank_{rank}_alpha_{alpha}_lr_{lr}",
                    num_epochs=1,  # Reduced for search
                    batch_size=8,
                    learning_rate=lr,
                    lora_rank=rank,
                    lora_alpha=alpha,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                )
                
                # Train and evaluate
                trainer = LoRATrainer(config)
                trainer.setup_model()
                trainer.load_data()
                trainer.setup_trainer()
                trainer.train()
                results_eval = trainer.evaluate()
                
                score = results_eval['val']['eval_accuracy']
                results.append({
                    'rank': rank,
                    'alpha': alpha,
                    'learning_rate': lr,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_config = config
                
                print(f"  Score: {score:.4f}")
    
    # Save results
    save_config(results, "outputs/hyperparameter_search_results.json")
    
    print(f"\n🏆 Best Configuration:")
    print(f"  Rank: {best_config.lora_rank}")
    print(f"  Alpha: {best_config.lora_alpha}")
    print(f"  Learning Rate: {best_config.learning_rate}")
    print(f"  Score: {best_score:.4f}")


def custom_data_processing():
    """Demonstrate custom data processing."""
    print("🎯 Custom Data Processing")
    print("=" * 50)
    
    # Load and preprocess data
    data = load_data("data/classification_data.json")
    print(f"Original data size: {len(data)}")
    
    # Preprocess
    processed_data = preprocess_data(data)
    print(f"Processed data size: {len(processed_data)}")
    
    # Filter by length
    from lora_qlora.utils.data import filter_data
    filtered_data = filter_data(processed_data, min_length=20, max_length=500)
    print(f"Filtered data size: {len(filtered_data)}")
    
    # Balance data
    balanced_data = balance_data(filtered_data, method='undersample')
    print(f"Balanced data size: {len(balanced_data)}")
    
    # Get statistics
    from lora_qlora.utils.data import get_data_stats
    stats = get_data_stats(balanced_data)
    print(f"Data statistics: {stats}")
    
    # Train with processed data
    config = LoRATrainingConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        data_path="data/classification_data.json",  # Will be overridden
        output_dir="outputs/custom_processing",
        num_epochs=1,
        batch_size=8,
        learning_rate=2e-4,
        lora_rank=16,
        lora_alpha=32.0
    )
    
    trainer = LoRATrainer(config)
    trainer.setup_model()
    
    # Override data
    trainer.train_dataset = trainer.train_dataset.__class__(balanced_data, trainer.tokenizer, config.max_length)
    trainer.val_dataset = trainer.val_dataset.__class__(balanced_data[:100], trainer.tokenizer, config.max_length)
    trainer.test_dataset = trainer.test_dataset.__class__(balanced_data[:50], trainer.tokenizer, config.max_length)
    
    trainer.setup_trainer()
    trainer.train()
    results = trainer.evaluate()
    
    print(f"Results with custom processing: {results['test']}")


def main():
    """Main function for advanced examples."""
    print("🎯 Advanced LoRA/QLoRA Examples")
    print("=" * 50)
    
    # Run examples
    compare_lora_qlora()
    print("\n" + "=" * 50)
    
    hyperparameter_search()
    print("\n" + "=" * 50)
    
    custom_data_processing()
    
    print("\n✅ Advanced examples complete!")


if __name__ == "__main__":
    main()
