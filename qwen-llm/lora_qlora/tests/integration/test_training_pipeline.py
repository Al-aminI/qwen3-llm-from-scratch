"""
Integration tests for training pipeline.

This module provides integration tests for the complete training pipeline.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from lora_qlora.core.training.config import LoRATrainingConfig, QLoRATrainingConfig
from lora_qlora.core.training.dataset import LoRADataset, QLoRADataset
from lora_qlora.core.training.trainer import LoRATrainer, QLoRATrainer
from lora_qlora.utils.data import load_data, preprocess_data, split_data


class TestTrainingPipeline(unittest.TestCase):
    """Test cases for training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = [
            {"text": "This is a positive review", "label": 1},
            {"text": "This is a negative review", "label": 0},
            {"text": "Great product, highly recommended", "label": 1},
            {"text": "Poor quality, not worth it", "label": 0},
            {"text": "Amazing experience, love it", "label": 1},
            {"text": "Terrible service, avoid", "label": 0},
            {"text": "Excellent value for money", "label": 1},
            {"text": "Waste of time and money", "label": 0},
            {"text": "Outstanding performance", "label": 1},
            {"text": "Disappointing results", "label": 0}
        ]
        
        # Save sample data
        import json
        self.data_path = os.path.join(self.temp_dir, "sample_data.json")
        with open(self.data_path, 'w') as f:
            json.dump(self.sample_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_lora_training_pipeline(self):
        """Test complete LoRA training pipeline."""
        # Create LoRA config
        config = LoRATrainingConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            data_path=self.data_path,
            output_dir=os.path.join(self.temp_dir, "lora_output"),
            num_epochs=1,
            batch_size=2,
            learning_rate=2e-4,
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        # Initialize trainer
        trainer = LoRATrainer(config)
        
        # Setup model
        trainer.setup_model()
        
        # Load data
        trainer.load_data()
        
        # Setup trainer
        trainer.setup_trainer()
        
        # Train
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        
        # Check results
        self.assertIn('val', results)
        self.assertIn('test', results)
        self.assertIn('eval_accuracy', results['val'])
        self.assertIn('eval_accuracy', results['test'])
        
        # Save model
        trainer.save_model(config.output_dir)
        
        # Check that model was saved
        self.assertTrue(os.path.exists(config.output_dir))
        self.assertTrue(os.path.exists(os.path.join(config.output_dir, "lora_weights.pt")))
        self.assertTrue(os.path.exists(os.path.join(config.output_dir, "config.pt")))
    
    def test_qlora_training_pipeline(self):
        """Test complete QLoRA training pipeline."""
        # Create QLoRA config
        config = QLoRATrainingConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            data_path=self.data_path,
            output_dir=os.path.join(self.temp_dir, "qlora_output"),
            num_epochs=1,
            batch_size=2,
            learning_rate=2e-4,
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            quantization_bits=4
        )
        
        # Initialize trainer
        trainer = QLoRATrainer(config)
        
        # Setup model
        trainer.setup_model()
        
        # Load data
        trainer.load_data()
        
        # Setup trainer
        trainer.setup_trainer()
        
        # Train
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        
        # Check results
        self.assertIn('val', results)
        self.assertIn('test', results)
        self.assertIn('eval_accuracy', results['val'])
        self.assertIn('eval_accuracy', results['test'])
        
        # Save model
        trainer.save_model(config.output_dir)
        
        # Check that model was saved
        self.assertTrue(os.path.exists(config.output_dir))
        self.assertTrue(os.path.exists(os.path.join(config.output_dir, "qlora_weights.pt")))
        self.assertTrue(os.path.exists(os.path.join(config.output_dir, "config.pt")))
    
    def test_data_processing_pipeline(self):
        """Test data processing pipeline."""
        # Load data
        data = load_data(self.data_path)
        self.assertEqual(len(data), len(self.sample_data))
        
        # Preprocess data
        processed_data = preprocess_data(data)
        self.assertEqual(len(processed_data), len(data))
        
        # Split data
        train_data, val_data, test_data = split_data(processed_data, train_split=0.6, val_split=0.2, test_split=0.2)
        
        # Check splits
        self.assertEqual(len(train_data), 6)  # 60% of 10
        self.assertEqual(len(val_data), 2)    # 20% of 10
        self.assertEqual(len(test_data), 2)   # 20% of 10
        
        # Check that all data is accounted for
        self.assertEqual(len(train_data) + len(val_data) + len(test_data), len(processed_data))
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        # Create tokenizer (mock)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create LoRA dataset
        lora_dataset = LoRADataset(self.sample_data, tokenizer, max_length=128)
        self.assertEqual(len(lora_dataset), len(self.sample_data))
        
        # Test getting a sample
        sample = lora_dataset[0]
        self.assertIn('input_ids', sample)
        self.assertIn('attention_mask', sample)
        self.assertIn('labels', sample)
        
        # Create QLoRA dataset
        qlora_dataset = QLoRADataset(self.sample_data, tokenizer, max_length=128)
        self.assertEqual(len(qlora_dataset), len(self.sample_data))
        
        # Test getting a sample
        sample = qlora_dataset[0]
        self.assertIn('input_ids', sample)
        self.assertIn('attention_mask', sample)
        self.assertIn('labels', sample)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        # Create config
        config = LoRATrainingConfig(
            model_name="Qwen/Qwen2.5-0.5B",
            data_path=self.data_path,
            output_dir=os.path.join(self.temp_dir, "config_test"),
            num_epochs=2,
            batch_size=4,
            learning_rate=2e-4,
            lora_rank=16,
            lora_alpha=32.0
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['model_name'], config.model_name)
        self.assertEqual(config_dict['num_epochs'], config.num_epochs)
        self.assertEqual(config_dict['lora_rank'], config.lora_rank)
        
        # Save and load config
        config_path = os.path.join(self.temp_dir, "config.json")
        import json
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)
        
        with open(config_path, 'r') as f:
            loaded_config_dict = json.load(f)
        
        self.assertEqual(config_dict, loaded_config_dict)


if __name__ == '__main__':
    unittest.main()
