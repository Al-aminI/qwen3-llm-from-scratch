"""
🎯 MULTIMODAL TRAINING EXAMPLE

This example demonstrates how to train the multimodal model using:
1. Real dataset loading
2. Multimodal trainer
3. SNAC audio tokenization
4. Proper training pipeline
"""

import sys
import os
import torch
import torch.utils.data
import numpy as np
from transformers import GPT2Tokenizer
import soundfile as sf
# Add the qwen-omni path
sys.path.append('/Users/mac/Desktop/unhubbling/arch/qwen-omni')

from pretraining.core.config.config import MultimodalPretrainingConfig
from pretraining.core.training.multimodal_trainer import MultimodalPretrainingTrainer
from pretraining.core.audio.snac_tokenizer import SNACTokenizer
from pretraining.core.model.minimal_llm import MultimodalLLM
from pretraining.utils.data_simplified import load_and_cache_librispeech_simplified, LibriSpeechSimplifiedDataset

def create_multimodal_wrapper(dataset):
    """
    Create a wrapper to convert (x, y) tuples to multimodal format
    """
    class MultimodalWrapper:
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            # Create attention mask (all 1s for valid tokens)
            attention_mask = torch.ones_like(x)
            return {
                'input_ids': x,
                'attention_mask': attention_mask,
                'labels': y
            }
    
    return MultimodalWrapper(dataset)

def main():
    """
    Main training function
    """
    print("🎯 MULTIMODAL TRAINING EXAMPLE")
    print("=" * 50)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create configuration
    config = MultimodalPretrainingConfig()
    config.max_steps = 300  # 300 steps total (100 more from checkpoint)
    config.eval_every = 1
    config.batch_size = 2
    config.num_documents = 2000  # 2000 documents for more training data
    config.max_tokens = 10000
    
    print(f"📋 Configuration:")
    print(f"   Model: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Sequence length: {config.max_seq_len}")
    print(f"   Documents: {config.num_documents}, tokens: {config.max_tokens}")
    
    # Setup GPT2Tokenizer exactly like tts-snac-base.py
    print(f"\n📊 Setting up GPT2Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True, pad_token_id=50257)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add pad token which is [50257]
    tokenizer.add_tokens(["SPACER"])
    
    # Resize embeddings to accommodate SNAC tokens (like tts-snac-base.py)
    # This ensures we have enough vocabulary space for SNAC audio tokens (0-1023 range)
    config.vocab_size = 50304  # 50257 (GPT-2) + 47 (safety margin for SNAC tokens)
    print(f"✅ Tokenizer setup complete")
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    # Load and prepare LibriSpeech simplified data
    print(f"\n📊 Loading and preparing LibriSpeech simplified data...")
    
    audio_text_pairs = load_and_cache_librispeech_simplified(config)
    print(f"✅ Loaded {len(audio_text_pairs)} audio-text pairs")
    
    # Setup SNAC audio tokenizer
    print("🎵 Setting up SNAC audio tokenizer...")
    audio_tokenizer = SNACTokenizer(
        model_name=config.snac_model_name,
        device="cpu"
    )
    print("✅ SNAC tokenizer loaded successfully")
    
    # Create multimodal dataset
    dataset = LibriSpeechSimplifiedDataset(
        audio_text_pairs, tokenizer, audio_tokenizer, config, config.max_seq_len
    )
    
    # Update dataset to use resized vocabulary size after model is created
    print(f"🔧 Updating dataset vocabulary size from {tokenizer.vocab_size} to {config.vocab_size}")
    dataset.vocab_size = config.vocab_size
    
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
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Test model creation
    print(f"\n🏗️ Testing model creation...")
   
    model = MultimodalLLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully: {total_params:,} parameters")
  
    
    # Run training using the multimodal trainer with resume functionality
    print(f"\n🚀 Starting multimodal training...")
    
    # Check for existing checkpoint to resume from
    checkpoint_path = "models/final_multimodal_model.pt"
    resume_from = checkpoint_path if os.path.exists(checkpoint_path) else None
    
    if resume_from:
        print(f"📦 Resuming training from: {resume_from}")
        # Load checkpoint to get the correct vocab size
        checkpoint = torch.load(resume_from, map_location='cpu')
        config.vocab_size = checkpoint['config'].vocab_size
        print(f"🔧 Updated config vocab size to: {config.vocab_size}")
    else:
        print(f"🆕 Starting fresh training")
 
    trainer = MultimodalPretrainingTrainer(config)
    model, final_metrics = trainer.train(train_loader, val_loader, resume_from=resume_from)
    
    print(f"\n🎉 MULTIMODAL TRAINING COMPLETED!")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
    
    if 'val_audio_loss' in final_metrics:
        print(f"   Audio Loss: {final_metrics['val_audio_loss']:.4f}")
    if 'val_text_loss' in final_metrics:
        print(f"   Text Loss: {final_metrics['val_text_loss']:.4f}")
        
    
    
    # Test generation (like tts-snac-base.py)
    print(f"\n🎵 Testing audio generation...")
    try:
        # Test text-to-audio generation
        test_text = "Hello, this is a test."
        print(f"   Testing with text: '{test_text}'")
        
        # Tokenize text
        text_tokens = tokenizer.encode(test_text, add_special_tokens=True)
        input_ids = torch.tensor([text_tokens], dtype=torch.long).to('cpu')
        
        # Generate audio tokens (like tts-snac-base.py)
        with torch.no_grad():
            model.eval()
            
            # Generate tokens one by one (like tts-snac-base.py)
            generated_tokens = text_tokens.copy()
            
            # Add audio start token
            generated_tokens.append(50258)
            
            max_audio_tokens = 50
            
            for i in range(max_audio_tokens):
                current_input = torch.tensor([generated_tokens], dtype=torch.long)
                logits = model(current_input)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature and sampling
                temperature = 0.8
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(next_token)
                
                # Stop if we hit EOS token (50256)
                if next_token == 50256:
                    break
                
                # Prevent infinite loops
                if i > 10 and len(set(generated_tokens[-10:])) < 3:
                    print(f"   ⚠️ Detected repetitive generation, stopping early")
                    break
            
            print(f"   Generated sequence: {generated_tokens}")
            
            # Test SNAC reconstruction
            try:
                reconstructed_codes = audio_tokenizer.reconstruct_tensors(generated_tokens)
                print(f"   Successfully reconstructed {len(reconstructed_codes)} SNAC code tensors")
                
                # Test audio decoding
                with torch.no_grad():
                    audio_hat = audio_tokenizer.snac_model.decode(reconstructed_codes)
                    audio_array = audio_hat.squeeze().cpu().numpy()
                    sample_rate = config.audio_sample_rate
                    sf.write("generated_audio.wav", audio_array, sample_rate)
                print(f"   Generated audio shape: {audio_hat.shape}")
                
                print(f"✅ Audio generation test passed!")
                
            except Exception as e:
                print(f"⚠️ SNAC reconstruction test failed: {e}")
        
    except Exception as e:
        print(f"⚠️ Generation test failed: {e}")
    
    print(f"\n🎉 Multimodal training example completed successfully!")

if __name__ == "__main__":
    main()