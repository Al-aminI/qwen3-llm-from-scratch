"""
🎵 MULTIMODAL INFERENCE EXAMPLE

This example demonstrates how to use the trained multimodal model for:
1. Text-to-audio generation
2. Audio understanding
3. Multimodal conversation
"""

import sys
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import GPT2Tokenizer

# Add the qwen-omni path
sys.path.append('/Users/mac/Desktop/unhubbling/arch/qwen-omni')

from pretraining.core.config.config import MultimodalPretrainingConfig
from pretraining.core.model.minimal_llm import MultimodalLLM
from pretraining.core.audio.snac_tokenizer import SNACTokenizer

def load_trained_model(model_path="models/final_multimodal_model.pt"):
    """
    Load the trained multimodal model
    """
    print("📦 Loading trained multimodal model...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train a model first using the training examples")
        return None, None, None
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = MultimodalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Step: {checkpoint.get('step', 'Unknown')}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config, checkpoint

def setup_tokenizers(config):
    """
    Setup text and audio tokenizers
    """
    print("🔤 Setting up tokenizers...")
    
    # Text tokenizer (GPT2Tokenizer like training)
    text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True, pad_token_id=50257)
    text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # add pad token which is [50257]
    text_tokenizer.add_tokens(["SPACER"])
    
    # Audio tokenizer (SNAC)

    audio_tokenizer = SNACTokenizer(
        model_name=config.snac_model_name,
        device="cpu"
    )
    print(f"✅ SNAC tokenizer loaded successfully")
    
    return text_tokenizer, audio_tokenizer


def generate_audio_from_text(model, text_tokenizer, audio_tokenizer, config, text, output_path, max_audio_tokens=50):
    """Generate audio from text using the trained model"""
    print(f"🎵 Generating audio for: '{text}'")
    
    try:
        # Tokenize text
        text_tokens = text_tokenizer.encode(text, add_special_tokens=True)
        print(f"   Text tokens: {text_tokens}")
        
        # Create input sequence with text + audio start token
        audio_start_token = config.audio_start_token
        audio_end_token = config.audio_end_token
        eos_token = config.eos_token_id
        
        # Prepare input for generation
        input_tokens = text_tokens + [audio_start_token]
        
        print(f"   Input sequence: {input_tokens}")
        
        # Generate audio tokens using the model (like tts-snac-base.py)
        model.eval()
        with torch.no_grad():
            # Generate tokens one by one
            generated_tokens = input_tokens.copy()
            
            # Add audio start token (like tts-snac-base.py)
            generated_tokens.append(audio_start_token)
            
            for i in range(max_audio_tokens):
                # Prepare current input
                current_input = torch.tensor([generated_tokens], dtype=torch.long)
                
                # Get model output
                logits = model(current_input)
                next_token_logits = logits[0, -1, :]  # Last token logits
                
                # Apply temperature and sampling (like tts-snac-base.py)
                temperature = 0.8
                next_token_logits = next_token_logits / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Add to generated sequence
                generated_tokens.append(next_token)
                
                # Stop if we hit EOS token (50256) like tts-snac-base.py
                if next_token == 50256:
                    break
                
                # Prevent infinite loops by stopping if we get too many repeated tokens
                if i > 10 and len(set(generated_tokens[-10:])) < 3:
                    print(f"   ⚠️ Detected repetitive generation, stopping early")
                    break
        
        print(f"   Generated sequence: {generated_tokens}")
        
        # Use SNAC reconstruction function (like tts-snac-base.py)
        try:
            print(f"   Reconstructing SNAC tensors from generated tokens...")
            
            # Use the reconstruct_tensors function from audio_tokenizer
            reconstructed_codes = audio_tokenizer.reconstruct_tensors(generated_tokens)
            
            if reconstructed_codes and len(reconstructed_codes) > 0:
                print(f"   Successfully reconstructed {len(reconstructed_codes)} SNAC code tensors")
                
                # Decode audio using SNAC model (like tts-snac-base.py)
                print(f"   Decoding audio using SNAC model...")
                with torch.no_grad():
                    audio_hat = audio_tokenizer.snac_model.decode(reconstructed_codes)
                
                # Convert to numpy and save
                audio_array = audio_hat.squeeze().cpu().numpy()
                sample_rate = config.audio_sample_rate
                
                print(f"   Generated audio shape: {audio_array.shape}")
                print(f"   Audio duration: {len(audio_array) / sample_rate:.2f} seconds")
                
                # Save audio file
                sf.write(output_path, audio_array, sample_rate)
                print(f"   ✅ Audio saved to: {output_path}")
                
                return audio_array
            else:
                print("   ⚠️ Could not reconstruct SNAC codes from generated tokens")
                return None
                
        except Exception as e:
            print(f"   ⚠️ SNAC reconstruction failed: {e}")
            print(f"   Generated tokens: {generated_tokens}")
            return None
            
    except Exception as e:
        print(f"⚠️ Audio generation failed: {e}")
        return None
    

def test_text_to_audio_generation():
    """
    Test text-to-audio generation with multiple samples
    """
    print("🎯 TEXT-TO-AUDIO GENERATION TEST")
    print("=" * 50)
    
    # Load trained model
    model, config, checkpoint = load_trained_model()
    if model is None:
        return
    
    # Setup tokenizers
    text_tokenizer, audio_tokenizer = setup_tokenizers(config)
    
    # Load the same training data used for training
    from pretraining.utils.data_simplified import load_and_cache_librispeech_simplified
    
    print("📊 Loading LibriSpeech training data for audio generation...")
    audio_text_pairs = load_and_cache_librispeech_simplified(config)
    
    # Use the same texts from training data
    test_texts = [pair['text'] for pair in audio_text_pairs[:5]]  # First 5 training samples
    print(f"🎵 Using training data texts for audio generation: {test_texts}")
    
    print(f"\n🎵 Generating audio for {len(test_texts)} test texts...")
    
    # Create output directory
    os.makedirs("generated_audio", exist_ok=True)
    
    for i, text in enumerate(test_texts):
        print(f"\n--- Sample {i+1} ---")
        output_path = f"generated_audio/sample_{i+1}.wav"
        
        try:
            audio = generate_audio_from_text(
                model=model,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
                config=config,
                text=text,
                output_path=output_path
            )
            
            if audio is not None:
                print(f"   ✅ Successfully generated audio for: '{text}'")
                print(f"   📁 Saved to: {output_path}")
            else:
                print(f"   ❌ Failed to generate audio for: '{text}'")
                
        except Exception as e:
            print(f"   ❌ Error generating audio for '{text}': {e}")
    
    print(f"\n🎉 Audio generation test completed!")
    print(f"📁 Check the 'generated_audio' folder for output files")

def test_model_forward_pass():
    """
    Test the model's forward pass with different inputs
    """
    print("\n🧠 Testing model forward pass...")
    
    # Load model
    model, config, checkpoint = load_trained_model()
    if model is None:
        return
    
    # Setup tokenizer
    text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True, pad_token_id=50257)
    text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    text_tokenizer.add_tokens(["SPACER"])
    
    # Test different inputs
    test_inputs = [
        "Hello world",
        "The quick brown fox",
        "Artificial intelligence is amazing"
    ]
    
    for i, text in enumerate(test_inputs):
        print(f"\n--- Forward Pass Test {i+1} ---")
        print(f"Input: '{text}'")
        
        try:
            # Tokenize
            tokens = text_tokenizer.encode(text, return_tensors='pt')
            print(f"Tokens: {tokens.shape}")
            
            # Forward pass
            with torch.no_grad():
                logits = model(tokens)
                print(f"Output logits: {logits.shape}")
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                print(f"Predictions: {predictions.shape}")
                
                # Decode some predictions
                if predictions.shape[1] > 0:
                    next_token = predictions[0, -1].item()
                    print(f"Next token ID: {next_token}")
                    
                    # Try to decode the next token
                    try:
                        next_word = text_tokenizer.decode([next_token])
                        print(f"Next word: '{next_word}'")
                    except:
                        print(f"Next word: [unable to decode]")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")

def test_text_generation():
    """
    Test text generation with the trained model
    """
    print("\n📝 TEXT GENERATION TEST")
    print("=" * 50)
    
    # Load trained model
    model, config, checkpoint = load_trained_model()
    if model is None:
        return
    
    # Setup tokenizer
    text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True, pad_token_id=50257)
    text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    text_tokenizer.add_tokens(["SPACER"])
    
    # Load the same training data used for training
    from pretraining.utils.data_simplified import load_and_cache_librispeech_simplified
    
    print("📊 Loading LibriSpeech training data...")
    audio_text_pairs = load_and_cache_librispeech_simplified(config)
    
    # Use the same texts from training data
    test_prompts = [pair['text'] for pair in audio_text_pairs[-5:]]  # Last 5 samples
    print(f"📝 Using training data texts: {test_prompts}")
    
    print(f"📝 Testing text generation with {len(test_prompts)} prompts...")
    
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"\n--- Text Generation {i+1} ---")
            print(f"Prompt: '{prompt}'")
            
            # Tokenize prompt
            input_tokens = text_tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([input_tokens], dtype=torch.long)
            
            print(f"Input tokens: {input_tokens}")
            
            # Generate text
            generated_tokens = input_tokens.copy()
            max_new_tokens = 20
            
            for _ in range(max_new_tokens):
                # Get model output
                logits = model(input_ids)
                next_token_logits = logits[0, -1, :]  # Last token logits
                
                # Sample next token (greedy for now)
                next_token = torch.argmax(next_token_logits).item()
                
                # Add to generated sequence
                generated_tokens.append(next_token)
                
                # Update input for next iteration
                input_ids = torch.tensor([generated_tokens], dtype=torch.long)
                
                # Stop if we hit EOS token
                if next_token == text_tokenizer.eos_token_id:
                    break
            
            # Decode generated text
            generated_text = text_tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Generated: '{generated_text}'")
            
            # Show only the new part
            new_text = generated_text[len(prompt):].strip()
            print(f"New text: '{new_text}'")

def main():
    """
    Main function for multimodal inference
    """
    print("🎵 MULTIMODAL INFERENCE EXAMPLE")
    print("=" * 50)
    
    # Test model forward pass first
    # test_model_forward_pass()
    
    # Test text generation
    # test_text_generation()
    
    # Test text-to-audio generation
    test_text_to_audio_generation()
    
    print(f"\n🎉 All inference tests completed!")
    print(f"📁 Generated audio files are in the 'generated_audio' folder")

if __name__ == "__main__":
    main()