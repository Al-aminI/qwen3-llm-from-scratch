import os
import pickle
import torch
import numpy as np
import json
import requests
import soundfile as sf
from typing import List, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from pretraining.core.config.config import MultimodalPretrainingConfig

def load_and_cache_librispeech_simplified(config: MultimodalPretrainingConfig, cache_dir: str = "data_cache"):
    """
    Load LibriSpeech dataset from simplified JSON with actual audio data
    """
    cache_path = os.path.join(cache_dir, f"librispeech_simplified_{config.num_documents}_{config.max_tokens}.pkl")
    
    # Check if cached data exists
    if os.path.exists(cache_path):
        print(f"📦 Loading cached LibriSpeech simplified data from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            audio_text_pairs = cached_data['audio_text_pairs']
            tokenizer = cached_data['tokenizer']
            
            print(f"✅ Loaded {len(audio_text_pairs)} audio-text pairs from cache")
            return audio_text_pairs
            
        except Exception as e:
            print(f"⚠️ Could not load cached data: {e}")
            print("🔄 Re-processing LibriSpeech simplified data...")
    
    print("🔄 Processing new LibriSpeech simplified data (will cache for future use)")
    
    # Load simplified JSON data
    json_path = os.path.join(cache_dir, "librispeech_simplified_100.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Simplified JSON file not found: {json_path}")
    
    print(f"📥 Loading LibriSpeech simplified data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Take only the number of documents we need
    num_samples = min(config.num_documents, len(data))
    data = data[:num_samples]
    
    print(f"📊 Processing {len(data)} audio-text pairs...")
    
    # Process the samples to get actual audio data
    audio_text_pairs = []
    for i, item in enumerate(data):
        print(f"   Processing sample {i + 1}/{len(data)}: {item['id']}")
        
        # Download audio from the URL
        audio_url = item['audio_src']
        response = requests.get(audio_url, timeout=30)
        response.raise_for_status()
        
        # Save temporary audio file
        temp_audio_path = f"/tmp/temp_audio_{i}.wav"
        with open(temp_audio_path, 'wb') as f:
            f.write(response.content)
        
        # Load audio using soundfile
        audio_array, sampling_rate = sf.read(temp_audio_path)
        
        # Clean up temp file
        os.remove(temp_audio_path)
        
        # Store the actual audio data
        audio_text_pairs.append({
            'audio': {
                'array': audio_array.astype(np.float32),
                'sampling_rate': sampling_rate
            },
            'text': item['text']
        })
            
       
    
    print(f"✅ Loaded {len(audio_text_pairs)} audio-text pairs with actual audio data")
    
    # Cache the data
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'audio_text_pairs': audio_text_pairs
        }, f)
    print(f"💾 Cached data saved to {cache_path}")
    
    return audio_text_pairs

class TextTokenDataset(Dataset):
    """Dataset for text-only tokenized data"""
    def __init__(self, tokens, max_seq_len):
        self.tokens = tokens
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

class LibriSpeechSimplifiedDataset(Dataset):
    """Dataset for LibriSpeech simplified audio-text pairs with real SNAC tokenization"""
    def __init__(self, audio_text_pairs, text_tokenizer, audio_tokenizer, config, max_seq_len=256):
        self.audio_text_pairs = audio_text_pairs
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.config = config
        self.max_seq_len = max_seq_len
        
        # Special tokens
        self.audio_start_token = config.audio_start_token
        self.audio_end_token = config.audio_end_token
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        
    def __len__(self):
        return len(self.audio_text_pairs)
    
    def __getitem__(self, idx):
        pair = self.audio_text_pairs[idx]
        text = pair['text']
        audio = pair['audio']
        
        # Process text
        text_tokens = self.text_tokenizer.encode(text, add_special_tokens=True)
        
        # Process audio with real SNAC tokenization (exactly like tts-snac-base.py)
        if audio is not None and isinstance(audio, dict) and 'array' in audio:
            # Extract audio array and sample rate
            audio_array = audio['array']
            sample_rate = audio.get('sampling_rate', 24000)
            
            # Use real SNAC tokenization - returns flattened codes with separators
            audio_codes = self.audio_tokenizer.encode_audio(audio_array, sample_rate)
            
            # Don't map SNAC tokens - use them directly like tts-snac-base.py
            # This preserves the full SNAC vocabulary range (0-1023 per scale)
            audio_tokens = audio_codes
        else:
            raise ValueError("No valid audio data available")
        
        # Combine text and audio tokens (exactly like tts-snac-base.py)
        # Format: [text_tokens] + [audio_codes_with_separators] + [eos_token]
        combined_tokens = text_tokens + audio_tokens + [self.eos_token_id]
        
        # Check token ranges and clamp if necessary
        max_token = max(combined_tokens) if combined_tokens else 0
        vocab_size = getattr(self, 'vocab_size', self.text_tokenizer.vocab_size)
        if max_token >= vocab_size:
            # Clamp tokens to vocabulary range
            combined_tokens = [min(token, vocab_size - 1) for token in combined_tokens]
        
        # Truncate if too long
        if len(combined_tokens) > self.max_seq_len:
            combined_tokens = combined_tokens[:self.max_seq_len]
        
        # Create input and labels
        input_ids = torch.tensor(combined_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(combined_tokens[1:], dtype=torch.long)
        
        # Padding for variable sequence lengths
        if len(input_ids) < self.max_seq_len:
            pad_length = self.max_seq_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_length,), self.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_length,), self.pad_token_id, dtype=torch.long)])
        else:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.pad_token_id] = 0
        
        # Create weights for multimodal loss (like tts-snac-base.py)
        weights = torch.ones_like(input_ids, dtype=torch.float)
        
        # Create text attention mask for weight calculation
        text_attention_mask = torch.zeros_like(input_ids)
        text_len = len(text_tokens)
        text_attention_mask[:text_len] = 1
        
        # Apply weights like tts-snac-base.py
        # Text tokens get lower weight, audio tokens get higher weight
        text_token_weight = self.config.text_weight
        audio_token_weight = self.config.audio_weight
        
        weights = torch.where(text_attention_mask == 1, text_token_weight, audio_token_weight)
        
        # Zero weight for padding tokens
        weights[input_ids == self.pad_token_id] = 0.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'weights': weights
        }

def create_multimodal_dataloader(dataset, batch_size=2, shuffle=True):
    """Create a DataLoader for multimodal dataset"""
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'weights': torch.stack([item['weights'] for item in batch])
        }
    )
