"""
🎵 Multimodal Dataset for Audio-Text Training

This module provides dataset classes for training multimodal language models
with both text and audio tokens.
"""

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
import librosa
from transformers import AutoTokenizer
from ..audio.snac_tokenizer import SNACTokenizer

class MultimodalDataset(Dataset):
    """
    🎵 Multimodal Dataset for Audio-Text Training
    
    Handles audio-text pairs for training multimodal language models.
    
    Features:
    - Audio preprocessing and tokenization
    - Text tokenization
    - Sequence padding and truncation
    - Special token handling
    - Efficient data loading
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        text_tokenizer: AutoTokenizer,
        audio_tokenizer: SNACTokenizer,
        max_seq_len: int = 1024,
        max_audio_duration: float = 10.0,
        sample_rate: int = 24000,
        device: str = "cpu"
    ):
        """
        Initialize multimodal dataset
        
        Args:
            data: List of dictionaries with 'text' and 'audio' keys
            text_tokenizer: Text tokenizer (e.g., GPT-2 tokenizer)
            audio_tokenizer: SNAC audio tokenizer
            max_seq_len: Maximum sequence length
            max_audio_duration: Maximum audio duration in seconds
            sample_rate: Audio sample rate
            device: Device to process data on
        """
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.max_seq_len = max_seq_len
        self.max_audio_duration = max_audio_duration
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        
        # Special tokens
        self.pad_token_id = text_tokenizer.pad_token_id or text_tokenizer.eos_token_id
        self.eos_token_id = text_tokenizer.eos_token_id
        self.bos_token_id = text_tokenizer.bos_token_id
        
        # Audio special tokens
        self.audio_special_tokens = audio_tokenizer.get_special_tokens()
        self.audio_start_token = self.audio_special_tokens['audio_start']
        self.audio_end_token = self.audio_special_tokens['audio_end']
        
        print(f"📊 Loaded {len(self.data)} audio-text pairs")
        print(f"🎵 Max sequence length: {max_seq_len}")
        print(f"🎵 Max audio duration: {max_audio_duration}s")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single audio-text pair
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        sample = self.data[idx]
        
        # Process text
        text_tokens = self._process_text(sample['text'])
        
        # Process audio
        audio_tokens = self._process_audio(sample['audio'])
        
        # Combine text and audio tokens
        combined_tokens = self._combine_tokens(text_tokens, audio_tokens)
        
        # Create attention mask and labels
        attention_mask = [1] * len(combined_tokens)
        labels = combined_tokens.copy()
        
        # Pad or truncate to max_seq_len
        if len(combined_tokens) > self.max_seq_len:
            combined_tokens = combined_tokens[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            # Pad with pad tokens
            pad_length = self.max_seq_len - len(combined_tokens)
            combined_tokens.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
            labels.extend([self.pad_token_id] * pad_length)
        
        return {
            'input_ids': torch.tensor(combined_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _process_text(self, text: str) -> List[int]:
        """
        Process text into tokens
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        # Tokenize text
        tokens = self.text_tokenizer.encode(text, add_special_tokens=True)
        
        # Add BOS token if not present
        if self.bos_token_id and tokens[0] != self.bos_token_id:
            tokens = [self.bos_token_id] + tokens
        
        return tokens
    
    def _process_audio(self, audio_path: str) -> List[int]:
        """
        Process audio into tokens using SNAC
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of audio token IDs
        """
        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Truncate if too long
            max_samples = int(self.max_audio_duration * self.sample_rate)
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
            
            # Encode audio to tokens
            audio_tokens = self.audio_tokenizer.encode_audio(audio.squeeze(0))
            
            # Add special tokens
            audio_tokens = [self.audio_start_token] + audio_tokens + [self.audio_end_token]
            
            return audio_tokens
            
        except Exception as e:
            print(f"⚠️ Error processing audio {audio_path}: {e}")
            # Return empty audio tokens
            return [self.audio_start_token, self.audio_end_token]
    
    def _combine_tokens(self, text_tokens: List[int], audio_tokens: List[int]) -> List[int]:
        """
        Combine text and audio tokens into a single sequence
        
        Args:
            text_tokens: Text token IDs
            audio_tokens: Audio token IDs
            
        Returns:
            Combined token sequence
        """
        # Combine tokens: text + audio + EOS
        combined = text_tokens + audio_tokens + [self.eos_token_id]
        
        return combined
    
    def get_sample_weights(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get sample weights for training (useful for balancing text vs audio)
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Weights tensor
        """
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        
        # Create weights: lower weight for text tokens, higher for audio tokens
        weights = torch.ones_like(input_ids, dtype=torch.float)
        
        # Find audio tokens (between start and end tokens)
        audio_start_indices = (input_ids == self.audio_start_token).nonzero(as_tuple=True)[0]
        audio_end_indices = (input_ids == self.audio_end_token).nonzero(as_tuple=True)[0]
        
        # Set different weights for text vs audio
        text_weight = 0.1  # Lower weight for text
        audio_weight = 1.0  # Higher weight for audio
        
        # Apply weights
        for start_idx, end_idx in zip(audio_start_indices, audio_end_indices):
            weights[start_idx:end_idx+1] = audio_weight
        
        # Apply attention mask
        weights = weights * attention_mask.float()
        
        return weights

class LibriSpeechDataset(MultimodalDataset):
    """
    📚 LibriSpeech Dataset for Audio-Text Training
    
    Specialized dataset for LibriSpeech audio-text pairs.
    """
    
    def __init__(
        self,
        data_path: str,
        text_tokenizer: AutoTokenizer,
        audio_tokenizer: SNACTokenizer,
        max_seq_len: int = 1024,
        max_audio_duration: float = 10.0,
        sample_rate: int = 24000,
        device: str = "cpu"
    ):
        """
        Initialize LibriSpeech dataset
        
        Args:
            data_path: Path to LibriSpeech data
            text_tokenizer: Text tokenizer
            audio_tokenizer: Audio tokenizer
            max_seq_len: Maximum sequence length
            max_audio_duration: Maximum audio duration
            sample_rate: Audio sample rate
            device: Device to process data on
        """
        # Load LibriSpeech data
        data = self._load_librispeech_data(data_path)
        
        super().__init__(
            data=data,
            text_tokenizer=text_tokenizer,
            audio_tokenizer=audio_tokenizer,
            max_seq_len=max_seq_len,
            max_audio_duration=max_audio_duration,
            sample_rate=sample_rate,
            device=device
        )
    
    def _load_librispeech_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load LibriSpeech data from path
        
        Args:
            data_path: Path to LibriSpeech data
            
        Returns:
            List of audio-text pairs
        """
        # This is a placeholder - in practice, you would load from
        # LibriSpeech dataset using datasets library or similar
        data = []
        
        # Example structure:
        # data.append({
        #     'text': "Hello world",
        #     'audio': "/path/to/audio.wav"
        # })
        
        return data

def create_multimodal_dataloader(
    dataset: MultimodalDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for multimodal training
    
    Args:
        dataset: Multimodal dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for multimodal training
    """
    def collate_fn(batch):
        """Custom collate function for multimodal data"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
