"""
🎵 SNAC AUDIO TOKENIZER

This module provides SNAC (Multi-Scale Neural Audio Codec) integration for
audio tokenization in the multimodal language model.
"""

import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Optional, Union
from snac import SNAC

class SNACTokenizer:
    """
    🎵 SNAC Audio Tokenizer
    
    Handles audio tokenization using the SNAC model for multimodal training.
    
    Features:
    - Audio encoding to discrete tokens
    - Audio decoding from tokens
    - Multi-scale audio representation
    - Efficient audio processing
    """
    
    def __init__(self, model_name: str = "hubertsiuzdak/snac_24khz", device: str = "auto"):
        """
        Initialize SNAC tokenizer
        
        Args:
            model_name: HuggingFace model name for SNAC
            device: Device to load model on ("auto", "cpu", "cuda")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu") if device == "auto" else torch.device(device)
        
        print(f"🎵 Loading SNAC model: {model_name}")
        self.snac_model = SNAC.from_pretrained(model_name).eval().to(self.device)
        
        # Special tokens for audio (aligned with tts-snac-base.py)
        self.audio_start_token = 50258  # Special token to mark start of audio
        self.audio_separator_token = 50258  # Use same token as separator (like tts-snac-base.py)
        self.audio_end_token = 50256  # EOS token to mark end of audio
        
        print(f"✅ SNAC model loaded on {self.device}")
    
    def encode_audio(self, audio: Union[torch.Tensor, np.ndarray], sample_rate: int = 24000) -> List[List[int]]:
        """
        Encode audio to discrete tokens using SNAC
        
        Args:
            audio: Audio tensor of shape (channels, samples) or (samples,)
            sample_rate: Sample rate of the audio
            
        Returns:
            List of token sequences for each scale
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure audio is in the right format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        elif audio.dim() == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.transpose(0, 1)  # Transpose to (channels, samples)
        
        # Ensure we have the right number of channels
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono
        
        # Add batch dimension
        audio = audio.unsqueeze(0)  # Shape: (1, 1, samples)
        
        with torch.no_grad():
            # Encode audio to tokens
            _, codes = self.snac_model(audio)
        
        # Flatten the codes into a sequence
        flattened_codes = self._flatten_codes(codes)
        
        return flattened_codes
    
    def decode_audio(self, tokens: List[int], sample_rate: int = 24000) -> torch.Tensor:
        """
        Decode tokens back to audio using SNAC
        
        Args:
            tokens: List of token IDs
            sample_rate: Target sample rate for output audio
            
        Returns:
            Reconstructed audio tensor
        """
        # Reconstruct the codes from tokens
        codes = self._reconstruct_codes(tokens)
        
        with torch.no_grad():
            # Decode audio from tokens
            audio_hat = self.snac_model.decode(codes)
        
        return audio_hat.squeeze().cpu()
    
    def _flatten_codes(self, codes: List[torch.Tensor]) -> List[int]:
        """
        Flatten SNAC codes into a sequence of tokens (exactly like tts-snac-base.py)
        
        Args:
            codes: List of tensors from SNAC model
            
        Returns:
            Flattened list of token IDs
        """
        flattened_list = []
        
        if len(codes) == 3:
            # 3-scale SNAC (exactly like tts-snac-base.py flatten_tensors_adjusted)
            for i in range(codes[0].size()[1]):
                flattened_list.append(50258)  # audio_start_token
                flattened_list.append(codes[0][0][i].item())
                for j in range(2):
                    flattened_list.append(codes[1][0][j + i * 2].item())
                    for k in range(2):
                        flattened_list.append(codes[2][0][k + j * 2 + i * 4].item())
        
        elif len(codes) == 4:
            # 4-scale SNAC (exactly like tts-snac-base.py flatten_tensors_adjusted)
            for i in range(codes[0].size()[1]):
                flattened_list.append(50258)  # audio_start_token
                flattened_list.append(codes[0][0][i].item())
                for j in range(2):
                    flattened_list.append(codes[1][0][j + i * 2].item())
                    for k in range(2):
                        flattened_list.append(codes[2][0][k + j * 2 + i * 4].item())
                        for l in range(2):
                            flattened_list.append(codes[3][0][l + k * 2 + j * 4 + i * 8].item())
        
        return flattened_list
    
    def _reconstruct_codes(self, tokens: List[int]) -> List[torch.Tensor]:
        """
        Reconstruct SNAC codes from flattened tokens
        
        Args:
            tokens: Flattened list of token IDs
            
        Returns:
            List of tensors for SNAC decoding
        """
        # Find audio tokens (between start and end tokens)
        start_indices = [i for i, token in enumerate(tokens) if token == self.audio_start_token]
        
        if not start_indices:
            return []
        
        # Process each audio segment
        codes = []
        for start_idx in start_indices:
            # Extract tokens for this audio segment
            audio_tokens = tokens[start_idx:]
            # Find the end of this segment (next start token or end of sequence)
            end_idx = len(audio_tokens)
            for i, token in enumerate(audio_tokens[1:], 1):
                if token == self.audio_start_token:
                    end_idx = i
                    break
            
            segment_tokens = audio_tokens[:end_idx]
            
            # Reconstruct codes for this segment
            segment_codes = self._reconstruct_segment_codes(segment_tokens)
            if segment_codes:
                codes.extend(segment_codes)
        
        return codes
    
    def _reconstruct_segment_codes(self, tokens: List[int]) -> List[torch.Tensor]:
        """
        Reconstruct codes for a single audio segment
        
        Args:
            tokens: Tokens for a single audio segment
            
        Returns:
            List of tensors for this segment
        """
        if not tokens or tokens[0] != self.audio_start_token:
            return []
        
        # Remove start token
        tokens = tokens[1:]
        
        # Determine the number of scales based on token count
        if len(tokens) % 7 == 0:
            # 3-scale SNAC
            return self._reconstruct_3scale_codes(tokens)
        elif len(tokens) % 15 == 0:
            # 4-scale SNAC
            return self._reconstruct_4scale_codes(tokens)
        else:
            # Try to infer from pattern
            return self._reconstruct_adaptive_codes(tokens)
    
    def _reconstruct_3scale_codes(self, tokens: List[int]) -> List[torch.Tensor]:
        """Reconstruct 3-scale SNAC codes"""
        tensor1, tensor2, tensor3 = [], [], []
        
        for i in range(0, len(tokens), 7):
            if i + 6 < len(tokens):
                tensor1.append(tokens[i])
                tensor2.append(tokens[i + 1])
                tensor2.append(tokens[i + 2])
                tensor3.append(tokens[i + 3])
                tensor3.append(tokens[i + 4])
                tensor3.append(tokens[i + 5])
                tensor3.append(tokens[i + 6])
        
        return [
            torch.tensor(tensor1).unsqueeze(0).to(self.device),
            torch.tensor(tensor2).unsqueeze(0).to(self.device),
            torch.tensor(tensor3).unsqueeze(0).to(self.device)
        ]
    
    def _reconstruct_4scale_codes(self, tokens: List[int]) -> List[torch.Tensor]:
        """Reconstruct 4-scale SNAC codes"""
        tensor1, tensor2, tensor3, tensor4 = [], [], [], []
        
        for i in range(0, len(tokens), 15):
            if i + 14 < len(tokens):
                tensor1.append(tokens[i])
                tensor2.append(tokens[i + 1])
                tensor3.append(tokens[i + 2])
                tensor4.append(tokens[i + 3])
                tensor4.append(tokens[i + 4])
                tensor3.append(tokens[i + 5])
                tensor4.append(tokens[i + 6])
                tensor4.append(tokens[i + 7])
                tensor4.append(tokens[i + 8])
                tensor2.append(tokens[i + 9])
                tensor3.append(tokens[i + 10])
                tensor4.append(tokens[i + 11])
                tensor4.append(tokens[i + 12])
                tensor3.append(tokens[i + 13])
                tensor4.append(tokens[i + 14])
        
        return [
            torch.tensor(tensor1).unsqueeze(0).to(self.device),
            torch.tensor(tensor2).unsqueeze(0).to(self.device),
            torch.tensor(tensor3).unsqueeze(0).to(self.device),
            torch.tensor(tensor4).unsqueeze(0).to(self.device)
        ]
    
    def _reconstruct_adaptive_codes(self, tokens: List[int]) -> List[torch.Tensor]:
        """Adaptively reconstruct codes based on token pattern"""
        # Simple heuristic: try to group tokens
        if len(tokens) == 0:
            return []
        
        # For now, return a simple reconstruction
        # This can be improved with more sophisticated pattern recognition
        return [torch.tensor(tokens).unsqueeze(0).to(self.device)]
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size for audio tokens"""
        # SNAC typically uses a vocabulary of ~1024 tokens per scale
        return 1024
    
    def get_special_tokens(self) -> dict:
        """Get special tokens for audio processing"""
        return {
            'audio_start': self.audio_start_token,
            'audio_separator': self.audio_separator_token,
            'audio_end': self.audio_end_token
        }
    
    def reconstruct_tensors(self, flattened_output):
        """
        Reconstruct SNAC tensors from flattened output (exactly like tts-snac-base.py)
        
        Args:
            flattened_output: List of token IDs from model generation
            
        Returns:
            List of tensors for SNAC decoding
        """
        def find_last_instance_of_seperator(lst, element=50256):
            """Find the last occurrence of separator token (like tts-snac-base.py)"""
            reversed_list = lst[::-1]
            try:
                reversed_index = reversed_list.index(element)
                return len(lst) - 1 - reversed_index
            except ValueError:
                # If no EOS token found, use the end of the sequence
                print(f"   ⚠️ No EOS token {element} found, using end of sequence")
                return len(lst)
        
        def remove_elements_before_hash(flattened_list):
            """Remove elements before first hash token (like tts-snac-base.py)"""
            try:
                first_hash_index = flattened_list.index(50258)
                return flattened_list[first_hash_index:]
            except ValueError:
                raise ValueError("Audio start token (50258) not found")
        
        def count_elements_between_hashes(lst):
            """Count elements between hash tokens (like tts-snac-base.py)"""
            try:
                first_index = lst.index(50258)
                second_index = lst.index(50258, first_index + 1)
                return second_index - first_index - 1
            except ValueError:
                return "List does not contain two '#' symbols"
        
        def list_to_torch_tensor(tensor_list):
            """Convert list to torch tensor (like tts-snac-base.py)"""
            tensor = torch.tensor(tensor_list)
            tensor = tensor.unsqueeze(0)
            return tensor
        
        # Process flattened output (exactly like tts-snac-base.py)
        try:
            flattened_output = remove_elements_before_hash(flattened_output)
            last_index = find_last_instance_of_seperator(flattened_output)
            flattened_output = flattened_output[:last_index]
        except ValueError as e:
            print(f"   ⚠️ Could not process generated tokens: {e}")
            return []
        
        codes = []
        tensor1, tensor2, tensor3, tensor4 = [], [], [], []
        
        n_tensors = count_elements_between_hashes(flattened_output)
        
        # If we don't have proper SNAC structure, return empty codes
        if n_tensors == "List does not contain two '#' symbols" or n_tensors < 7:
            print(f"   ⚠️ Generated tokens don't have proper SNAC structure")
            return []
        
        if n_tensors == 7:  # 3-scale SNAC
            for i in range(0, len(flattened_output), 8):
                if i + 7 < len(flattened_output):
                    tensor1.append(flattened_output[i+1])
                    tensor2.append(flattened_output[i+2])
                    tensor3.append(flattened_output[i+3])
                    tensor3.append(flattened_output[i+4])
                    tensor2.append(flattened_output[i+5])
                    tensor3.append(flattened_output[i+6])
                    tensor3.append(flattened_output[i+7])
            
            codes = [
                list_to_torch_tensor(tensor1).to(self.device),
                list_to_torch_tensor(tensor2).to(self.device),
                list_to_torch_tensor(tensor3).to(self.device)
            ]
        
        elif n_tensors == 15:  # 4-scale SNAC
            for i in range(0, len(flattened_output), 16):
                if i + 15 < len(flattened_output):
                    tensor1.append(flattened_output[i+1])
                    tensor2.append(flattened_output[i+2])
                    tensor3.append(flattened_output[i+3])
                    tensor4.append(flattened_output[i+4])
                    tensor4.append(flattened_output[i+5])
                    tensor3.append(flattened_output[i+6])
                    tensor4.append(flattened_output[i+7])
                    tensor4.append(flattened_output[i+8])
                    tensor2.append(flattened_output[i+9])
                    tensor3.append(flattened_output[i+10])
                    tensor4.append(flattened_output[i+11])
                    tensor4.append(flattened_output[i+12])
                    tensor3.append(flattened_output[i+13])
                    tensor4.append(flattened_output[i+14])
                    tensor4.append(flattened_output[i+15])
            
            codes = [
                list_to_torch_tensor(tensor1).to(self.device),
                list_to_torch_tensor(tensor2).to(self.device),
                list_to_torch_tensor(tensor3).to(self.device),
                list_to_torch_tensor(tensor4).to(self.device)
            ]
        
        return codes
