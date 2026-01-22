"""
🔮 TEXT GENERATION UTILITIES

This module provides utilities for text generation from trained models.
"""

import torch
import torch.nn.functional as F
from typing import Optional

def generate_text(model, tokenizer, prompt: str, max_length: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9):
    """
    🔮 TEXT GENERATION FUNCTION
    
    This function generates text using the trained model with advanced sampling:
    
    🎯 Sampling Strategies:
    1. Temperature Scaling: Controls randomness (0.1 = focused, 2.0 = random)
    2. Top-k Sampling: Only consider top k most likely tokens
    3. Top-p (Nucleus) Sampling: Consider tokens until cumulative probability reaches p
    
    🔍 How it works:
    1. Tokenize the prompt
    2. For each position, get model predictions
    3. Apply temperature scaling to logits
    4. Apply top-k filtering
    5. Apply top-p filtering
    6. Sample from the filtered distribution
    7. Append to sequence and repeat
    
    📊 Parameter Effects:
    - Temperature: Lower = more focused, Higher = more creative
    - Top-k: Lower = more focused, Higher = more diverse
    - Top-p: Lower = more focused, Higher = more diverse
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(generated_ids)
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            next_token = next_token.unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if we reach the end token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text
