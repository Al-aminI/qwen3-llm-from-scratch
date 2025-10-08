"""
DATA LOADING AND CACHING UTILITIES

This module provides utilities for loading and processing training data.
"""

import os
import pickle
from typing import List, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_cache_data(config, cache_dir: str = "data_cache"):
    """
    This function demonstrates modern ML data handling:
    
    Key features:
    1. Caching: Avoids reprocessing the same data
    2. Streaming: Loads large datasets without memory issues
    3. Tokenization: Converts text to numbers the model can understand
    4. Efficient storage: Uses pickle for fast loading
    
    The process:
    1. Check if we already processed this data (cache hit)
    2. If not, load from HuggingFace datasets
    3. Tokenize the text (convert words → numbers)
    4. Cache the result for next time
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check cache first (smart optimization!)
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer (the "dictionary" that converts text ↔ numbers)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset (streaming = memory efficient)
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)

    # Load only a small subset for fast training
    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])  # Limit text length

    print(f"Loaded {len(texts)} documents")

    # Tokenize (convert text to numbers)
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache for next time
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    """
    This creates training examples for our language model:
    
    What it does:
    - Takes a long sequence of tokens
    - Creates sliding windows of fixed length
    - Each example: input sequence + target sequence (shifted by 1)
    
    Example:
    Original text: "The cat sat on the mat"
    Tokens: [1, 2, 3, 4, 5, 6]
    Window size: 4
    
    Example 1: input=[1,2,3,4], target=[2,3,4,5]
    Example 2: input=[2,3,4,5], target=[3,4,5,6]
    
    This teaches the model to predict the next token!
    """
    def __init__(self, tokens: List[int], seq_len: int = 256):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y
