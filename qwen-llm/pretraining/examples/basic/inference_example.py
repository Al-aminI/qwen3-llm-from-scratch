"""
This example demonstrates how to use a trained model for inference.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
from transformers import AutoTokenizer
from pretraining import MinimalLLM, generate_text

def main():
    """
    This function demonstrates how to load and use a trained model.
    """
    
    model_path = "models/final_model1.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print(" Please run training first with: python pretraining/examples/basic/train_example.py")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f" Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")
    
    # Demo prompts
    demo_prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a distant galaxy",
        "The most important thing to remember is",
        "In the year 2050, technology will",
        "The best way to learn programming is"
    ]
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n Demo {i}: '{prompt}'")
        print("-" * 50)
        
        generated_text = generate_text(
            model, tokenizer, prompt,
            max_length=100,
            temperature=0.7,
            top_k=40,
            top_p=0.85
        )
        
        print(f" {generated_text}")
        print()

if __name__ == "__main__":
    main()
