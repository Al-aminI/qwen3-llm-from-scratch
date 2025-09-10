#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 QUANTIZED MODEL SERVING SCRIPT

This script serves quantized models (LoRA, QLoRA, and regular quantized models)
via a Flask API for efficient inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
import json
import time
import math
from typing import Dict, List, Optional
import argparse

# Import our custom components
from config.qwen3_small_config import SmallModelConfig
from qwen3_complete_model import MinimalLLM
from quantization_tutorial import LoRALayer, LoRALinear, LoRAManager, QLoRALayer, QLoRALinear, QLoRAManager, QuantizationConfig
from transformers import AutoTokenizer

class QuantizedModelServer:
    """
    🎯 QUANTIZED MODEL SERVER
    
    Serves different types of quantized models for inference.
    """
    
    def __init__(self, model_path: str, model_type: str = "lora"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Load the quantized model"""
        print(f"📦 Loading {self.model_type.upper()} model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create model config
        model_config = SmallModelConfig()
        model_config.vocab_size = self.tokenizer.vocab_size
        
        # Create base model
        base_model = MinimalLLM(model_config)
        
        if self.model_type == "lora":
            self._load_lora_model(base_model, checkpoint)
        elif self.model_type == "qlora":
            self._load_qlora_model(base_model, checkpoint)
        elif self.model_type == "quantized":
            self._load_quantized_model(base_model, checkpoint)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ {self.model_type.upper()} model loaded successfully")
    
    def _load_lora_model(self, base_model: MinimalLLM, checkpoint: Dict):
        """Load LoRA model"""
        # Load base model weights
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply LoRA
        lora_config = QuantizationConfig()
        lora_config.lora_rank = checkpoint.get('lora_rank', 16)
        lora_config.lora_alpha = checkpoint.get('lora_alpha', 32.0)
        lora_config.lora_dropout = checkpoint.get('lora_dropout', 0.1)
        
        lora_manager = LoRAManager(base_model, lora_config)
        lora_manager.apply_lora()
        
        # Load LoRA weights
        if 'lora_weights' in checkpoint:
            for name, lora_layer in lora_manager.lora_layers.items():
                if name in checkpoint['lora_weights']:
                    lora_layer.lora.lora_A.load_state_dict(checkpoint['lora_weights'][name]['lora_A'])
                    lora_layer.lora.lora_B.load_state_dict(checkpoint['lora_weights'][name]['lora_B'])
        
        self.model = base_model
    
    def _load_qlora_model(self, base_model: MinimalLLM, checkpoint: Dict):
        """Load QLoRA model"""
        # Load base model weights
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply QLoRA
        qlora_config = QuantizationConfig()
        qlora_config.lora_rank = checkpoint.get('lora_rank', 16)
        qlora_config.lora_alpha = checkpoint.get('lora_alpha', 32.0)
        qlora_config.lora_dropout = checkpoint.get('lora_dropout', 0.1)
        qlora_config.qlora_bits = checkpoint.get('qlora_bits', 4)
        
        qlora_manager = QLoRAManager(base_model, qlora_config)
        qlora_manager.apply_qlora()
        
        # Load QLoRA weights
        if 'qlora_weights' in checkpoint:
            for name, qlora_layer in qlora_manager.qlora_layers.items():
                if name in checkpoint['qlora_weights']:
                    qlora_layer.qlora.lora_A.load_state_dict(checkpoint['qlora_weights'][name]['lora_A'])
                    qlora_layer.qlora.lora_B.load_state_dict(checkpoint['qlora_weights'][name]['lora_B'])
                    qlora_layer.qlora.register_buffer('quantized_weights', checkpoint['qlora_weights'][name]['quantized_weights'])
                    qlora_layer.qlora.register_buffer('scale', checkpoint['qlora_weights'][name]['scale'])
                    qlora_layer.qlora.register_buffer('zero_point', checkpoint['qlora_weights'][name]['zero_point'])
        
        self.model = base_model
    
    def _load_quantized_model(self, base_model: MinimalLLM, checkpoint: Dict):
        """Load regular quantized model"""
        # Load model weights
        base_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Apply quantization if specified
        if checkpoint.get('quantized', False):
            # Apply dynamic quantization
            base_model = torch.quantization.quantize_dynamic(
                base_model, 
                {nn.Linear}, 
                dtype=torch.qint8
            )
        
        self.model = base_model
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, 
                     top_k: int = 50, top_p: float = 0.9) -> str:
        """
        🎯 GENERATE TEXT
        
        Generates text using the quantized model.
        """
        self.model.eval()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text[len(prompt):]  # Return only the generated part
    
    def classify_sentiment(self, text: str) -> Dict:
        """
        🎯 CLASSIFY SENTIMENT
        
        Classifies sentiment of the input text.
        """
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_ids)
            
            # Use first token for classification (like [CLS])
            cls_representation = outputs[:, 0, :]
            
            # Simple classification head (you might want to load a trained classifier)
            # For demo, we'll use a simple linear layer
            classifier = nn.Linear(self.model.config.d_model, 2).to(self.device)
            logits = classifier(cls_representation)
            
            # Get probabilities
            probabilities = F.softmax(logits, dim=-1)
            prediction = logits.argmax(dim=-1).item()
            confidence = probabilities[0][prediction].item()
            
            sentiment = "Positive" if prediction == 1 else "Negative"
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    'negative': probabilities[0][0].item(),
                    'positive': probabilities[0][1].item()
                }
            }
    
    def get_model_info(self) -> Dict:
        """
        📊 GET MODEL INFORMATION
        
        Returns information about the loaded model.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'vocab_size': self.model.config.vocab_size,
            'd_model': self.model.config.d_model,
            'n_layers': self.model.config.n_layers
        }

# Flask app
app = Flask(__name__)
server = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_type': server.model_type})

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify(server.get_model_info())

@app.route('/generate', methods=['POST'])
def generate():
    """Generate text endpoint"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.9)
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        start_time = time.time()
        generated_text = server.generate_text(prompt, max_length, temperature, top_k, top_p)
        generation_time = time.time() - start_time
        
        return jsonify({
            'prompt': prompt,
            'generated_text': generated_text,
            'generation_time': generation_time,
            'model_type': server.model_type
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify():
    """Classify sentiment endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        start_time = time.time()
        result = server.classify_sentiment(text)
        classification_time = time.time() - start_time
        
        result['classification_time'] = classification_time
        result['model_type'] = server.model_type
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Benchmark model performance"""
    try:
        data = request.get_json()
        num_samples = data.get('num_samples', 10)
        max_length = data.get('max_length', 50)
        
        # Benchmark text generation
        prompts = [
            "The weather today is",
            "I think that",
            "In my opinion",
            "The best way to",
            "Technology has"
        ] * (num_samples // 5 + 1)
        prompts = prompts[:num_samples]
        
        generation_times = []
        for prompt in prompts:
            start_time = time.time()
            _ = server.generate_text(prompt, max_length=max_length)
            generation_times.append(time.time() - start_time)
        
        avg_generation_time = sum(generation_times) / len(generation_times)
        
        # Benchmark sentiment classification
        test_texts = [
            "I love this movie!",
            "This is terrible.",
            "It's okay, nothing special.",
            "Amazing performance!",
            "Waste of time."
        ] * (num_samples // 5 + 1)
        test_texts = test_texts[:num_samples]
        
        classification_times = []
        for text in test_texts:
            start_time = time.time()
            _ = server.classify_sentiment(text)
            classification_times.append(time.time() - start_time)
        
        avg_classification_time = sum(classification_times) / len(classification_times)
        
        return jsonify({
            'model_type': server.model_type,
            'num_samples': num_samples,
            'avg_generation_time': avg_generation_time,
            'avg_classification_time': avg_classification_time,
            'generation_speed': 1.0 / avg_generation_time,
            'classification_speed': 1.0 / avg_classification_time
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """Main function to start the server"""
    global server
    
    parser = argparse.ArgumentParser(description='Serve quantized models')
    parser.add_argument('--model_path', required=True, help='Path to the quantized model')
    parser.add_argument('--model_type', choices=['lora', 'qlora', 'quantized'], 
                       default='lora', help='Type of quantized model')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize server
    server = QuantizedModelServer(args.model_path, args.model_type)
    
    print(f"🚀 Starting {args.model_type.upper()} model server...")
    print(f"   Model: {args.model_path}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Device: {server.device}")
    
    # Start Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
