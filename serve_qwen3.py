#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 SIMPLE SERVING SCRIPT FOR QWEN3 MODEL

This script provides a simple HTTP server to serve the trained Qwen3 model.
It's perfect for testing and simple deployments.
"""

import torch
import json
import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from qwen3_complete_model import MinimalLLM, generate_text
from qwen3_small_config import SmallModelConfig

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
config = None

def load_model(model_path: str = "final_model.pt"):
    """
    📦 LOAD THE TRAINED MODEL
    
    This function loads the trained model and tokenizer for serving.
    """
    global model, tokenizer, config
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    print(f"📦 Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = MinimalLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {device}")
    
    return model, tokenizer, config

@app.route('/health', methods=['GET'])
def health_check():
    """
    🏥 HEALTH CHECK ENDPOINT
    
    Returns the health status of the model server.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(next(model.parameters()).device) if model else None
    })

@app.route('/generate', methods=['POST'])
def generate():
    """
    🔮 TEXT GENERATION ENDPOINT
    
    Generates text based on the provided prompt.
    
    Expected JSON payload:
    {
        "prompt": "Your text prompt here",
        "max_length": 100,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get request data
        data = request.get_json()
        
        # Validate required fields
        if 'prompt' not in data:
            return jsonify({'error': 'Missing required field: prompt'}), 400
        
        # Extract parameters with defaults
        prompt = data['prompt']
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.8)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.9)
        
        # Validate parameters
        if max_length <= 0 or max_length > 500:
            return jsonify({'error': 'max_length must be between 1 and 500'}), 400
        
        if temperature <= 0 or temperature > 2.0:
            return jsonify({'error': 'temperature must be between 0 and 2.0'}), 400
        
        if top_k <= 0 or top_k > 1000:
            return jsonify({'error': 'top_k must be between 1 and 1000'}), 400
        
        if top_p <= 0 or top_p > 1.0:
            return jsonify({'error': 'top_p must be between 0 and 1.0'}), 400
        
        # Generate text
        generated_text = generate_text(
            model, tokenizer, prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        return jsonify({
            'prompt': prompt,
            'generated_text': generated_text,
            'parameters': {
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    📊 MODEL INFORMATION ENDPOINT
    
    Returns information about the loaded model.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Qwen3-style Language Model',
        'parameters': sum(p.numel() for p in model.parameters()),
        'config': {
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'vocab_size': config.vocab_size,
            'max_seq_len': config.max_seq_len
        },
        'device': str(next(model.parameters()).device)
    })

@app.route('/', methods=['GET'])
def home():
    """
    🏠 HOME ENDPOINT
    
    Returns a simple HTML interface for testing the model.
    """
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qwen3 Model Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 4px; }
            .error { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Qwen3 Model Server</h1>
            <p>Generate text using your trained Qwen3 model!</p>
            
            <form id="generateForm">
                <div class="form-group">
                    <label for="prompt">Prompt:</label>
                    <textarea id="prompt" rows="3" placeholder="Enter your prompt here...">The future of artificial intelligence</textarea>
                </div>
                
                <div class="form-group">
                    <label for="max_length">Max Length:</label>
                    <input type="number" id="max_length" value="100" min="1" max="500">
                </div>
                
                <div class="form-group">
                    <label for="temperature">Temperature:</label>
                    <input type="number" id="temperature" value="0.8" min="0.1" max="2.0" step="0.1">
                </div>
                
                <div class="form-group">
                    <label for="top_k">Top-k:</label>
                    <input type="number" id="top_k" value="50" min="1" max="1000">
                </div>
                
                <div class="form-group">
                    <label for="top_p">Top-p:</label>
                    <input type="number" id="top_p" value="0.9" min="0.1" max="1.0" step="0.1">
                </div>
                
                <button type="submit">Generate Text</button>
            </form>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('generateForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const prompt = document.getElementById('prompt').value;
                const max_length = parseInt(document.getElementById('max_length').value);
                const temperature = parseFloat(document.getElementById('temperature').value);
                const top_k = parseInt(document.getElementById('top_k').value);
                const top_p = parseFloat(document.getElementById('top_p').value);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<p>🔄 Generating...</p>';
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            prompt: prompt,
                            max_length: max_length,
                            temperature: temperature,
                            top_k: top_k,
                            top_p: top_p
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        resultDiv.innerHTML = `
                            <div class="result">
                                <h3>Generated Text:</h3>
                                <p>${data.generated_text}</p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="result error">
                                <h3>Error:</h3>
                                <p>${data.error}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>Error:</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            });
        </script>
    </body>
    </html>
    '''

def main():
    """
    🚀 MAIN SERVER FUNCTION
    
    Starts the Flask server to serve the Qwen3 model.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Serve Qwen3 model')
    parser.add_argument('--model', default='final_model.pt', help='Path to model file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5003, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load model
    try:
        load_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Start server
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    print(f"📊 Model info: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔗 Open http://localhost:{args.port} in your browser to test the model!")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
