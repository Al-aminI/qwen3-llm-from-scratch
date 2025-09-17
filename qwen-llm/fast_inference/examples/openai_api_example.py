#!/usr/bin/env python3
"""
OpenAI-Compatible API Server Example

This example demonstrates how to use the OpenAI-compatible API server
with the fast_inference engine.
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any


class OpenAIClient:
    """Simple OpenAI client for testing the API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "dummy"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Create a chat completion."""
        data = {
            "model": "fast-inference-model",
            "messages": messages,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Create a text completion."""
        data = {
            "model": "fast-inference-model",
            "prompt": prompt,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/v1/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def embedding(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Create embeddings."""
        data = {
            "model": "fast-inference-model",
            "input": input_text,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        response = requests.get(
            f"{self.base_url}/v1/models",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def test_chat_completion():
    """Test chat completion endpoint."""
    print("🧪 Testing Chat Completion...")
    
    client = OpenAIClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you today?"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        
        print("✅ Chat completion successful!")
        print(f"Response: {response['choices'][0]['message']['content']}")
        print(f"Usage: {response['usage']}")
        
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")


def test_completion():
    """Test text completion endpoint."""
    print("\n🧪 Testing Text Completion...")
    
    client = OpenAIClient()
    
    prompt = "The future of artificial intelligence is"
    
    try:
        response = client.completion(
            prompt=prompt,
            max_tokens=30,
            temperature=0.8
        )
        
        print("✅ Text completion successful!")
        print(f"Response: {response['choices'][0]['text']}")
        print(f"Usage: {response['usage']}")
        
    except Exception as e:
        print(f"❌ Text completion failed: {e}")


def test_embedding():
    """Test embedding endpoint."""
    print("\n🧪 Testing Embeddings...")
    
    client = OpenAIClient()
    
    text = "This is a test sentence for embedding generation."
    
    try:
        response = client.embedding(input_text=text)
        
        print("✅ Embedding successful!")
        print(f"Embedding dimension: {len(response['data'][0]['embedding'])}")
        print(f"Usage: {response['usage']}")
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")


def test_models():
    """Test models endpoint."""
    print("\n🧪 Testing Models List...")
    
    client = OpenAIClient()
    
    try:
        response = client.list_models()
        
        print("✅ Models list successful!")
        print(f"Available models: {[model['id'] for model in response['data']]}")
        
    except Exception as e:
        print(f"❌ Models list failed: {e}")


def test_health():
    """Test health check endpoint."""
    print("\n🧪 Testing Health Check...")
    
    client = OpenAIClient()
    
    try:
        response = client.health_check()
        
        print("✅ Health check successful!")
        print(f"Status: {response['status']}")
        print(f"Model: {response.get('model', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")


def test_streaming():
    """Test streaming chat completion."""
    print("\n🧪 Testing Streaming Chat Completion...")
    
    client = OpenAIClient()
    
    messages = [
        {"role": "user", "content": "Tell me a short story about a robot."}
    ]
    
    try:
        data = {
            "model": "fast-inference-model",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.8,
            "stream": True
        }
        
        response = requests.post(
            f"{client.base_url}/v1/chat/completions",
            headers=client.headers,
            json=data,
            stream=True
        )
        response.raise_for_status()
        
        print("✅ Streaming chat completion successful!")
        print("Streaming response:")
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                print(delta['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Streaming chat completion failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Fast Inference OpenAI-Compatible API Test Suite")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Run tests
    test_health()
    test_models()
    test_chat_completion()
    test_completion()
    test_embedding()
    test_streaming()
    
    print("\n" + "=" * 60)
    print("🎉 Test suite completed!")


if __name__ == "__main__":
    main()
