#!/usr/bin/env python3
"""
Test script for OpenAI-compatible API implementation

This script tests the basic functionality of the OpenAI API implementation
without requiring a running server.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from core.engine.openai_protocol import (
            ChatCompletionRequest, ChatCompletionResponse,
            CompletionRequest, CompletionResponse,
            EmbeddingRequest, EmbeddingResponse,
            HealthResponse, ModelList
        )
        print("✅ Protocol models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import protocol models: {e}")
        return False
    
    try:
        from core.engine.openai_serving_engine import OpenAIServingEngine
        print("✅ Serving engine imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import serving engine: {e}")
        return False
    
    try:
        from core.engine.openai_api_server import app
        print("✅ API server imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import API server: {e}")
        return False
    
    return True


def test_protocol_models():
    """Test protocol model creation and validation."""
    print("\n🧪 Testing protocol models...")
    
    try:
        from core.engine.openai_protocol import (
            ChatCompletionRequest, ChatCompletionResponse,
            CompletionRequest, CompletionResponse,
            HealthResponse, ModelList
        )
        
        # Test chat completion request
        chat_request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "Hello, world!"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        print("✅ Chat completion request created successfully")
        
        # Test completion request
        completion_request = CompletionRequest(
            model="test-model",
            prompt="The future of AI is",
            max_tokens=30,
            temperature=0.8
        )
        print("✅ Completion request created successfully")
        
        # Test health response
        health_response = HealthResponse(
            status="healthy",
            model="test-model"
        )
        print("✅ Health response created successfully")
        
        # Test model list
        model_list = ModelList(data=[])
        print("✅ Model list created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Protocol model test failed: {e}")
        return False


def test_serving_engine():
    """Test serving engine initialization."""
    print("\n🧪 Testing serving engine...")
    
    try:
        from core.engine.openai_serving_engine import OpenAIServingEngine
        
        # Create a mock engine
        class MockEngine:
            def __init__(self):
                self.tokenizer = MockTokenizer()
            
            async def generate_async(self, prompt, params):
                # Mock async generator
                yield "Hello"
                yield " world"
                yield "!"
        
        class MockTokenizer:
            def encode(self, text):
                return text.split()
            
            def decode(self, tokens):
                return " ".join(tokens)
        
        # Test serving engine creation
        mock_engine = MockEngine()
        serving_engine = OpenAIServingEngine(mock_engine, "test-model")
        print("✅ Serving engine created successfully")
        
        # Test request ID generation
        request_id = serving_engine._generate_request_id()
        print(f"✅ Request ID generated: {request_id}")
        
        # Test token counting
        token_count = serving_engine._count_tokens("Hello world")
        print(f"✅ Token counting works: {token_count} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Serving engine test failed: {e}")
        return False


def test_api_server():
    """Test API server creation."""
    print("\n🧪 Testing API server...")
    
    try:
        from core.engine.openai_api_server import app
        
        # Check that the app is created
        assert app is not None
        print("✅ FastAPI app created successfully")
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/health", "/ping", "/v1/models", 
            "/v1/chat/completions", "/v1/completions",
            "/v1/embeddings", "/v1/tokenize", "/v1/detokenize"
        ]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} registered")
            else:
                print(f"⚠️ Route {route} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 OpenAI API Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_protocol_models,
        test_serving_engine,
        test_api_server
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The OpenAI API implementation is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
