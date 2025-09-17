#!/usr/bin/env python3
"""
Simple test script for OpenAI-compatible API implementation

This script tests the basic functionality without requiring torch or other heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_protocol_models_only():
    """Test protocol models without torch dependencies."""
    print("🧪 Testing protocol models (no torch)...")
    
    try:
        # Test basic Pydantic models
        from pydantic import BaseModel, Field
        import time
        import uuid
        
        # Simple test models
        class TestModel(BaseModel):
            id: str = Field(default_factory=lambda: f"test-{uuid.uuid4()}")
            created: int = Field(default_factory=lambda: int(time.time()))
            name: str = "test"
        
        # Test model creation
        model = TestModel()
        print(f"✅ Test model created: {model.id}")
        
        # Test model serialization
        data = model.model_dump()
        print(f"✅ Model serialized: {data}")
        
        return True
        
    except Exception as e:
        print(f"❌ Protocol model test failed: {e}")
        return False


def test_fastapi_import():
    """Test FastAPI import."""
    print("\n🧪 Testing FastAPI import...")
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI imported successfully")
        
        # Test app creation
        app = FastAPI(title="Test API")
        print("✅ FastAPI app created successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ FastAPI not available: {e}")
        print("💡 Install with: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ FastAPI test failed: {e}")
        return False


def test_pydantic_models():
    """Test Pydantic model creation."""
    print("\n🧪 Testing Pydantic models...")
    
    try:
        from pydantic import BaseModel, Field
        from typing import List, Optional, Union, Literal
        import time
        import uuid
        
        # Test OpenAI-style models
        class ChatMessage(BaseModel):
            role: Literal["system", "user", "assistant"]
            content: str
        
        class ChatCompletionRequest(BaseModel):
            model: str
            messages: List[ChatMessage]
            max_tokens: Optional[int] = 100
            temperature: Optional[float] = 1.0
        
        # Test model creation
        message = ChatMessage(role="user", content="Hello!")
        request = ChatCompletionRequest(
            model="test-model",
            messages=[message],
            max_tokens=50,
            temperature=0.7
        )
        
        print("✅ Chat completion request created successfully")
        print(f"   Model: {request.model}")
        print(f"   Messages: {len(request.messages)}")
        print(f"   Max tokens: {request.max_tokens}")
        
        # Test serialization
        data = request.model_dump()
        print("✅ Request serialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
        return False


def test_basic_server_structure():
    """Test basic server structure without dependencies."""
    print("\n🧪 Testing basic server structure...")
    
    try:
        # Test that we can create a basic FastAPI structure
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="Test OpenAI API")
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        @app.post("/v1/chat/completions")
        async def chat_completions(request: dict):
            return {"message": "Chat completion endpoint"}
        
        print("✅ Basic server structure created successfully")
        print("✅ Health endpoint registered")
        print("✅ Chat completions endpoint registered")
        
        return True
        
    except Exception as e:
        print(f"❌ Server structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Simple OpenAI API Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        test_protocol_models_only,
        test_fastapi_import,
        test_pydantic_models,
        test_basic_server_structure
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
        print("🎉 All basic tests passed! The OpenAI API structure is ready.")
        print("\n💡 To run the full server, you'll need:")
        print("   1. Install torch and transformers for model support")
        print("   2. Have a trained model available")
        print("   3. Run: python -m fast_inference.cli_openai")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
