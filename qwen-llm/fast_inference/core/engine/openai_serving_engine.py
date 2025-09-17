"""
OpenAI-Compatible Serving Engine

This module provides the base serving engine for OpenAI-compatible API endpoints,
handling common functionality like request validation, response formatting, and error handling.
"""

import asyncio
import json
import time
import traceback
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .openai_protocol import (
    ErrorResponse, ErrorInfo, UsageInfo, HealthResponse,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse,
    CompletionRequest, CompletionResponse, CompletionStreamResponse,
    EmbeddingRequest, EmbeddingResponse, EmbeddingData,
    TokenizeRequest, TokenizeResponse, DetokenizeRequest, DetokenizeResponse
)
from .universal_vllm_engine import UniversalVLLMStyleEngine
try:
    from ..utils.sampling import SamplingParams
except ImportError:
    # Fallback for when sampling module is not available
    class SamplingParams:
        def __init__(self, **kwargs):
            self.max_new_tokens = kwargs.get('max_new_tokens', 100)
            self.temperature = kwargs.get('temperature', 1.0)
            self.top_p = kwargs.get('top_p', 1.0)
            self.top_k = kwargs.get('top_k', 50)
            self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
            self.stop = kwargs.get('stop', None)
        
        def model_dump(self):
            return {
                'max_new_tokens': self.max_new_tokens,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'repetition_penalty': self.repetition_penalty,
                'stop': self.stop
            }


class OpenAIServingEngine:
    """
    Base serving engine for OpenAI-compatible API endpoints.
    
    Provides common functionality for:
    - Request validation and processing
    - Response formatting
    - Error handling
    - Streaming support
    - Token usage tracking
    """
    
    def __init__(self, engine: UniversalVLLMStyleEngine, model_name: str = "fast-inference-model"):
        """
        Initialize the serving engine.
        
        Args:
            engine: The underlying inference engine
            model_name: Name of the model for API responses
        """
        self.engine = engine
        self.model_name = model_name
        self.request_id_counter = 0
        
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self.request_id_counter += 1
        return f"req-{int(time.time())}-{self.request_id_counter}"
    
    def _create_error_response(self, message: str, error_type: str = "invalid_request_error", 
                              status_code: int = 400, param: Optional[str] = None) -> JSONResponse:
        """Create a standardized error response."""
        error_info = ErrorInfo(
            message=message,
            type=error_type,
            param=param,
            code=status_code
        )
        error_response = ErrorResponse(error=error_info)
        return JSONResponse(
            content=error_response.model_dump(),
            status_code=status_code
        )
    
    def _convert_to_sampling_params(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> SamplingParams:
        """Convert OpenAI request parameters to internal sampling parameters."""
        return SamplingParams(
            max_new_tokens=request.max_tokens or 100,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            top_k=50,  # Default top_k
            repetition_penalty=1.0 + (request.frequency_penalty or 0.0),
            stop=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None
        )
    
    def _calculate_usage(self, prompt_tokens: int, completion_tokens: int) -> UsageInfo:
        """Calculate token usage information."""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the engine's tokenizer."""
        try:
            tokens = self.engine.tokenizer.encode(text)
            return len(tokens)
        except Exception:
            # Fallback: rough estimation
            return len(text.split()) * 1.3
    
    async def health_check(self) -> HealthResponse:
        """Perform health check."""
        try:
            # Test basic functionality
            test_prompt = "Hello"
            test_result = await self.engine.generate_async(test_prompt, {"max_new_tokens": 1})
            
            # Consume the generator to test it works
            async for _ in test_result:
                break
            
            return HealthResponse(
                status="healthy",
                model=self.model_name
            )
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                model=self.model_name
            )
    
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request) -> Union[JSONResponse, StreamingResponse]:
        """Create chat completion."""
        try:
            # Validate request
            if not request.messages:
                return self._create_error_response("Messages cannot be empty", "invalid_request_error")
            
            # Convert messages to prompt
            prompt = self._convert_messages_to_prompt(request.messages)
            
            # Convert to sampling parameters
            sampling_params = self._convert_to_sampling_params(request)
            
            # Count prompt tokens
            prompt_tokens = self._count_tokens(prompt)
            
            if request.stream:
                return await self._create_chat_completion_stream(request, prompt, sampling_params, prompt_tokens)
            else:
                return await self._create_chat_completion_non_stream(request, prompt, sampling_params, prompt_tokens)
                
        except Exception as e:
            return self._create_error_response(f"Internal server error: {str(e)}", "internal_error", 500)
    
    def _convert_messages_to_prompt(self, messages: List[Any]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            # Handle both dict and ChatMessage objects
            if hasattr(message, 'role'):
                role = message.role
                content = message.content or ""
            else:
                role = message.get("role", "")
                content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                prompt_parts.append(f"Tool: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
    
    async def _create_chat_completion_non_stream(self, request: ChatCompletionRequest, prompt: str, 
                                               sampling_params: SamplingParams, prompt_tokens: int) -> JSONResponse:
        """Create non-streaming chat completion."""
        # Generate completion
        generated_text = ""
        completion_tokens = 0
        
        async for token in self.engine.generate_async(prompt, sampling_params.model_dump()):
            generated_text += token
            completion_tokens += 1
        
        # Create response
        response = ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            usage=self._calculate_usage(prompt_tokens, completion_tokens)
        )
        
        return JSONResponse(content=response.model_dump())
    
    async def _create_chat_completion_stream(self, request: ChatCompletionRequest, prompt: str,
                                           sampling_params: SamplingParams, prompt_tokens: int) -> StreamingResponse:
        """Create streaming chat completion."""
        async def generate_stream():
            completion_tokens = 0
            async for token in self.engine.generate_async(prompt, sampling_params.model_dump()):
                completion_tokens += 1
                
                # Create stream response
                stream_response = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": token
                        },
                        "finish_reason": None
                    }]
                )
                
                yield f"data: {json.dumps(stream_response.model_dump())}\n\n"
            
            # Send final chunk
            final_response = ChatCompletionStreamResponse(
                model=request.model,
                choices=[{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None
                    },
                    "finish_reason": "stop"
                }]
            )
            yield f"data: {json.dumps(final_response.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    async def create_completion(self, request: CompletionRequest, raw_request: Request) -> Union[JSONResponse, StreamingResponse]:
        """Create text completion."""
        try:
            # Validate request
            if not request.prompt:
                return self._create_error_response("Prompt cannot be empty", "invalid_request_error")
            
            # Convert prompt to string
            if isinstance(request.prompt, list):
                prompt = " ".join(str(p) for p in request.prompt)
            else:
                prompt = str(request.prompt)
            
            # Convert to sampling parameters
            sampling_params = self._convert_to_sampling_params(request)
            
            # Count prompt tokens
            prompt_tokens = self._count_tokens(prompt)
            
            if request.stream:
                return await self._create_completion_stream(request, prompt, sampling_params, prompt_tokens)
            else:
                return await self._create_completion_non_stream(request, prompt, sampling_params, prompt_tokens)
                
        except Exception as e:
            return self._create_error_response(f"Internal server error: {str(e)}", "internal_error", 500)
    
    async def _create_completion_non_stream(self, request: CompletionRequest, prompt: str,
                                          sampling_params: SamplingParams, prompt_tokens: int) -> JSONResponse:
        """Create non-streaming completion."""
        # Generate completion
        generated_text = ""
        completion_tokens = 0
        
        async for token in self.engine.generate_async(prompt, sampling_params.model_dump()):
            generated_text += token
            completion_tokens += 1
        
        # Create response
        response = CompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "text": generated_text,
                "finish_reason": "stop"
            }],
            usage=self._calculate_usage(prompt_tokens, completion_tokens)
        )
        
        return JSONResponse(content=response.model_dump())
    
    async def _create_completion_stream(self, request: CompletionRequest, prompt: str,
                                      sampling_params: SamplingParams, prompt_tokens: int) -> StreamingResponse:
        """Create streaming completion."""
        async def generate_stream():
            completion_tokens = 0
            async for token in self.engine.generate_async(prompt, sampling_params.model_dump()):
                completion_tokens += 1
                
                # Create stream response
                stream_response = CompletionStreamResponse(
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "text": token,
                        "finish_reason": None
                    }]
                )
                
                yield f"data: {json.dumps(stream_response.model_dump())}\n\n"
            
            # Send final chunk
            final_response = CompletionStreamResponse(
                model=request.model,
                choices=[{
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop"
                }]
            )
            yield f"data: {json.dumps(final_response.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    async def create_embedding(self, request: EmbeddingRequest, raw_request: Request) -> JSONResponse:
        """Create embeddings."""
        try:
            # For now, return dummy embeddings since we don't have embedding support
            # In a real implementation, you would generate actual embeddings
            
            inputs = request.input
            if isinstance(inputs, str):
                inputs = [inputs]
            
            embeddings = []
            total_tokens = 0
            
            for i, text in enumerate(inputs):
                # Generate dummy embedding (in practice, use actual embedding model)
                embedding_dim = request.dimensions or 768
                dummy_embedding = [0.1] * embedding_dim
                
                embeddings.append(EmbeddingData(
                    index=i,
                    embedding=dummy_embedding
                ))
                total_tokens += self._count_tokens(text)
            
            response = EmbeddingResponse(
                model=request.model,
                data=embeddings,
                usage=UsageInfo(
                    prompt_tokens=total_tokens,
                    total_tokens=total_tokens,
                    completion_tokens=0
                )
            )
            
            return JSONResponse(content=response.model_dump())
            
        except Exception as e:
            return self._create_error_response(f"Internal server error: {str(e)}", "internal_error", 500)
    
    async def tokenize(self, request: TokenizeRequest, raw_request: Request) -> JSONResponse:
        """Tokenize text."""
        try:
            tokens = self.engine.tokenizer.encode(request.input)
            response = TokenizeResponse(tokens=tokens)
            return JSONResponse(content=response.model_dump())
            
        except Exception as e:
            return self._create_error_response(f"Tokenization error: {str(e)}", "invalid_request_error", 400)
    
    async def detokenize(self, request: DetokenizeRequest, raw_request: Request) -> JSONResponse:
        """Detokenize tokens."""
        try:
            text = self.engine.tokenizer.decode(request.tokens)
            response = DetokenizeResponse(text=text)
            return JSONResponse(content=response.model_dump())
            
        except Exception as e:
            return self._create_error_response(f"Detokenization error: {str(e)}", "invalid_request_error", 400)
