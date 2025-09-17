"""
OpenAI-Compatible API Protocol Models

This module defines the request and response models for OpenAI-compatible API endpoints,
following the OpenAI API specification for maximum compatibility.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import uuid


class OpenAIBaseModel(BaseModel):
    """Base model for OpenAI API compatibility."""
    model_config = ConfigDict(extra="allow")


class ErrorInfo(OpenAIBaseModel):
    """Error information model."""
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ErrorResponse(OpenAIBaseModel):
    """Error response model."""
    error: ErrorInfo


class ModelPermission(OpenAIBaseModel):
    """Model permission model."""
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    """Model card model."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "fast_inference"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    """Model list response."""
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(OpenAIBaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


# Chat Completions Models
class ChatMessage(OpenAIBaseModel):
    """Chat message model."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(OpenAIBaseModel):
    """Chat completion request model."""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class ChatCompletionChoice(OpenAIBaseModel):
    """Chat completion choice model."""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "null"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    """Chat completion response model."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionStreamChoice(OpenAIBaseModel):
    """Chat completion stream choice model."""
    index: int
    delta: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "null"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    """Chat completion stream response model."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]


# Text Completions Models
class CompletionRequest(OpenAIBaseModel):
    """Text completion request model."""
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    echo: Optional[bool] = False
    suffix: Optional[str] = None
    seed: Optional[int] = None


class CompletionChoice(OpenAIBaseModel):
    """Completion choice model."""
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length", "content_filter", "null"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(OpenAIBaseModel):
    """Completion response model."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class CompletionStreamChoice(OpenAIBaseModel):
    """Completion stream choice model."""
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length", "content_filter", "null"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class CompletionStreamResponse(OpenAIBaseModel):
    """Completion stream response model."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionStreamChoice]


# Embeddings Models
class EmbeddingRequest(OpenAIBaseModel):
    """Embedding request model."""
    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(OpenAIBaseModel):
    """Embedding data model."""
    index: int
    object: str = "embedding"
    embedding: Union[List[float], str]  # float list or base64 string


class EmbeddingResponse(OpenAIBaseModel):
    """Embedding response model."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo


# Tokenization Models
class TokenizeRequest(OpenAIBaseModel):
    """Tokenize request model."""
    model: str
    input: str


class TokenizeResponse(OpenAIBaseModel):
    """Tokenize response model."""
    tokens: List[int]


class DetokenizeRequest(OpenAIBaseModel):
    """Detokenize request model."""
    model: str
    tokens: List[int]


class DetokenizeResponse(OpenAIBaseModel):
    """Detokenize response model."""
    text: str


# Health Check Models
class HealthResponse(OpenAIBaseModel):
    """Health check response model."""
    status: str = "healthy"
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    version: str = "1.0.0"
    model: Optional[str] = None
