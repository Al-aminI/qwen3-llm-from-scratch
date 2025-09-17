"""
OpenAI-Compatible API Server

This module provides a FastAPI-based OpenAI-compatible API server that can serve
any model supported by the fast_inference engine with high performance and
comprehensive feature support.
"""

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError

from .openai_protocol import (
    ModelList, ModelCard, ModelPermission,
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest,
    TokenizeRequest, DetokenizeRequest, HealthResponse,
    ErrorResponse, ErrorInfo
)
from .openai_serving_engine import OpenAIServingEngine
from .universal_vllm_engine import UniversalVLLMStyleEngine, create_universal_vllm_style_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
engine: Optional[UniversalVLLMStyleEngine] = None
serving_engine: Optional[OpenAIServingEngine] = None


async def run_inference_loop(engine_instance):
    """Run the background inference loop to process requests."""
    logger.info("Starting background inference loop...")
    try:
        while True:
            try:
                # Process one step of inference
                finished_sequences = await engine_instance.step_async()
                
                # If no sequences to process, sleep briefly
                if not finished_sequences:
                    await asyncio.sleep(0.01)
                else:
                    # Log progress
                    logger.debug(f"Processed {len(finished_sequences)} sequences")
                    
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                await asyncio.sleep(0.1)  # Wait before retrying
                
    except asyncio.CancelledError:
        logger.info("Inference loop cancelled")
        raise
    except Exception as e:
        logger.error(f"Fatal error in inference loop: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global engine, serving_engine
    
    # Startup
    logger.info("Starting OpenAI-compatible API server...")
    
    # Initialize engine
    try:
        model_path = os.getenv("MODEL_PATH", "models/final_model1.pt")
        tokenizer_path = os.getenv("TOKENIZER_PATH", "HuggingFaceTB/SmolLM-135M")
        model_name = os.getenv("MODEL_NAME", "fast-inference-model")
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        
        engine = create_universal_vllm_style_engine(
            model=model_path,
            tokenizer=tokenizer_path,
            max_seq_len=2048,
            num_blocks=1000,
            block_size=128,
            max_batch_size=32
        )
        
        serving_engine = OpenAIServingEngine(engine, model_name)
        
        # Start background inference loop
        inference_task = asyncio.create_task(run_inference_loop(engine))
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cancel background task
    if 'inference_task' in locals():
        inference_task.cancel()
        try:
            await inference_task
        except asyncio.CancelledError:
            pass
    
    # Shutdown
    logger.info("Shutting down API server...")
    if engine:
        # Clean up resources
        pass


# Create FastAPI app
app = FastAPI(
    title="Fast Inference OpenAI-Compatible API",
    description="High-performance OpenAI-compatible API server powered by fast_inference",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_serving_engine() -> OpenAIServingEngine:
    """Get the serving engine instance."""
    if serving_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return serving_engine


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=ErrorInfo(
                message=f"Validation error: {exc.errors()}",
                type="invalid_request_error",
                code=422
            )
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorInfo(
                message="Internal server error",
                type="internal_error",
                code=500
            )
        ).model_dump()
    )


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    serving_engine = get_serving_engine()
    health = await serving_engine.health_check()
    return health


@app.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"status": "ok", "timestamp": int(time.time())}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    serving_engine = get_serving_engine()
    
    model_card = ModelCard(
        id=serving_engine.model_name,
        max_model_len=2048,
        permission=[ModelPermission()]
    )
    
    return ModelList(data=[model_card])


# Chat completions endpoint
@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    raw_request: Request,
    serving_engine: OpenAIServingEngine = Depends(get_serving_engine)
):
    """Create chat completion."""
    return await serving_engine.create_chat_completion(request, raw_request)


# Text completions endpoint
@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest,
    raw_request: Request,
    serving_engine: OpenAIServingEngine = Depends(get_serving_engine)
):
    """Create text completion."""
    return await serving_engine.create_completion(request, raw_request)


# Embeddings endpoint
@app.post("/v1/embeddings")
async def create_embedding(
    request: EmbeddingRequest,
    raw_request: Request,
    serving_engine: OpenAIServingEngine = Depends(get_serving_engine)
):
    """Create embeddings."""
    return await serving_engine.create_embedding(request, raw_request)


# Tokenization endpoints
@app.post("/v1/tokenize")
async def tokenize(
    request: TokenizeRequest,
    raw_request: Request,
    serving_engine: OpenAIServingEngine = Depends(get_serving_engine)
):
    """Tokenize text."""
    return await serving_engine.tokenize(request, raw_request)


@app.post("/v1/detokenize")
async def detokenize(
    request: DetokenizeRequest,
    raw_request: Request,
    serving_engine: OpenAIServingEngine = Depends(get_serving_engine)
):
    """Detokenize tokens."""
    return await serving_engine.detokenize(request, raw_request)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fast Inference OpenAI-Compatible API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/v1/models"
    }


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main entry point for the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast Inference OpenAI-Compatible API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer")
    parser.add_argument("--model-name", type=str, default="fast-inference-model", help="Model name")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = args.tokenizer_path
    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "fast_inference.core.engine.openai_api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False
    )


if __name__ == "__main__":
    main()
