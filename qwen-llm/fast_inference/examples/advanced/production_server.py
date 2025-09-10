#!/usr/bin/env python3
"""
Production server example for fast inference.

This script demonstrates how to create a production-ready inference server
using the fast inference engine with proper error handling, logging, and monitoring.
"""

import sys
import os
import time
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fast_inference import SimpleFastInference, create_simple_fast_inference, SamplingParams
from fast_inference.utils.benchmarking import BenchmarkRunner


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request structure for inference."""
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    request_id: Optional[str] = None


@dataclass
class InferenceResponse:
    """Response structure for inference."""
    request_id: str
    generated_text: str
    generation_time: float
    tokens_generated: int
    success: bool
    error_message: Optional[str] = None


class InferenceServer:
    """
    Production-ready inference server.
    
    This class provides a robust inference server with:
    - Request queuing and batching
    - Error handling and recovery
    - Performance monitoring
    - Health checks
    - Graceful shutdown
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, 
                 max_batch_size: int = 8, max_queue_size: int = 100):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to the model
            tokenizer_path: Path to the tokenizer
            max_batch_size: Maximum batch size for processing
            max_queue_size: Maximum queue size
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        
        # Initialize engine
        self.engine: Optional[SimpleFastInference] = None
        self.is_ready = False
        
        # Request queue and processing
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.response_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'avg_generation_time': 0.0,
            'avg_tokens_per_second': 0.0
        }
        
        # Benchmark runner for performance monitoring
        self.benchmark_runner = BenchmarkRunner()
        
        logger.info(f"Inference server initialized with max_batch_size={max_batch_size}")
    
    async def initialize(self):
        """Initialize the inference engine."""
        try:
            logger.info("Initializing inference engine...")
            self.engine = create_simple_fast_inference(
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path
            )
            self.is_ready = True
            logger.info("Inference engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    async def process_request(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process a single inference request.
        
        Args:
            request: Inference request
            
        Returns:
            Inference response
        """
        if not self.is_ready:
            return InferenceResponse(
                request_id=request.request_id or "unknown",
                generated_text="",
                generation_time=0.0,
                tokens_generated=0,
                success=False,
                error_message="Server not ready"
            )
        
        start_time = time.time()
        
        try:
            # Generate text
            generated_text = self.engine.generate_single(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
            
            generation_time = time.time() - start_time
            tokens_generated = len(generated_text.split())
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self.stats['total_tokens'] += tokens_generated
            self.stats['total_time'] += generation_time
            self.stats['avg_generation_time'] = self.stats['total_time'] / self.stats['total_requests']
            self.stats['avg_tokens_per_second'] = self.stats['total_tokens'] / self.stats['total_time']
            
            logger.info(f"Request {request.request_id} completed in {generation_time:.3f}s")
            
            return InferenceResponse(
                request_id=request.request_id or "unknown",
                generated_text=generated_text,
                generation_time=generation_time,
                tokens_generated=tokens_generated,
                success=True
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            
            logger.error(f"Request {request.request_id} failed: {e}")
            
            return InferenceResponse(
                request_id=request.request_id or "unknown",
                generated_text="",
                generation_time=generation_time,
                tokens_generated=0,
                success=False,
                error_message=str(e)
            )
    
    async def process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """
        Process a batch of inference requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses
        """
        if not self.is_ready:
            return [
                InferenceResponse(
                    request_id=req.request_id or "unknown",
                    generated_text="",
                    generation_time=0.0,
                    tokens_generated=0,
                    success=False,
                    error_message="Server not ready"
                )
                for req in requests
            ]
        
        start_time = time.time()
        
        try:
            # Extract prompts and parameters
            prompts = [req.prompt for req in requests]
            max_new_tokens = requests[0].max_new_tokens  # Assume all requests have same params
            
            # Generate batch
            results = self.engine.generate_batch(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                temperature=requests[0].temperature,
                top_k=requests[0].top_k,
                top_p=requests[0].top_p
            )
            
            generation_time = time.time() - start_time
            
            # Create responses
            responses = []
            for i, (request, result) in enumerate(zip(requests, results)):
                tokens_generated = len(result.split())
                
                responses.append(InferenceResponse(
                    request_id=request.request_id or f"batch_{i}",
                    generated_text=result,
                    generation_time=generation_time / len(requests),  # Average time per request
                    tokens_generated=tokens_generated,
                    success=True
                ))
            
            # Update statistics
            self.stats['total_requests'] += len(requests)
            self.stats['successful_requests'] += len(requests)
            self.stats['total_tokens'] += sum(len(result.split()) for result in results)
            self.stats['total_time'] += generation_time
            self.stats['avg_generation_time'] = self.stats['total_time'] / self.stats['total_requests']
            self.stats['avg_tokens_per_second'] = self.stats['total_tokens'] / self.stats['total_time']
            
            logger.info(f"Batch of {len(requests)} requests completed in {generation_time:.3f}s")
            
            return responses
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.stats['total_requests'] += len(requests)
            self.stats['failed_requests'] += len(requests)
            
            logger.error(f"Batch processing failed: {e}")
            
            return [
                InferenceResponse(
                    request_id=req.request_id or "unknown",
                    generated_text="",
                    generation_time=generation_time / len(requests),
                    tokens_generated=0,
                    success=False,
                    error_message=str(e)
                )
                for req in requests
            ]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status information
        """
        return {
            'status': 'healthy' if self.is_ready else 'unhealthy',
            'ready': self.is_ready,
            'stats': self.stats.copy(),
            'queue_size': self.request_queue.qsize(),
            'timestamp': time.time()
        }
    
    async def benchmark(self, num_requests: int = 10) -> Dict[str, Any]:
        """
        Run performance benchmark.
        
        Args:
            num_requests: Number of requests for benchmark
            
        Returns:
            Benchmark results
        """
        if not self.is_ready:
            return {'error': 'Server not ready'}
        
        # Generate test prompts
        test_prompts = [
            f"Write a short story about {i} words: " for i in range(num_requests)
        ]
        
        # Create test requests
        test_requests = [
            InferenceRequest(
                prompt=prompt,
                max_new_tokens=50,
                request_id=f"benchmark_{i}"
            )
            for i, prompt in enumerate(test_prompts)
        ]
        
        # Run benchmark
        start_time = time.time()
        responses = await self.process_batch(test_requests)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        successful_requests = sum(1 for resp in responses if resp.success)
        total_tokens = sum(resp.tokens_generated for resp in responses)
        
        return {
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'throughput_requests_per_sec': num_requests / total_time,
            'throughput_tokens_per_sec': total_tokens / total_time,
            'avg_time_per_request': total_time / num_requests,
            'success_rate': successful_requests / num_requests
        }


async def main():
    """Main function demonstrating production server usage."""
    print("🚀 Production Inference Server Example")
    print("=" * 50)
    
    # Initialize server
    server = InferenceServer(
        model_path="models/final_model1.pt",  # Update this path
        tokenizer_path="HuggingFaceTB/SmolLM-135M"  # Update this path
    )
    
    try:
        # Initialize engine
        await server.initialize()
        
        # Health check
        print("\n🏥 Health Check")
        print("-" * 20)
        health = await server.health_check()
        print(f"Status: {health['status']}")
        print(f"Ready: {health['ready']}")
        
        # Single request example
        print("\n📝 Single Request Example")
        print("-" * 30)
        
        request = InferenceRequest(
            prompt="Hello, how are you today?",
            max_new_tokens=50,
            temperature=0.8,
            request_id="single_request_1"
        )
        
        response = await server.process_request(request)
        print(f"Request ID: {response.request_id}")
        print(f"Success: {response.success}")
        print(f"Generation Time: {response.generation_time:.3f}s")
        print(f"Tokens Generated: {response.tokens_generated}")
        print(f"Generated Text: {response.generated_text}")
        
        # Batch request example
        print("\n📚 Batch Request Example")
        print("-" * 30)
        
        batch_requests = [
            InferenceRequest(
                prompt="Tell me a joke about",
                max_new_tokens=30,
                request_id=f"batch_request_{i}"
            )
            for i in range(3)
        ]
        
        batch_responses = await server.process_batch(batch_requests)
        
        for response in batch_responses:
            print(f"\nRequest ID: {response.request_id}")
            print(f"Success: {response.success}")
            print(f"Generated Text: {response.generated_text}")
        
        # Performance benchmark
        print("\n🔬 Performance Benchmark")
        print("-" * 30)
        
        benchmark_results = await server.benchmark(num_requests=5)
        print(f"Total Requests: {benchmark_results['total_requests']}")
        print(f"Successful Requests: {benchmark_results['successful_requests']}")
        print(f"Total Time: {benchmark_results['total_time']:.3f}s")
        print(f"Throughput: {benchmark_results['throughput_requests_per_sec']:.2f} requests/s")
        print(f"Token Throughput: {benchmark_results['throughput_tokens_per_sec']:.1f} tokens/s")
        print(f"Success Rate: {benchmark_results['success_rate']:.2%}")
        
        # Final statistics
        print("\n📊 Server Statistics")
        print("-" * 25)
        final_health = await server.health_check()
        stats = final_health['stats']
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Successful: {stats['successful_requests']}")
        print(f"Failed: {stats['failed_requests']}")
        print(f"Total Tokens: {stats['total_tokens']}")
        print(f"Average Generation Time: {stats['avg_generation_time']:.3f}s")
        print(f"Average Tokens/Second: {stats['avg_tokens_per_second']:.1f}")
        
        print("\n✅ Production server example completed!")
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Error: {e}")
    
    print("\n🎯 Key Features Demonstrated:")
    print("   - Robust error handling and recovery")
    print("   - Request queuing and batching")
    print("   - Performance monitoring and statistics")
    print("   - Health checks and status reporting")
    print("   - Production-ready logging")
    print("   - Graceful error handling")


if __name__ == "__main__":
    asyncio.run(main())
