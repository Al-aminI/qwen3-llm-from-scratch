#!/usr/bin/env python3
"""
Command-line interface for fast inference.

This module provides a CLI for running inference, benchmarking,
and other operations with the fast inference engine.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

from .core.engine import create_simple_fast_inference, create_fast_inference_engine
from .utils.sampling import SamplingParams
from .utils.benchmarking import BenchmarkRunner, generate_test_prompts


def create_engine(args):
    """Create inference engine based on arguments."""
    if args.advanced:
        return create_fast_inference_engine(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            max_batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            n_pages=args.n_pages,
            page_size=args.page_size
        )
    else:
        return create_simple_fast_inference(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_len
        )


def cmd_generate(args):
    """Generate text from prompts."""
    print("🚀 Fast Inference - Text Generation")
    print("=" * 40)
    
    try:
        # Create engine
        print(f"Loading model from {args.model_path}...")
        engine = create_engine(args)
        print("✅ Model loaded successfully!")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        # Read prompts
        if args.prompts:
            prompts = args.prompts
        elif args.prompt_file:
            with open(args.prompt_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Interactive mode
            prompts = []
            print("\nEnter prompts (empty line to finish):")
            while True:
                prompt = input("Prompt: ").strip()
                if not prompt:
                    break
                prompts.append(prompt)
        
        if not prompts:
            print("❌ No prompts provided")
            return 1
        
        # Generate text
        print(f"\nGenerating text for {len(prompts)} prompt(s)...")
        start_time = time.time()
        
        if len(prompts) == 1:
            result = engine.generate_single(
                prompts[0],
                max_new_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p
            )
            results = [result]
        else:
            results = engine.generate_batch(
                prompts,
                max_new_tokens=sampling_params.max_new_tokens,
                temperature=sampling_params.temperature,
                top_k=sampling_params.top_k,
                top_p=sampling_params.top_p
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Display results
        print(f"\n📝 Results (generated in {generation_time:.3f}s):")
        print("-" * 50)
        
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"\n{i+1}. Prompt: {prompt}")
            print(f"   Generated: {result}")
        
        # Save results if requested
        if args.output:
            output_data = {
                'prompts': prompts,
                'results': results,
                'generation_time': generation_time,
                'sampling_params': {
                    'max_new_tokens': sampling_params.max_new_tokens,
                    'temperature': sampling_params.temperature,
                    'top_k': sampling_params.top_k,
                    'top_p': sampling_params.top_p,
                    'repetition_penalty': sampling_params.repetition_penalty
                }
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n💾 Results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_benchmark(args):
    """Run performance benchmark."""
    print("📊 Fast Inference - Performance Benchmark")
    print("=" * 45)
    
    try:
        # Create engine
        print(f"Loading model from {args.model_path}...")
        engine = create_engine(args)
        print("✅ Model loaded successfully!")
        
        # Generate test prompts
        if args.prompts:
            test_prompts = args.prompts
        else:
            test_prompts = generate_test_prompts(
                num_prompts=args.num_requests,
                min_length=args.min_length,
                max_length=args.max_length
            )
        
        print(f"Generated {len(test_prompts)} test prompts")
        
        # Create benchmark runner
        runner = BenchmarkRunner()
        
        # Define inference function
        def inference_func(prompts, max_tokens):
            return engine.generate_batch(prompts, max_new_tokens=max_tokens)
        
        # Run benchmark
        print(f"\nRunning benchmark...")
        result = runner.run_benchmark(
            method_name="Fast Inference",
            inference_func=inference_func,
            test_prompts=test_prompts,
            max_new_tokens=args.max_tokens,
            warmup_runs=args.warmup,
            num_runs=args.runs
        )
        
        # Display results
        print(f"\n📈 Benchmark Results:")
        print("-" * 30)
        print(f"Method: {result.method_name}")
        print(f"Total Time: {result.total_time:.3f}s")
        print(f"Total Tokens: {result.total_tokens:,}")
        print(f"Total Requests: {result.total_requests}")
        print(f"Throughput: {result.throughput_tokens_per_sec:.1f} tokens/s")
        print(f"Request Rate: {result.throughput_requests_per_sec:.2f} requests/s")
        print(f"Memory Usage: {result.memory_usage_mb:.1f} MB")
        print(f"Avg Time/Request: {result.avg_time_per_request:.3f}s")
        print(f"Avg Tokens/Request: {result.avg_tokens_per_request:.1f}")
        
        # Save results if requested
        if args.output:
            runner.save_results(args.output)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_compare(args):
    """Compare different inference methods."""
    print("⚖️ Fast Inference - Method Comparison")
    print("=" * 40)
    
    try:
        # Create engines
        print(f"Loading models from {args.model_path}...")
        
        # Simple engine
        simple_engine = create_simple_fast_inference(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path
        )
        
        # Advanced engine (if requested)
        if args.include_advanced:
            advanced_engine = create_fast_inference_engine(
                model_path=args.model_path,
                tokenizer_path=args.tokenizer_path
            )
        
        print("✅ Models loaded successfully!")
        
        # Generate test prompts
        test_prompts = generate_test_prompts(
            num_prompts=args.num_requests,
            min_length=args.min_length,
            max_length=args.max_length
        )
        
        # Create benchmark runner
        runner = BenchmarkRunner()
        
        # Benchmark simple engine
        def simple_inference_func(prompts, max_tokens):
            return simple_engine.generate_batch(prompts, max_new_tokens=max_tokens)
        
        simple_result = runner.run_benchmark(
            "Simple Fast Inference",
            simple_inference_func,
            test_prompts,
            args.max_tokens
        )
        
        # Benchmark advanced engine (if requested)
        if args.include_advanced:
            def advanced_inference_func(prompts, max_tokens):
                return advanced_engine.generate_batch(prompts, max_new_tokens=max_tokens)
            
            advanced_result = runner.run_benchmark(
                "Advanced Fast Inference",
                advanced_inference_func,
                test_prompts,
                args.max_tokens
            )
        
        # Display comparison
        runner.print_summary()
        
        # Save results if requested
        if args.output:
            runner.save_results(args.output)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fast Inference Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate text from a prompt
  fast-inference generate --model-path model.pt --tokenizer-path tokenizer --prompts "Hello, world!"

  # Generate from file
  fast-inference generate --model-path model.pt --tokenizer-path tokenizer --prompt-file prompts.txt

  # Run benchmark
  fast-inference benchmark --model-path model.pt --tokenizer-path tokenizer --num-requests 10

  # Compare methods
  fast-inference compare --model-path model.pt --tokenizer-path tokenizer --include-advanced
        """
    )
    
    # Global arguments
    parser.add_argument('--model-path', required=True, help='Path to model file')
    parser.add_argument('--tokenizer-path', required=True, help='Path to tokenizer')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text from prompts')
    gen_parser.add_argument('--prompts', nargs='+', help='Input prompts')
    gen_parser.add_argument('--prompt-file', help='File containing prompts (one per line)')
    gen_parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    gen_parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    gen_parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty')
    gen_parser.add_argument('--output', help='Output file for results (JSON)')
    gen_parser.add_argument('--advanced', action='store_true', help='Use advanced engine')
    gen_parser.add_argument('--batch-size', type=int, default=32, help='Maximum batch size')
    gen_parser.add_argument('--max-seq-len', type=int, default=2048, help='Maximum sequence length')
    gen_parser.add_argument('--n-pages', type=int, default=1000, help='Number of KV cache pages')
    gen_parser.add_argument('--page-size', type=int, default=128, help='Size of each page')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    bench_parser.add_argument('--num-requests', type=int, default=10, help='Number of test requests')
    bench_parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens per request')
    bench_parser.add_argument('--min-length', type=int, default=20, help='Minimum prompt length')
    bench_parser.add_argument('--max-length', type=int, default=100, help='Maximum prompt length')
    bench_parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs')
    bench_parser.add_argument('--runs', type=int, default=1, help='Number of benchmark runs')
    bench_parser.add_argument('--output', help='Output file for results (JSON)')
    bench_parser.add_argument('--advanced', action='store_true', help='Use advanced engine')
    bench_parser.add_argument('--batch-size', type=int, default=32, help='Maximum batch size')
    bench_parser.add_argument('--max-seq-len', type=int, default=2048, help='Maximum sequence length')
    bench_parser.add_argument('--n-pages', type=int, default=1000, help='Number of KV cache pages')
    gen_parser.add_argument('--page-size', type=int, default=128, help='Size of each page')
    
    # Compare command
    comp_parser = subparsers.add_parser('compare', help='Compare different inference methods')
    comp_parser.add_argument('--num-requests', type=int, default=10, help='Number of test requests')
    comp_parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens per request')
    comp_parser.add_argument('--min-length', type=int, default=20, help='Minimum prompt length')
    comp_parser.add_argument('--max-length', type=int, default=100, help='Maximum prompt length')
    comp_parser.add_argument('--include-advanced', action='store_true', help='Include advanced engine comparison')
    comp_parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Execute command
    if args.command == 'generate':
        return cmd_generate(args)
    elif args.command == 'benchmark':
        return cmd_benchmark(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
