"""
🚀 Triton Tutorials CLI

Command-line interface for running Triton tutorials and examples.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """
    🎯 MAIN CLI FUNCTION
    
    Provides a command-line interface for running tutorials.
    """
    parser = argparse.ArgumentParser(
        description="🚀 Triton Tutorials CLI - Learn Triton from beginner to expert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  triton-tutorial run beginner 1          # Run lesson 1 from beginner level
  triton-tutorial run intermediate 4      # Run lesson 4 from intermediate level
  triton-tutorial list                    # List all available lessons
  triton-tutorial benchmark               # Run performance benchmarks
  triton-tutorial install                 # Install dependencies
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a specific lesson')
    run_parser.add_argument('level', choices=['beginner', 'intermediate', 'advanced', 'expert'],
                           help='Tutorial level')
    run_parser.add_argument('lesson', type=int, help='Lesson number')
    run_parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all available lessons')
    list_parser.add_argument('--level', choices=['beginner', 'intermediate', 'advanced', 'expert'],
                           help='Filter by level')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--lesson', type=int, help='Specific lesson to benchmark')
    benchmark_parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    
    # Install command
    install_parser = subparsers.add_parser('install', help='Install dependencies')
    install_parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_lesson(args.level, args.lesson, args.gpu)
    elif args.command == 'list':
        list_lessons(args.level)
    elif args.command == 'benchmark':
        run_benchmarks(args.lesson, args.all)
    elif args.command == 'install':
        install_dependencies(args.dev)
    elif args.command == 'info':
        show_system_info()
    else:
        parser.print_help()

def run_lesson(level: str, lesson: int, force_gpu: bool):
    """
    🎯 RUN A SPECIFIC LESSON
    
    Args:
        level: Tutorial level (beginner, intermediate, advanced, expert)
        lesson: Lesson number
        force_gpu: Force GPU usage
    """
    print(f"🚀 Running {level} lesson {lesson}")
    
    # Check CUDA availability
    if force_gpu:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Please use a GPU-enabled environment.")
            sys.exit(1)
    
    # Import and run the lesson
    try:
        if level == 'beginner':
            if lesson == 1:
                from lessons.beginner.lesson_01_gpu_fundamentals import main
            elif lesson == 2:
                from lessons.beginner.lesson_02_memory_management import main
            elif lesson == 3:
                from lessons.beginner.lesson_03_basic_operations import main
            else:
                print(f"❌ Lesson {lesson} not found in {level} level")
                sys.exit(1)
        elif level == 'intermediate':
            if lesson == 4:
                from lessons.intermediate.lesson_04_matrix_operations import main
            else:
                print(f"❌ Lesson {lesson} not found in {level} level")
                sys.exit(1)
        else:
            print(f"❌ Level {level} not implemented yet")
            sys.exit(1)
        
        main()
        
    except ImportError as e:
        print(f"❌ Error importing lesson: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running lesson: {e}")
        sys.exit(1)

def list_lessons(level_filter: str = None):
    """
    📚 LIST ALL AVAILABLE LESSONS
    
    Args:
        level_filter: Optional level filter
    """
    print("📚 Available Triton Tutorials:")
    print("=" * 50)
    
    lessons = {
        'beginner': [
            "1. GPU Fundamentals & Triton Basics",
            "2. Memory Management & Data Types",
            "3. Basic Operations & Kernels"
        ],
        'intermediate': [
            "4. Matrix Operations & Tiling Strategies",
            "5. Advanced Memory Patterns & Optimization",
            "6. Kernel Fusion & Performance Tuning"
        ],
        'advanced': [
            "7. Attention Mechanisms & FlashAttention",
            "8. Transformer Components & Optimization",
            "9. MoE (Mixture of Experts) Kernels"
        ],
        'expert': [
            "10. Autotuning & Advanced Optimization",
            "11. Custom Data Types & Quantization",
            "12. Production Systems & Deployment"
        ]
    }
    
    for level, lesson_list in lessons.items():
        if level_filter and level != level_filter:
            continue
            
        print(f"\n🎯 {level.upper()} LEVEL:")
        for lesson in lesson_list:
            print(f"  {lesson}")
    
    print(f"\n💡 Usage: triton-tutorial run <level> <lesson_number>")
    print(f"   Example: triton-tutorial run beginner 1")

def run_benchmarks(lesson: int = None, run_all: bool = False):
    """
    📊 RUN PERFORMANCE BENCHMARKS
    
    Args:
        lesson: Specific lesson to benchmark
        run_all: Run all benchmarks
    """
    print("📊 Running Performance Benchmarks:")
    print("=" * 50)
    
    if lesson:
        print(f"🎯 Benchmarking lesson {lesson}")
        # Run specific lesson benchmark
        if lesson == 1:
            from lessons.beginner.lesson_01_gpu_fundamentals import benchmark_vector_addition
            benchmark_vector_addition()
        elif lesson == 2:
            from lessons.beginner.lesson_02_memory_management import benchmark_memory_access
            benchmark_memory_access()
        elif lesson == 3:
            from lessons.beginner.lesson_03_basic_operations import benchmark_basic_operations
            benchmark_basic_operations()
        elif lesson == 4:
            from lessons.intermediate.lesson_04_matrix_operations import benchmark_matrix_operations
            benchmark_matrix_operations()
        else:
            print(f"❌ Lesson {lesson} benchmark not available")
    elif run_all:
        print("🎯 Running all available benchmarks")
        # Run all benchmarks
        try:
            from lessons.beginner.lesson_01_gpu_fundamentals import benchmark_vector_addition
            benchmark_vector_addition()
        except ImportError:
            pass
        
        try:
            from lessons.beginner.lesson_02_memory_management import benchmark_memory_access
            benchmark_memory_access()
        except ImportError:
            pass
        
        try:
            from lessons.beginner.lesson_03_basic_operations import benchmark_basic_operations
            benchmark_basic_operations()
        except ImportError:
            pass
        
        try:
            from lessons.intermediate.lesson_04_matrix_operations import benchmark_matrix_operations
            benchmark_matrix_operations()
        except ImportError:
            pass
    else:
        print("❌ Please specify --lesson <number> or --all")

def install_dependencies(dev: bool = False):
    """
    📦 INSTALL DEPENDENCIES
    
    Args:
        dev: Install development dependencies
    """
    print("📦 Installing Dependencies:")
    print("=" * 50)
    
    import subprocess
    import sys
    
    # Install core dependencies
    print("Installing core dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    if dev:
        print("Installing development dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    
    print("✅ Dependencies installed successfully!")

def show_system_info():
    """
    ℹ️ SHOW SYSTEM INFORMATION
    
    Displays system information relevant to Triton development.
    """
    print("ℹ️ System Information:")
    print("=" * 50)
    
    # Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # PyTorch version
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Triton version
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: Not installed")
    
    # NumPy version
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")

if __name__ == "__main__":
    main()
