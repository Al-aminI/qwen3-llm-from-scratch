# 📚 Triton Tutorials API Reference

This document provides a comprehensive API reference for the Triton Tutorials package.

## 🎯 Table of Contents

- [Core Modules](#core-modules)
- [Lesson Modules](#lesson-modules)
- [Utility Modules](#utility-modules)
- [Example Modules](#example-modules)
- [Benchmark Modules](#benchmark-modules)
- [Test Modules](#test-modules)

## 🧠 Core Modules

### `triton_tutorials.__init__`

Main package initialization module.

```python
from triton_tutorials import *
```

**Exports:**
- `BenchmarkSuite`: Comprehensive benchmarking suite
- `PerformanceProfiler`: Performance profiling utilities
- `ValidationSuite`: Validation and testing utilities
- `DataGenerator`: Test data generation utilities
- `PerformanceAnalyzer`: Performance analysis utilities

## 📖 Lesson Modules

### Beginner Lessons

#### `lessons.beginner.lesson_01_gpu_fundamentals`

**Functions:**
- `vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`
  - Performs vector addition using Triton
  - **Parameters:**
    - `a`: First input vector
    - `b`: Second input vector
  - **Returns:** Sum of input vectors
  - **Example:**
    ```python
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    result = vector_add(a, b)
    ```

#### `lessons.beginner.lesson_02_memory_management`

**Kernels:**
- `coalesced_access_kernel(input_ptr, output_ptr, n_elements, stride, BLOCK_SIZE)`
  - Demonstrates coalesced memory access
- `non_coalesced_access_kernel(input_ptr, output_ptr, n_elements, stride, BLOCK_SIZE)`
  - Demonstrates non-coalesced memory access

#### `lessons.beginner.lesson_03_basic_operations`

**Kernels:**
- `element_wise_add_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE)`
  - Performs element-wise addition
- `sum_reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE)`
  - Performs sum reduction

### Intermediate Lessons

#### `lessons.intermediate.lesson_04_matrix_operations`

**Functions:**
- `basic_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`
  - Basic matrix multiplication implementation
- `optimized_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`
  - Optimized matrix multiplication implementation
- `batch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`
  - Batch matrix multiplication
- `matrix_transpose(a: torch.Tensor) -> torch.Tensor`
  - Matrix transpose operation

#### `lessons.intermediate.lesson_05_advanced_memory`

**Functions:**
- `shared_memory_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`
  - Matrix multiplication using shared memory
- `cache_friendly_reduction(input_tensor: torch.Tensor) -> torch.Tensor`
  - Cache-friendly reduction operation

#### `lessons.intermediate.lesson_06_kernel_fusion`

**Functions:**
- `fused_add_multiply(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor`
  - Fused add-multiply operation: `(a + b) * c`
- `fused_matmul_activation(a: torch.Tensor, b: torch.Tensor, activation: str) -> torch.Tensor`
  - Fused matrix multiplication + activation
  - **Parameters:**
    - `activation`: Activation function ("relu", "tanh", "sigmoid", "gelu")
- `fused_loop(input_tensor: torch.Tensor, num_iterations: int) -> torch.Tensor`
  - Fused loop operation

## 🛠️ Utility Modules

### `utils.benchmarking`

#### `BenchmarkSuite`

Comprehensive benchmarking suite for Triton kernels.

```python
from utils.benchmarking import BenchmarkSuite

suite = BenchmarkSuite(warmup_runs=10, benchmark_runs=100)
```

**Methods:**
- `benchmark_function(triton_func, pytorch_func, name, *args, **kwargs) -> BenchmarkResult`
  - Benchmarks a Triton function against PyTorch
- `benchmark_memory_bandwidth(func, name, input_size, dtype) -> BenchmarkResult`
  - Benchmarks memory bandwidth utilization
- `benchmark_matrix_sizes(triton_func, pytorch_func, name, sizes) -> List[BenchmarkResult]`
  - Benchmarks multiple matrix sizes
- `print_results()`
  - Prints benchmark results in formatted table
- `save_results(filename: str)`
  - Saves results to JSON file
- `clear_results()`
  - Clears all benchmark results

**Example:**
```python
suite = BenchmarkSuite()
result = suite.benchmark_function(
    triton_func=lambda x, y: x + y,
    pytorch_func=lambda x, y: x + y,
    name="Vector Addition",
    a, b
)
suite.print_results()
```

#### `BenchmarkResult`

Result from a benchmark run.

**Attributes:**
- `name: str`: Name of the benchmark
- `triton_time: float`: Triton execution time (ms)
- `pytorch_time: float`: PyTorch execution time (ms)
- `speedup: float`: Speedup ratio
- `memory_usage: Optional[float]`: Memory usage (GB)
- `throughput: Optional[float]`: Throughput (GB/s)
- `error: Optional[str]`: Error message if any

### `utils.profiling`

#### `PerformanceProfiler`

Performance profiler for Triton kernels.

```python
from utils.profiling import PerformanceProfiler

profiler = PerformanceProfiler()
```

**Methods:**
- `profile_function(func, name, *args, **kwargs) -> ProfileResult`
  - Profiles a function's performance
- `profile_memory_bandwidth(func, name, input_size, dtype) -> ProfileResult`
  - Profiles memory bandwidth utilization
- `profile_kernel_launch_overhead(kernel_func, name, num_launches, *args, **kwargs) -> ProfileResult`
  - Profiles kernel launch overhead
- `print_results()`
  - Prints profiling results
- `save_results(filename: str)`
  - Saves results to JSON file
- `clear_results()`
  - Clears all profiling results

#### `ProfileResult`

Result from a profiling run.

**Attributes:**
- `name: str`: Name of the profiled function
- `execution_time: float`: Execution time (ms)
- `memory_usage: float`: Memory usage (GB)
- `gpu_memory_usage: float`: GPU memory usage (GB)
- `gpu_utilization: float`: GPU utilization (%)
- `cpu_utilization: float`: CPU utilization (%)
- `throughput: Optional[float]`: Throughput (GB/s)
- `error: Optional[str]`: Error message if any

### `utils.validation`

#### `ValidationSuite`

Validation suite for Triton kernels.

```python
from utils.validation import ValidationSuite

suite = ValidationSuite(rtol=1e-5, atol=1e-6)
```

**Methods:**
- `validate_function(triton_func, reference_func, name, *args, **kwargs) -> ValidationResult`
  - Validates a Triton function against reference
- `validate_shapes(triton_func, reference_func, name, *args, **kwargs) -> ValidationResult`
  - Validates output shapes
- `validate_dtypes(triton_func, reference_func, name, *args, **kwargs) -> ValidationResult`
  - Validates output data types
- `validate_edge_cases(triton_func, reference_func, name, test_cases) -> List[ValidationResult]`
  - Validates with multiple test cases
- `validate_numerical_stability(triton_func, reference_func, name, *args, **kwargs) -> ValidationResult`
  - Validates numerical stability
- `print_results()`
  - Prints validation results
- `save_results(filename: str)`
  - Saves results to JSON file
- `get_summary() -> Dict[str, Any]`
  - Gets validation summary
- `clear_results()`
  - Clears all validation results

#### `ValidationResult`

Result from a validation test.

**Attributes:**
- `name: str`: Name of the validation test
- `passed: bool`: Whether the test passed
- `max_error: float`: Maximum error
- `mean_error: float`: Mean error
- `relative_error: float`: Relative error
- `error_message: Optional[str]`: Error message if any

### `utils.data_generation`

#### `DataGenerator`

Test data generator for Triton kernels.

```python
from utils.data_generation import DataGenerator, DataConfig

generator = DataGenerator(seed=42)
```

**Methods:**
- `generate_vector(config: DataConfig) -> torch.Tensor`
  - Generates a vector with specified configuration
- `generate_matrix(rows: int, cols: int, config: DataConfig) -> torch.Tensor`
  - Generates a matrix with specified configuration
- `generate_batch_matrices(batch_size: int, rows: int, cols: int, config: DataConfig) -> torch.Tensor`
  - Generates a batch of matrices
- `generate_attention_data(batch_size, num_heads, seq_len, head_dim, config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
  - Generates attention data (Q, K, V)
- `generate_sparse_data(size: int, sparsity: float, config: DataConfig) -> torch.Tensor`
  - Generates sparse data
- `generate_sequence_data(batch_size, seq_len, vocab_size, config) -> torch.Tensor`
  - Generates sequence data (token IDs)
- `generate_benchmark_data(sizes: List[int], dtype) -> List[torch.Tensor]`
  - Generates data for benchmarking
- `generate_matrix_benchmark_data(sizes: List[Tuple[int, int, int]], dtype) -> List[Tuple[torch.Tensor, torch.Tensor]]`
  - Generates matrix data for benchmarking

#### `DataConfig`

Configuration for data generation.

**Attributes:**
- `size: int`: Size of the tensor
- `dtype: torch.dtype`: Data type (default: torch.float32)
- `device: str`: Device (default: 'cuda' if available, else 'cpu')
- `seed: Optional[int]`: Random seed for reproducibility
- `distribution: str`: Distribution type ('normal', 'uniform', 'zeros', 'ones')
- `mean: float`: Mean for normal distribution (default: 0.0)
- `std: float`: Standard deviation for normal distribution (default: 1.0)
- `min_val: float`: Minimum value for uniform distribution (default: 0.0)
- `max_val: float`: Maximum value for uniform distribution (default: 1.0)

### `utils.performance_analysis`

#### `PerformanceAnalyzer`

Performance analyzer for Triton kernels.

```python
from utils.performance_analysis import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
```

**Methods:**
- `analyze_kernel(kernel_func, name, input_size, dtype, num_runs) -> PerformanceMetrics`
  - Analyzes kernel performance
- `compare_kernels(triton_func, pytorch_func, name, input_size, dtype, num_runs) -> Tuple[PerformanceMetrics, PerformanceMetrics]`
  - Compares Triton and PyTorch implementations
- `analyze_scaling(kernel_func, name, sizes, dtype, num_runs) -> List[PerformanceMetrics]`
  - Analyzes scaling performance
- `plot_performance(metrics_list, metric, title)`
  - Plots performance metrics
- `plot_scaling(scaling_metrics, metric, title)`
  - Plots scaling analysis
- `plot_comparison(triton_metrics, pytorch_metrics, title)`
  - Plots comparison between implementations
- `generate_report(filename: str)`
  - Generates performance report
- `print_summary()`
  - Prints performance summary
- `clear_metrics()`
  - Clears all performance metrics

#### `PerformanceMetrics`

Performance metrics for a kernel.

**Attributes:**
- `name: str`: Name of the kernel
- `execution_time: float`: Execution time (ms)
- `memory_bandwidth: float`: Memory bandwidth (GB/s)
- `arithmetic_intensity: float`: Arithmetic intensity
- `gpu_utilization: float`: GPU utilization (%)
- `cache_hit_rate: float`: Cache hit rate (%)
- `throughput: float`: Throughput (elements/s)
- `efficiency: float`: Efficiency (0-1)

## 🎯 Example Modules

### `examples.llm_inference_optimization`

**Functions:**
- `optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor`
  - Optimized attention mechanism
- `transformer_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`
  - Optimized transformer matrix multiplication
- `layer_norm(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor`
  - Optimized layer normalization

## 📊 Benchmark Modules

### `benchmarks.benchmark_beginner`

#### `BenchmarkBeginnerLessons`

Benchmark suite for beginner lessons.

```python
from benchmarks.benchmark_beginner import BenchmarkBeginnerLessons

benchmark = BenchmarkBeginnerLessons()
benchmark.run_all_benchmarks()
```

**Methods:**
- `benchmark_vector_addition()`
- `benchmark_memory_access_patterns()`
- `benchmark_element_wise_operations()`
- `benchmark_reduction_operations()`
- `benchmark_activation_functions()`
- `benchmark_memory_bandwidth()`
- `benchmark_data_types()`
- `run_all_benchmarks()`

### `benchmarks.benchmark_intermediate`

#### `BenchmarkIntermediateLessons`

Benchmark suite for intermediate lessons.

```python
from benchmarks.benchmark_intermediate import BenchmarkIntermediateLessons

benchmark = BenchmarkIntermediateLessons()
benchmark.run_all_benchmarks()
```

**Methods:**
- `benchmark_matrix_operations()`
- `benchmark_batch_matrix_operations()`
- `benchmark_matrix_transpose()`
- `benchmark_shared_memory_operations()`
- `benchmark_cache_friendly_operations()`
- `benchmark_kernel_fusion()`
- `run_all_benchmarks()`

### `benchmarks.benchmark_examples`

#### `BenchmarkExamples`

Benchmark suite for examples.

```python
from benchmarks.benchmark_examples import BenchmarkExamples

benchmark = BenchmarkExamples()
benchmark.run_all_benchmarks()
```

**Methods:**
- `benchmark_transformer_matmul()`
- `benchmark_layer_norm()`
- `benchmark_attention_mechanism()`
- `benchmark_llm_pipeline()`
- `benchmark_memory_usage()`
- `benchmark_different_dtypes()`
- `run_all_benchmarks()`

### `benchmarks.benchmark_utils`

#### `BenchmarkUtilities`

Benchmark suite for utilities.

```python
from benchmarks.benchmark_utils import BenchmarkUtilities

benchmark = BenchmarkUtilities()
benchmark.run_all_benchmarks()
```

**Methods:**
- `benchmark_benchmarking_suite()`
- `benchmark_profiling_suite()`
- `benchmark_validation_suite()`
- `benchmark_data_generation()`
- `benchmark_performance_analysis()`
- `benchmark_utility_integration()`
- `run_all_benchmarks()`

## 🧪 Test Modules

### `tests.test_beginner`

#### `TestBeginnerLessons`

Test suite for beginner lessons.

```python
from tests.test_beginner import TestBeginnerLessons

# Run tests
unittest.main()
```

**Test Methods:**
- `test_lesson_01_vector_addition_small()`
- `test_lesson_01_vector_addition_medium()`
- `test_lesson_01_vector_addition_large()`
- `test_lesson_01_vector_addition_edge_cases()`
- `test_lesson_01_vector_addition_different_dtypes()`
- `test_lesson_02_memory_coalescing()`
- `test_lesson_03_element_wise_operations()`
- `test_lesson_03_reduction_operations()`
- `test_lesson_03_reduction_edge_cases()`
- `test_lesson_03_broadcasting()`
- `test_lesson_03_activation_functions()`
- `test_lesson_03_error_handling()`
- `test_lesson_03_performance_characteristics()`
- `test_lesson_03_memory_bandwidth()`

#### `TestBeginnerLessonsIntegration`

Integration tests for beginner lessons.

**Test Methods:**
- `test_lesson_integration_vector_operations()`
- `test_lesson_integration_reduction_operations()`

### `tests.test_intermediate`

#### `TestIntermediateLessons`

Test suite for intermediate lessons.

**Test Methods:**
- `test_lesson_04_basic_matrix_multiplication()`
- `test_lesson_04_optimized_matrix_multiplication()`
- `test_lesson_04_batch_matrix_multiplication()`
- `test_lesson_04_matrix_transpose()`
- `test_lesson_04_matrix_operations_edge_cases()`
- `test_lesson_05_shared_memory_matrix_multiplication()`
- `test_lesson_05_cache_friendly_reduction()`
- `test_lesson_05_advanced_memory_edge_cases()`
- `test_lesson_06_fused_add_multiply()`
- `test_lesson_06_fused_matmul_activation()`
- `test_lesson_06_fused_loop()`
- `test_lesson_06_kernel_fusion_edge_cases()`
- `test_lesson_06_fusion_performance_characteristics()`

#### `TestIntermediateLessonsIntegration`

Integration tests for intermediate lessons.

**Test Methods:**
- `test_lesson_integration_matrix_operations()`
- `test_lesson_integration_memory_optimization()`
- `test_lesson_integration_performance_optimization()`
- `test_lesson_integration_kernel_fusion()`

### `tests.test_utils`

#### `TestBenchmarking`

Test suite for benchmarking utilities.

**Test Methods:**
- `test_benchmark_suite_initialization()`
- `test_benchmark_function()`
- `test_benchmark_memory_bandwidth()`
- `test_benchmark_matrix_sizes()`
- `test_benchmark_error_handling()`
- `test_benchmark_results_management()`

#### `TestProfiling`

Test suite for profiling utilities.

**Test Methods:**
- `test_profiler_initialization()`
- `test_profile_function()`
- `test_profile_memory_bandwidth()`
- `test_profile_kernel_launch_overhead()`
- `test_profile_error_handling()`

#### `TestValidation`

Test suite for validation utilities.

**Test Methods:**
- `test_validation_suite_initialization()`
- `test_validate_function()`
- `test_validate_shapes()`
- `test_validate_dtypes()`
- `test_validate_edge_cases()`
- `test_validate_numerical_stability()`
- `test_validation_error_handling()`
- `test_validation_summary()`

#### `TestDataGeneration`

Test suite for data generation utilities.

**Test Methods:**
- `test_data_generator_initialization()`
- `test_generate_vector()`
- `test_generate_matrix()`
- `test_generate_batch_matrices()`
- `test_generate_attention_data()`
- `test_generate_sparse_data()`
- `test_generate_sequence_data()`
- `test_generate_benchmark_data()`
- `test_generate_matrix_benchmark_data()`
- `test_data_distributions()`

#### `TestPerformanceAnalysis`

Test suite for performance analysis utilities.

**Test Methods:**
- `test_analyzer_initialization()`
- `test_analyze_kernel()`
- `test_compare_kernels()`
- `test_analyze_scaling()`
- `test_analyzer_error_handling()`
- `test_analyzer_summary()`

### `tests.test_examples`

#### `TestExamples`

Test suite for examples.

**Test Methods:**
- `test_optimized_attention()`
- `test_transformer_matmul()`
- `test_transformer_matmul_edge_cases()`
- `test_layer_norm()`
- `test_layer_norm_edge_cases()`
- `test_layer_norm_different_eps()`
- `test_examples_performance_characteristics()`
- `test_examples_memory_usage()`
- `test_examples_different_dtypes()`
- `test_examples_error_handling()`
- `test_examples_integration()`

#### `TestExamplesIntegration`

Integration tests for examples.

**Test Methods:**
- `test_transformer_pipeline()`
- `test_attention_mechanism()`
- `test_memory_efficiency()`

## 🚀 Usage Examples

### Basic Usage

```python
from triton_tutorials import BenchmarkSuite, ValidationSuite, DataGenerator

# Generate test data
generator = DataGenerator(seed=42)
config = DataConfig(size=1024, dtype=torch.float32, device='cuda')
a = generator.generate_vector(config)
b = generator.generate_vector(config)

# Benchmark
suite = BenchmarkSuite()
result = suite.benchmark_function(
    triton_func=lambda x, y: x + y,
    pytorch_func=lambda x, y: x + y,
    name="Vector Addition",
    a, b
)
suite.print_results()

# Validate
validation = ValidationSuite()
validation_result = validation.validate_function(
    triton_func=lambda x, y: x + y,
    reference_func=lambda x, y: x + y,
    name="Vector Addition",
    a, b
)
validation.print_results()
```

### Advanced Usage

```python
from triton_tutorials import PerformanceProfiler, PerformanceAnalyzer
from lessons.intermediate.lesson_04_matrix_operations import optimized_matmul

# Profile performance
profiler = PerformanceProfiler()
profile_result = profiler.profile_function(
    optimized_matmul,
    "Optimized Matrix Multiplication",
    a, b
)
profiler.print_results()

# Analyze performance
analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze_kernel(
    optimized_matmul,
    "Optimized Matrix Multiplication",
    a.numel(),
    torch.float32,
    num_runs=100
)
analyzer.print_summary()
```

### Running Benchmarks

```python
from benchmarks.benchmark_beginner import BenchmarkBeginnerLessons
from benchmarks.benchmark_intermediate import BenchmarkIntermediateLessons

# Run beginner benchmarks
beginner_benchmark = BenchmarkBeginnerLessons()
beginner_benchmark.run_all_benchmarks()

# Run intermediate benchmarks
intermediate_benchmark = BenchmarkIntermediateLessons()
intermediate_benchmark.run_all_benchmarks()
```

### Running Tests

```python
import unittest
from tests.test_beginner import TestBeginnerLessons
from tests.test_intermediate import TestIntermediateLessons

# Run tests
unittest.main()
```

## 📝 Notes

- All functions require CUDA-enabled GPUs for optimal performance
- CPU fallback is available but with reduced performance
- Memory usage is optimized for large-scale operations
- All benchmarks include correctness verification
- Error handling is comprehensive with detailed error messages
- Results can be saved to JSON files for further analysis

## 🔗 Related Documentation

- [README.md](../README.md) - Package overview and installation
- [QUICK_START.md](../QUICK_START.md) - Quick start guide
- [TRITON_TUTORIALS_SUMMARY.md](../TRITON_TUTORIALS_SUMMARY.md) - Complete tutorial summary
