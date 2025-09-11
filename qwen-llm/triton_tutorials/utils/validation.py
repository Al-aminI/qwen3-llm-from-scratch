"""
✅ Validation Utilities

This module provides utilities for validating Triton kernels and ensuring
correctness against reference implementations.
"""

import torch
import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class ValidationResult:
    """Results from a validation test."""
    name: str
    passed: bool
    max_error: float
    mean_error: float
    relative_error: float
    error_message: Optional[str] = None

class ValidationSuite:
    """
    ✅ VALIDATION SUITE
    
    A comprehensive validation suite for Triton kernels.
    """
    
    def __init__(self, rtol: float = 1e-5, atol: float = 1e-6):
        """
        Initialize the validation suite.
        
        Args:
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
        """
        self.rtol = rtol
        self.atol = atol
        self.results = []
    
    def validate_function(self,
                         triton_func: Callable,
                         reference_func: Callable,
                         name: str,
                         *args, **kwargs) -> ValidationResult:
        """
        Validate a Triton function against a reference implementation.
        
        Args:
            triton_func: Triton implementation
            reference_func: Reference implementation (e.g., PyTorch)
            name: Name of the validation test
            *args: Arguments to pass to both functions
            **kwargs: Keyword arguments to pass to both functions
            
        Returns:
            ValidationResult object
        """
        try:
            # Run both implementations
            triton_result = triton_func(*args, **kwargs)
            reference_result = reference_func(*args, **kwargs)
            
            # Check if results are close
            if torch.allclose(triton_result, reference_result, rtol=self.rtol, atol=self.atol):
                passed = True
                max_error = 0.0
                mean_error = 0.0
                relative_error = 0.0
                error_message = None
            else:
                passed = False
                
                # Calculate error metrics
                diff = torch.abs(triton_result - reference_result)
                max_error = torch.max(diff).item()
                mean_error = torch.mean(diff).item()
                
                # Calculate relative error
                ref_abs = torch.abs(reference_result)
                relative_error = torch.mean(diff / (ref_abs + 1e-8)).item()
                
                error_message = f"Results don't match within tolerance (rtol={self.rtol}, atol={self.atol})"
            
            result = ValidationResult(
                name=name,
                passed=passed,
                max_error=max_error,
                mean_error=mean_error,
                relative_error=relative_error,
                error_message=error_message
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error_result = ValidationResult(
                name=name,
                passed=False,
                max_error=float('inf'),
                mean_error=float('inf'),
                relative_error=float('inf'),
                error_message=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def validate_shapes(self,
                       triton_func: Callable,
                       reference_func: Callable,
                       name: str,
                       *args, **kwargs) -> ValidationResult:
        """
        Validate that Triton and reference functions produce the same output shapes.
        
        Args:
            triton_func: Triton implementation
            reference_func: Reference implementation
            name: Name of the validation test
            *args: Arguments to pass to both functions
            **kwargs: Keyword arguments to pass to both functions
            
        Returns:
            ValidationResult object
        """
        try:
            # Run both implementations
            triton_result = triton_func(*args, **kwargs)
            reference_result = reference_func(*args, **kwargs)
            
            # Check shapes
            if triton_result.shape == reference_result.shape:
                passed = True
                max_error = 0.0
                mean_error = 0.0
                relative_error = 0.0
                error_message = None
            else:
                passed = False
                max_error = float('inf')
                mean_error = float('inf')
                relative_error = float('inf')
                error_message = f"Shape mismatch: Triton {triton_result.shape} vs Reference {reference_result.shape}"
            
            result = ValidationResult(
                name=name,
                passed=passed,
                max_error=max_error,
                mean_error=mean_error,
                relative_error=relative_error,
                error_message=error_message
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error_result = ValidationResult(
                name=name,
                passed=False,
                max_error=float('inf'),
                mean_error=float('inf'),
                relative_error=float('inf'),
                error_message=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def validate_dtypes(self,
                       triton_func: Callable,
                       reference_func: Callable,
                       name: str,
                       *args, **kwargs) -> ValidationResult:
        """
        Validate that Triton and reference functions produce the same output dtypes.
        
        Args:
            triton_func: Triton implementation
            reference_func: Reference implementation
            name: Name of the validation test
            *args: Arguments to pass to both functions
            **kwargs: Keyword arguments to pass to both functions
            
        Returns:
            ValidationResult object
        """
        try:
            # Run both implementations
            triton_result = triton_func(*args, **kwargs)
            reference_result = reference_func(*args, **kwargs)
            
            # Check dtypes
            if triton_result.dtype == reference_result.dtype:
                passed = True
                max_error = 0.0
                mean_error = 0.0
                relative_error = 0.0
                error_message = None
            else:
                passed = False
                max_error = float('inf')
                mean_error = float('inf')
                relative_error = float('inf')
                error_message = f"Dtype mismatch: Triton {triton_result.dtype} vs Reference {reference_result.dtype}"
            
            result = ValidationResult(
                name=name,
                passed=passed,
                max_error=max_error,
                mean_error=mean_error,
                relative_error=relative_error,
                error_message=error_message
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error_result = ValidationResult(
                name=name,
                passed=False,
                max_error=float('inf'),
                mean_error=float('inf'),
                relative_error=float('inf'),
                error_message=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def validate_edge_cases(self,
                           triton_func: Callable,
                           reference_func: Callable,
                           name: str,
                           test_cases: List[tuple]) -> List[ValidationResult]:
        """
        Validate function with multiple test cases.
        
        Args:
            triton_func: Triton implementation
            reference_func: Reference implementation
            name: Base name for validation tests
            test_cases: List of argument tuples for testing
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for i, args in enumerate(test_cases):
            result = self.validate_function(
                triton_func, reference_func,
                f"{name}_case_{i}",
                *args
            )
            results.append(result)
        
        return results
    
    def validate_numerical_stability(self,
                                   triton_func: Callable,
                                   reference_func: Callable,
                                   name: str,
                                   *args, **kwargs) -> ValidationResult:
        """
        Validate numerical stability of a function.
        
        Args:
            triton_func: Triton implementation
            reference_func: Reference implementation
            name: Name of the validation test
            *args: Arguments to pass to both functions
            **kwargs: Keyword arguments to pass to both functions
            
        Returns:
            ValidationResult object
        """
        try:
            # Run both implementations multiple times
            triton_results = []
            reference_results = []
            
            for _ in range(10):
                triton_result = triton_func(*args, **kwargs)
                reference_result = reference_func(*args, **kwargs)
                triton_results.append(triton_result)
                reference_results.append(reference_result)
            
            # Check consistency
            triton_consistent = all(torch.allclose(triton_results[0], result, rtol=1e-6) for result in triton_results[1:])
            reference_consistent = all(torch.allclose(reference_results[0], result, rtol=1e-6) for result in reference_results[1:])
            
            if triton_consistent and reference_consistent:
                # Check if results are close
                if torch.allclose(triton_results[0], reference_results[0], rtol=self.rtol, atol=self.atol):
                    passed = True
                    max_error = 0.0
                    mean_error = 0.0
                    relative_error = 0.0
                    error_message = None
                else:
                    passed = False
                    diff = torch.abs(triton_results[0] - reference_results[0])
                    max_error = torch.max(diff).item()
                    mean_error = torch.mean(diff).item()
                    ref_abs = torch.abs(reference_results[0])
                    relative_error = torch.mean(diff / (ref_abs + 1e-8)).item()
                    error_message = "Results don't match within tolerance"
            else:
                passed = False
                max_error = float('inf')
                mean_error = float('inf')
                relative_error = float('inf')
                error_message = "Function is not numerically stable"
            
            result = ValidationResult(
                name=name,
                passed=passed,
                max_error=max_error,
                mean_error=mean_error,
                relative_error=relative_error,
                error_message=error_message
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            error_result = ValidationResult(
                name=name,
                passed=False,
                max_error=float('inf'),
                mean_error=float('inf'),
                relative_error=float('inf'),
                error_message=str(e)
            )
            self.results.append(error_result)
            return error_result
    
    def print_results(self):
        """Print validation results in a formatted table."""
        if not self.results:
            print("No validation results available.")
            return
        
        print("\n✅ Validation Results:")
        print("=" * 80)
        print(f"{'Name':<30} {'Status':<10} {'Max Error':<12} {'Mean Error':<12} {'Rel Error':<12}")
        print("-" * 80)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            max_error = f"{result.max_error:.2e}" if result.max_error != float('inf') else "∞"
            mean_error = f"{result.mean_error:.2e}" if result.mean_error != float('inf') else "∞"
            rel_error = f"{result.relative_error:.2e}" if result.relative_error != float('inf') else "∞"
            
            print(f"{result.name:<30} {status:<10} {max_error:<12} {mean_error:<12} {rel_error:<12}")
            
            if result.error_message:
                print(f"  Error: {result.error_message}")
    
    def save_results(self, filename: str):
        """Save validation results to a JSON file."""
        results_dict = []
        for result in self.results:
            results_dict.append({
                'name': result.name,
                'passed': result.passed,
                'max_error': result.max_error,
                'mean_error': result.mean_error,
                'relative_error': result.relative_error,
                'error_message': result.error_message
            })
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Validation results saved to {filename}")
    
    def clear_results(self):
        """Clear all validation results."""
        self.results = []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        
        total = len(self.results)
        passed = sum(1 for result in self.results if result.passed)
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate
        }

def validate_vector_operations():
    """Validate vector operations."""
    print("✅ Validating Vector Operations:")
    print("=" * 50)
    
    suite = ValidationSuite()
    
    # Test different sizes
    sizes = [1024, 4096, 16384]
    
    for size in sizes:
        print(f"\n📊 Size: {size:,} elements")
        
        # Create test data
        a = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        b = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        
        # Validate addition
        suite.validate_function(
            lambda x, y: x + y,
            lambda x, y: x + y,
            f"Vector Addition ({size:,})",
            a, b
        )
        
        # Validate multiplication
        suite.validate_function(
            lambda x, y: x * y,
            lambda x, y: x * y,
            f"Vector Multiplication ({size:,})",
            a, b
        )
    
    suite.print_results()
    return suite

def validate_matrix_operations():
    """Validate matrix operations."""
    print("\n✅ Validating Matrix Operations:")
    print("=" * 50)
    
    suite = ValidationSuite()
    
    # Test different matrix sizes
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    for M, K, N in sizes:
        print(f"\n📊 Size: {M}x{K} @ {K}x{N} = {M}x{N}")
        
        # Create test data
        a = torch.randn(M, K, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float32)
        
        # Validate matrix multiplication
        suite.validate_function(
            torch.matmul,
            torch.matmul,
            f"Matrix Multiplication ({M}x{K}x{N})",
            a, b
        )
    
    suite.print_results()
    return suite

def validate_edge_cases():
    """Validate edge cases."""
    print("\n✅ Validating Edge Cases:")
    print("=" * 50)
    
    suite = ValidationSuite()
    
    # Test edge cases
    edge_cases = [
        # Single element
        (torch.tensor([1.0], device='cuda' if torch.cuda.is_available() else 'cpu'),
         torch.tensor([2.0], device='cuda' if torch.cuda.is_available() else 'cpu')),
        
        # Zero tensor
        (torch.zeros(100, device='cuda' if torch.cuda.is_available() else 'cpu'),
         torch.zeros(100, device='cuda' if torch.cuda.is_available() else 'cpu')),
        
        # Large values
        (torch.full((100,), 1e6, device='cuda' if torch.cuda.is_available() else 'cpu'),
         torch.full((100,), 1e6, device='cuda' if torch.cuda.is_available() else 'cpu')),
        
        # Small values
        (torch.full((100,), 1e-6, device='cuda' if torch.cuda.is_available() else 'cpu'),
         torch.full((100,), 1e-6, device='cuda' if torch.cuda.is_available() else 'cpu')),
    ]
    
    for i, (a, b) in enumerate(edge_cases):
        suite.validate_function(
            lambda x, y: x + y,
            lambda x, y: x + y,
            f"Edge Case {i}",
            a, b
        )
    
    suite.print_results()
    return suite

if __name__ == "__main__":
    # Run all validations
    vector_suite = validate_vector_operations()
    matrix_suite = validate_matrix_operations()
    edge_suite = validate_edge_cases()
    
    # Print summaries
    print("\n📊 Validation Summary:")
    print("=" * 50)
    
    vector_summary = vector_suite.get_summary()
    matrix_summary = matrix_suite.get_summary()
    edge_summary = edge_suite.get_summary()
    
    print(f"Vector Operations: {vector_summary['passed']}/{vector_summary['total']} passed ({vector_summary['pass_rate']:.1%})")
    print(f"Matrix Operations: {matrix_summary['passed']}/{matrix_summary['total']} passed ({matrix_summary['pass_rate']:.1%})")
    print(f"Edge Cases: {edge_summary['passed']}/{edge_summary['total']} passed ({edge_summary['pass_rate']:.1%})")
    
    # Save results
    vector_suite.save_results("vector_validation.json")
    matrix_suite.save_results("matrix_validation.json")
    edge_suite.save_results("edge_validation.json")
