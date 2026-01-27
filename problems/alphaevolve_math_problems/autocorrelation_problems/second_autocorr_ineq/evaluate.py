import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 0.962

def verify_c2_solution(f_values: np.ndarray):
    """
    Verifies the C2 lower bound solution using the rigorous, unitless, piecewise linear integral method.
    """
    n_points = len(f_values)
    if n_points == 0 or f_values is None:
        raise ValueError("Received empty function values.")
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")
    if np.any(f_values < -1e-6):  # Allow for small floating point errors
        raise ValueError("Function must be non-negative.")
    
    f_nonneg = np.maximum(f_values, 0.0)
    
    # The raw, unscaled convolution is used
    convolution = np.convolve(f_nonneg, f_nonneg, mode="full")
    
    # Calculate the L2-norm squared: ||f*f||_2^2 via piecewise linear integration
    num_conv_points = len(convolution)
    x_points = np.linspace(-0.5, 0.5, num_conv_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution, [0]))
    
    l2_norm_squared = 0.0
    for i in range(len(convolution) + 1):
        y1, y2, h = y_points[i], y_points[i + 1], x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared
    
    # Calculate the L1-norm: ||f*f||_1
    # This is an approximation of the integral of the absolute value of the autoconvolution
    norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)
    
    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution))
    
    # Check for division by zero
    if norm_1 * norm_inf < 1e-12:
        raise ValueError(f"Norm product too close to zero: norm_1={norm_1}, norm_inf={norm_inf}")
    
    computed_c2 = l2_norm_squared / (norm_1 * norm_inf)
    return computed_c2

def validate_wrapper(f_values_list: Any) -> Tuple[bool, Optional[str]]:
    try:
        # Convert to numpy array
        if not isinstance(f_values_list, (list, np.ndarray)):
            return False, f"construct_function must return list or np.ndarray, got {type(f_values_list)}"
        f_values = np.array(f_values_list, dtype=float)
        
        verify_c2_solution(f_values)
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    f_values_list = results[0]
    f_values = np.array(f_values_list, dtype=float)
    c2 = verify_c2_solution(f_values)
    
    return {
        "combined_score": float(c2),
        "c2": float(c2),
        "benchmark_ratio": float(c2) / BENCHMARK,
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="construct_function",
        num_runs=1,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_wrapper,
        aggregate_metrics_fn=aggregate_metrics,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path", type=str)
    parser.add_argument("results_dir", type=str)
    args = parser.parse_args()

    main(args.program_path, args.results_dir)
