import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 1.4556

def verify_c3_solution(f_values: np.ndarray) -> float:
    """Verify the solution for the C3 UPPER BOUND optimization."""

    n_points = len(f_values)
    if n_points == 0 or f_values is None:
        raise ValueError("Received empty function values.")
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")

    convolution = np.convolve(f_values, f_values)
    den = (np.sum(f_values)**2)
    if den < 1e-12:
        raise ValueError(f"Sum of squared values of the function is too close to zero {den}.")
    c3 = abs(2 * len(f_values) * np.max(convolution) / den)

    return c3

def validate_wrapper(f_values_list: Any) -> Tuple[bool, Optional[str]]:
    try:
        # Convert to numpy array
        if not isinstance(f_values_list, (list, np.ndarray)):
            return False, f"construct_function must return list or np.ndarray, got {type(f_values_list)}"
        f_values = np.array(f_values_list, dtype=float)
        verify_c3_solution(f_values)
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    f_values_list = results[0]
    f_values = np.array(f_values_list, dtype=float)
    c3 = verify_c3_solution(f_values)
    
    return {
        "combined_score": float(1/c3), # Maximize 1/C3
        "inv_c3": float(1/c3),
        "benchmark_ratio": BENCHMARK / float(c3),
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
