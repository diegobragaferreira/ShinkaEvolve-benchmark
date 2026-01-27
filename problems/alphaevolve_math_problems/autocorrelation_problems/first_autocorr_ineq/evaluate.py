import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK: float = 1.5031

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    """
    if not isinstance(sequence, list):
        raise ValueError(f"Sequence type expected to be list, received {type(sequence)}")

    if not sequence:
        raise ValueError("Sequence cannot be None.")

    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Sequence entries must be integers or floats.")
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Sequence cannot contain nans or infs.")

    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    if sum_a < 0.01:
        raise ValueError(f"Sum of sequence entries too close to zero: {sum_a}.")

    return float(2 * n * max_b / (sum_a**2))

def validate_wrapper(sequence: Any) -> Tuple[bool, Optional[str]]:
    try:
        evaluate_sequence(sequence)
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    sequence = results[0]
    c1 = evaluate_sequence(sequence)
    
    return {
        "combined_score": float(1/c1), # Maximizing 1/C1
        "inv_c1": float(1/c1),
        "benchmark_ratio": float(BENCHMARK/c1),
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="search_for_best_sequence",
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
