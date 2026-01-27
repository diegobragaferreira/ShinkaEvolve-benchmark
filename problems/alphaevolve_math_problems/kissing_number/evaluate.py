import argparse
import numpy as np
import itertools
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 593
TOL = 1e-6

def compute_squared_norm(point: list[int]) -> int:
    """Returns the squared norm of an integer vector using exact computation."""
    return sum(pow(int(x), 2) for x in point)

def verify_sphere_packing(sphere_centers: np.ndarray, tol: float = 1e-6):
    """Checks that after normalizing, the points correspond to a valid sphere packing for kissing numbers."""
    # Rounding to integers to guarantee exact computation throughout.
    sphere_centers = np.around(sphere_centers).astype(np.int64)
    squared_norms = [compute_squared_norm(list(center)) for center in sphere_centers]

    # Checks that the set doesn't contain 0.
    min_squared_norm = min(squared_norms)
    if min_squared_norm <= tol:
        raise AssertionError("Verification failed because the set contains 0.")

    # Checks that the minimum pairwise distance between centers >= the maximum norm of the centers.
    max_squared_norm = max(squared_norms)
    min_squared_distance = min(
        compute_squared_norm(list(a - b)) for a, b in itertools.combinations(sphere_centers, 2)
    )
    if min_squared_distance < max_squared_norm:
        raise AssertionError(f"Verification failed because the minimum squared distance = {min_squared_distance} < {max_squared_norm} = maximum squared norm.")

def validate_wrapper(points: Any) -> Tuple[bool, Optional[str]]:
    try:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape[1] != 11:
            return False, f"Invalid shapes: points = {points.shape}, expected ({points.shape[1]},11)"
        
        verify_sphere_packing(points, TOL)
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    points = results[0]
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    num_points = len(points)
    
    return {
        "combined_score": float(num_points),
        "num_points": float(num_points),
        "benchmark_ratio": float(num_points / BENCHMARK),
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="kissing_number11",
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
