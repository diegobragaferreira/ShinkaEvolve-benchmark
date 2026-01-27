import argparse
import numpy as np
import itertools
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 0.036529889880030156
TOL = 1e-6
NUM_POINTS = 11

def check_inside_triangle_wtol(points: np.ndarray, tol: float = 1e-6):
    """Checks that all points are inside the triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    for x, y in points:
        cond1 = y >= -tol
        cond2 = np.sqrt(3) * x <= np.sqrt(3) - y + tol
        cond3 = y <= np.sqrt(3) * x + tol

        if not (cond1 and cond2 and cond3):
            raise ValueError(
                f"Point ({x}, {y}) is outside the equilateral triangle (tolerance: {tol})."
            )

def triangle_area(a: np.array, b: np.array, c: np.array) -> float:
    return np.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2

def validate_wrapper(points: Any) -> Tuple[bool, Optional[str]]:
    try:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape != (NUM_POINTS, 2):
            return False, f"Invalid shapes: points = {points.shape}, expected {(NUM_POINTS,2)}"
        
        check_inside_triangle_wtol(points, TOL)
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    points = results[0]
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3) / 2])
    min_triangle_area = min(
        [triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)]
    )
    min_area_normalized = min_triangle_area / triangle_area(a, b, c)
    
    return {
        "combined_score": float(min_area_normalized),
        "min_area_normalized": float(min_area_normalized),
        "benchmark_ratio": float(min_area_normalized / BENCHMARK),
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="heilbronn_triangle11",
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
