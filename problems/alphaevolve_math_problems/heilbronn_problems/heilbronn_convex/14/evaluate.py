import argparse
import numpy as np
import itertools
from scipy.spatial import ConvexHull
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 0.027835571458482138
NUM_POINTS = 14

def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculates the area of a triangle given its vertices p1, p2, and p3."""
    return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2

def validate_wrapper(points: Any) -> Tuple[bool, Optional[str]]:
    try:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape != (NUM_POINTS, 2):
            return False, f"Invalid shapes: points = {points.shape}, expected {(NUM_POINTS,2)}"
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    points = results[0]
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    min_triangle_area = min(
        [triangle_area(p1, p2, p3) for p1, p2, p3 in itertools.combinations(points, 3)]
    )
    convex_hull_area = ConvexHull(points).volume
    min_area_normalized = min_triangle_area / convex_hull_area
    
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
        experiment_fn_name="heilbronn_convex14",
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
