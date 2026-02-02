import argparse
import numpy as np
import scipy as sp
import scipy.spatial
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 1 / 4.165849767
NUM_POINTS = 14
DIMENSION = 3

def validate_wrapper(points: Any) -> Tuple[bool, Optional[str]]:
    try:
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape != (NUM_POINTS, DIMENSION):
            return False, f"Invalid shapes: points = {points.shape}, expected {(NUM_POINTS,DIMENSION)}"
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    points = results[0]
    if not isinstance(points, np.ndarray):
        points = np.array(points)

    pairwise_distances = sp.spatial.distance.pdist(points)
    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)

    min_max_ratio = (min_distance / max_distance) ** 2 if max_distance > 0 else 0
    
    return {
        "combined_score": float(min_max_ratio),
        "min_max_ratio": float(min_max_ratio),
        "benchmark_ratio": float(min_max_ratio / BENCHMARK),
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="min_max_dist_dim3_14",
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
