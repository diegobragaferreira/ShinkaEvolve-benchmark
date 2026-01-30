import os
import argparse
import numpy as np
import sys
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 2.3658321334167627
NUM_CIRCLES = 21
TOL = 1e-6


def minimum_circumscribing_rectangle(circles: np.ndarray):
    """Returns the width and height of the minimum circumscribing rectangle.

    Args:
    circles: A numpy array of shape (num_circles, 3), where each row is of the
        form (x, y, radius), specifying a circle.

    Returns:
    A tuple (width, height) of the minimum circumscribing rectangle.
    """
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    return max_x - min_x, max_y - min_y


def validate_packing_radii(radii: np.ndarray) -> None:
    n = len(radii)
    for i in range(n):
        if radii[i] < 0:
            raise ValueError(f"Circle {i} has negative radius {radii[i]}")
        elif np.isnan(radii[i]):
            raise ValueError(f"Circle {i} has nan radius")


def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i,2]+circles[j,2]}"
                )


def validate_packing_inside_rect_wtol(circles: np.array, tol: float = 1e-6) -> None:
    width, height = minimum_circumscribing_rectangle(circles)
    if width + height > (2 + tol):
        raise ValueError("Circles are not contained inside a rectangle of perimeter 4.")


def validate_wrapper(circles: Any) -> Tuple[bool, Optional[str]]:
    """Wraps the existing validation logic to return (bool, error_msg)."""
    try:
        if not isinstance(circles, np.ndarray):
            circles = np.array(circles)

        if circles.shape != (NUM_CIRCLES, 3):
            return False, f"Invalid shapes: circles = {circles.shape}, expected {(NUM_CIRCLES,3)}"
        
        if np.isnan(circles).any():
             return False, "nan entry found in answer!"

        validate_packing_radii(circles[:, -1])
        validate_packing_overlap_wtol(circles, TOL)
        validate_packing_inside_rect_wtol(circles, TOL)
        return True, None
    except Exception as e:
        return False, str(e)


def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    """Aggregates results from runs (expecting just 1 run)."""
    if not results:
        return {"combined_score": 0.0}

    # results[0] is the circles array from the single run
    circles = results[0]
    if not isinstance(circles, np.ndarray):
        circles = np.array(circles)

    radii_sum = np.sum(circles[:, -1])
    return {
        "combined_score": float(radii_sum),
        "radii_sum": float(radii_sum),
        "benchmark_ratio": float(radii_sum / BENCHMARK),
    }


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def main(program_path: str, results_dir: str):
    """Runs the evaluation using shinka.core.run_shinka_eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="circle_packing21",
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

