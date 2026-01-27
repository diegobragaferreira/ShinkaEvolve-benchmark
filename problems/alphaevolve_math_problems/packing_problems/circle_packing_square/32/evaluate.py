import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARK = 2.937944526205518
NUM_CIRCLES = 32
TOL = 1e-6

def validate_packing_radii(radii: np.ndarray) -> None:
    n = len(radii)
    for i in range(n):
        if radii[i] < 0:
            raise ValueError(f"Circle {i} has negative radius {radii[i]}")
        elif np.isnan(radii[i]):
            raise ValueError(f"Circle {i} has nan radius")

def validate_packing_unit_square_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if (x - r < -tol) or (x + r > 1 + tol) or (y - r < -tol) or (y + r > 1 + tol):
            raise ValueError(
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )

def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i,2]+circles[j,2]}"
                )

def validate_wrapper(circles: Any) -> Tuple[bool, Optional[str]]:
    try:
        if not isinstance(circles, np.ndarray):
            circles = np.array(circles)

        if circles.shape != (NUM_CIRCLES, 3):
            return False, f"Invalid shapes: circles = {circles.shape}, expected {(NUM_CIRCLES,3)}"
        
        if np.isnan(circles).any():
             return False, "nan entry found in answer!"

        validate_packing_radii(circles[:, -1])
        validate_packing_overlap_wtol(circles, TOL)
        validate_packing_unit_square_wtol(circles, TOL)
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    circles = results[0]
    if not isinstance(circles, np.ndarray):
        circles = np.array(circles)

    radii_sum = np.sum(circles[:, -1])
    
    return {
        "combined_score": float(radii_sum),
        "sum_radii": float(radii_sum),
        "benchmark_ratio": float(radii_sum / BENCHMARK),
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="circle_packing32",
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
