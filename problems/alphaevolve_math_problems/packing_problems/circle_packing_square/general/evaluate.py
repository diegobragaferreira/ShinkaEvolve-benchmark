import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

BENCHMARKS = {
    1: 0.5,
    2: 0.586,
    3: 0.796,
    4: 1.007,
    5: 1.104,
    6: 1.203,
    7: 1.307,
    8: 1.424,
    9: 1.525,
    10: 1.592,
    11: 1.681,
    12: 1.766,
    13: 1.830,
    14: 1.906,
    15: 1.981,
    16: 2.054,
    17: 2.112,
    18: 2.179,
    19: 2.237,
    20: 2.302,
    21: 2.363,
    22: 2.421,
    23: 2.479,
    24: 2.531,
    25: 2.588,
    26: 2.636,
    27: 2.686,
    28: 2.738,
    29: 2.791,
    30: 2.843,
    31: 2.890,
    32: 2.938,
}

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

def validate_wrapper(circles: Any, num_circles: int) -> Tuple[bool, Optional[str]]:
    try:
        if not isinstance(circles, np.ndarray):
            circles = np.array(circles)

        if circles.shape != (num_circles, 3):
            return False, f"Invalid shapes: circles = {circles.shape}, expected {(num_circles,3)}"
        
        if np.isnan(circles).any():
             return False, "nan entry found in answer!"

        validate_packing_radii(circles[:, -1])
        validate_packing_overlap_wtol(circles, TOL)
        validate_packing_unit_square_wtol(circles, TOL)
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any], num_circles_list: List[int]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    total_benchmark_ratio = 0.0
    for i, res in enumerate(results):
        circles = res
        num_circles = num_circles_list[i]
        
        if not isinstance(circles, np.ndarray):
            circles = np.array(circles)
            
        radii_sum = np.sum(circles[:, -1])
        total_benchmark_ratio += radii_sum / BENCHMARKS[num_circles]
    
    avg_benchmark_ratio = total_benchmark_ratio / len(results)
    
    return {
        "combined_score": float(avg_benchmark_ratio),
        "avg_benchmark_ratio": float(avg_benchmark_ratio),
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    # We are using run_index to map to N (num_circles)
    # run_shinka_eval will call this num_runs times. 
    # But here we have multiple benchmarks.
    # The standard run_shinka_eval executes the experiment function num_runs times.
    # We need to adapt it. 
    # Actually, run_shinka_eval is designed for a single task repeated multiple times or with different args.
    # Here we have 32 different tasks.
    # We can use get_experiment_kwargs to pass 'num_circles'.
    # We need num_runs = 32.
    # N will be run_index + 1
    return {"num_circles": run_index + 1}

def validate_wrapper_dynamic(result: Any, kwargs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    num_circles = kwargs['num_circles']
    return validate_wrapper(result, num_circles)

def aggregate_metrics_dynamic(results: List[Any], kwargs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_circles_list = [k['num_circles'] for k in kwargs_list]
    return aggregate_metrics(results, num_circles_list)

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    # We run 32 runs, one for each N from 1 to 32
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="circle_packing_square",
        num_runs=32,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_wrapper_dynamic,
        aggregate_metrics_fn=aggregate_metrics_dynamic,
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
