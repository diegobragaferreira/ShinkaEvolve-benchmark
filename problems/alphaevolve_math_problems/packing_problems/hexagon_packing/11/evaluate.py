import argparse
import numpy as np
import math
from typing import Tuple, Optional, List, Dict, Any
from shinka.core import run_shinka_eval

N_HEX = 11
BENCHMARK = 1 / 3.930092

def hexagon_vertices(
    center_x: float,
    center_y: float,
    side_length: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    vertices = []
    angle_radians = math.radians(angle_degrees)
    for i in range(6):
        angle = angle_radians + 2 * math.pi * i / 6
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        vertices.append((x, y))
    return vertices

def normalize_vector(v: tuple[float, float]) -> tuple[float, float]:
    magnitude = math.sqrt(v[0] ** 2 + v[1] ** 2)
    return (v[0] / magnitude, v[1] / magnitude) if magnitude != 0 else (0.0, 0.0)

def get_normals(vertices: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normals = []
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = normalize_vector((-edge[1], edge[0]))
        normals.append(normal)
    return normals

def project_polygon(
    vertices: list[tuple[float, float]],
    axis: tuple[float, float],
) -> tuple[float, float]:
    min_proj = float("inf")
    max_proj = float("-inf")
    for vertex in vertices:
        projection = vertex[0] * axis[0] + vertex[1] * axis[1]
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj

def overlap_1d(min1: float, max1: float, min2: float, max2: float, tol: float = 1e-6) -> bool:
    return max1 >= min2 - tol and max2 >= min1 - tol

def polygons_intersect(
    vertices1: list[tuple[float, float]],
    vertices2: list[tuple[float, float]],
    tol: float = 1e-6,
) -> bool:
    normals1 = get_normals(vertices1)
    normals2 = get_normals(vertices2)
    axes = normals1 + normals2
    for axis in axes:
        min1, max1 = project_polygon(vertices1, axis)
        min2, max2 = project_polygon(vertices2, axis)
        if not overlap_1d(min1, max1, min2, max2, tol):
            return False
    return True

def hexagons_are_disjoint(
    hex1_params: tuple[float, float, float, float],
    hex2_params: tuple[float, float, float, float],
    tol: float = 1e-6,
) -> bool:
    hex1_vertices = hexagon_vertices(*hex1_params)
    hex2_vertices = hexagon_vertices(*hex2_params)
    return not polygons_intersect(hex1_vertices, hex2_vertices, tol)

def is_inside_hexagon(
    point: tuple[float, float],
    hex_params: tuple[float, float, float, float],
    tol: float = 1e-6,
) -> bool:
    hex_vertices = hexagon_vertices(*hex_params)
    for i in range(len(hex_vertices)):
        p1 = hex_vertices[i]
        p2 = hex_vertices[(i + 1) % len(hex_vertices)]
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        point_vector = (point[0] - p1[0], point[1] - p1[1])
        cross_product = edge_vector[0] * point_vector[1] - edge_vector[1] * point_vector[0]
        if cross_product < -tol:
            return False
    return True

def all_hexagons_contained(
    inner_hex_params_list: list[tuple[float, float, float, float]],
    outer_hex_params: tuple[float, float, float, float],
    tol: float = 1e-6,
) -> bool:
    for inner_hex_params in inner_hex_params_list:
        inner_hex_vertices = hexagon_vertices(*inner_hex_params)
        for vertex in inner_hex_vertices:
            if not is_inside_hexagon(vertex, outer_hex_params, tol):
                return False
    return True

def verify_construction(
    inner_hex_data: tuple[float, float, float],
    outer_hex_center: tuple[float, float],
    outer_hex_side_length: float,
    outer_hex_angle_degrees: float,
    tol: float = 1e-6,
):
    inner_hex_params_list = [
        (x, y, 1, angle) for x, y, angle in inner_hex_data
    ]
    outer_hex_params = (
        outer_hex_center[0],
        outer_hex_center[1],
        outer_hex_side_length,
        outer_hex_angle_degrees,
    )
    for i in range(len(inner_hex_params_list)):
        for j in range(i + 1, len(inner_hex_params_list)):
            if not hexagons_are_disjoint(
                inner_hex_params_list[i], inner_hex_params_list[j], tol
            ):
                raise AssertionError(f"Hexagons {i+1} and {j+1} intersect!")
    if not all_hexagons_contained(inner_hex_params_list, outer_hex_params, tol):
        raise AssertionError("Not all inner hexagons are contained in the outer hexagon!")

def validate_wrapper(result: Any) -> Tuple[bool, Optional[str]]:
    try:
        inner_hex_data, outer_hex_data, outer_hex_side_length = result
        
        if not isinstance(inner_hex_data, np.ndarray):
            inner_hex_data = np.array(inner_hex_data)
        if not isinstance(outer_hex_data, np.ndarray):
            outer_hex_data = np.array(outer_hex_data)

        if outer_hex_side_length <= 0:
             return False, "Outer hex side length must be positive!"
        
        if np.isnan(inner_hex_data).any():
             return False, "nan entry found in inner_hex_data!"
        if np.isnan(outer_hex_data).any():
             return False, "nan entry found in outer_hex_data!"

        if inner_hex_data.shape != (N_HEX, 3):
            return False, f"Invalid shapes: inner_hex_data = {inner_hex_data.shape}, expected {(N_HEX,3)}"

        if outer_hex_data.shape != (3,):
            return False, f"Invalid shapes: outer_hex_data = {outer_hex_data.shape}, expected {(3,)}"

        outer_hex_center = outer_hex_data[:2]
        outer_hex_angle_degrees = outer_hex_data[-1]
        verify_construction(
            inner_hex_data, outer_hex_center, outer_hex_side_length, outer_hex_angle_degrees
        )
        return True, None
    except Exception as e:
        return False, str(e)

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0}
    
    inner_hex_data, outer_hex_data, outer_hex_side_length = results[0]
    inv_outer_hex_side_length = float(1 / outer_hex_side_length)
    
    return {
        "combined_score": float(inv_outer_hex_side_length),
        "inv_outer_hex_side_length": float(inv_outer_hex_side_length),
        "benchmark_ratio": float(inv_outer_hex_side_length / BENCHMARK),
    }

def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="hexagon_packing_11",
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
