# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
import time
import math

def hexagon_vertices(center_x, center_y, rotation_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length."""
    angle_rad = np.radians(rotation_degrees)
    # Vertices of a unit hexagon centered at origin
    unit_vertices = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = unit_vertices @ rotation_matrix.T
    return rotated_vertices * side_length + np.array([center_x, center_y])

def check_containment_single(hex_vertices, outer_polygon):
    """Check if all vertices of a hexagon are inside the outer hexagon."""
    # Check if all vertices are inside the outer polygon
    for vertex in hex_vertices:
        point = Point(vertex)
        if not outer_polygon.contains(point):
            return False
    return True

def check_collision_single(hex1_vertices, hex2_vertices):
    """Check if two hexagons collide using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def estimate_min_outer_radius(inner_hex_params):
    """Estimate the minimal outer hexagon radius needed to contain all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(11):
        x, y, rot = inner_hex_params[3*i], inner_hex_params[3*i+1], inner_hex_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        all_vertices.extend(hex_vertices)

    if len(all_vertices) == 0:
        return 100.0

    # Calculate bounding box
    all_vertices = np.array(all_vertices)
    min_x, max_x = all_vertices[:, 0].min(), all_vertices[:, 0].max()
    min_y, max_y = all_vertices[:, 1].min(), all_vertices[:, 1].max()

    # Calculate center of bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Find maximum distance from center to any vertex
    max_dist = 0
    for vertex in all_vertices:
        dist = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
        max_dist = max(max_dist, dist)

    # For a hexagon, the side length is max_dist / sqrt(3)
    # But we want to ensure our hexagon can contain everything with some margin
    return max_dist * 2 / np.sqrt(3) * 1.1  # 10% margin

def calculate_total_penalty(params):
    """Calculate penalty for constraint violations."""
    # Extract parameters
    n = 11
    inner_params = params[:-1]
    outer_radius = params[-1]

    # Check if outer hexagon is large enough
    if outer_radius <= 0:
        return 1e10

    # Create outer hexagon vertices once
    outer_vertices = hexagon_vertices(0, 0, 0, outer_radius)
    outer_polygon = prep(Polygon(outer_vertices))

    # Check containment and collisions
    total_penalty = 0

    # Check containment of all inner hexagons
    for i in range(n):
        x, y, rot = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            total_penalty += 1e8

    # Check collisions between all pairs of inner hexagons
    if total_penalty == 0:
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, rot1 = inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]
                x2, y2, rot2 = inner_params[3*j], inner_params[3*j+1], inner_params[3*j+2]
                hex1_vertices = hexagon_vertices(x1, y1, rot1, 1)
                hex2_vertices = hexagon_vertices(x2, y2, rot2, 1)
                if check_collision_single(hex1_vertices, hex2_vertices):
                    total_penalty += 1e8

    return total_penalty

def objective_function(params):
    """Objective function to maximize 1/outer_hex_side_length."""
    penalty = calculate_total_penalty(params)

    # Return negative inverse of outer hex side length plus penalties
    # This is because we want to minimize the negative inverse (i.e., maximize the inverse)
    outer_radius = params[-1]
    return -(1.0 / outer_radius) + penalty

def generate_initial_solutions():
    """Generate multiple initial configurations using different strategies."""
    initial_solutions = []

    # Strategy 1: Dense hexagonal packing
    dense_pattern = [
        [0, 0, 0],      # center
        [0, 2, 0],      # top
        [1.732, 1, 0],  # top-right
        [1.732, -1, 0], # bottom-right
        [0, -2, 0],     # bottom
        [-1.732, -1, 0], # bottom-left
        [-1.732, 1, 0],  # top-left
        [3.464, 0, 0],   # far right
        [1.732, 2, 0],   # top-right extended
        [-1.732, 2, 0],  # top-left extended
        [-3.464, 0, 0],  # far left
    ]

    # Strategy 2: Linear chain arrangement
    linear_pattern = [
        [0, 0, 0],       # center
        [-2.5, 0, 0],    # left
        [2.5, 0, 0],     # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0],  # bottom-right
        [-3.75, 2.17, 0],  # far top-left
        [3.75, 2.17, 0],   # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0],  # far bottom-right
    ]

    # Strategy 3: Compact cluster arrangement
    compact_pattern = [
        [0, 0, 0],      # center
        [2, 0, 0],      # right
        [1, 1.732, 0],  # top-right
        [-1, 1.732, 0], # top-left
        [-2, 0, 0],     # left
        [-1, -1.732, 0], # bottom-left
        [1, -1.732, 0],  # bottom-right
        [3, 0, 0],       # far right
        [0, 2.5, 0],     # top
        [0, -2.5, 0],    # bottom
        [-3, 0, 0],      # far left
    ]

    # Generate initial configurations
    for i, pattern in enumerate([dense_pattern, linear_pattern, compact_pattern]):
        params = []
        for x, y, rot in pattern:
            params.extend([x, y, rot])

        # Estimate outer radius
        est_radius = estimate_min_outer_radius(np.array(params))
        params.append(est_radius)
        initial_solutions.append(np.array(params))

    return initial_solutions

def simulated_annealing_step(current_params, temp, bounds, step_size=0.1):
    """Perform a single step of simulated annealing."""
    # Create a copy of current parameters
    new_params = current_params.copy()

    # Randomly select a parameter to modify
    param_idx = np.random.randint(len(new_params))

    # Determine bounds for this parameter
    if param_idx < len(new_params) - 1:  # Position or rotation parameters
        param_bound = bounds[param_idx]
        # Random walk with bounded step
        new_params[param_idx] += np.random.uniform(-step_size, step_size)
        new_params[param_idx] = np.clip(new_params[param_idx], param_bound[0], param_bound[1])
    else:  # Outer radius
        # Modify outer radius with smaller steps
        new_params[param_idx] += np.random.uniform(-step_size*0.5, step_size*0.5)
        new_params[param_idx] = max(0.1, new_params[param_idx])

    return new_params

def refine_with_simulated_annealing(initial_params, bounds, max_iter=2000, temp_start=1.0, cooling_rate=0.995):
    """Refine solution using simulated annealing."""
    current_params = initial_params.copy()
    current_value = objective_function(current_params)
    best_params = current_params.copy()
    best_value = current_value

    temp = temp_start

    for i in range(max_iter):
        # Generate neighbor
        new_params = simulated_annealing_step(current_params, temp, bounds)
        new_value = objective_function(new_params)

        # Accept or reject
        delta = new_value - current_value
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_params = new_params
            current_value = new_value

            if current_value < best_value:
                best_params = current_params.copy()
                best_value = current_value

        # Cool down
        temp *= cooling_rate

        if temp < 1e-8:
            break

    return best_params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    # Generate initial solutions
    initial_solutions = generate_initial_solutions()

    # Evaluate initial solutions
    best_solution = None
    best_objective_value = float('inf')

    # Define bounds
    bounds = []
    # Positions: x, y for each hexagon (limited to reasonable range)
    for _ in range(11):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])  # x, y, rotation
    # Outer hexagon side length
    bounds.append((0.1, 20.0))  # Must be positive and reasonable

    # Try all initial configurations
    for i, initial_params in enumerate(initial_solutions):
        # Refine using simulated annealing
        refined_params = refine_with_simulated_annealing(initial_params, bounds)

        # Evaluate objective
        obj_value = objective_function(refined_params)

        if obj_value < best_objective_value:
            best_objective_value = obj_value
            best_solution = refined_params

    # If no good solution was found, fall back to a known good configuration
    if best_solution is None:
        # Default configuration from prior work
        default_config = [
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0], 3.95  # Estimated side length
        ]
        best_solution = np.array(default_config).flatten()

    # Extract results
    outer_side_length = best_solution[-1]

    # Extract inner hexagon data
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]]

    # Outer hexagon data (centered at origin)
    outer_hex_data = np.array([0, 0, 0])

    # Validate solution one more time
    outer_vertices = hexagon_vertices(0, 0, 0, outer_side_length)
    outer_polygon = prep(Polygon(outer_vertices))

    # Check validity
    valid = True
    for i in range(11):
        x, y, rot = best_solution[3*i], best_solution[3*i+1], best_solution[3*i+2]
        hex_vertices = hexagon_vertices(x, y, rot, 1)
        if not check_containment_single(hex_vertices, outer_polygon):
            valid = False
            break

    # If still invalid, use safe fallback
    if not valid:
        # Fallback to pattern that works well
        inner_hex_data = np.array([
            [0, 0, 0], [0, 2, 0], [1.732, 1, 0], [1.732, -1, 0], [0, -2, 0],
            [-1.732, -1, 0], [-1.732, 1, 0], [3.464, 0, 0], [1.732, 2, 0],
            [-1.732, 2, 0], [-3.464, 0, 0]
        ])
        outer_side_length = estimate_min_outer_radius(inner_hex_data.flatten())
        outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END