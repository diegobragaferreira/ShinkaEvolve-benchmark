# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time
import random

def hexagon_vertices(center_x, center_y, size=1, angle_deg=0):
    """Generate vertices of a regular hexagon given center, size, and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        angle = angle_rad + i * np.pi / 3
        x = center_x + size * np.cos(angle)
        y = center_y + size * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y):
    """Compute minimum outer hexagon radius that contains all inner hexagons."""
    max_distance = 0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        distance = np.sqrt((cx - outer_center_x)**2 + (cy - outer_center_y)**2)
        max_distance = max(max_distance, distance + 1)  # Add radius of unit hexagon
    return max_distance

def evaluate_configuration(inner_hex_data, outer_center_x, outer_center_y):
    """Evaluate current configuration: returns (validity, inv_radius)."""
    # Check for overlaps
    for i in range(len(inner_hex_data)):
        hex1_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for j in range(i+1, len(inner_hex_data)):
            hex2_vertices = hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], 1, inner_hex_data[j][2])
            if check_overlap(hex1_vertices, hex2_vertices):
                return False, 0

    # Check containment
    outer_radius = compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y)
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_radius, 0)
    outer_polygon = Polygon(outer_vertices)

    for i in range(len(inner_hex_data)):
        hex_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, 0

    # Return inverse of outer radius
    return True, 1.0 / outer_radius

def generate_initial_config():
    """Generate an initial configuration based on hexagonal tiling principles."""
    # Start with a better initial arrangement than simple grid
    # Inspired by optimal hexagon packings but adjusted for 12 hexagons
    config = []

    # Central hexagon
    config.append([0, 0, 0])

    # First ring - 6 hexagons
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    # Second ring - 5 hexagons arranged to optimize space
    # Place them in a pattern that's close to optimal
    angles = [30, 90, 150, 210, 270]
    radius = 3.464  # Approximately sqrt(12)
    for i, angle in enumerate(angles):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    # Add one more hexagon to make 12 total
    config.append([0, -3.5, 0])

    return np.array(config)

def adaptive_mutation_strength(iteration, max_iterations, initial_strength=0.5):
    """Adaptively reduce mutation strength as optimization progresses."""
    # Exponential decay schedule for mutation strength
    decay_factor = 0.95 ** (iteration / max_iterations * 10)
    return initial_strength * decay_factor

def multi_start_optimization(initial_configs, outer_center_x, outer_center_y, max_time_seconds=170):
    """Run optimization from multiple initial configurations to avoid local minima."""
    best_config = None
    best_inv_radius = 0
    start_time = time.time()

    # Try multiple configurations
    for i, initial_config in enumerate(initial_configs):
        if (time.time() - start_time) > max_time_seconds:
            break

        print(f"Starting optimization from configuration {i+1}")

        # Run optimization with adaptive approach
        optimized_config = optimize_positions_adaptive(initial_config, outer_center_x, outer_center_y,
                                                      max_time_seconds - (time.time() - start_time))

        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = optimized_config.copy()

        # Add small random perturbation for next start
        for j in range(len(optimized_config)):
            optimized_config[j][0] += np.random.normal(0, 0.02)
            optimized_config[j][1] += np.random.normal(0, 0.02)

    # If we didn't find a better solution, return the original
    if best_config is None:
        return initial_configs[0]

    return best_config

def optimize_positions_adaptive(initial_config, outer_center_x, outer_center_y, remaining_time=170):
    """Optimize positions using adaptive approach with early termination."""
    start_time = time.time()

    def objective(params):
        # Reconstruct configuration from flattened parameters
        config = initial_config.copy()
        # Update positions only (leave angles as they are for now)
        idx = 0
        for i in range(len(config)):
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2

        validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
        if not validity:
            return 1e10  # Large penalty for invalid configurations
        return -inv_radius  # Negative because we want to maximize

    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(len(initial_config)):
        initial_params.extend([initial_config[i][0], initial_config[i][1]])

    # Multi-stage optimization with adaptive parameters
    stages = [
        {'bounds': [(-8, 8), (-8, 8)] * len(initial_config), 'maxiter': 300, 'ftol': 1e-8},
        {'bounds': [(-6, 6), (-6, 6)] * len(initial_config), 'maxiter': 200, 'ftol': 1e-10},
        {'bounds': [(-5, 5), (-5, 5)] * len(initial_config), 'maxiter': 100, 'ftol': 1e-12}
    ]

    current_params = initial_params.copy()

    for stage_idx, stage in enumerate(stages):
        if (time.time() - start_time) > remaining_time * 0.9:
            break

        try:
            result = minimize(objective, current_params, method='L-BFGS-B',
                            bounds=stage['bounds'], options={'maxiter': stage['maxiter'], 'ftol': stage['ftol']})
            current_params = result.x.copy()
        except:
            # If optimization fails, continue with what we have
            break

    # Reconstruct optimized configuration
    optimized_config = initial_config.copy()
    idx = 0
    for i in range(len(optimized_config)):
        optimized_config[i][0] = current_params[idx]
        optimized_config[i][1] = current_params[idx + 1]
        idx += 2

    return optimized_config

def generate_multiple_initial_configs():
    """Generate several different initial configurations for multi-start approach."""
    configs = []

    # Configuration 1: Base symmetric arrangement
    config1 = generate_initial_config()
    configs.append(config1)

    # Configuration 2: Perturbed version of the base
    config2 = config1.copy()
    for i in range(len(config2)):
        if i > 0:  # Skip central hexagon
            config2[i][0] += np.random.normal(0, 0.2)
            config2[i][1] += np.random.normal(0, 0.2)
    configs.append(config2)

    # Configuration 3: Another variation with different ring radii
    config3 = []
    config3.append([0, 0, 0])
    for i in range(6):
        angle = i * 60
        radius = 1.9
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config3.append([x, y, 0])

    angles = [0, 72, 144, 216, 288]
    radius = 3.5
    for i, angle in enumerate(angles):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config3.append([x, y, 0])

    config3.append([0, -radius - 1.0, 0])
    configs.append(np.array(config3))

    return configs

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate multiple initial configurations
    initial_configs = generate_multiple_initial_configs()

    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0

    # Multi-start optimization approach
    optimized_config = multi_start_optimization(initial_configs, outer_center_x, outer_center_y)

    # Final verification and refinement
    max_attempts = 5
    for attempt in range(max_attempts):
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity:
            break

        # If not valid, try small random adjustments to positions
        for i in range(len(optimized_config)):
            optimized_config[i][0] += np.random.normal(0, 0.03)
            optimized_config[i][1] += np.random.normal(0, 0.03)

    # Compute final outer hexagon radius
    outer_radius = 1.0 / inv_radius if inv_radius > 0 else 10.0

    # Ensure that we have exactly 12 hexagons
    inner_hex_data = np.array(optimized_config)
    if len(inner_hex_data) != 12:
        # Fallback to simple configuration if needed (shouldn't happen)
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        outer_radius = 8.0

    outer_hex_data = np.array([outer_center_x, outer_center_y, 0])
    outer_hex_side_length = outer_radius * 2  # approximate

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END