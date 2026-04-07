# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
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
    """Check if two hexagons overlap using Shapely with buffer for precision."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    # Use small buffer to handle floating point precision issues
    return poly1.buffer(1e-10).intersects(poly2.buffer(1e-10))

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

def generate_symmetric_initial_configs():
    """Generate multiple symmetric initial configurations to increase chances of finding better solutions."""
    configs = []

    # Configuration 1: Standard hexagonal packing pattern
    config1 = []
    config1.append([0, 0, 0])  # center

    # First ring - 6 hexagons
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config1.append([x, y, 0])

    # Second ring - 5 hexagons
    angles = [0, 72, 144, 216, 288]
    radius = 3.4
    for i, angle in enumerate(angles):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config1.append([x, y, 0])

    config1.append([0, -radius - 1.0, 0])
    configs.append(np.array(config1[:12]))

    # Configuration 2: Perturbed version with slight asymmetry to escape local minima
    config2 = config1.copy()
    for i in range(len(config2)):
        if i != 0:  # Don't perturb central hexagon
            config2[i][0] += random.uniform(-0.2, 0.2)
            config2[i][1] += random.uniform(-0.2, 0.2)
    configs.append(np.array(config2[:12]))

    # Configuration 3: Different ring arrangement
    config3 = []
    config3.append([0, 0, 0])

    # First ring - 6 hexagons arranged in a tighter pattern
    for i in range(6):
        angle = i * 60
        radius = 1.9  # Slightly smaller radius
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config3.append([x, y, 0])

    # Second ring - 6 hexagons in different positions
    angles = [30, 90, 150, 210, 270, 330]  # Rotated positions
    radius = 3.3
    for i, angle in enumerate(angles):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config3.append([x, y, 0])

    configs.append(np.array(config3[:12]))

    # Configuration 4: Another variation with different spacing
    config4 = []
    config4.append([0, 0, 0])

    # First ring - 6 hexagons with varied spacing
    for i in range(6):
        angle = i * 60
        radius = 2.0 + random.uniform(-0.1, 0.1)  # Slight variation
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config4.append([x, y, 0])

    # Second ring - 5 hexagons
    angles = [0, 70, 140, 210, 280]
    radius = 3.5
    for i, angle in enumerate(angles):
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        config4.append([x, y, 0])

    config4.append([0, -radius - 1.0, 0])
    configs.append(np.array(config4[:12]))

    return configs

def optimize_positions_symmetric(initial_config, outer_center_x, outer_center_y):
    """Optimize positions using staged optimization with symmetry preservation."""

    def objective(params):
        # Reconstruct configuration from flattened parameters
        config = initial_config.copy()
        # Update positions only (leave angles as they are)
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

    # Stage 1: Coarse optimization with relaxed bounds
    bounds_stage1 = [(-8, 8), (-8, 8)] * len(initial_config)
    result1 = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds_stage1,
                      options={'maxiter': 200, 'ftol': 1e-8})

    # Stage 2: Refine with tighter bounds and different method
    refined_params = result1.x.copy()
    bounds_stage2 = [(-6, 6), (-6, 6)] * len(initial_config)
    result2 = minimize(objective, refined_params, method='L-BFGS-B', bounds=bounds_stage2,
                      options={'maxiter': 150, 'ftol': 1e-10})

    # Stage 3: Fine-tune with even tighter bounds
    final_params = result2.x.copy()
    bounds_stage3 = [(-5, 5), (-5, 5)] * len(initial_config)
    result3 = minimize(objective, final_params, method='L-BFGS-B', bounds=bounds_stage3,
                      options={'maxiter': 100, 'ftol': 1e-12})

    # Reconstruct optimized configuration
    optimized_config = initial_config.copy()
    idx = 0
    for i in range(len(optimized_config)):
        optimized_config[i][0] = result3.x[idx]
        optimized_config[i][1] = result3.x[idx + 1]
        idx += 2

    return optimized_config

def mutate_symmetrically(config, mut_pb=0.3, mut_strength=0.2):
    """Mutate configuration while preserving some symmetry properties."""
    mutated = config.copy()

    # Mutate central hexagon
    if random.random() < mut_pb:
        mutated[0, 0] += random.uniform(-mut_strength, mut_strength)
        mutated[0, 1] += random.uniform(-mut_strength, mut_strength)

    # Mutate first ring (indices 1-6) keeping some rotational symmetry
    if random.random() < mut_pb:
        offset_x = random.uniform(-mut_strength, mut_strength)
        offset_y = random.uniform(-mut_strength, mut_strength)
        for i in range(1, 7):
            mutated[i, 0] += offset_x
            mutated[i, 1] += offset_y

    # Mutate second ring (indices 7-11) keeping rotational symmetry
    if random.random() < mut_pb:
        offset_x = random.uniform(-mut_strength, mut_strength)
        offset_y = random.uniform(-mut_strength, mut_strength)
        for i in range(7, 12):
            mutated[i, 0] += offset_x
            mutated[i, 1] += offset_y

    return mutated

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate multiple symmetric initial configurations
    initial_configs = generate_symmetric_initial_configs()

    best_inv_radius = 0
    best_config = None

    # Try multiple initial configurations
    for i, initial_config in enumerate(initial_configs):
        # Multi-stage optimization approach
        optimized_config = optimize_positions_symmetric(initial_config, 0.0, 0.0)

        # Final verification and refinement
        max_attempts = 5
        for attempt in range(max_attempts):
            validity, inv_radius = evaluate_configuration(optimized_config, 0.0, 0.0)
            if validity:
                break

            # If not valid, try small random adjustments to positions
            for j in range(len(optimized_config)):
                optimized_config[j][0] += np.random.normal(0, 0.02)
                optimized_config[j][1] += np.random.normal(0, 0.02)

        if validity and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = optimized_config.copy()

    # If no valid configuration found, use a fallback
    if best_config is None:
        # Fallback to simple configuration
        best_config = np.array([
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
        best_inv_radius = 1.0 / 8.0  # Approximate value for fallback

    # Compute final outer hexagon radius
    outer_radius = 1.0 / best_inv_radius if best_inv_radius > 0 else 10.0

    inner_hex_data = np.array(best_config)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = outer_radius * 2  # approximate

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END