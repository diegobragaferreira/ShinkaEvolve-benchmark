# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time
import random
from itertools import combinations

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
    # Early check for overlaps
    for i, j in combinations(range(len(inner_hex_data)), 2):
        hex1_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        hex2_vertices = hexagon_vertices(inner_hex_data[j][0], inner_hex_data[j][1], 1, inner_hex_data[j][2])
        if check_overlap(hex1_vertices, hex2_vertices):
            return False, 0

    # Check containment
    outer_radius = compute_outer_hex_radius(inner_hex_data, outer_center_x, outer_center_y)
    outer_vertices = hexagon_vertices(outer_center_x, outer_center_y, outer_radius, 0)
    outer_polygon = Polygon(outer_vertices)

    # Check all vertices of all hexagons
    for i in range(len(inner_hex_data)):
        hex_vertices = hexagon_vertices(inner_hex_data[i][0], inner_hex_data[i][1], 1, inner_hex_data[i][2])
        for vertex in hex_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, 0

    # Return inverse of outer radius
    return True, 1.0 / outer_radius

def generate_better_initial_config():
    """Generate an improved initial configuration with better packing density."""
    # Start with a better arrangement based on known dense packings
    config = []

    # Central hexagon
    config.append([0, 0, 0])

    # First ring - 6 hexagons arranged at 60 degree intervals
    ring1_radius = 2.0
    for i in range(6):
        angle = i * 60
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    # Second ring - 5 hexagons arranged in a pentagonal pattern
    # Using 72-degree increments for better uniformity
    ring2_angles = [0, 72, 144, 216, 288]
    ring2_radius = 3.5  # Slightly larger to allow for better packing
    for angle in ring2_angles:
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    # Add final hexagon to complete 12 total
    config.append([0, -ring2_radius - 1.0, 0])

    # Add small random perturbations to break symmetries that might cause issues
    for i in range(len(config)):
        if i > 0:  # Don't perturb the central hexagon
            config[i][0] += random.uniform(-0.02, 0.02)
            config[i][1] += random.uniform(-0.02, 0.02)

    return np.array(config)

def optimize_with_multi_start(initial_config, outer_center_x, outer_center_y, max_time_seconds=170):
    """Multi-start optimization approach to avoid local minima."""
    start_time = time.time()

    best_config = initial_config.copy()
    best_inv_radius = 0

    # Try multiple different starting configurations
    for start_iter in range(5):
        if (time.time() - start_time) > max_time_seconds * 0.95:
            break

        # Create slightly different initial configuration
        current_config = initial_config.copy()
        if start_iter > 0:
            # Apply random perturbations to create diversity
            for i in range(len(current_config)):
                if i > 0:  # Don't perturb the central hexagon
                    current_config[i][0] += random.uniform(-0.05, 0.05)
                    current_config[i][1] += random.uniform(-0.05, 0.05)

        # Stage 1: Coarse optimization
        optimized_config = stage_optimization(current_config, outer_center_x, outer_center_y,
                                            bounds=[(-8, 8), (-8, 8)] * len(current_config),
                                            maxiter=300, ftol=1e-8)

        # Stage 2: Medium refinement
        optimized_config = stage_optimization(optimized_config, outer_center_x, outer_center_y,
                                            bounds=[(-6, 6), (-6, 6)] * len(optimized_config),
                                            maxiter=200, ftol=1e-10)

        # Stage 3: Fine tuning
        optimized_config = stage_optimization(optimized_config, outer_center_x, outer_center_y,
                                            bounds=[(-5, 5), (-5, 5)] * len(optimized_config),
                                            maxiter=150, ftol=1e-12)

        # Validate and compare
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity and inv_radius > best_inv_radius:
            best_inv_radius = inv_radius
            best_config = optimized_config.copy()

    return best_config

def stage_optimization(initial_config, outer_center_x, outer_center_y, bounds, maxiter, ftol):
    """Perform single stage optimization."""
    def objective(params):
        config = initial_config.copy()
        idx = 0
        for i in range(len(config)):
            config[i][0] = params[idx]
            config[i][1] = params[idx + 1]
            idx += 2

        validity, inv_radius = evaluate_configuration(config, outer_center_x, outer_center_y)
        if not validity:
            return 1e10
        return -inv_radius  # Negative because we want to maximize

    initial_params = []
    for i in range(len(initial_config)):
        initial_params.extend([initial_config[i][0], initial_config[i][1]])

    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': maxiter, 'ftol': ftol})
        optimized_config = initial_config.copy()
        idx = 0
        for i in range(len(optimized_config)):
            optimized_config[i][0] = result.x[idx]
            optimized_config[i][1] = result.x[idx + 1]
            idx += 2
        return optimized_config
    except:
        return initial_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate improved initial configuration
    initial_config = generate_better_initial_config()

    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0

    # Multi-start optimization approach
    optimized_config = optimize_with_multi_start(initial_config, outer_center_x, outer_center_y)

    # Final verification and refinement
    max_attempts = 10
    for attempt in range(max_attempts):
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity:
            break

        # If not valid, try small adjustments to positions
        for i in range(len(optimized_config)):
            optimized_config[i][0] += np.random.normal(0, 0.01)
            optimized_config[i][1] += np.random.normal(0, 0.01)

    # Compute final outer hexagon radius
    outer_radius = 1.0 / inv_radius if inv_radius > 0 else 10.0

    # Ensure that we have exactly 12 hexagons
    inner_hex_data = np.array(optimized_config)
    if len(inner_hex_data) != 12:
        # Fallback to simple configuration if needed
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