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

def generate_symmetric_initial_config():
    """Generate a highly symmetric initial configuration."""
    # Use a known good symmetric pattern for 12 hexagons
    # This configuration has 6-fold rotational symmetry in the arrangement
    config = []

    # Central hexagon
    config.append([0, 0, 0])

    # First ring: 6 hexagons at radius 2
    ring1_radius = 2.0
    for i in range(6):
        angle = i * 60
        x = ring1_radius * np.cos(np.radians(angle))
        y = ring1_radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    # Second ring: 5 hexagons arranged in a pentagonal pattern around the first ring
    # This provides better coverage and avoids the straight-line gaps found in simpler approaches
    ring2_angles = [0, 72, 144, 216, 288]
    ring2_radius = 3.3  # Slightly smaller than previous attempts for better packing
    for angle in ring2_angles:
        x = ring2_radius * np.cos(np.radians(angle))
        y = ring2_radius * np.sin(np.radians(angle))
        config.append([x, y, 0])

    # Add final hexagon to complete 12 total - place it along the negative y-axis
    config.append([0, -ring2_radius - 1.0, 0])

    # Apply small random perturbations to break any potential degenerate cases
    # but maintain overall symmetric structure
    for i in range(len(config)):
        if i > 0:  # Don't perturb the central hexagon
            config[i][0] += random.uniform(-0.05, 0.05)
            config[i][1] += random.uniform(-0.05, 0.05)

    return np.array(config)

def optimize_with_symmetry_guided_search(initial_config, outer_center_x, outer_center_y):
    """Optimize using a symmetry-guided approach that maintains structural relationships."""

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
                      options={'maxiter': 500, 'ftol': 1e-8})

    # Apply symmetry preservation for the refined optimization
    # We slightly adjust the approach to preserve meaningful relationships
    refined_params = result1.x.copy()
    # Add symmetry-breaking noise around the optimized result to escape local minima
    for i in range(0, len(refined_params), 2):
        refined_params[i] += random.uniform(-0.1, 0.1)  # Small perturbation
        refined_params[i+1] += random.uniform(-0.1, 0.1)

    # Stage 2: Refine with tighter bounds and different method
    bounds_stage2 = [(-6, 6), (-6, 6)] * len(initial_config)
    result2 = minimize(objective, refined_params, method='L-BFGS-B', bounds=bounds_stage2,
                      options={'maxiter': 300, 'ftol': 1e-10})

    # Stage 3: Fine-tune with even tighter bounds
    final_params = result2.x.copy()
    bounds_stage3 = [(-5, 5), (-5, 5)] * len(initial_config)
    result3 = minimize(objective, final_params, method='L-BFGS-B', bounds=bounds_stage3,
                      options={'maxiter': 200, 'ftol': 1e-12})

    # Reconstruct optimized configuration
    optimized_config = initial_config.copy()
    idx = 0
    for i in range(len(optimized_config)):
        optimized_config[i][0] = result3.x[idx]
        optimized_config[i][1] = result3.x[idx + 1]
        idx += 2

    return optimized_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate symmetric initial configuration
    initial_config = generate_symmetric_initial_config()

    # Set outer hexagon at center
    outer_center_x, outer_center_y = 0.0, 0.0

    # Multi-stage optimization approach with symmetry awareness
    optimized_config = optimize_with_symmetry_guided_search(initial_config, outer_center_x, outer_center_y)

    # Final verification and refinement
    max_attempts = 10
    for attempt in range(max_attempts):
        validity, inv_radius = evaluate_configuration(optimized_config, outer_center_x, outer_center_y)
        if validity:
            break

        # If not valid, try small random adjustments to positions
        for i in range(len(optimized_config)):
            optimized_config[i][0] += np.random.normal(0, 0.02)
            optimized_config[i][1] += np.random.normal(0, 0.02)

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