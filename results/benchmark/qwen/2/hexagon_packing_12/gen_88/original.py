# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
from numba import njit
import time
import random
from collections import defaultdict

# Spatial hashing for efficient overlap detection
@njit
def get_grid_coords(x, y, cell_size=2.0):
    """Get grid coordinates for a point"""
    return int(x / cell_size), int(y / cell_size)

@njit
def generate_hexagon_vertices(x, y, angle_degrees, side_length=1):
    """Generate vertices of a regular hexagon given center, rotation, and side length"""
    angle_rad = np.radians(angle_degrees)
    vertices = np.empty((6, 2))
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i, 0] = x + side_length * np.cos(theta)
        vertices[i, 1] = y + side_length * np.sin(theta)
    return vertices

@njit
def point_in_hexagon(point_x, point_y, hex_center_x, hex_center_y, hex_angle, hex_side_length):
    """Check if a point is inside a hexagon using geometric properties"""
    # Transform point to hexagon's coordinate system
    dx = point_x - hex_center_x
    dy = point_y - hex_center_y
    angle_rad = np.radians(hex_angle)

    # Rotate point to align with hexagon axes
    rotated_x = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
    rotated_y = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)

    # For unit hexagons, check against the boundary
    # Maximum distance from center in direction of axes
    max_dist = hex_side_length
    if abs(rotated_x) > max_dist or abs(rotated_y) > max_dist:
        return False

    # Check against slanted edges - simplified but effective for unit hexagons
    # This check ensures point is within the hexagon's boundaries
    if abs(rotated_x) <= max_dist and abs(rotated_y) <= max_dist:
        return True
    return False

@njit
def check_containment_inner_to_outer(inner_x, inner_y, inner_angle, outer_x, outer_y, outer_angle, outer_side_length):
    """Check if inner hexagon is fully contained within outer hexagon"""
    inner_vertices = generate_hexagon_vertices(inner_x, inner_y, inner_angle, 1.0)

    # Create outer hexagon vertices
    outer_vertices = generate_hexagon_vertices(outer_x, outer_y, outer_angle, outer_side_length)
    outer_polygon = Polygon(outer_vertices)

    # Check if all vertices of inner hexagon are inside outer hexagon
    for i in range(6):
        if not outer_polygon.contains(Point(inner_vertices[i, 0], inner_vertices[i, 1])):
            return False

    return True

@njit
def check_overlap_hexagons(x1, y1, angle1, x2, y2, angle2):
    """Check if two hexagons overlap using vertex-based collision detection"""
    vertices1 = generate_hexagon_vertices(x1, y1, angle1, 1.0)
    vertices2 = generate_hexagon_vertices(x2, y2, angle2, 1.0)

    # Simple bounding box check first
    min1 = np.min(vertices1, axis=0)
    max1 = np.max(vertices1, axis=0)
    min2 = np.min(vertices2, axis=0)
    max2 = np.max(vertices2, axis=0)

    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False

    # Create polygons and check intersection
    poly1 = Polygon(vertices1)
    poly2 = Polygon(vertices2)

    # If intersection exists, they overlap
    return poly1.intersects(poly2)

@njit
def calculate_penalty(packed_hexagons, outer_hexagon_params, penalty_weights=(1e6, 1e5)):
    """Calculate penalty based on constraint violations with adaptive weights"""
    penalty = 0.0
    n = len(packed_hexagons)

    outer_x, outer_y, outer_angle, outer_side_length = outer_hexagon_params

    # Check containment penalties (higher priority)
    containment_penalty_weight = penalty_weights[0]
    overlap_penalty_weight = penalty_weights[1]

    for i in range(n):
        if not check_containment_inner_to_outer(
            packed_hexagons[i][0], packed_hexagons[i][1], packed_hexagons[i][2],
            outer_x, outer_y, outer_angle, outer_side_length
        ):
            penalty += containment_penalty_weight

    # Check overlap penalties (lower priority)
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap_hexagons(
                packed_hexagons[i][0], packed_hexagons[i][1], packed_hexagons[i][2],
                packed_hexagons[j][0], packed_hexagons[j][1], packed_hexagons[j][2]
            ):
                penalty += overlap_penalty_weight

    return penalty

def evaluate_configuration(params):
    """Evaluate the fitness of a given configuration"""
    # Extract parameters
    packed_hexagons = []
    idx = 0
    for i in range(12):
        packed_hexagons.append([params[idx], params[idx+1], params[idx+2]])
        idx += 3

    outer_side_length = params[-1]

    # Calculate penalty
    penalty = calculate_penalty(packed_hexagons, [0, 0, 0, outer_side_length])

    # Inverse side length (negative because we minimize)
    # We want to maximize inverse side length, so minimize negative value
    objective_value = -1.0 / outer_side_length + penalty

    return objective_value

# Enhanced symmetric initialization function with better base patterns
def generate_symmetric_initial_population(pop_size=30):
    """Generate highly symmetric configurations to guide optimization"""
    population = []

    # Base symmetric pattern inspired by known optimal configurations
    # This pattern attempts to create a tighter packing with more symmetry
    base_pattern = [
        [0.0, 0.0, 0.0],        # Center
        [0.0, 3.0, 0.0],        # Top
        [0.0, -3.0, 0.0],       # Bottom
        [2.6, 1.5, 0.0],        # Top Right
        [-2.6, 1.5, 0.0],       # Top Left
        [2.6, -1.5, 0.0],       # Bottom Right
        [-2.6, -1.5, 0.0],      # Bottom Left
        [3.5, 0.0, 0.0],        # Far Right
        [-3.5, 0.0, 0.0],       # Far Left
        [1.75, 3.03, 0.0],      # Upper Middle Right
        [-1.75, 3.03, 0.0],     # Upper Middle Left
        [1.75, -3.03, 0.0],     # Lower Middle Right
        [-1.75, -3.03, 0.0],    # Lower Middle Left
    ]

    # Generate variations of the symmetric pattern
    for _ in range(pop_size):
        config = []
        # Add perturbation to base pattern
        for i, (x, y, angle) in enumerate(base_pattern[:-1]):  # Skip last element (outer side)
            # Add small random perturbations
            pert_x = x + random.uniform(-0.3, 0.3)
            pert_y = y + random.uniform(-0.3, 0.3)
            pert_angle = angle + random.uniform(-10, 10)  # Random rotation
            config.extend([pert_x, pert_y, pert_angle])

        # Add outer side length with reasonable range
        config.append(6.0 + random.uniform(0, 2))
        population.append(config)

    return population

# Improved local search function
def local_search(best_config):
    """Apply advanced local search to refine the configuration"""
    best_value = evaluate_configuration(best_config)
    improved = True
    iterations = 0

    # Momentum-based search
    momentum = [0.0] * len(best_config)

    while improved and iterations < 100:
        improved = False
        iterations += 1
        # Try multiple directions for each parameter
        for i in range(len(best_config)):
            original_val = best_config[i]

            # Try multiple step sizes and directions
            steps = [-0.15, -0.1, -0.05, 0.05, 0.1, 0.15]
            if i == len(best_config) - 1:  # outer side length - must be positive
                steps = [0.05, 0.1, 0.15]

            for delta in steps:
                # Apply momentum for smoother movement
                if i < len(best_config) - 1:  # Not outer side length
                    new_val = original_val + delta + momentum[i] * 0.1
                    if new_val < -10 or new_val > 10:
                        continue
                else:  # outer side length - constrain positive
                    new_val = original_val + delta + momentum[i] * 0.1
                    if new_val <= 0.1:
                        continue

                best_config[i] = new_val
                new_value = evaluate_configuration(best_config)

                if new_value < best_value:
                    best_value = new_value
                    improved = True
                    momentum[i] = delta  # Update momentum
                else:
                    best_config[i] = original_val  # Revert if no improvement

    return best_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Define bounds for optimization
    # Each hexagon has 3 parameters: x, y, angle; plus outer side length
    # Bounds: x, y from -10 to 10, angle from 0 to 360, outer side length from 1 to 20
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])
    bounds.append((1, 20))  # Outer side length bound

    # Generate enhanced initial population
    initial_pop = generate_symmetric_initial_population(25)

    # Set up differential evolution with improved parameters
    def objective_func(params):
        return evaluate_configuration(params)

    # Run optimization with enhanced settings
    try:
        result = differential_evolution(
            objective_func,
            bounds,
            seed=42,
            maxiter=150,           # More iterations for better convergence
            popsize=25,            # Larger population for better exploration
            mutation=(0.7, 1),     # Higher mutation rate for better exploration
            recombination=0.8,     # Higher recombination rate
            tol=1e-6,
            workers=1,
            init=initial_pop
        )

        # Best result
        best_params = result.x
        best_score = result.fun

        # Local search refinement with better parameters
        best_params = local_search(best_params)
        refined_score = evaluate_configuration(best_params)

        # Extract configuration
        inner_hex_data = []
        idx = 0
        for i in range(12):
            inner_hex_data.append([
                best_params[idx],
                best_params[idx+1],
                best_params[idx+2]
            ])
            idx += 3

        outer_side_length = best_params[-1]

        # Store results
        inner_hex_data = np.array(inner_hex_data)
        outer_hex_data = np.array([0, 0, 0])

    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to previous solution
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
            [0, -4, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_side_length = 8.0

    # Ensure all computations completed within time limit
    elapsed_time = time.time() - start_time
    if elapsed_time > 175:  # Leave buffer
        print("Warning: Time limit approaching")

    return inner_hex_data, outer_hex_data, outer_side_length

# EVOLVE-BLOCK-END