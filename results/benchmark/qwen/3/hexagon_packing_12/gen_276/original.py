# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import math
import time
from numba import jit
from scipy.spatial.distance import cdist

# Constants for hexagons
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_WIDTH = 2.0  # Distance between parallel sides
UNIT_HEX_SIDE_LENGTH = 1.0  # Side length of unit hexagon
MAX_EVAL_TIME = 180.0  # seconds
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices_numba(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius using numba JIT"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def compute_distance_fast(x1, y1, x2, y2):
    """Fast Euclidean distance computation"""
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx*dx + dy*dy)

@jit(nopython=True)
def check_bbox_overlap_fast(x1_min, y1_min, x1_max, y1_max, x2_min, y2_min, x2_max, y2_max):
    """Fast bounding box overlap check"""
    return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)

@jit(nopython=True)
def compute_bounding_box_fast(vertices):
    """Compute tight bounding box for hexagon vertices"""
    if len(vertices) == 0:
        return 0.0, 0.0, 0.0, 0.0
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    return np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)

def hexagon_to_polygon(x, y, angle_deg, radius=1.0):
    """Convert hexagon parameters to shapely polygon"""
    vertices = get_hexagon_vertices_numba(x, y, angle_deg, radius)
    return Polygon(vertices)

def is_contained_in_outer_fast(hex_poly, outer_poly):
    """Fast containment check with early rejection"""
    # Quick bounding box check first
    bbox1 = hex_poly.bounds
    bbox2 = outer_poly.bounds
    if not check_bbox_overlap_fast(bbox1[0], bbox1[1], bbox1[2], bbox1[3],
                                   bbox2[0], bbox2[1], bbox2[2], bbox2[3]):
        return False
    return outer_poly.contains(hex_poly) or outer_poly.covers(hex_poly)

def check_overlap_fast(hex1_poly, hex2_poly):
    """Fast overlap check with bounding box pre-filter"""
    # Quick bounding box check first
    bbox1 = hex1_poly.bounds
    bbox2 = hex2_poly.bounds

    if not check_bbox_overlap_fast(bbox1[0], bbox1[1], bbox1[2], bbox1[3],
                                   bbox2[0], bbox2[1], bbox2[2], bbox2[3]):
        return False

    # Detailed overlap check
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def compute_outer_hexagon_radius_fast(inner_hex_data):
    """Fast computation of minimum outer hexagon radius"""
    if len(inner_hex_data) == 0:
        return 0.0

    # Get all vertices of all inner hexagons efficiently
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices_numba(x, y, angle)
        all_vertices.extend(vertices)

    if len(all_vertices) == 0:
        return 0.0

    # Compute centroid and max distance
    centroid_x = np.mean([v[0] for v in all_vertices])
    centroid_y = np.mean([v[1] for v in all_vertices])

    max_distance = 0.0
    for x, y in all_vertices:
        distance = compute_distance_fast(x, y, centroid_x, centroid_y)
        max_distance = max(max_distance, distance)

    # Add buffer for hexagon radius calculation
    return max_distance + UNIT_HEX_RADIUS + 1e-10

def compute_outer_hexagon_side_length(inner_hex_data):
    """Compute the side length of the minimal outer hexagon"""
    return compute_outer_hexagon_radius_fast(inner_hex_data)

def validate_solution_fast(inner_hex_data, outer_hex_data=None):
    """Fast validation with early exit conditions"""
    if len(inner_hex_data) != 12:
        return False, "Wrong number of hexagons"

    # Create outer hexagon
    if outer_hex_data is None:
        outer_radius = compute_outer_hexagon_radius_fast(inner_hex_data)
        outer_x, outer_y, outer_angle = 0, 0, 0
    else:
        outer_x, outer_y, outer_angle = outer_hex_data
        outer_radius = compute_outer_hexagon_radius_fast(inner_hex_data)

    outer_hex = hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)

    # Check each inner hexagon efficiently
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        inner_hex = hexagon_to_polygon(x, y, angle)

        # Quick containment check
        if not is_contained_in_outer_fast(inner_hex, outer_hex):
            return False, f"Inner hexagon {i} not contained"

        # Check overlaps with others - early exit
        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            inner_hex2 = hexagon_to_polygon(x2, y2, angle2)

            if check_overlap_fast(inner_hex, inner_hex2):
                return False, f"Overlapping hexagons {i} and {j}"

    return True, "Valid solution"

def objective_function_fast(params, inner_hex_data=None):
    """
    Fast objective function to minimize (negative of 1/outer_radius)
    """
    # Reshape params into hexagon data
    hex_data = params.reshape(-1, 3)

    # Compute outer radius
    outer_radius = compute_outer_hexagon_radius_fast(hex_data)

    # We want to maximize 1/outer_radius, so we minimize -1/outer_radius
    if outer_radius <= 0:
        return 1e10  # Large penalty for invalid configurations

    return -1.0 / outer_radius

def create_better_initial_config():
    """Create a better initial configuration based on known optimal packing arrangements"""
    # Use a known good configuration from mathematical analysis of hexagon packings
    positions = np.array([
        [0.0, 0.0, 0.0],      # Center
        [0.0, 2.0, 0.0],      # Top
        [1.732050808, 1.0, 0.0],   # Top right
        [1.732050808, -1.0, 0.0],  # Bottom right
        [0.0, -2.0, 0.0],     # Bottom
        [-1.732050808, -1.0, 0.0],  # Bottom left
        [-1.732050808, 1.0, 0.0],   # Top left
        [3.464101616, 2.0, 0.0],    # Far top right
        [3.464101616, -2.0, 0.0],   # Far bottom right
        [-3.464101616, -2.0, 0.0],  # Far bottom left
        [-3.464101616, 2.0, 0.0],   # Far top left
        [0.0, -4.0, 0.0],     # Far bottom
    ], dtype=np.float64)

    return positions

def create_symmetric_configs():
    """Generate multiple symmetric initial configurations for multi-start optimization"""
    configs = []

    # Configuration 1: Standard hexagonal arrangement
    config1 = create_better_initial_config()
    configs.append(config1)

    # Configuration 2: Rotated version to explore different orientations
    config2 = config1.copy()
    for i in range(12):
        # Rotate by 30 degrees (π/6 radians)
        x, y, _ = config2[i]
        angle_rad = np.pi / 6
        new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        config2[i] = [new_x, new_y, 0.0]
    configs.append(config2)

    # Configuration 3: Reflection of the first (mirror across y-axis)
    config3 = config1.copy()
    for i in range(12):
        config3[i][0] = -config3[i][0]  # Flip x-coordinate
        config3[i][2] = (config3[i][2] + 180) % 360  # Reverse rotation
    configs.append(config3)

    # Configuration 4: Perturbed version to escape local minima
    config4 = config1.copy()
    for i in range(12):
        config4[i][0] += np.random.normal(0, 0.03)
        config4[i][1] += np.random.normal(0, 0.03)
    configs.append(config4)

    return configs

def symmetry_aware_mutation(params, mutation_strength=0.05):
    """Apply symmetry-aware mutation that respects hexagonal symmetry properties"""
    mutated_params = params.copy()

    # Mutate positions with consideration for symmetry
    for i in range(0, len(mutated_params), 3):  # Process each hexagon
        # Mutate x coordinate
        mutated_params[i] += np.random.normal(0, mutation_strength * 0.8)
        # Mutate y coordinate
        mutated_params[i+1] += np.random.normal(0, mutation_strength * 0.8)
        # Mutate angle (but keep within reasonable range)
        mutated_params[i+2] += np.random.normal(0, mutation_strength * 3)
        mutated_params[i+2] = mutated_params[i+2] % 360  # Keep angle in [0,360)

    # Apply symmetry constraint: enforce that opposite positions remain symmetric
    # This maintains the hexagonal symmetry structure
    for i in range(0, 12, 2):  # Process pairs of opposite hexagons
        idx1 = i
        idx2 = (i + 6) % 12  # Opposite hexagon

        if idx1 < 6 and idx2 >= 6:
            # Maintain symmetry: opposite hexagons should be mirror images
            x1, y1 = mutated_params[idx1*3], mutated_params[idx1*3 + 1]
            x2, y2 = mutated_params[idx2*3], mutated_params[idx2*3 + 1]

            # Force symmetry: y-coordinates should be negated, x-coordinates same magnitude
            mutated_params[idx2*3] = -x1  # Mirror x
            mutated_params[idx2*3 + 1] = -y1  # Mirror y

    return mutated_params

def multi_stage_optimization(initial_params, start_time):
    """Perform multi-stage optimization with timeout checks"""
    # Stage 1: Optimize positions only with fixed rotations (fast)
    try:
        # Fix rotations to accelerate first stage
        fixed_params = initial_params.copy()
        for i in range(12):
            fixed_params[i*3 + 2] = 0.0  # Set all rotations to 0

        # Create bounds for positions only
        bounds = [(-5.0, 5.0)] * 24 + [(0.0, 0.0)] * 12

        result1 = minimize(
            objective_function_fast,
            fixed_params,
            args=(None,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-6}
        )

        if result1.success:
            current_params = result1.x.copy()
        else:
            current_params = initial_params.copy()

        # Timeout check
        if time.time() - start_time > MAX_EVAL_TIME - 30:
            return current_params

    except Exception as e:
        current_params = initial_params.copy()

    # Stage 2: Local search with symmetry-aware mutation
    try:
        # Use a local search approach with symmetry preservation
        best_variant = current_params.copy()
        best_objective = objective_function_fast(current_params)

        # Try several mutated variants
        for _ in range(3):  # Test fewer variants for speed
            if time.time() - start_time > MAX_EVAL_TIME - 15:
                break
            mutated = symmetry_aware_mutation(current_params, mutation_strength=0.03)
            obj_val = objective_function_fast(mutated)
            if obj_val < best_objective:  # Minimize objective (maximize 1/outer_radius)
                best_objective = obj_val
                best_variant = mutated.copy()

        current_params = best_variant.copy()

    except Exception as e:
        pass  # Continue with current_params

    # Stage 3: Optimize with both positions and rotations (finest refinement)
    try:
        # Now optimize with full parameters
        bounds = [(-5.0, 5.0)] * 24 + [(-180.0, 180.0)] * 12

        result2 = minimize(
            objective_function_fast,
            current_params,
            args=(None,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-7}
        )

        if result2.success:
            final_params = result2.x.copy()
        else:
            final_params = current_params.copy()

    except Exception as e:
        final_params = current_params.copy()

    return final_params

def optimize_hexagon_arrangement():
    """Use a sophisticated optimization approach to find the best packing"""
    start_time = time.time()

    # Multi-start optimization approach with early termination
    best_objective = -float('inf')  # We want to maximize 1/outer_radius
    best_hex_data = None

    # Try multiple starting configurations with symmetry awareness
    initial_configs = create_symmetric_configs()

    # Add a few more diverse configurations
    for _ in range(2):
        # Randomly perturbed configurations
        random_config = create_better_initial_config().copy()
        for i in range(12):
            # Add small random perturbations
            random_config[i][0] += np.random.normal(0, 0.03)
            random_config[i][1] += np.random.normal(0, 0.03)
        initial_configs.append(random_config)

    for i, initial_positions in enumerate(initial_configs):
        if time.time() - start_time > MAX_EVAL_TIME - 10:
            break

        try:
            # Flatten the initial positions to use as starting point for optimization
            initial_params = initial_positions.flatten()

            # Multi-stage optimization approach with symmetry awareness
            best_params = multi_stage_optimization(initial_params, start_time)

            # Reshape to hexagon data
            current_hex_data = best_params.reshape(-1, 3)

            # Validate the solution
            valid, message = validate_solution_fast(current_hex_data)

            # If valid, compute objective value
            if valid:
                # Compute outer radius for this solution
                outer_radius = compute_outer_hexagon_radius_fast(current_hex_data)
                objective_value = 1.0 / outer_radius

                # Update best if this is better
                if objective_value > best_objective:
                    best_objective = objective_value
                    best_hex_data = current_hex_data.copy()

        except Exception as e:
            continue  # Skip this configuration if it fails

    # If no valid configurations found, return the best known configuration
    if best_hex_data is None:
        # Use the better initial configuration as fallback
        return create_better_initial_config()

    return best_hex_data

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """

    try:
        # Run sophisticated optimization
        inner_hex_data = optimize_hexagon_arrangement()

        # Compute the outer hexagon size required
        outer_hex_side_length = compute_outer_hexagon_side_length(inner_hex_data)

        # Outer hexagon centered at origin, no rotation
        outer_hex_data = np.array([0, 0, 0])

        # Final validation
        valid, message = validate_solution_fast(inner_hex_data)

        # If validation still fails, use fallback
        if not valid:
            inner_hex_data = np.array([
                [0, 0, 0],  # center
                [-2.5, 0, 0],  # left
                [2.5, 0, 0],  # right
                [-1.25, 2.17, 0],  # top-left
                [1.25, 2.17, 0],  # top-right
                [-1.25, -2.17, 0],  # bottom-left
                [1.25, -2.17, 0],  # bottom-right
                [-3.75, 2.17, 0],  # far top-left
                [3.75, 2.17, 0],  # far top-right
                [-3.75, -2.17, 0],  # far bottom-left
                [3.75, -2.17, 0],  # far bottom-right,
                [0, -4, 0],  # far bottom-center
            ])
            outer_hex_side_length = 8
            outer_hex_data = np.array([0, 0, 0])

    except Exception as e:
        # Fallback to original approach
        print(f"Fallback due to error: {e}")
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right,
            [0, -4, 0],  # far bottom-center
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END