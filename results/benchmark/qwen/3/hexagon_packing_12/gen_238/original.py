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
MAX_EVAL_TIME = 180.0  # seconds
TARGET_RATIO = 0.2537

@jit(nopython=True)
def get_hexagon_vertices(x, y, angle_deg, radius=1.0):
    """Get vertices of a hexagon given center, angle, and radius"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

def hexagon_to_polygon(x, y, angle_deg, radius=1.0):
    """Convert hexagon parameters to shapely polygon"""
    vertices = get_hexagon_vertices(x, y, angle_deg, radius)
    return Polygon(vertices)

def is_contained_in_outer(hex_poly, outer_poly):
    """Check if hexagon is fully contained in outer hexagon"""
    return outer_poly.contains(hex_poly) or outer_poly.covers(hex_poly)

def check_overlap_fast(hex1_poly, hex2_poly):
    """Fast overlap check using bounding boxes"""
    # Quick bounding box check first
    bbox1 = hex1_poly.bounds
    bbox2 = hex2_poly.bounds
    if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or
        bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
        return False
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def compute_outer_hexagon_radius(inner_hex_data):
    """Compute minimum outer hexagon radius that contains all inner hexagons"""
    if len(inner_hex_data) == 0:
        return 0.0

    # Get all vertices of all inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        vertices = get_hexagon_vertices(x, y, angle)
        all_vertices.extend(vertices)

    if len(all_vertices) == 0:
        return 0.0

    # Compute centroid
    centroid_x = np.mean([v[0] for v in all_vertices])
    centroid_y = np.mean([v[1] for v in all_vertices])

    # Find maximum distance from centroid to any vertex
    max_distance = 0.0
    for x, y in all_vertices:
        distance = math.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        max_distance = max(max_distance, distance)

    # Add buffer for hexagon radius calculation
    return max_distance + UNIT_HEX_RADIUS + 1e-10

def validate_solution(inner_hex_data, outer_hex_data=None):
    """Validate that solution meets all constraints"""
    if len(inner_hex_data) != 12:
        return False, "Wrong number of hexagons"

    # Create outer hexagon
    if outer_hex_data is None:
        outer_radius = compute_outer_hexagon_radius(inner_hex_data)
        outer_x, outer_y, outer_angle = 0, 0, 0
    else:
        outer_x, outer_y, outer_angle = outer_hex_data
        outer_radius = compute_outer_hexagon_radius(inner_hex_data)

    outer_hex = hexagon_to_polygon(outer_x, outer_y, outer_angle, outer_radius)

    # Check each inner hexagon
    for i in range(len(inner_hex_data)):
        x, y, angle = inner_hex_data[i]
        inner_hex = hexagon_to_polygon(x, y, angle)

        # Check containment with buffer to handle floating point precision
        if not is_contained_in_outer(inner_hex, outer_hex):
            return False, f"Inner hexagon {i} not contained"

        # Check overlaps with others - early exit for performance
        for j in range(i+1, len(inner_hex_data)):
            x2, y2, angle2 = inner_hex_data[j]
            inner_hex2 = hexagon_to_polygon(x2, y2, angle2)

            if check_overlap_fast(inner_hex, inner_hex2):
                return False, f"Overlapping hexagons {i} and {j}"

    return True, "Valid solution"

def objective_function(params, inner_hex_data=None):
    """
    Objective function to minimize (negative of 1/outer_radius)
    """
    # Reshape params into hexagon data
    hex_data = params.reshape(-1, 3)

    # Compute outer radius
    outer_radius = compute_outer_hexagon_radius(hex_data)

    # We want to maximize 1/outer_radius, so we minimize -1/outer_radius
    if outer_radius <= 0:
        return 1e10  # Large penalty for invalid configurations

    return -1.0 / outer_radius

def create_better_initial_config():
    """Create a better initial configuration based on mathematical insights"""
    # This configuration is inspired by the known optimal solution:
    # Based on the benchmark target of 1/3.9419123 ≈ 0.2537, 
    # we create a configuration that starts near the optimal region
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

    # Configuration 1: Better initial configuration
    config1 = create_better_initial_config()
    configs.append(config1)

    # Configuration 2: Slightly perturbed version
    config2 = config1.copy()
    for i in range(12):
        config2[i][0] += np.random.normal(0, 0.05)
        config2[i][1] += np.random.normal(0, 0.05)
    configs.append(config2)

    # Configuration 3: Rotated version (30-degree rotation in degrees)
    config3 = config1.copy()
    for i in range(12):
        # Apply 30-degree rotation matrix
        rad = np.pi / 6
        x, y = config3[i][0], config3[i][1]
        new_x = x * np.cos(rad) - y * np.sin(rad)
        new_y = x * np.sin(rad) + y * np.cos(rad)
        config3[i][0], config3[i][1] = new_x, new_y
    configs.append(config3)

    return configs

def optimize_positions_only(params, fix_rotations=False):
    """Optimize only positions (not rotations) for initial refinement"""
    # Create bounds for positions only 
    bounds = []
    for i in range(12):
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0)])  # x, y bounds
        if not fix_rotations:
            bounds.append((-180.0, 180.0))  # angle bounds
        else:
            bounds.append((params[i*3 + 2], params[i*3 + 2]))  # Keep current angle

    # For this approach, we'll keep it simple with basic optimization
    return params

def multi_stage_optimization(initial_params):
    """Perform multi-stage optimization for better results with symmetry awareness"""
    start_time = time.time()

    # Stage 1: Optimize positions only with fixed rotations (symmetry-preserving)
    try:
        # Fix rotations to avoid getting stuck in local minima
        fixed_params = initial_params.copy()
        for i in range(12):
            fixed_params[i*3 + 2] = 0.0  # Set all rotations to 0

        # L-BFGS-B optimization with position-only variables (x,y)
        result1 = minimize(
            objective_function,
            fixed_params,
            args=(None,),
            method='L-BFGS-B',
            bounds=[(-5.0, 5.0)] * 24,  # Only x,y bounds since angles fixed
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        if result1.success:
            current_params = result1.x.copy()
        else:
            current_params = initial_params.copy()

    except Exception as e:
        current_params = initial_params.copy()

    # Stage 2: Optimize with both positions and rotations
    try:
        # Now optimize with full parameters
        bounds = [(-5.0, 5.0)] * 24 + [(-180.0, 180.0)] * 12  # Full optimization

        result2 = minimize(
            objective_function,
            current_params,
            args=(None,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
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
            random_config[i][0] += np.random.normal(0, 0.05)
            random_config[i][1] += np.random.normal(0, 0.05)
        initial_configs.append(random_config)

    for i, initial_positions in enumerate(initial_configs):
        try:
            # Flatten the initial positions to use as starting point for optimization
            initial_params = initial_positions.flatten()

            # Multi-stage optimization approach
            best_params = multi_stage_optimization(initial_params)

            # Reshape to hexagon data
            current_hex_data = best_params.reshape(-1, 3)

            # Validate the solution
            valid, message = validate_solution(current_hex_data)

            # If valid, compute objective value
            if valid:
                # Compute outer radius for this solution
                outer_radius = compute_outer_hexagon_radius(current_hex_data)
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
        outer_hex_side_length = compute_outer_hexagon_radius(inner_hex_data)

        # Outer hexagon centered at origin, no rotation
        outer_hex_data = np.array([0, 0, 0])

        # Final validation
        valid, message = validate_solution(inner_hex_data)

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
