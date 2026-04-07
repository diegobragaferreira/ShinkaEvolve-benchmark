# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import warnings
import time
import random

# Constants
UNIT_HEX_RADIUS = 1.0  # Side length of unit hexagon
UNIT_HEX_APOGEE = np.sqrt(3)/2  # Distance from center to corner of unit hexagon

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon as a Shapely Polygon"""
    angle_offset = np.deg2rad(rotation)
    points = []
    for i in range(6):
        angle = angle_offset + i * np.pi/3
        x = center[0] + UNIT_HEX_RADIUS * np.cos(angle)
        y = center[1] + UNIT_HEX_RADIUS * np.sin(angle)
        points.append((x, y))
    return Polygon(points)

def check_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer_hexagon with robust buffer validation"""
    # First try with a very small buffer to catch obvious containment issues
    try:
        buffered_hexagon = hexagon.buffer(-1e-12)
        if outer_hexagon.contains(buffered_hexagon):
            return True
    except:
        pass

    # If that fails, use a more conservative approach with slightly larger buffer
    try:
        buffered_hexagon = hexagon.buffer(-1e-10)
        if outer_hexagon.contains(buffered_hexagon):
            return True
    except:
        pass

    # Fallback to vertex-by-vertex check for robustness
    for point in hexagon.exterior.coords[:-1]:
        if not outer_hexagon.contains(Point(point)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap with robust buffer validation"""
    # Try with very small positive buffer first
    try:
        buffered_hex1 = hex1.buffer(1e-12)
        buffered_hex2 = hex2.buffer(1e-12)
        if buffered_hex1.intersects(buffered_hex2):
            return True
    except:
        pass

    # Try with moderate buffer
    try:
        buffered_hex1 = hex1.buffer(1e-10)
        buffered_hex2 = hex2.buffer(1e-10)
        if buffered_hex1.intersects(buffered_hex2):
            return True
    except:
        pass

    # Fallback to direct intersection test
    return hex1.intersects(hex2)

def calculate_packing_density(inner_params, outer_radius):
    """Calculate packing density as ratio of total area occupied by inner hexagons to outer hexagon area"""
    inner_area = 0
    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        hexagon = create_unit_hexagon((x, y), angle)
        inner_area += hexagon.area

    outer_hexagon = create_unit_hexagon((0, 0), 0)
    outer_coords = list(outer_hexagon.exterior.coords)
    scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
    outer_hexagon_scaled = Polygon(scaled_coords)

    return inner_area / outer_hexagon_scaled.area

def calculate_outer_hex_radius(inner_params, outer_center=(0,0)):
    """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
    max_dist = 0
    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        center = (x, y)
        dist = np.linalg.norm(np.array(center) - np.array(outer_center))
        # Add distance from center to corner of unit hexagon
        dist += UNIT_HEX_APOGEE
        max_dist = max(max_dist, dist)
    return max_dist

def evaluate_constraints(inner_params, outer_radius):
    """Comprehensive constraint evaluation with early termination"""
    inner_hexagons = []

    # Create inner hexagons
    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        hexagon = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(hexagon)

    # Create outer hexagon
    outer_hexagon = create_unit_hexagon((0, 0), 0)
    outer_coords = list(outer_hexagon.exterior.coords)
    scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
    outer_hexagon_scaled = Polygon(scaled_coords)

    # Check containment (early termination)
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hexagon_scaled):
            return False, False, 0.0  # containment violated

    # Check overlaps (early termination)
    for i in range(11):
        for j in range(i+1, 11):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                return False, False, 0.0  # overlap violated

    return True, True, 1.0 / outer_radius  # valid solution

def fitness_function(params):
    """Fitness function that maximizes 1/outer_radius while penalizing constraint violations"""
    # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
    outer_radius = params[-1]

    # Extract inner hexagon parameters
    inner_params = params[:-1]

    # Check constraints
    containment_ok, overlap_ok, inv_radius = evaluate_constraints(inner_params, outer_radius)

    # If any constraint violated, return large penalty
    if not (containment_ok and overlap_ok):
        # Heavy penalty for constraint violations
        return 100000.0 + abs(outer_radius)

    # Return negative of inverse radius to minimize (maximize 1/outer_radius)
    # Also incorporate packing density for better optimization guidance
    packing_density = calculate_packing_density(inner_params, outer_radius)
    return -inv_radius + (1 - packing_density) * 10000.0

def generate_diverse_initial_configurations(num_configs=10):
    """Generate multiple diverse initial configurations using various heuristics"""
    configs = []

    # Configuration 1: Honeycomb-like arrangement
    honeycomb_positions = [
        (0, 0),           # center
        (-2, 0),          # left
        (2, 0),           # right
        (0, 2),           # top
        (0, -2),          # bottom
        (-1, 1),          # top-left
        (1, 1),           # top-right
        (-1, -1),         # bottom-left
        (1, -1),          # bottom-right
        (-2.5, 1.5),      # far top-left
        (2.5, 1.5),       # far top-right
    ]

    # Configuration 2: Spiral arrangement
    spiral_positions = [
        (0, 0),           # center
        (0, 2),           # top
        (1.73, 1),        # top-right
        (1.73, -1),       # bottom-right
        (0, -2),          # bottom
        (-1.73, -1),      # bottom-left
        (-1.73, 1),       # top-left
        (0, 1.5),         # upper-middle
        (0, -1.5),        # lower-middle
        (1.5, 0),         # right-middle
        (-1.5, 0),        # left-middle
    ]

    # Configuration 3: Grid arrangement
    grid_positions = [
        (0, 0),           # center
        (-2.5, 0),        # left
        (2.5, 0),         # right
        (0, 2.5),         # top
        (0, -2.5),        # bottom
        (-1.5, 1.5),      # top-left
        (1.5, 1.5),       # top-right
        (-1.5, -1.5),     # bottom-left
        (1.5, -1.5),      # bottom-right
        (-3, 0),          # further left
        (3, 0),           # further right
    ]

    # Configuration 4: Clustered arrangement
    cluster_positions = [
        (0, 0),           # center
        (-2, 0),          # left
        (2, 0),           # right
        (0, 2),           # top
        (0, -2),          # bottom
        (-1.5, 1.5),      # top-left
        (1.5, 1.5),       # top-right
        (-1.5, -1.5),     # bottom-left
        (1.5, -1.5),      # bottom-right
        (-2.5, 2.5),      # far top-left
        (2.5, 2.5),       # far top-right
    ]

    # Generate configs using the different strategies
    strategies = [
        honeycomb_positions,
        spiral_positions,
        grid_positions,
        cluster_positions
    ]

    for i, positions in enumerate(strategies):
        for _ in range(num_configs // len(strategies)):
            config = []
            # Add randomness to positions
            for j, (cx, cy) in enumerate(positions):
                # Add small random variation to avoid exact symmetries
                jitter_x = np.random.normal(0, 0.15)
                jitter_y = np.random.normal(0, 0.15)
                angle = np.random.uniform(0, 360)
                config.extend([cx + jitter_x, cy + jitter_y, angle])

            # Add outer radius estimate
            estimated_radius = 0
            for cx, cy in positions:
                dist = np.sqrt(cx**2 + cy**2) + UNIT_HEX_APOGEE
                estimated_radius = max(estimated_radius, dist)

            config.append(estimated_radius + 0.5 + np.random.uniform(0, 0.2))
            configs.append(config)

    # Add some completely random configurations
    for _ in range(num_configs // 4):
        config = []
        for _ in range(11):
            # Random positions within a reasonable range
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            angle = np.random.uniform(0, 360)
            config.extend([x, y, angle])

        # Random outer radius estimate
        config.append(np.random.uniform(4, 8))
        configs.append(config)

    return configs

def refine_with_local_search(initial_params, bounds):
    """Refine solution using local optimization after global search"""
    try:
        # Use L-BFGS-B for fine-tuning with tighter tolerances
        result = minimize(
            fitness_function,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=lambda x: None
        )

        if result.success:
            return result.x
    except Exception as e:
        warnings.warn(f"Local search failed: {str(e)}")

    return initial_params

def optimize_solution():
    """Main optimization routine using multiple strategies"""
    best_solution = None
    best_inv_radius = 0

    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_configurations(20)

    # Try multiple optimization strategies
    for config_idx, initial_guess in enumerate(initial_configs):
        try:
            # Set up bounds for optimization
            bounds = []
            # Bounds for inner hexagon positions and rotations
            for _ in range(11):
                bounds.extend([(-8.0, 8.0), (-8.0, 8.0), (0, 360)])  # x, y, angle
            # Bound for outer radius
            bounds.append((3.0, 10.0))  # Reasonable range for outer radius

            # Try different optimization methods
            # First try differential evolution for global search
            de_result = differential_evolution(
                fitness_function,
                bounds,
                seed=config_idx,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )

            if de_result.success:
                # Refine with local search
                refined_params = refine_with_local_search(de_result.x, bounds)

                # Evaluate refined solution
                inner_params = refined_params[:-1]
                outer_radius = refined_params[-1]
                inv_radius, containment_ok, overlap_ok = evaluate_constraints(inner_params, outer_radius)

                if containment_ok and overlap_ok and inv_radius > best_inv_radius:
                    best_inv_radius = inv_radius
                    best_solution = refined_params[:]

            # If DE didn't work, try a direct L-BFGS approach
            elif config_idx < 5:  # Only try direct optimization for first few configs
                lbfgs_result = minimize(
                    fitness_function,
                    initial_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )

                if lbfgs_result.success:
                    # Evaluate solution
                    inner_params = lbfgs_result.x[:-1]
                    outer_radius = lbfgs_result.x[-1]
                    inv_radius, containment_ok, overlap_ok = evaluate_constraints(inner_params, outer_radius)

                    if containment_ok and overlap_ok and inv_radius > best_inv_radius:
                        best_inv_radius = inv_radius
                        best_solution = lbfgs_result.x[:]

        except Exception as e:
            warnings.warn(f"Optimization attempt {config_idx} failed: {str(e)}")
            continue

    return best_solution, best_inv_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses evolutionary optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Run optimization
        final_params, inv_radius = optimize_solution()

        if final_params is not None:
            # Extract results
            inner_params = final_params[:-1]
            outer_radius = final_params[-1]

            # Validate solution
            containment_ok, overlap_ok, test_inv_radius = evaluate_constraints(inner_params, outer_radius)

            if containment_ok and overlap_ok and test_inv_radius > 0.25:
                # Format output
                inner_hex_data = np.zeros((11, 3))
                for i in range(11):
                    inner_hex_data[i] = inner_params[3*i:3*i+3]

                outer_hex_data = np.array([0, 0, 0])

                return inner_hex_data, outer_hex_data, outer_radius

    except Exception as e:
        warnings.warn(f"Error in optimization: {str(e)}")
        pass

    # Fallback to original method if optimization fails
    inner_hex_data = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 8  # large enough to contain all inner hexagons

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END