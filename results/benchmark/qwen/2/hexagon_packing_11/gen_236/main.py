# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import warnings
from joblib import Parallel, delayed

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
    """Check if hexagon is fully contained within outer_hexagon with buffer for precision"""
    # Use a small buffer to avoid floating point precision issues
    buffered_hexagon = hexagon.buffer(-1e-10)
    return outer_hexagon.contains(buffered_hexagon)

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap with buffer for precision"""
    # Use a small buffer to avoid floating point precision issues
    buffered_hex1 = hex1.buffer(1e-10)
    buffered_hex2 = hex2.buffer(1e-10)
    return buffered_hex1.intersects(buffered_hex2)

def calculate_bounding_circle_radius(inner_params):
    """Calculate the minimal bounding circle radius for all hexagon vertices"""
    # Get all hexagon vertices
    all_vertices = []

    for i in range(11):
        x, y, angle = inner_params[3*i:3*i+3]
        hexagon = create_unit_hexagon((x, y), angle)
        # Get all vertices of this hexagon
        for point in hexagon.exterior.coords[:-1]:  # exclude closing point
            all_vertices.append(point)

    if not all_vertices:
        return 1.0

    # Convert to numpy array for easier computation
    vertices_array = np.array(all_vertices)

    # Find the minimum bounding circle using a simple approach:
    # Compute centroid and max distance from centroid
    centroid = np.mean(vertices_array, axis=0)
    distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))
    min_bounding_radius = np.max(distances) + 1e-6

    return min_bounding_radius

def voronoi_based_objective(params):
    """Objective function using Voronoi-based approach"""
    # params: [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, outer_radius]
    n = 11
    outer_radius = params[-1]

    # Extract inner hexagon parameters
    inner_params = params[:-1]

    # Create inner hexagons
    inner_hexagons = []
    centers = []
    for i in range(n):
        x, y, angle = inner_params[3*i:3*i+3]
        centers.append([x, y])
        hexagon = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(hexagon)

    # Create outer hexagon
    outer_hexagon = create_unit_hexagon((0, 0), 0)
    outer_coords = list(outer_hexagon.exterior.coords)
    scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
    outer_hexagon_scaled = Polygon(scaled_coords)

    # Check constraints
    total_penalty = 0

    # Check containment
    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hexagon_scaled):
            total_penalty += 10000  # Large penalty for violation

    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                total_penalty += 10000  # Large penalty for overlap

    # If any constraint violated, return large value
    if total_penalty > 0:
        return total_penalty + 100000

    # Calculate the actual tight radius using the improved method
    # This gives us a more accurate measure of the true packing efficiency
    actual_tight_radius = calculate_bounding_circle_radius(inner_params)

    # Return negative of inverse radius to minimize (maximize 1/outer_radius)
    return -1.0 / actual_tight_radius

def construct_honeycomb_packing():
    """Construct initial configuration using honeycomb-inspired approach"""
    # Use a more sophisticated honeycomb-like arrangement based on known good solutions
    centers = [
        (0.0, 0.0),       # center
        (-1.9, 0.0),      # left
        (1.9, 0.0),       # right
        (0.0, 1.9),       # top
        (0.0, -1.9),      # bottom
        (-1.4, 1.4),      # top-left
        (1.4, 1.4),       # top-right
        (-1.4, -1.4),     # bottom-left
        (1.4, -1.4),      # bottom-right
        (-2.3, 0.0),      # further left
        (2.3, 0.0),       # further right
    ]

    # Add some randomness to avoid symmetric solutions
    initial_guess = []
    for i, (cx, cy) in enumerate(centers):
        # Add small random variation with controlled magnitude
        jitter_x = np.random.normal(0, 0.15)
        jitter_y = np.random.normal(0, 0.15)
        # Use a wider angle range for better exploration
        angle = np.random.uniform(0, 360)
        initial_guess.extend([cx + jitter_x, cy + jitter_y, angle])

    # Estimate outer radius based on the honeycomb arrangement
    max_dist = 0
    for cx, cy in centers:
        dist = np.sqrt(cx**2 + cy**2) + UNIT_HEX_APOGEE
        max_dist = max(max_dist, dist)

    initial_guess.append(max_dist + 0.5)  # Add margin for safety

    return initial_guess

def construct_grid_packing():
    """Construct initial configuration using grid-based approach"""
    # Grid-like arrangement with slight perturbation
    centers = [
        (0.0, 0.0),       # center
        (-2.0, 0.0),      # left
        (2.0, 0.0),       # right
        (0.0, 2.0),       # top
        (0.0, -2.0),      # bottom
        (-1.5, 1.5),      # top-left
        (1.5, 1.5),       # top-right
        (-1.5, -1.5),     # bottom-left
        (1.5, -1.5),      # bottom-right
        (-2.5, 0.0),      # further left
        (2.5, 0.0),       # further right
    ]

    # Add some randomness to avoid symmetric solutions
    initial_guess = []
    for i, (cx, cy) in enumerate(centers):
        # Add small random variation with controlled magnitude
        jitter_x = np.random.normal(0, 0.2)
        jitter_y = np.random.normal(0, 0.2)
        # Use a wider angle range for better exploration
        angle = np.random.uniform(0, 360)
        initial_guess.extend([cx + jitter_x, cy + jitter_y, angle])

    # Estimate outer radius
    max_dist = 0
    for cx, cy in centers:
        dist = np.sqrt(cx**2 + cy**2) + UNIT_HEX_APOGEE
        max_dist = max(max_dist, dist)

    initial_guess.append(max_dist + 0.5)

    return initial_guess

def construct_spiral_packing():
    """Construct initial configuration using spiral arrangement"""
    # Create a spiral-like pattern
    centers = [
        (0.0, 0.0),       # center
        (0.0, 2.0),       # top
        (1.73, 1.0),      # top-right
        (1.73, -1.0),     # bottom-right
        (0.0, -2.0),      # bottom
        (-1.73, -1.0),    # bottom-left
        (-1.73, 1.0),     # top-left
        (0.0, 1.5),       # upper-middle
        (0.0, -1.5),      # lower-middle
        (1.5, 0.0),       # right-middle
        (-1.5, 0.0),      # left-middle
    ]

    # Add some randomness
    initial_guess = []
    for i, (cx, cy) in enumerate(centers):
        # Add small random variation
        jitter_x = np.random.normal(0, 0.15)
        jitter_y = np.random.normal(0, 0.15)
        angle = np.random.uniform(0, 360)
        initial_guess.extend([cx + jitter_x, cy + jitter_y, angle])

    # Estimate outer radius
    max_dist = 0
    for cx, cy in centers:
        dist = np.sqrt(cx**2 + cy**2) + UNIT_HEX_APOGEE
        max_dist = max(max_dist, dist)

    initial_guess.append(max_dist + 0.5)

    return initial_guess

def run_differential_evolution_with_seed(seed, bounds):
    """Run differential evolution with a specific seed"""
    try:
        np.random.seed(seed)
        result = differential_evolution(
            voronoi_based_objective,
            bounds,
            seed=seed,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        return result
    except Exception as e:
        warnings.warn(f"Differential evolution failed with seed {seed}: {str(e)}")
        return None

def run_local_refinement(final_params, bounds):
    """Run local refinement on the result from global optimization"""
    try:
        # Use L-BFGS-B for fine-tuning with higher precision
        result = minimize(
            voronoi_based_objective,
            final_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
        )
        return result
    except Exception as e:
        warnings.warn(f"Local refinement failed: {str(e)}")
        return None

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses parallel evolutionary optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Generate bounds for optimization
    bounds = []
    # Bounds for inner hexagon positions (more constrained)
    for _ in range(11):
        bounds.extend([(-5.0, 5.0), (-5.0, 5.0), (0, 360)])  # x, y, angle
    # Bound for outer radius
    bounds.append((2.5, 8.0))  # Tightened range

    # Run multiple differential evolution processes in parallel with different seeds
    seeds = [42, 123, 456, 789, 999, 111, 222, 333]

    # Generate multiple diversified initial configurations
    initial_configs = []
    for i in range(4):  # Create 4 different initialization strategies
        if i == 0:
            initial_configs.append(construct_honeycomb_packing())
        elif i == 1:
            initial_configs.append(construct_grid_packing())
        elif i == 2:
            initial_configs.append(construct_spiral_packing())
        else:  # Random configurations
            config = []
            for j in range(11):
                x = np.random.uniform(-4, 4)
                y = np.random.uniform(-4, 4)
                angle = np.random.uniform(0, 360)
                config.extend([x, y, angle])
            # Estimate outer radius
            max_dist = 0
            for j in range(11):
                cx, cy = config[3*j:3*j+2]
                dist = np.sqrt(cx**2 + cy**2) + UNIT_HEX_APOGEE
                max_dist = max(max_dist, dist)
            config.append(max_dist + 0.5)
            initial_configs.append(config)

    # Run parallel differential evolution
    results = Parallel(n_jobs=-1)(
        delayed(run_differential_evolution_with_seed)(seed, bounds)
        for seed in seeds
    )

    # Collect successful results and their fitness values
    successful_results = []
    for i, result in enumerate(results):
        if result is not None and result.success:
            successful_results.append((result, i))

    # Collect all results including from different initial configs
    all_results = successful_results.copy()

    # Now try local refinement on the initial configurations
    for initial_config in initial_configs:
        # Run local refinement on initial configs
        try:
            refined_result = run_local_refinement(initial_config, bounds)
            if refined_result is not None and refined_result.success:
                all_results.append((refined_result, f"initial_{len(all_results)}"))
        except:
            continue

    if not all_results:
        # If no successful optimization runs, fall back to original method
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

    # Find the best result among all successful runs
    best_result = min(all_results, key=lambda x: x[0].fun)
    best_de_result = best_result[0]

    # Final validation and formatting
    final_params = best_de_result.x
    inner_params = final_params[:-1]
    outer_radius = final_params[-1]

    # Validate solution
    n = 11
    inner_hexagons = []
    for i in range(n):
        x, y, angle = inner_params[3*i:3*i+3]
        hexagon = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(hexagon)

    # Create outer hexagon
    outer_hexagon = create_unit_hexagon((0, 0), 0)
    outer_coords = list(outer_hexagon.exterior.coords)
    scaled_coords = [(x*outer_radius, y*outer_radius) for x, y in outer_coords]
    outer_hexagon_scaled = Polygon(scaled_coords)

    # Check constraints
    containment_ok = True
    overlap_ok = True

    for hexagon in inner_hexagons:
        if not check_containment(hexagon, outer_hexagon_scaled):
            containment_ok = False
            break

    if containment_ok:
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    overlap_ok = False
                    break
            if not overlap_ok:
                break

    if containment_ok and overlap_ok:
        # Format output
        inner_hex_data = np.zeros((n, 3))
        for i in range(n):
            inner_hex_data[i] = inner_params[3*i:3*i+3]

        outer_hex_data = np.array([0, 0, 0])

        return inner_hex_data, outer_hex_data, outer_radius

    # Fallback to original method if validation fails
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