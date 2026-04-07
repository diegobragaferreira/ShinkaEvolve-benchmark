# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import warnings
from joblib import Parallel, delayed
import time

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
    """Check if hexagon is fully contained within outer_hexagon"""
    # Check if all vertices of inner hexagon are inside outer hexagon
    for point in hexagon.exterior.coords[:-1]:  # Exclude closing point
        if not outer_hexagon.contains(Point(point)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    return hex1.intersects(hex2)

def calculate_tight_outer_radius(inner_params):
    """Calculate tightest possible outer hexagon radius using actual vertex positions"""
    # Get all hexagon vertices and find bounding circle
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

    # Find centroid of all vertices
    centroid = np.mean(vertices_array, axis=0)

    # Calculate distances from centroid to all vertices
    distances = np.sqrt(np.sum((vertices_array - centroid)**2, axis=1))

    # Outer radius is the maximum distance plus a small margin for numerical stability
    outer_radius = np.max(distances) + 1e-6

    return outer_radius

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

    # Calculate tight outer radius for better objective
    actual_tight_radius = calculate_tight_outer_radius(inner_params)

    # Return negative of inverse radius to minimize (maximize 1/outer_radius)
    return -1.0 / actual_tight_radius

def compute_voronoi_regions(centers, outer_radius):
    """Compute Voronoi regions for given centers within outer hexagon"""
    # Add boundary points to ensure finite regions
    boundary_points = []
    # Create a square around the outer hexagon
    half_side = outer_radius * 1.2
    boundary_points.extend([
        [-half_side, -half_side],
        [half_side, -half_side],
        [half_side, half_side],
        [-half_side, half_side]
    ])

    all_points = np.array(centers + boundary_points)

    try:
        vor = Voronoi(all_points)
        return vor
    except:
        # Fallback if Voronoi computation fails
        return None

def construct_voronoi_packing(seed=None):
    """Construct initial configuration using Voronoi-inspired approach"""
    if seed is not None:
        np.random.seed(seed)

    # Start with a more structured configuration based on hexagonal lattice
    centers = [
        (0, 0),           # center
        (-2.0, 0),        # left
        (2.0, 0),         # right
        (0, 2.0),         # top
        (0, -2.0),        # bottom
        (-1.5, 1.5),      # top-left
        (1.5, 1.5),       # top-right
        (-1.5, -1.5),     # bottom-left
        (1.5, -1.5),      # bottom-right
        (-2.5, 0),        # further left
        (2.5, 0),         # further right
    ]

    # Add some randomness to avoid symmetric solutions
    initial_guess = []
    for i, (cx, cy) in enumerate(centers):
        # Add small random variation
        jitter_x = np.random.normal(0, 0.1)
        jitter_y = np.random.normal(0, 0.1)
        initial_guess.extend([cx + jitter_x, cy + jitter_y, np.random.uniform(0, 30)])  # Smaller angle range

    # Estimate outer radius - based on hexagon positions and size
    max_dist = 0
    for cx, cy in centers:
        dist = np.sqrt(cx**2 + cy**2) + UNIT_HEX_APOGEE
        max_dist = max(max_dist, dist)

    initial_guess.append(max_dist + 0.3)  # Add margin

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

    if not successful_results:
        # If no successful differential evolution runs, fall back to a single run
        try:
            np.random.seed(42)
            de_result = differential_evolution(
                voronoi_based_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )

            if de_result.success:
                # Local refinement
                refined_result = run_local_refinement(de_result.x, bounds)
                if refined_result is not None and refined_result.success:
                    final_params = refined_result.x
                else:
                    final_params = de_result.x
            else:
                raise Exception("Differential evolution failed")
        except Exception as e:
            warnings.warn(f"Single optimization failed: {str(e)}")
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

    else:
        # Find the best result among all successful runs
        best_result = min(successful_results, key=lambda x: x[0].fun)
        best_de_result = best_result[0]

        # Local refinement
        refined_result = run_local_refinement(best_de_result.x, bounds)
        if refined_result is not None and refined_result.success:
            final_params = refined_result.x
        else:
            final_params = best_de_result.x

    # Extract final results
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