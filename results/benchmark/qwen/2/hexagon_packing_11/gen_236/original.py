# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import warnings

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
    return buffered_hex1.intersects(buffered2)

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

def construct_voronoi_packing():
    """Construct initial configuration using honeycomb-inspired approach"""
    # Use a more sophisticated honeycomb-like arrangement based on known good solutions
    # This arrangement places hexagons in a pattern that's known to work well for 11 hexagons
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

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses Voronoi-based optimization to find the best arrangement.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    try:
        # Generate initial configuration using Voronoi-inspired approach
        initial_guess = construct_voronoi_packing()

        # Set up bounds for optimization
        bounds = []
        # Bounds for inner hexagon positions (more constrained)
        for _ in range(11):
            bounds.extend([(-4.0, 4.0), (-4.0, 4.0), (0, 360)])  # x, y, angle
        # Bound for outer radius
        bounds.append((2.5, 7.0))  # Tightened range

        # Optimization settings with stricter tolerances
        options = {'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}

        # Use L-BFGS-B for fine-tuning with higher precision
        result = minimize(
            voronoi_based_objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )

        if result.success:
            final_params = result.x
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

    except Exception as e:
        warnings.warn(f"Voronoi optimization failed: {str(e)}")
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