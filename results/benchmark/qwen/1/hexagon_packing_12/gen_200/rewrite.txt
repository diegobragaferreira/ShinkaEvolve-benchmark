# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point
from numba import jit, prange
import time
from collections import namedtuple
import warnings

# Define a structure for hexagon data
HexagonData = namedtuple('HexagonData', ['x', 'y', 'rotation'])

@jit(nopython=True, parallel=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Vertices of regular hexagon with side length 1 centered at origin
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])

    # Rotate and translate
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]

    return rotated_verts

@jit(nopython=True)
def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Calculate distance from point to line segment."""
    # Vector from (x1,y1) to (x2,y2)
    dx = x2 - x1
    dy = y2 - y1
    
    # Length squared of line segment
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        # Line segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0, min(1, t))  # Clamp projection to line segment
    
    # Find closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    # Distance to closest point
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True)
def compute_min_distance_hexagon_hexagon(h1_x, h1_y, h1_angle, h2_x, h2_y, h2_angle):
    """Compute minimum distance between two hexagons using analytical approach."""
    v1 = hexagon_vertices(h1_x, h1_y, h1_angle)
    v2 = hexagon_vertices(h2_x, h2_y, h2_angle)
    
    min_dist = np.inf
    # Check vertex-to-vertex distances
    for i in range(6):
        for j in range(6):
            dist = np.sqrt((v1[i,0]-v2[j,0])**2 + (v1[i,1]-v2[j,1])**2)
            if dist < min_dist:
                min_dist = dist
    
    # Check vertex-to-edge distances
    for i in range(6):
        for j in range(6):
            # Distance from vertex v1[i] to edge v2[j]-v2[(j+1)%6]
            dist = distance_point_to_line(v1[i,0], v1[i,1], v2[j,0], v2[j,1], v2[(j+1)%6,0], v2[(j+1)%6,1])
            if dist < min_dist:
                min_dist = dist
            
            # Distance from vertex v2[j] to edge v1[i]-v1[(i+1)%6]
            dist = distance_point_to_line(v2[j,0], v2[j,1], v1[i,0], v1[i,1], v1[(i+1)%6,0], v1[(i+1)%6,1])
            if dist < min_dist:
                min_dist = dist
    
    return min_dist

def compute_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Convert hexagon parameters to shapely polygon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def fast_check_overlap_pair_fast(hex1, hex2):
    """Fast preliminary overlap check using bounding circles before detailed Shapely check."""
    # Calculate centers and radii
    hex1_center = np.mean([list(p) for p in hex1.exterior.coords[:-1]], axis=0)
    hex2_center = np.mean([list(p) for p in hex2.exterior.coords[:-1]], axis=0)

    # Distance between centers
    dist = np.linalg.norm(hex1_center - hex2_center)

    # For unit hexagons, the circumradius is 1 and inradius is sqrt(3)/2
    # If distance is greater than sum of radii (2), they don't overlap
    if dist > 2.0:
        return False

    # Perform detailed overlap check only if close enough
    return hex1.intersects(hex2)

def efficient_parallel_overlap_check(hexagons, tree, max_distance=2.0):
    """Efficient overlap checking using spatial indexing."""
    overlaps = []
    n_hexagons = len(hexagons)

    # Query nearby hexagons using cKDTree
    for i in range(n_hexagons):
        # Find neighbors within max_distance
        nearby_indices = tree.query_ball_point(list(hexagons[i].exterior.centroid.coords)[0], max_distance)

        # Check overlaps only with nearby hexagons
        for j in nearby_indices:
            if i < j:  # Only check each pair once
                if fast_check_overlap_pair_fast(hexagons[i], hexagons[j]):
                    overlaps.append((i, j))

    return overlaps

def evaluate_configuration_parallel(config):
    """
    Evaluate a configuration with parallel constraint checking.
    config: array of shape (37,) - [x1,y1,theta1,...,x12,y12,theta12,R]
    Returns negative inverse side length (to maximize inverse side length)
    """
    # Extract parameters
    positions_angles = config[:-1].reshape(-1, 3)
    outer_radius = config[-1]

    # Create outer hexagon
    outer_hex = Polygon([
        (outer_radius * np.cos(i * np.pi / 3), outer_radius * np.sin(i * np.pi / 3))
        for i in range(6)
    ])

    # Create inner hexagons
    inner_hexagons = []
    for i in range(12):
        x, y, angle = positions_angles[i]
        inner_hex = compute_hexagon_polygon(x, y, angle)
        inner_hexagons.append(inner_hex)

        # Check containment early
        if not outer_hex.contains(inner_hex.centroid):
            return 1e10  # Penalty for violation

    # Build spatial index for efficient overlap checking
    hex_centers = []
    for hexagon in inner_hexagons:
        center = list(hexagon.exterior.centroid.coords)[0]
        hex_centers.append(center)

    tree = cKDTree(hex_centers)

    # Efficiently check pairwise overlaps using spatial indexing
    overlaps = efficient_parallel_overlap_check(inner_hexagons, tree)

    # Check if any overlaps were found
    if overlaps:
        return 1e10  # Penalty for overlap

    # Return negative inverse side length (we want to maximize 1/R)
    return -1.0 / outer_radius

def get_initial_guess_better():
    """Get a better initial guess based on known hexagon packing patterns"""
    # Start with a known dense configuration
    # Arrange in a hexagonal pattern with strategic positioning
    positions_angles = []

    # Central hexagon
    positions_angles.append([0.0, 0.0, 0.0])

    # First ring (6 hexagons) - distance sqrt(3)
    for i in range(6):
        angle = i * np.pi/3
        x = 1.732 * np.cos(angle)  # ~sqrt(3)
        y = 1.732 * np.sin(angle)
        positions_angles.append([x, y, 0.0])

    # Second ring (6 hexagons) - distance 2*sqrt(3) with offset
    for i in range(6):
        angle = i * np.pi/3 + np.pi/6
        x = 3.464 * np.cos(angle)  # ~2*sqrt(3)
        y = 3.464 * np.sin(angle)
        positions_angles.append([x, y, 0.0])

    # Add reasonable starting outer radius
    initial_radius = 5.5

    # Flatten for optimization
    flat_config = np.array(positions_angles).flatten()
    flat_config = np.append(flat_config, initial_radius)

    return flat_config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses an enhanced optimization approach.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define bounds for optimization
    # Positions: x,y in [-10, 10], angles in [0, 360]
    # Outer radius should be reasonable
    bounds = []
    for _ in range(12):
        bounds.extend([(-10, 10), (-10, 10), (0, 360)])  # x, y, angle
    bounds.append((2.0, 15.0))  # outer_radius

    # Get initial configuration
    initial_guess = get_initial_guess_better()

    # Phase 1: Coarse global optimization with larger population
    start_time = time.time()

    # Use differential evolution for global optimization with increased population
    try:
        result = differential_evolution(
            evaluate_configuration_parallel,
            bounds,
            maxiter=100,
            popsize=30,  # Larger population for better exploration
            seed=42,
            disp=False,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-6
        )
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
        result = type('obj', (object,), {'x': initial_guess, 'fun': 1e10})()

    # Phase 2: Local refinement with L-BFGS-B if needed
    if result.fun < -0.25:  # If we haven't reached target yet, do local refinement
        try:
            refined_result = minimize(
                evaluate_configuration_parallel,
                result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            if refined_result.fun < result.fun:
                result = refined_result
        except Exception as e:
            warnings.warn(f"L-BFGS-B refinement failed: {e}")

    end_time = time.time()

    # Extract results
    final_config = result.x
    positions_angles = final_config[:-1].reshape(-1, 3)
    outer_hex_side_length = final_config[-1]

    # Convert back to required format
    # The inner hex data is positions_angles
    inner_hex_data = positions_angles.copy()

    # Outer hex is centered at origin
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END