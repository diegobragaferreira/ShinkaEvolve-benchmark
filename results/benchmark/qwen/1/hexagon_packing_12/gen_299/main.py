# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from joblib import Parallel, delayed
import time
from numba import njit
from scipy.spatial import cKDTree


@njit
def create_hexagon_vertices_numba(center_x, center_y, rotation_rad, side_length=1.0):
    """Create vertices of a regular hexagon using numba JIT for speed."""
    vertices = np.empty((6, 2))
    angle_step = 2 * np.pi / 6
    for i in range(6):
        angle = angle_step * i + rotation_rad
        x = center_x + side_length * np.cos(angle)
        y = center_y + side_length * np.sin(angle)
        vertices[i] = (x, y)
    return vertices


@njit
def point_in_hexagon_fast_numba(point_x, point_y, hex_center_x, hex_center_y, rotation_rad, side_length=1.0):
    """Fast point-in-hexagon test using numba JIT."""
    # Transform point to hexagon's coordinate system
    cos_rot = np.cos(rotation_rad)
    sin_rot = np.sin(rotation_rad)
    dx = point_x - hex_center_x
    dy = point_y - hex_center_y
    rot_x = dx * cos_rot + dy * sin_rot
    rot_y = -dx * sin_rot + dy * cos_rot

    # For unit hexagon, distance from center to edge = sqrt(3)/2
    edge_distance = side_length * np.sqrt(3) / 2
    half_side = side_length / 2.0

    # Check bounds
    if abs(rot_x) > edge_distance or abs(rot_y) > half_side:
        return False
    # Additional constraint for hexagon shape
    if abs(rot_x) + abs(rot_y) > side_length * np.sqrt(3):
        return False
    return True


@njit
def distance_point_to_segment(point_x, point_y, seg_start_x, seg_start_y, seg_end_x, seg_end_y):
    """Calculate the shortest distance from a point to a line segment."""
    px, py = point_x, point_y
    x1, y1 = seg_start_x, seg_start_y
    x2, y2 = seg_end_x, seg_end_y

    # Vector from line_start to line_end
    dx, dy = x2 - x1, y2 - y1

    # Length squared of line segment
    length_sq = dx*dx + dy*dy

    if length_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)

    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)


@njit
def hexagon_distance_numba(hex1_vertices, hex2_vertices):
    """Calculate minimum distance between two hexagons using numba JIT."""
    # For unit hexagons, we can quickly estimate minimum distance
    # by checking distance from one hexagon's vertices to the other's edges
    min_dist = 1e10

    for i in range(6):
        v1 = hex1_vertices[i]
        v2 = hex1_vertices[(i + 1) % 6]

        for j in range(6):
            v3 = hex2_vertices[j]
            v4 = hex2_vertices[(j + 1) % 6]

            dist = distance_point_to_segment(v3[0], v3[1], v1[0], v1[1], v2[0], v2[1])
            min_dist = min(min_dist, dist)

    return min_dist


def create_unit_hexagon_numba(center=(0,0), rotation=0):
    """Create a unit regular hexagon with given center and rotation using numba."""
    rotation_rad = rotation * np.pi / 180
    vertices = create_hexagon_vertices_numba(center[0], center[1], rotation_rad, 1.0)
    return Polygon(vertices)


def check_containment_numba(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon using numba."""
    # Check if all vertices of inner hex are inside outer hex
    for point in list(inner_hex.exterior.coords):
        if not outer_hex.contains(Point(point[0], point[1])):
            return False
    return True


def check_overlap_numba(hex1, hex2):
    """Check if two hexagons overlap using numba."""
    return hex1.intersects(hex2)


def build_hexagon_tree(hexagons):
    """Build spatial tree for faster overlap checking."""
    centers = []
    for hexagon in hexagons:
        vertices = list(hexagon.exterior.coords)
        center = np.mean(vertices[:-1], axis=0)  # Exclude repeated last vertex
        centers.append(center)
    return cKDTree(centers)


def fast_overlaps_check_numba(hexagons, tree=None, max_dist=2.0):
    """Fast overlap checking using spatial indexing."""
    if tree is None:
        tree = build_hexagon_tree(hexagons)

    # Get neighbors within max_dist
    centers = []
    for hexagon in hexagons:
        vertices = list(hexagon.exterior.coords)
        center = np.mean(vertices[:-1], axis=0)
        centers.append(center)

    # Check overlap efficiently using the spatial tree
    for i, center in enumerate(centers):
        nearby_indices = tree.query_ball_point(center, max_dist)
        for j in nearby_indices:
            if j > i:  # Only check each pair once
                if check_overlap_numba(hexagons[i], hexagons[j]):
                    return True
    return False


def evaluate_configuration_fast_numba(config):
    """
    Evaluate a configuration with fast constraint checking using numba and spatial acceleration.
    config: array of shape (37,) - [x1,y1,theta1,...,x12,y12,theta12,R]
    Returns negative inverse side length (to maximize inverse side length)
    """
    # Extract parameters
    positions_angles = config[:-1].reshape(-1, 3)
    outer_radius = config[-1]

    # Create outer hexagon
    outer_hex = create_unit_hexagon_numba((0, 0), 0)
    # Scale the outer hexagon to have side length = outer_radius
    scaled_outer_vertices = []
    for i in range(6):
        theta = i * np.pi / 3
        x = outer_radius * np.cos(theta)
        y = outer_radius * np.sin(theta)
        scaled_outer_vertices.append((x, y))
    outer_hex = Polygon(scaled_outer_vertices)

    # Create inner hexagons
    inner_hexagons = []
    for i in range(12):
        x, y, angle = positions_angles[i]
        inner_hex = create_unit_hexagon_numba((x, y), angle)
        inner_hexagons.append(inner_hex)

        # Check containment early - fast version
        if not check_containment_numba(inner_hex, outer_hex):
            return 1e10  # Penalty for violation

    # Fast overlap checking using spatial indexing
    if len(inner_hexagons) > 1:
        tree = build_hexagon_tree(inner_hexagons)
        if fast_overlaps_check_numba(inner_hexagons, tree):
            return 1e10  # Penalty for overlap

    # Return negative inverse side length (we want to maximize 1/R)
    return -1.0 / outer_radius


def get_initial_guess_symmetric():
    """Get a symmetric initial guess that leverages known packing symmetries."""
    # Create a configuration that respects 6-fold rotational symmetry and mirror symmetry
    # This starts closer to a known good configuration

    positions_angles = []

    # Central hexagon
    positions_angles.append([0.0, 0.0, 0.0])

    # First ring - 6 hexagons placed radially
    ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)  # 6 directions
    ring1_radius = 1.732  # Approximately sqrt(3) for good packing

    for angle in ring1_angles:
        x = ring1_radius * np.cos(angle)
        y = ring1_radius * np.sin(angle)
        positions_angles.append([x, y, 0.0])

    # Second ring - 6 hexagons
    ring2_angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 7, endpoint=False)  # Offset by π/6 for better packing
    ring2_radius = 3.464  # Approximately 2*sqrt(3)

    for angle in ring2_angles:
        x = ring2_radius * np.cos(angle)
        y = ring2_radius * np.sin(angle)
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
    Uses an enhanced optimization approach with spatial acceleration and symmetry.
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

    # Get symmetric initial configuration
    initial_guess = get_initial_guess_symmetric()

    # Phase 1: Coarse global optimization with larger population
    start_time = time.time()

    # Use differential evolution for global optimization with increased population
    result = differential_evolution(
        evaluate_configuration_fast_numba,
        bounds,
        maxiter=100,
        popsize=30,  # Larger population for better exploration
        seed=42,
        disp=False,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-6
    )

    # Phase 2: Local refinement with L-BFGS-B if needed
    if result.fun < -0.25:  # If we haven't reached target yet, do local refinement
        # Refine using L-BFGS-B
        refined_result = minimize(
            evaluate_configuration_fast_numba,
            result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-9}
        )
        if refined_result.fun < result.fun:
            result = refined_result

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