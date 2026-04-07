# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import itertools

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0  # Distance from center to corner for unit hexagon
UNIT_HEXAGON_WIDTH = 2.0  # Diameter of unit hexagon
MAX_EVAL_TIME = 180.0  # seconds

# Precomputed unit hexagon vertices (centered at origin)
def get_unit_hexagon_vertices():
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    return vertices

UNIT_HEXAGON_VERTICES = get_unit_hexagon_vertices()

def rotate_point(point, angle_rad):
    """Rotate a point around origin"""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([point[0]*cos_a - point[1]*sin_a, point[0]*sin_a + point[1]*cos_a])

def hexagon_vertices(center, angle_rad, scale=1.0):
    """Get vertices of a hexagon at given position and rotation"""
    rotated_vertices = np.array([rotate_point(v, angle_rad) for v in UNIT_HEXAGON_VERTICES])
    return rotated_vertices * scale + np.array(center)

def point_in_polygon(point, polygon):
    """Fast point-in-polygon check"""
    return polygon.contains(Point(point))

def is_contained_in_outer_hexagon(hexagon_vertices_list, outer_center, outer_angle, outer_radius):
    """Check if hexagon is fully contained in outer hexagon using optimized approach"""
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Fast check: test if all vertices are inside outer polygon
    for vertex in hexagon_vertices_list:
        if not point_in_polygon(vertex, outer_polygon):
            return False
    return True

def check_overlap_fast(hex1_vertices, hex2_vertices):
    """Fast overlap check using spatial indexing"""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    except:
        # Fallback for degenerate cases
        return False

def calculate_outer_hexagon_radius(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
    max_dist = 0
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])

        # Get all vertices of this hexagon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Calculate max distance from outer center to any vertex
        for vertex in vertices:
            dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
            max_dist = max(max_dist, dist)

    return max_dist

def validate_solution(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution: check containment and non-overlap"""
    # Precompute all hexagon polygons once for reuse
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center = inner_hex_data[i][:2]
        angle = np.radians(inner_hex_data[i][2])
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
        hex_polygons.append(Polygon(vertices))

    # Calculate outer radius once
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # Check containment using the outer hexagon polygon
    outer_vertices = hexagon_vertices(outer_center, outer_angle, outer_radius)
    outer_polygon = Polygon(outer_vertices)

    # Check if all inner hexagons are contained within outer hexagon
    for hex_poly in hex_polygons:
        # Fast check: if any vertex is outside, reject
        for vertex in hex_poly.exterior.coords[:-1]:  # Exclude closing vertex
            if not outer_polygon.contains(Point(vertex)):
                return False

    # Check overlaps using optimized spatial indexing
    # Create a simple grid-based spatial hash for O(1) neighbor lookup
    grid_size = 2.5  # Size of grid cells (slightly larger than hexagon diameter)
    grid = {}

    # Place each hexagon into grid cells
    for i, hex_poly in enumerate(hex_polygons):
        # Get bounding box of hexagon
        min_x, min_y, max_x, max_y = hex_poly.bounds
        # Determine grid cell range
        min_grid_x = int(min_x // grid_size)
        max_grid_x = int(max_x // grid_size)
        min_grid_y = int(min_y // grid_size)
        max_grid_y = int(max_y // grid_size)

        # Add to all overlapping grid cells
        for gx in range(min_grid_x, max_grid_x + 1):
            for gy in range(min_grid_y, max_grid_y + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)

    # Check overlaps only between hexagons in the same or adjacent cells
    for i in range(len(hex_polygons)):
        # Get the grid cells this hexagon occupies
        min_x, min_y, max_x, max_y = hex_polygons[i].bounds
        min_grid_x = int(min_x // grid_size)
        max_grid_x = int(max_x // grid_size)
        min_grid_y = int(min_y // grid_size)
        max_grid_y = int(max_y // grid_size)

        # Check neighbors in this and adjacent cells
        for gx in range(min_grid_x - 1, max_grid_x + 2):
            for gy in range(min_grid_y - 1, max_grid_y + 2):
                if (gx, gy) in grid:
                    for j in grid[(gx, gy)]:
                        # Only check pairs once and skip self
                        if i < j:
                            try:
                                if hex_polygons[i].intersects(hex_polygons[j]):
                                    return False
                            except:
                                return False
    return True

def evaluate_fitness(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness (negative of outer hexagon radius for maximization)"""
    # Calculate minimum outer radius needed
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # If solution is invalid, penalize heavily
    if not validate_solution(inner_hex_data, outer_center, outer_angle):
        return -1e10  # Very poor fitness

    # Return negative radius (we want to minimize radius, so maximize negative value)
    return -outer_radius

def generate_lattice_points():
    """Generate a coarse lattice of possible hexagon positions and rotations"""
    # Create a 3D lattice in (x,y,rotation) space
    x_range = np.arange(-6, 7, 0.5)
    y_range = np.arange(-6, 7, 0.5)
    rot_range = np.arange(0, 360, 30)  # Every 30 degrees

    # Generate all combinations
    lattice_points = []
    for x, y, rot in itertools.product(x_range, y_range, rot_range):
        lattice_points.append([x, y, rot])

    return np.array(lattice_points)

def is_valid_configuration(hex_data, exclude_idx=None):
    """Quick validity check for a configuration"""
    if exclude_idx is not None:
        # Test all hexagons except the excluded one
        test_hexes = [hex_data[i] for i in range(len(hex_data)) if i != exclude_idx]
    else:
        test_hexes = hex_data

    # Check if all hexagons are valid individually
    for i in range(len(test_hexes)):
        center = test_hexes[i][:2]
        angle = np.radians(test_hexes[i][2])

        # Create hexagon polygon
        vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

        # Check containment in reasonable bounds
        for vertex in vertices:
            if abs(vertex[0]) > 10 or abs(vertex[1]) > 10:
                return False

    # Quick overlap check between first few hexagons
    for i in range(min(len(test_hexes), 5)):
        for j in range(i+1, min(len(test_hexes), 5)):
            center1 = test_hexes[i][:2]
            angle1 = np.radians(test_hexes[i][2])
            center2 = test_hexes[j][:2]
            angle2 = np.radians(test_hexes[j][2])

            vertices1 = hexagon_vertices(center1, angle1, UNIT_HEXAGON_RADIUS)
            vertices2 = hexagon_vertices(center2, angle2, UNIT_HEXAGON_RADIUS)

            try:
                poly1 = Polygon(vertices1)
                poly2 = Polygon(vertices2)
                if poly1.intersects(poly2):
                    return False
            except:
                # If we encounter geometry errors, assume invalid
                return False

    return True

def optimize_outer_hexagon_radius(hex_data):
    """Refine the outer hexagon radius using local optimization"""
    def objective(r):
        # r is the radius of the outer hexagon
        # We want to minimize this (minimize outer radius)
        if r <= 0:
            return 1e10

        # Calculate outer radius needed
        max_dist = 0
        for i in range(len(hex_data)):
            center = hex_data[i][:2]
            angle = np.radians(hex_data[i][2])
            vertices = hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
            for vertex in vertices:
                dist = np.linalg.norm(np.array(vertex))
                max_dist = max(max_dist, dist)

        # Make sure we're still within bounds (safety check)
        if max_dist > 100:
            return 1e10

        return max_dist

    # Use scipy minimize to find optimized radius
    try:
        result = minimize(objective, [calculate_outer_hexagon_radius(hex_data)],
                         method='Nelder-Mead', options={'maxiter': 50})
        return max(result.x[0], 0.1)  # Ensure positive radius
    except:
        return calculate_outer_hexagon_radius(hex_data)

def hexagon_lattice_search():
    """Main lattice-based searching algorithm"""
    # Start with a good baseline configuration
    base_config = [
        [0, 0, 0],           # center
        [-2.5, 0, 0],       # left
        [2.5, 0, 0],        # right
        [-1.25, 2.17, 0],   # top-left
        [1.25, 2.17, 0],    # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],   # bottom-right
        [-3.75, 2.17, 0],   # far top-left
        [3.75, 2.17, 0],    # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],   # far bottom-right
    ]

    best_config = np.array(base_config)
    best_radius = calculate_outer_hexagon_radius(best_config)

    # Generate lattice points for exploration
    lattice_points = generate_lattice_points()

    # Try different configurations based on lattice points
    for _ in range(1000):  # Limit iterations
        # Randomly select some positions from lattice
        candidate_config = []
        used_indices = set()

        # Try to build a valid configuration
        for i in range(NUM_INNER_HEXAGONS):
            # Pick a random lattice point for this hexagon
            if len(lattice_points) > 0:
                idx = random.randint(0, min(100, len(lattice_points)-1))
                candidate_config.append(lattice_points[idx])
            else:
                # Fallback to random placement
                candidate_config.append([
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(0, 360)
                ])

        # Convert to array
        candidate_array = np.array(candidate_config)

        # Quick validation
        if is_valid_configuration(candidate_array):
            # Full validation
            if validate_solution(candidate_array):
                radius = calculate_outer_hexagon_radius(candidate_array)
                if radius < best_radius:
                    best_radius = radius
                    best_config = candidate_array.copy()

    return best_config, best_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Use the lattice-based approach for better exploration
    best_inner_config, best_radius = hexagon_lattice_search()

    # Apply local optimization to refine the solution
    try:
        refined_config = best_inner_config.copy()

        # Simple local refinement: tweak positions to try to decrease outer radius
        for _ in range(500):  # Local refinement iterations
            if time.time() - start_time > MAX_EVAL_TIME - 1:
                break

            # Create small perturbations
            test_config = refined_config.copy()

            # Pick a random hexagon to perturb
            hex_idx = random.randint(0, NUM_INNER_HEXAGONS-1)

            # Small random changes
            test_config[hex_idx][0] += random.uniform(-0.1, 0.1)
            test_config[hex_idx][1] += random.uniform(-0.1, 0.1)
            test_config[hex_idx][2] += random.uniform(-5, 5)
            test_config[hex_idx][2] %= 360

            # Validate and accept if better
            if validate_solution(test_config):
                test_radius = calculate_outer_hexagon_radius(test_config)
                if test_radius < best_radius:
                    best_radius = test_radius
                    refined_config = test_config.copy()

        best_inner_config = refined_config
    except:
        pass  # If refinement fails, keep the best config found

    # Validate final solution
    if not validate_solution(best_inner_config):
        # Fallback to known good configuration
        best_inner_config = np.array([
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
        ])
        best_radius = 8.0

    # Return result
    inner_hex_data = best_inner_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = best_radius

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END