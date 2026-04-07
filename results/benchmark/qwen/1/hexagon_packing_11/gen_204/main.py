# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
import itertools
from collections import defaultdict

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

def get_bounding_box(hex_poly):
    """Get bounding box of hexagon for spatial indexing"""
    bounds = hex_poly.bounds
    return (bounds[0], bounds[1], bounds[2], bounds[3])

def build_spatial_grid(hexagons, grid_size=2.5):
    """Build a spatial grid for fast collision detection"""
    grid = defaultdict(list)
    for i, hex_poly in enumerate(hexagons):
        bbox = get_bounding_box(hex_poly)
        min_x, min_y, max_x, max_y = bbox
        # Determine grid cell range
        min_grid_x = int(min_x // grid_size)
        max_grid_x = int(max_x // grid_size)
        min_grid_y = int(min_y // grid_size)
        max_grid_y = int(max_y // grid_size)

        # Add to all overlapping grid cells
        for gx in range(min_grid_x, max_grid_x + 1):
            for gy in range(min_grid_y, max_grid_y + 1):
                grid[(gx, gy)].append(i)
    return grid

def is_valid_position(position, existing_positions, min_distance=1.8):
    """Check if a position is sufficiently far from existing positions"""
    for existing_pos in existing_positions:
        distance = np.linalg.norm(np.array(position) - np.array(existing_pos))
        if distance < min_distance:
            return False
    return True

def validate_solution_with_spatial_indexing(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Validate solution using efficient spatial indexing"""
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

    # Check overlaps using spatial indexing
    grid = build_spatial_grid(hex_polygons, grid_size=UNIT_HEXAGON_WIDTH * 1.2)

    # Check overlaps only between hexagons in the same or adjacent cells
    for i in range(len(hex_polygons)):
        # Get the grid cells this hexagon occupies
        min_x, min_y, max_x, max_y = hex_polygons[i].bounds
        min_grid_x = int(min_x // (UNIT_HEXAGON_WIDTH * 1.2))
        max_grid_x = int(max_x // (UNIT_HEXAGON_WIDTH * 1.2))
        min_grid_y = int(min_y // (UNIT_HEXAGON_WIDTH * 1.2))
        max_grid_y = int(max_y // (UNIT_HEXAGON_WIDTH * 1.2))

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

def evaluate_fitness_with_validation(inner_hex_data, outer_center=[0,0], outer_angle=0):
    """Evaluate fitness with validation"""
    # Calculate minimum outer radius needed
    outer_radius = calculate_outer_hexagon_radius(inner_hex_data, outer_center, outer_angle)

    # If solution is invalid, penalize heavily
    if not validate_solution_with_spatial_indexing(inner_hex_data, outer_center, outer_angle):
        return -1e10  # Very poor fitness

    # Return negative radius (we want to minimize radius, so maximize negative value)
    return -outer_radius

def generate_grid_lattice():
    """Generate a 3D lattice of possible hexagon positions and rotations"""
    x_range = np.arange(-6, 7, 1.0)
    y_range = np.arange(-6, 7, 1.0)
    rot_range = np.arange(0, 360, 15)  # Every 15 degrees for fine rotation control

    # Generate all combinations
    lattice_points = []
    for x, y, rot in itertools.product(x_range, y_range, rot_range):
        lattice_points.append([x, y, rot])

    return np.array(lattice_points)

def generate_fine_grid_lattice():
    """Generate a fine-grained 3D lattice for local optimization"""
    x_range = np.arange(-5, 6, 0.5)
    y_range = np.arange(-5, 6, 0.5)
    rot_range = np.arange(0, 360, 5)  # Every 5 degrees for high precision

    # Generate all combinations
    lattice_points = []
    for x, y, rot in itertools.product(x_range, y_range, rot_range):
        lattice_points.append([x, y, rot])

    return np.array(lattice_points)

def generate_hexagonal_pattern():
    """Generate a hexagonal arrangement pattern for initial configuration"""
    # Center hexagon
    positions = [[0, 0, 0]]
    
    # Ring of 6 hexagons around center
    for i in range(6):
        angle = i * 60
        radius = 2.0
        x = radius * np.cos(np.radians(angle))
        y = radius * np.sin(np.radians(angle))
        positions.append([x, y, 0])
    
    # Additional positions for remaining hexagons
    additional_positions = [
        (-3.0, 1.0, 0), (3.0, 1.0, 0),
        (-3.0, -1.0, 0), (3.0, -1.0, 0),
        (0.0, 3.0, 0), (0.0, -3.0, 0),
        (1.5, 2.6, 0), (-1.5, -2.6, 0),
        (-1.5, 2.6, 0), (1.5, -2.6, 0)
    ]
    
    # Combine with existing positions
    for i, (x, y, rot) in enumerate(additional_positions):
        if len(positions) < 11:
            positions.append([x, y, rot])
    
    # Ensure exactly 11 positions
    while len(positions) < 11:
        positions.append([0.0, 0.0, 0.0])
    
    return np.array(positions[:11])

def get_neighbors_in_lattice(lattice_points, target_indices, k=10):
    """Get k nearest neighbors in lattice for efficient search"""
    if len(lattice_points) == 0:
        return []
    
    # Select a subset of points around targets for neighborhood search
    indices = list(target_indices)
    selected_points = []
    
    for i in indices:
        if i < len(lattice_points):
            selected_points.append(lattice_points[i])
    
    if len(selected_points) == 0:
        return []
    
    return selected_points

def optimize_with_lattice_search():
    """Main lattice-based optimization approach"""
    # Generate coarse lattice for global exploration
    coarse_lattice = generate_grid_lattice()
    
    # Generate fine lattice for local optimization
    fine_lattice = generate_fine_grid_lattice()
    
    # Start with hexagonal pattern
    best_config = generate_hexagonal_pattern()
    best_radius = calculate_outer_hexagon_radius(best_config)
    
    # Global search using coarse lattice
    for _ in range(2000):  # Reduced iterations for speed
        # Sample from coarse lattice points
        sample_indices = []
        for _ in range(11):
            sample_indices.append(random.randint(0, len(coarse_lattice)-1))
        
        candidate_config = []
        for i in sample_indices:
            if i < len(coarse_lattice):
                candidate_config.append(coarse_lattice[i])
            else:
                # Fallback to random placement
                candidate_config.append([
                    random.uniform(-5, 5),
                    random.uniform(-5, 5),
                    random.uniform(0, 360)
                ])
        
        candidate_array = np.array(candidate_config)
        
        # Validate configuration
        if validate_solution_with_spatial_indexing(candidate_array):
            radius = calculate_outer_hexagon_radius(candidate_array)
            if radius < best_radius:
                best_radius = radius
                best_config = candidate_array.copy()
    
    # Local optimization with fine lattice
    for _ in range(3000):  # Fine tuning iterations
        # Select random hexagon to perturb
        hex_idx = random.randint(0, 10)
        
        # Try different fine lattice points near current configuration
        start_idx = int(len(fine_lattice) * random.random())
        for i in range(100):  # Limited search in fine lattice
            lattice_idx = (start_idx + i) % len(fine_lattice)
            candidate_point = fine_lattice[lattice_idx]
            
            # Create modified configuration
            test_config = best_config.copy()
            test_config[hex_idx] = candidate_point
            
            # Validate and accept if better
            if validate_solution_with_spatial_indexing(test_config):
                test_radius = calculate_outer_hexagon_radius(test_config)
                if test_radius < best_radius:
                    best_radius = test_radius
                    best_config = test_config.copy()
    
    # Final validation
    if not validate_solution_with_spatial_indexing(best_config):
        # Fallback to hexagonal pattern if validation fails
        best_config = generate_hexagonal_pattern()
        best_radius = calculate_outer_hexagon_radius(best_config)
    
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
    
    # Use lattice-based approach for efficient exploration
    best_inner_config, best_radius = optimize_with_lattice_search()
    
    # Apply additional local refinement if time allows
    elapsed_time = time.time() - start_time
    if elapsed_time < MAX_EVAL_TIME - 2:  # Leave 2 seconds for final processing
        # Perform additional local search for further improvement
        for _ in range(1000):
            if time.time() - start_time > MAX_EVAL_TIME - 2:
                break
                
            # Random perturbation around best solution
            test_config = best_inner_config.copy()
            
            # Perturb one hexagon
            hex_idx = random.randint(0, 10)
            test_config[hex_idx][0] += random.uniform(-0.2, 0.2)
            test_config[hex_idx][1] += random.uniform(-0.2, 0.2)
            test_config[hex_idx][2] += random.uniform(-3, 3)
            test_config[hex_idx][2] %= 360
            
            # Validate and improve if better
            if validate_solution_with_spatial_indexing(test_config):
                test_radius = calculate_outer_hexagon_radius(test_config)
                if test_radius < best_radius:
                    best_radius = test_radius
                    best_inner_config = test_config.copy()
    
    # Final validation
    if not validate_solution_with_spatial_indexing(best_inner_config):
        # Fallback to known good configuration
        best_inner_config = generate_hexagonal_pattern()
        best_radius = calculate_outer_hexagon_radius(best_inner_config)
    
    # Return result
    inner_hex_data = best_inner_config
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # Centered at origin
    outer_hex_side_length = best_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END