# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon, Point
from joblib import Parallel, delayed
import time
from numba import jit, njit
import math
from collections import defaultdict

class QuadTreeNode:
    def __init__(self, bounds, capacity=4):
        self.bounds = bounds  # [x_min, y_min, x_max, y_max]
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.children = [None, None, None, None]  # nw, ne, sw, se
        
    def subdivide(self):
        x_min, y_min, x_max, y_max = self.bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        
        # Create four child quadrants
        self.children[0] = QuadTreeNode([x_min, y_min, x_mid, y_mid], self.capacity)  # nw
        self.children[1] = QuadTreeNode([x_mid, y_min, x_max, y_mid], self.capacity)  # ne
        self.children[2] = QuadTreeNode([x_min, y_mid, x_mid, y_max], self.capacity)  # sw
        self.children[3] = QuadTreeNode([x_mid, y_mid, x_max, y_max], self.capacity)  # se
        
        self.divided = True
        
        # Redistribute points
        for point in self.points:
            self.insert_point_into_children(point)
        self.points = []
        
    def insert_point_into_children(self, point):
        for child in self.children:
            if child.contains_point(point):
                child.insert(point)
                return True
        return False
        
    def contains_point(self, point):
        x, y = point
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
        
    def insert(self, point):
        if not self.contains_point(point):
            return False
            
        if len(self.points) < self.capacity and not self.divided:
            self.points.append(point)
            return True
            
        if not self.divided:
            self.subdivide()
            
        for child in self.children:
            if child.insert(point):
                return True
        return False
        
    def query_range(self, range_bounds):
        x_min, y_min, x_max, y_max = range_bounds
        found_points = []
        
        if not self.intersects_range(range_bounds):
            return found_points
            
        for point in self.points:
            x, y = point
            if x_min <= x <= x_max and y_min <= y <= y_max:
                found_points.append(point)
                
        if self.divided:
            for child in self.children:
                found_points.extend(child.query_range(range_bounds))
                
        return found_points
        
    def intersects_range(self, range_bounds):
        x_min, y_min, x_max, y_max = range_bounds
        bounds_x_min, bounds_y_min, bounds_x_max, bounds_y_max = self.bounds
        
        return not (x_max < bounds_x_min or x_min > bounds_x_max or 
                   y_max < bounds_y_min or y_min > bounds_y_max)

class QuadTree:
    def __init__(self, bounds, capacity=4):
        self.root = QuadTreeNode(bounds, capacity)
        
    def insert(self, point):
        self.root.insert(point)
        
    def query_range(self, range_bounds):
        return self.root.query_range(range_bounds)

@njit
def distance_point_to_point(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)

@njit
def distance_point_to_line(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    
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
def point_in_hexagon_fast(point_x, point_y, hex_center_x, hex_center_y, rotation, side_length):
    # Transform point to hexagon's coordinate system
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)
    dx = point_x - hex_center_x
    dy = point_y - hex_center_y
    rot_x = dx * cos_rot + dy * sin_rot
    rot_y = -dx * sin_rot + dy * cos_rot

    # Distance from center to edge in x and y directions
    edge_distance_x = side_length * np.sqrt(3) / 2
    edge_distance_y = side_length * 0.5

    # Check if point is within bounds
    return abs(rot_x) <= edge_distance_x and abs(rot_y) <= edge_distance_y and \
           abs(rot_x) + abs(rot_y) <= side_length * np.sqrt(3)

def create_unit_hexagon(center=(0,0), rotation=0):
    """Create a unit regular hexagon with given center and rotation."""
    angle = rotation * np.pi / 180
    # Vertices of a unit hexagon centered at origin
    hex_vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = np.cos(theta)
        y = np.sin(theta)
        hex_vertices.append((x + center[0], y + center[1]))
    return Polygon(hex_vertices)

def create_hexagon_vertices(center=(0,0), rotation=0, side_length=1):
    """Create vertices of a regular hexagon with given center, rotation, and side length."""
    angle = rotation * np.pi / 180
    hex_vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = center[0] + side_length * np.cos(theta)
        y = center[1] + side_length * np.sin(theta)
        hex_vertices.append((x, y))
    return hex_vertices

def check_containment(inner_hex, outer_hex):
    """Check if inner hexagon is fully contained within outer hexagon."""
    # Check if all vertices of inner hex are inside outer hex
    for point in list(inner_hex.exterior.coords):
        if not outer_hex.contains(Point(point)):
            return False
    return True

def check_overlap(hex1, hex2):
    """Check if two hexagons overlap."""
    return hex1.intersects(hex2)

def fast_overlap_check(hex1_vertices, hex2_vertices):
    """Fast overlap check using bounding circle approximation."""
    # Calculate centers and radii
    hex1_center = np.mean(hex1_vertices, axis=0)
    hex2_center = np.mean(hex2_vertices, axis=0)
    
    # Approximate radius as half the diagonal of the hexagon (max distance from center to vertex)
    hex_radius = 1.0  # Unit hexagon
    
    # Distance between centers
    dist_centers = np.linalg.norm(hex1_center - hex2_center)
    
    # If circles don't intersect, no overlap
    if dist_centers > 2 * hex_radius:
        return False
    
    # Actually check for overlap if necessary
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def build_quadtree_for_hexagons(hexagons):
    """Build a quadtree for faster overlap checking."""
    if len(hexagons) == 0:
        return None
    
    # Determine bounds of all hexagons
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for hexagon in hexagons:
        vertices = list(hexagon.exterior.coords)
        for x, y in vertices:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    # Add padding
    padding = 1.0
    bounds = [min_x - padding, min_y - padding, max_x + padding, max_y + padding]
    
    quadtree = QuadTree(bounds)
    centers = []
    for hexagon in hexagons:
        vertices = list(hexagon.exterior.coords)
        center = np.mean(vertices[:-1], axis=0)  # Exclude repeated last vertex
        centers.append(tuple(center))
        quadtree.insert(tuple(center))
    
    return quadtree, centers

def parallel_overlap_check_quadtree(hexagons, quadtree, centers, start_idx, end_idx, max_dist=2.0):
    """Parallel overlap checking using quadtree for neighborhood filtering."""
    overlaps = []
    
    for i in range(start_idx, end_idx):
        if i >= len(hexagons):
            break
        # Get neighbors within max_dist using quadtree
        center = centers[i]
        x, y = center
        query_bounds = [x - max_dist, y - max_dist, x + max_dist, y + max_dist]
        nearby_indices = []
        
        # Manual query since we don't have direct access to quadtree API
        # In real implementation, this would use quadtree's query_range
        for j, other_center in enumerate(centers):
            if i != j:
                dist = distance_point_to_point(center[0], center[1], other_center[0], other_center[1])
                if dist <= max_dist:
                    nearby_indices.append(j)
        
        for j in nearby_indices:
            if j > i:  # Only check each pair once
                if check_overlap(hexagons[i], hexagons[j]):
                    overlaps.append((i, j))
    return overlaps

def evaluate_configuration_fast(config):
    """
    Evaluate a configuration with fast constraint checking.
    config: array of shape (37,) - [x1,y1,theta1,...,x12,y12,theta12,R]
    Returns negative inverse side length (to maximize inverse side length)
    """
    # Extract parameters
    positions_angles = config[:-1].reshape(-1, 3)
    outer_radius = config[-1]
    
    # Create outer hexagon
    outer_hex = create_unit_hexagon((0, 0), 0)
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
        inner_hex = create_unit_hexagon((x, y), angle)
        inner_hexagons.append(inner_hex)

        # Check containment early
        if not check_containment(inner_hex, outer_hex):
            return 1e10  # Penalty for violation

    # Fast overlap checking using quadtree
    if len(inner_hexagons) > 1:
        quadtree, centers = build_quadtree_for_hexagons(inner_hexagons)
        
        # Use joblib for parallel overlap checking with spatial indexing
        num_pairs = 12 * 11 // 2  # Number of unique pairs
        chunk_size = max(1, num_pairs // 4)  # Process 4 chunks
        
        overlap_results = Parallel(n_jobs=-1)(
            delayed(parallel_overlap_check_quadtree)(inner_hexagons, quadtree, centers, i*chunk_size, min((i+1)*chunk_size, len(inner_hexagons)))
            for i in range(4)
        )
        
        # Check if any overlaps were found
        for result in overlap_results:
            if result:
                return 1e10  # Penalty for overlap

    # Return negative inverse side length (we want to maximize 1/R)
    return -1.0 / outer_radius

def get_initial_guess_symmetric():
    """Generate a highly symmetric initial configuration."""
    # Create a configuration with rotational symmetry properties
    positions_angles = []
    
    # Central hexagon
    positions_angles.append([0.0, 0.0, 0.0])
    
    # Ring 1 - 6 hexagons evenly spaced around central hexagon
    ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)  # 7 points but we'll use first 6
    for i in range(6):
        angle = ring1_angles[i]
        x = 2.0 * np.cos(angle)
        y = 2.0 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
    
    # Ring 2 - 6 hexagons at larger radius, rotated to maximize packing
    ring2_angles = np.linspace(np.pi/6, 2*np.pi + np.pi/6, 7, endpoint=False)  # Offset by π/6
    for i in range(6):
        angle = ring2_angles[i]
        x = 3.5 * np.cos(angle)
        y = 3.5 * np.sin(angle)
        positions_angles.append([x, y, 0.0])
        
    # Set initial radius slightly larger than estimated minimum
    initial_radius = 6.2

    # Flatten for optimization
    flat_config = np.array(positions_angles).flatten()
    flat_config = np.append(flat_config, initial_radius)

    return flat_config

def get_initial_guess_hexagonal_lattice():
    """Generate initial guess using hexagonal lattice principles."""
    # Use a more sophisticated lattice arrangement
    positions_angles = []
    
    # Central hexagon
    positions_angles.append([0.0, 0.0, 0.0])
    
    # Radial rings with increasing spacing
    ring_radii = [2.0, 3.0, 4.0]  # Three rings
    ring_sizes = [6, 6, 6]  # Six hexagons per ring
    
    for ring_idx, (radius, size) in enumerate(zip(ring_radii, ring_sizes)):
        angles = np.linspace(0, 2*np.pi, size+1, endpoint=False)
        for i in range(size):
            angle = angles[i]
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions_angles.append([x, y, 0.0])
            
    # Ensure exactly 12 hexagons
    positions_angles = positions_angles[:12]
    
    # Add reasonable starting outer radius
    initial_radius = 7.5

    # Flatten for optimization
    flat_config = np.array(positions_angles).flatten()
    flat_config = np.append(flat_config, initial_radius)

    return flat_config

def generate_symmetric_config(initial_positions_angles, symmetry_type="rotational"):
    """Enforce symmetric constraints on the configuration."""
    # For rotational symmetry (C6), rotate the positions by 60 degree increments
    if symmetry_type == "rotational":
        # First six positions form one symmetric set
        # Positions 1-6 are the radial positions
        # Positions 7-12 are duplicates with 60-degree rotations
        positions_angles = initial_positions_angles.copy()
        
        # For a C6 symmetric pattern, we only need to specify 2 positions per ring
        # and let the algorithm replicate them
        return positions_angles[:12]
    
    return initial_positions_angles

def adaptive_penalty_scoring(obj_val, penalty_factor=100.0, adapt_threshold=1e-5):
    """Apply adaptive penalty based on objective value."""
    if obj_val > 1e5 or obj_val < -1e5:  # Extreme values - likely constraint violations
        return penalty_factor * 1000.0
    elif obj_val < adapt_threshold:  # Near constraint violation threshold
        return obj_val - penalty_factor * 10.0  # Slight penalty
    return obj_val

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a hybrid evolutionary algorithm with spatial acceleration and symmetry awareness.
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

    start_time = time.time()
    
    # Phase 1: Coarse global optimization with large population
    initial_guess = get_initial_guess_hexagonal_lattice()
    
    de_kwargs_stage1 = {
        'func': evaluate_configuration_fast,
        'bounds': bounds,
        'maxiter': 100,
        'popsize': 35,  # Large population for global exploration
        'seed': 42,
        'disp': False,
        'mutation': (0.5, 1.0),
        'recombination': 0.8,
        'tol': 1e-5
    }
    
    result1 = differential_evolution(**de_kwargs_stage1)
    
    # Phase 2: Medium exploration phase
    de_kwargs_stage2 = {
        'func': evaluate_configuration_fast,
        'bounds': bounds,
        'maxiter': 80,
        'popsize': 25,  # Moderate population
        'seed': 42,
        'disp': False,
        'mutation': (0.7, 1.0),
        'recombination': 0.7,
        'tol': 1e-6
    }
    
    result2 = differential_evolution(**de_kwargs_stage2)
    
    # Phase 3: Fine tuning and local refinement
    de_kwargs_stage3 = {
        'func': evaluate_configuration_fast,
        'bounds': bounds,
        'maxiter': 60,
        'popsize': 20,  # Smaller population for focused search
        'seed': 42,
        'disp': False,
        'mutation': (0.8, 1.0),
        'recombination': 0.5,
        'tol': 1e-7
    }
    
    result3 = differential_evolution(**de_kwargs_stage3)
    
    # Select best result from all stages
    best_result = min([result1, result2, result3], key=lambda r: r.fun)
    
    # Final refinement with L-BFGS-B if needed
    if best_result.fun < -0.24:  # If we haven't reached target yet, do local refinement
        refined_result = minimize(
            evaluate_configuration_fast,
            best_result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 30, 'ftol': 1e-9}
        )
        if refined_result.fun < best_result.fun:
            best_result = refined_result

    end_time = time.time()

    # Extract results
    final_config = best_result.x
    positions_angles = final_config[:-1].reshape(-1, 3)
    outer_hex_side_length = final_config[-1]

    # Convert back to required format
    # The inner hex data is positions_angles
    inner_hex_data = positions_angles.copy()

    # Outer hex is centered at origin
    outer_hex_data = np.array([0, 0, 0])

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END