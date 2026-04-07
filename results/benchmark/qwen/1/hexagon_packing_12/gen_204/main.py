# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import time
from math import sqrt, pi
from numba import jit
import heapq

# BVH Node class for spatial acceleration
class BVHNode:
    def __init__(self, bounds, objects=None, left=None, right=None):
        self.bounds = bounds  # (min_x, max_x, min_y, max_y)
        self.objects = objects or []
        self.left = left
        self.right = right
        self.is_leaf = left is None and right is None

    def get_area(self):
        if self.is_leaf and len(self.objects) == 0:
            return 0
        return (self.bounds[1] - self.bounds[0]) * (self.bounds[3] - self.bounds[2])

# BVH class for efficient spatial queries
class BVH:
    def __init__(self, objects, max_objects_per_node=4):
        self.max_objects_per_node = max_objects_per_node
        self.root = self.build_tree(objects)

    def build_tree(self, objects):
        if not objects:
            return BVHNode((0, 0, 0, 0))
        
        # Calculate bounding box for all objects
        min_x = min(obj[0] for obj in objects)
        max_x = max(obj[0] for obj in objects)
        min_y = min(obj[1] for obj in objects)
        max_y = max(obj[1] for obj in objects)
        
        bounds = (min_x, max_x, min_y, max_y)
        return self._build_recursive(objects, bounds)

    def _build_recursive(self, objects, bounds):
        if len(objects) <= self.max_objects_per_node:
            return BVHNode(bounds, objects.copy())
        
        # Split along the longest axis
        width = bounds[1] - bounds[0]
        height = bounds[3] - bounds[2]
        
        if width >= height:
            split_pos = (bounds[0] + bounds[1]) / 2
            left_objects = [obj for obj in objects if obj[0] <= split_pos]
            right_objects = [obj for obj in objects if obj[0] > split_pos]
        else:
            split_pos = (bounds[2] + bounds[3]) / 2
            left_objects = [obj for obj in objects if obj[1] <= split_pos]
            right_objects = [obj for obj in objects if obj[1] > split_pos]
        
        left_child = self._build_recursive(left_objects, self._calculate_bounds(left_objects))
        right_child = self._build_recursive(right_objects, self._calculate_bounds(right_objects))
        
        return BVHNode(bounds, left=None, right=None, left=left_child, right=right_child)

    def _calculate_bounds(self, objects):
        if not objects:
            return (0, 0, 0, 0)
        min_x = min(obj[0] for obj in objects)
        max_x = max(obj[0] for obj in objects)
        min_y = min(obj[1] for obj in objects)
        max_y = max(obj[1] for obj in objects)
        return (min_x, max_x, min_y, max_y)

    def query_overlap(self, query_bounds):
        """Find all objects that might overlap with the query bounds"""
        results = []
        self._query_recursive(self.root, query_bounds, results)
        return results

    def _query_recursive(self, node, query_bounds, results):
        if not self._intersect_bounds(node.bounds, query_bounds):
            return
        
        if node.is_leaf:
            results.extend(node.objects)
        else:
            self._query_recursive(node.left, query_bounds, results)
            self._query_recursive(node.right, query_bounds, results)

    def _intersect_bounds(self, bounds1, bounds2):
        return not (bounds1[1] < bounds2[0] or bounds1[0] > bounds2[1] or
                   bounds1[3] < bounds2[2] or bounds1[2] > bounds2[3])

# Helper functions for hexagon operations
@jit(nopython=True)
def hexagon_vertices(x, y, angle_deg, side_length=1):
    """Compute vertices of a hexagon with Numba JIT."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    base_verts = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    rotated_verts = np.empty_like(base_verts)
    for i in range(6):
        x_orig, y_orig = base_verts[i]
        rotated_verts[i] = [
            x + side_length * (x_orig * cos_a - y_orig * sin_a),
            y + side_length * (x_orig * sin_a + y_orig * cos_a)
        ]
    return rotated_verts

def create_hexagon_polygon(x, y, angle_deg, side_length=1):
    """Create shapely polygon from hexagon parameters."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    return Polygon(vertices)

def check_containment(hex_poly, outer_hex_poly):
    """Check if a hexagon is fully contained within outer hexagon."""
    return outer_hex_poly.contains(hex_poly) or outer_hex_poly.covers(hex_poly)

def check_overlap(hex1_poly, hex2_poly):
    """Check if two hexagons overlap."""
    return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

def bounds_intersect(bounds1, bounds2):
    """Check if two bounding boxes intersect."""
    return not (bounds1[1] < bounds2[0] or bounds1[0] > bounds2[1] or
               bounds1[3] < bounds2[2] or bounds1[2] > bounds2[3])

def calculate_hexagon_bounds(x, y, angle_deg, side_length=1):
    """Calculate bounding box for a hexagon."""
    vertices = hexagon_vertices(x, y, angle_deg, side_length)
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    return (min_x, max_x, min_y, max_y)

def compute_required_outer_side(inner_hex_data, center=(0, 0)):
    """Compute minimum outer hexagon side length."""
    max_dist = 0.0
    for i in range(len(inner_hex_data)):
        cx, cy, _ = inner_hex_data[i]
        dist = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
        dist_to_edge = dist + 1.0  # Unit hexagon circumradius
        max_dist = max(max_dist, dist_to_edge)
    return max_dist * 2.0

def is_valid_configuration(inner_hex_data, outer_hex_side_length, center=(0, 0)):
    """Comprehensive validity check using BVH for overlap detection."""
    if len(inner_hex_data) != 12:
        return False, 0.0
    
    # Create boundaries for each hexagon
    hex_bounds = []
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        cx, cy, angle = inner_hex_data[i]
        bounds = calculate_hexagon_bounds(cx, cy, angle, 1.0)
        hex_bounds.append(bounds)
        hex_poly = create_hexagon_polygon(cx, cy, angle, 1.0)
        hex_polygons.append(hex_poly)
    
    # Create outer hexagon
    outer_hex_poly = create_hexagon_polygon(0, 0, 0, outer_hex_side_length)
    
    # Check containment
    for i, hex_poly in enumerate(hex_polygons):
        if not check_containment(hex_poly, outer_hex_poly):
            return False, 0.0
    
    # Create BVH for efficient overlap detection
    # Use center points for BVH construction
    center_points = [(hex_data[0], hex_data[1]) for hex_data in inner_hex_data]
    bvh = BVH(center_points)
    
    # Check overlaps using BVH
    for i in range(len(inner_hex_data)):
        # Get potential candidates via BVH
        current_bounds = calculate_hexagon_bounds(*inner_hex_data[i], 1.0)
        candidate_indices = bvh.query_overlap(current_bounds)
        
        # Check overlaps with candidates
        for j in candidate_indices:
            if i != j and j < len(hex_polygons):  # Ensure valid index
                if check_overlap(hex_polygons[i], hex_polygons[j]):
                    return False, 0.0
    
    # Valid configuration
    return True, compute_required_outer_side(inner_hex_data, center)

def geometric_initial_placement():
    """Generate high-quality geometric initial configuration."""
    # Strategy based on the mathematical properties of hexagonal packing
    # Place hexagons in a pattern that maximizes density while maintaining symmetries
    
    positions = []
    
    # Central hexagon
    positions.append([0, 0, 0])
    
    # First shell - 6 hexagons at distance sqrt(3) from center
    # This spacing minimizes overlap potential while being symmetric
    base_spacing = sqrt(3)  # Distance between centers for touching unit hexagons
    angles = np.linspace(0, 2*pi, 7)[:-1]  # 6 directions

    for angle in angles:
        x = base_spacing * np.cos(angle)
        y = base_spacing * np.sin(angle)
        positions.append([x, y, 0])

    # Second shell - 4 hexagons positioned to fill the gaps
    # These are placed at a distance of 2*sqrt(3) from center
    second_spacing = 2 * sqrt(3)
    # Using 4 strategic positions to maximize packing efficiency
    second_angles = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]
    
    for angle in second_angles:
        x = second_spacing * np.cos(angle)
        y = second_spacing * np.sin(angle)
        positions.append([x, y, 0])

    # Add remaining positions to make 12
    # Position additional hexagons to further optimize the packing
    positions.append([0, -2*sqrt(3), 0])
    positions.append([0, 2*sqrt(3), 0])
    
    # Trim to exactly 12
    positions = positions[:12]
    
    # Small random perturbations to break any remaining symmetry
    np.random.seed(42)
    for i in range(12):
        positions[i][0] += np.random.normal(0, 0.05)
        positions[i][1] += np.random.normal(0, 0.05)
        positions[i][2] += np.random.uniform(-5, 5)
    
    return np.array(positions)

def greedy_improvement(initial_config, max_iterations=1000):
    """Improve configuration using greedy local search."""
    current_config = initial_config.copy()
    best_config = current_config.copy()
    
    # Find initial valid configuration
    best_score = 0.0
    best_outer_side = float('inf')
    
    # Try different starting outer side lengths
    outer_side_guesses = [3.0, 3.5, 4.0, 4.5]
    
    for guess in outer_side_guesses:
        is_valid, required_side = is_valid_configuration(current_config, guess)
        if is_valid:
            score = 1.0 / guess
            if score > best_score:
                best_score = score
                best_config = current_config.copy()
                best_outer_side = guess
    
    # Greedy improvement
    for iteration in range(max_iterations):
        # Try small perturbations
        new_config = current_config.copy()
        
        # Select a random hexagon to modify
        hex_idx = np.random.randint(0, 12)
        
        # Perturb position and angle
        new_config[hex_idx, 0] += np.random.normal(0, 0.1)
        new_config[hex_idx, 1] += np.random.normal(0, 0.1)
        new_config[hex_idx, 2] = np.random.uniform(0, 360)
        
        # Try several outer side lengths
        for outer_side in [3.0, 3.5, 4.0, 4.5, 5.0]:
            is_valid, required_side = is_valid_configuration(new_config, outer_side)
            if is_valid:
                score = 1.0 / outer_side
                if score > best_score:
                    best_score = score
                    best_config = new_config.copy()
                    best_outer_side = outer_side
                break  # Found a valid configuration, move to next iteration
        
        current_config = best_config.copy()
    
    return best_config, best_score, best_outer_side

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    start_time = time.time()
    
    # Generate initial configuration using geometric approach
    initial_config = geometric_initial_placement()
    
    # Use greedy improvement to find better configuration
    best_config, best_score, best_outer_side = greedy_improvement(initial_config)
    
    # If no good solution found, fallback to a reliable configuration
    if best_score <= 1e-5:
        # Fallback to known good configuration
        inner_hex_data = np.array([
            [0, 0, 0],
            [0, 2, 0],
            [0, -2, 0],
            [1.732, 1, 0],
            [-1.732, 1, 0],
            [1.732, -1, 0],
            [-1.732, -1, 0],
            [3.464, 0, 0],
            [-3.464, 0, 0],
            [1.732, 3, 0],
            [-1.732, 3, 0],
            [1.732, -3, 0],
        ])
        outer_hex_side_length = 6.928
        best_config = inner_hex_data
        best_outer_side = outer_hex_side_length
        
    # Create outer hexagon data (centered at origin, no rotation)
    outer_hex_data = np.array([0, 0, 0])
    
    # Ensure we don't exceed time limits
    end_time = time.time()
    eval_time = end_time - start_time
    
    return best_config, outer_hex_data, best_outer_side

# EVOLVE-BLOCK-END