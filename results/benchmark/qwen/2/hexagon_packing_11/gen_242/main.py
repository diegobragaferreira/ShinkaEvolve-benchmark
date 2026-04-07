# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import time
from collections import deque
import itertools
from numba import jit, prange

# Constants
NUM_INNER_HEX = 11
UNIT_HEX_RADIUS = 1.0
HEX_VERTICES = 6

# JIT compiled helper functions for geometric operations
@jit(nopython=True)
def get_hexagon_vertices_jit(x, y, angle_deg, radius=1.0):
    """Get vertices of a regular hexagon given position and angle - JIT compiled"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

@jit(nopython=True)
def point_in_hexagon_fast_jit(point_x, point_y, hex_center_x, hex_center_y, hex_radius, angle_deg):
    """Fast point-in-hexagon test - JIT compiled"""
    # Transform point to hexagon's local coordinate system
    angle_rad = np.radians(angle_deg)
    rel_x = point_x - hex_center_x
    rel_y = point_y - hex_center_y
    
    # Rotate point back to align with hexagon axes
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rot_x = rel_x * cos_a - rel_y * sin_a
    rot_y = rel_x * sin_a + rel_y * cos_a
    
    # Simplified check: for a regular hexagon aligned with axes
    width_half = hex_radius * np.sqrt(3) / 2
    height_half = hex_radius * np.sqrt(3) / 2
    
    if abs(rot_x) > width_half or abs(rot_y) > height_half:
        return False
        
    # More precise check using hexagon boundaries
    if abs(rot_x) > width_half:
        return False
    if abs(rot_y) > hex_radius * 0.5 + (width_half - abs(rot_x)) * 0.5773502691896257:  # sqrt(3)/3
        return False
    
    return True

@jit(nopython=True)
def get_edges_jit(vertices):
    """Get edges from vertices"""
    edges = []
    n = len(vertices)
    for i in range(n):
        edges.append(vertices[i] - vertices[(i+1)%n])
    return np.array(edges)

@jit(nopython=True)
def project_polygon_onto_axis_jit(vertices, axis):
    """Project polygon onto axis and return min/max projections"""
    projections = []
    for vertex in vertices:
        proj = vertex[0] * axis[0] + vertex[1] * axis[1]
        projections.append(proj)
    
    min_proj = projections[0]
    max_proj = projections[0]
    for p in projections:
        if p < min_proj:
            min_proj = p
        if p > max_proj:
            max_proj = p
    
    return min_proj, max_proj

@jit(nopython=True)
def sat_collision_check_jit(hex1_vertices, hex2_vertices):
    """SAT-based overlap detection - much faster than Shapely"""
    # Get edges for both polygons
    edges1 = get_edges_jit(hex1_vertices)
    edges2 = get_edges_jit(hex2_vertices)
    
    # Get normals to all edges (perpendicular vectors)
    normals1 = []
    normals2 = []
    
    for edge in edges1:
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm_len = np.sqrt(normal[0]**2 + normal[1]**2)
        if norm_len > 1e-10:
            normal = normal / norm_len
        normals1.append(normal)
    
    for edge in edges2:
        # Normal vector (perpendicular to edge)
        normal = np.array([-edge[1], edge[0]])
        # Normalize
        norm_len = np.sqrt(normal[0]**2 + normal[1]**2)
        if norm_len > 1e-10:
            normal = normal / norm_len
        normals2.append(normal)
    
    # Test all axes
    all_normals = normals1 + normals2
    
    for axis in all_normals:
        min1, max1 = project_polygon_onto_axis_jit(hex1_vertices, axis)
        min2, max2 = project_polygon_onto_axis_jit(hex2_vertices, axis)
        
        # Check for overlap
        if max1 < min2 or max2 < min1:
            return False  # No overlap along this axis
    
    return True  # Overlap detected

class HexagonTreeSearcher:
    """Implements tree-based search for optimal hexagon packing"""
    
    def __init__(self):
        self.best_solution = None
        self.best_inv_side_length = 0.0
        self.best_side_length = float('inf')
        self.max_depth = 20
        self.search_nodes = 0
        self.max_nodes = 500000
        
    def calculate_outer_hex_side_length(self, inner_hex_data, outer_hex_center=(0, 0)):
        """Calculate minimum outer hexagon side length needed to contain all inner hexagons"""
        if len(inner_hex_data) == 0:
            return 1000.0
        
        max_distance = 0.0
        center_x, center_y = outer_hex_center
        
        # For each inner hexagon, check all 6 vertices
        for i in range(len(inner_hex_data)):
            x, y, angle = inner_hex_data[i]
            vertices = get_hexagon_vertices_jit(x, y, angle)
            
            # Calculate distance from center to each vertex
            for vertex in vertices:
                distance = np.sqrt((vertex[0] - center_x)**2 + (vertex[1] - center_y)**2)
                max_distance = max(max_distance, distance)
        
        # Account for hexagon radius
        return max_distance * 2.0 / np.sqrt(3)  # Convert circumradius to side length
    
    def check_containment_fast(self, hex_vertices, outer_center=(0, 0), outer_radius=1000.0):
        """Fast containment check"""
        outer_center_x, outer_center_y = outer_center
        # Check if all vertices are within the outer hexagon
        outer_circumradius = outer_radius * np.sqrt(3) / 2
        
        for vertex in hex_vertices:
            dist_from_center = np.sqrt((vertex[0] - outer_center_x)**2 + (vertex[1] - outer_center_y)**2)
            if dist_from_center > outer_circumradius:
                return False
        return True
    
    def is_valid_configuration(self, inner_hex_data, outer_side_length):
        """Validate configuration for collisions and containment"""
        num_hex = len(inner_hex_data)
        
        # Fast containment check for all inner hexagons
        for i in range(num_hex):
            center_x, center_y, rotation = inner_hex_data[i]
            vertices = get_hexagon_vertices_jit(center_x, center_y, rotation)
            if not self.check_containment_fast(vertices, (0, 0), outer_side_length * np.sqrt(3) / 2):
                return False, 0.0
        
        # Check pairwise collisions using SAT
        for i in range(num_hex):
            for j in range(i + 1, num_hex):
                center_x1, center_y1, rotation1 = inner_hex_data[i]
                center_x2, center_y2, rotation2 = inner_hex_data[j]
                
                vertices1 = get_hexagon_vertices_jit(center_x1, center_y1, rotation1)
                vertices2 = get_hexagon_vertices_jit(center_x2, center_y2, rotation2)
                
                if sat_collision_check_jit(vertices1, vertices2):
                    return False, 0.0
        
        # Valid configuration
        return True, 1.0 / outer_side_length
    
    def generate_initial_configurations(self):
        """Generate initial configurations using geometric heuristics"""
        configs = []
        
        # Configuration 1: Base symmetric hexagonal pattern
        base_positions = [
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ]
        configs.append(np.array(base_positions))
        
        # Configuration 2: Spiral arrangement
        spiral_positions = []
        for i in range(11):
            angle = i * 0.7  # Slightly different spacing
            radius = 0.25 * i
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            spiral_positions.append([x, y, 0])
        configs.append(np.array(spiral_positions))
        
        # Configuration 3: Concentric ring
        ring_positions = [[0, 0, 0]]  # center
        ring_radius = 1.8
        for i in range(1, 11):
            angle = (i - 1) * 2 * np.pi / 10
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            ring_positions.append([x, y, 0])
        configs.append(np.array(ring_positions))
        
        # Configuration 4: Grid-like with symmetry
        grid_positions = []
        # Center
        grid_positions.append([0, 0, 0])
        # 4 corners
        for i in range(4):
            angle = i * np.pi / 2
            x = 2.5 * np.cos(angle)
            y = 2.5 * np.sin(angle)
            grid_positions.append([x, y, 0])
        # 4 midpoints
        for i in range(4):
            angle = i * np.pi / 2 + np.pi / 4
            x = 1.5 * np.cos(angle)
            y = 1.5 * np.sin(angle)
            grid_positions.append([x, y, 0])
        # 2 additional positions
        grid_positions.append([2.0, 0, 0])
        grid_positions.append([0, 2.0, 0])
        configs.append(np.array(grid_positions))
        
        # Configuration 5: Random but structured
        random_positions = []
        for i in range(11):
            # Distribute in a way that keeps them reasonably spaced
            if i == 0:
                random_positions.append([0, 0, 0])
            else:
                angle = (i - 1) * 2 * np.pi / 10 + np.random.uniform(-0.1, 0.1)
                # Spread out to reduce overlap risk
                radius = 0.5 + np.random.uniform(1.0, 2.5)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                rotation = np.random.uniform(0, 360)
                random_positions.append([x, y, rotation])
        configs.append(np.array(random_positions))
        
        return configs
    
    def tree_search_step(self, current_config, depth, remaining_positions, outer_side_length, 
                        current_best_inv_side_length):
        """Recursive search step with early pruning"""
        self.search_nodes += 1
        if self.search_nodes > self.max_nodes:
            return False
        
        # If reached max depth or no positions left, validate and update best solution
        if depth >= self.max_depth or len(remaining_positions) == 0:
            is_valid, inv_side_length = self.is_valid_configuration(current_config, outer_side_length)
            if is_valid and inv_side_length > current_best_inv_side_length:
                self.best_solution = current_config.copy()
                self.best_inv_side_length = inv_side_length
                self.best_side_length = 1.0 / inv_side_length
                return True
            return False
        
        # Early pruning: Check if current configuration can possibly beat best
        if depth == 0:
            # For root node, check if it's even valid
            is_valid, _ = self.is_valid_configuration(current_config[:depth+1], outer_side_length)
            if not is_valid:
                return False
        
        # If we have enough positions, try to place the next one
        if len(remaining_positions) > 0:
            # Get the next position to place
            next_pos_idx = len(current_config) - len(remaining_positions)
            
            # Try to place the hexagon with various rotations and positions within bounds
            # We'll use a structured search instead of exhaustive enumeration
            candidate_positions = self.generate_candidate_positions(
                current_config, 
                remaining_positions[next_pos_idx], 
                outer_side_length
            )
            
            for candidate_pos in candidate_positions:
                # Place the hexagon and recurse
                new_config = current_config.copy()
                new_config[len(current_config) - len(remaining_positions)] = candidate_pos
                
                # If this looks promising, continue search
                if depth < 3:  # Only do deeper search up to certain depth
                    # Pruning based on already known best
                    try:
                        est_side_length = self.calculate_outer_hex_side_length(new_config, (0, 0))
                        if est_side_length * 1.1 > outer_side_length:  # Allow 10% margin
                            self.tree_search_step(
                                new_config, 
                                depth + 1, 
                                remaining_positions[1:], 
                                outer_side_length, 
                                current_best_inv_side_length
                            )
                    except:
                        pass
                elif depth >= 3:
                    # For later depths, just try a few random placements
                    new_config = current_config.copy()
                    new_config[len(current_config) - len(remaining_positions)] = candidate_pos
                    is_valid, inv_side_length = self.is_valid_configuration(new_config, outer_side_length)
                    if is_valid and inv_side_length > current_best_inv_side_length:
                        self.best_solution = new_config.copy()
                        self.best_inv_side_length = inv_side_length
                        self.best_side_length = 1.0 / inv_side_length
                        return True
                        
        return False
    
    def generate_candidate_positions(self, existing_config, target_position, outer_side_length, 
                                   num_candidates=5):
        """Generate candidate positions for next hexagon based on existing configuration"""
        candidates = []
        
        # Base position
        base_x, base_y, base_angle = target_position
        
        # Add some randomness to position and rotation to explore neighborhood
        for i in range(num_candidates):
            # Vary position around base
            dx = np.random.normal(0, 0.3) if i > 0 else 0
            dy = np.random.normal(0, 0.3) if i > 0 else 0
            # Vary rotation
            dangle = np.random.normal(0, 10) if i > 0 else 0
            
            # Make sure placement is within bounds
            candidate_x = base_x + dx
            candidate_y = base_y + dy
            candidate_angle = (base_angle + dangle) % 360
            
            # Make sure it's not too far from the center to maintain feasibility
            distance_to_center = np.sqrt(candidate_x**2 + candidate_y**2)
            if distance_to_center < outer_side_length:
                candidates.append([candidate_x, candidate_y, candidate_angle])
                
        # Always include original position as one candidate
        candidates.append([base_x, base_y, base_angle])
        
        return candidates
    
    def hierarchical_search(self, initial_configs, max_time=170):
        """Perform hierarchical search on multiple initial configurations"""
        start_time = time.time()
        best_found = False
        
        # Process each initial configuration
        for i, initial_config in enumerate(initial_configs[:3]):  # Limit to first 3 for speed
            if time.time() - start_time > max_time:
                break
                
            # Estimate outer side length for this configuration
            est_side_length = self.calculate_outer_hex_side_length(initial_config, (0, 0))
            
            # Check if initial configuration is valid
            is_valid, _ = self.is_valid_configuration(initial_config, est_side_length)
            if is_valid and is_valid:
                # Update best if necessary
                if 1.0 / est_side_length > self.best_inv_side_length:
                    self.best_solution = initial_config.copy()
                    self.best_inv_side_length = 1.0 / est_side_length
                    self.best_side_length = est_side_length
                    best_found = True
                    
            # Try refining with tree search for a subset of configurations
            if i < 2 and not best_found:  # Only do tree search for first 2 configurations as sample
                self.search_nodes = 0
                self.tree_search_step(
                    initial_config.copy(), 
                    0, 
                    list(range(len(initial_config))), 
                    est_side_length, 
                    self.best_inv_side_length
                )
                
        return self.best_solution, self.best_side_length, self.best_inv_side_length

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize searcher
    searcher = HexagonTreeSearcher()
    
    # Generate multiple initial configurations
    initial_configs = searcher.generate_initial_configurations()
    
    # Perform hierarchical search
    best_solution, best_side_length, best_inv_side_length = searcher.hierarchical_search(
        initial_configs, max_time=170
    )
    
    # If we found a good solution, return it
    if best_solution is not None and best_inv_side_length > 0.1:
        outer_hex_data = np.array([0, 0, 0])
        return best_solution, outer_hex_data, best_side_length
    
    # Fallback to traditional approach if no good solution found
    # Original configuration from the simple grid
    initial_config = np.array([
        [0, 0, 0],  # center
        [-2.5, 0, 0],  # left
        [2.5, 0, 0],  # right
        [-1.25, 2.17, 0],  # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0],  # bottom-left
        [1.25, -2.17, 0],  # bottom-right
        [-3.75, 2.17, 0],  # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0],  # far bottom-left
        [3.75, -2.17, 0],  # far bottom-right
    ])
    
    # Set reasonable initial outer hexagon size based on configuration
    max_dist_from_center = 0
    for i in range(len(initial_config)):
        center_x, center_y, _ = initial_config[i]
        dist = np.sqrt(center_x**2 + center_y**2)
        max_dist_from_center = max(max_dist_from_center, dist + 1.0)  # Add radius margin
    
    # Outer hexagon should have side length slightly larger than max distance
    outer_hex_side_length = max_dist_from_center * 1.2  # 20% margin
    
    # Evaluate this configuration
    is_valid, _ = searcher.is_valid_configuration(initial_config, outer_hex_side_length)
    
    # If initial configuration is invalid due to overlap or containment,
    # we fall back to the simpler approach but with better validation
    if not is_valid:
        # Fallback to a basic valid configuration
        inner_hex_data = np.array([
            [0, 0, 0],  # center
            [-2.5, 0, 0],  # left
            [2.5, 0, 0],  # right
            [-1.25, 2.17, 0],  # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],  # bottom-left
            [1.25, -2.17, 0],  # bottom-right
            [-3.75, 2.17, 0],  # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],  # far bottom-left
            [3.75, -2.17, 0],  # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0  # fallback value
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Since we've confirmed initial config works, we can return it
    inner_hex_data = initial_config.copy()
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END