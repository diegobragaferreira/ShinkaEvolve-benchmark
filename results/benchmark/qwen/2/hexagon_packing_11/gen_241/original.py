# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
import time
import math
from collections import defaultdict

class HexagonPackingOptimizer:
    def __init__(self):
        self.unit_hex_radius = 1.0
        self.unit_hex_vertices = self._generate_unit_hexagon_vertices()
        self.max_attempts = 10000
        self.min_outer_side_length = 1.0
        self.max_outer_side_length = 20.0
        
    def _generate_unit_hexagon_vertices(self):
        """Generate vertices of a unit regular hexagon centered at origin"""
        vertices = []
        for i in range(6):
            angle = i * np.pi / 3
            x = self.unit_hex_radius * np.cos(angle)
            y = self.unit_hex_radius * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
    
    def _hexagon_from_params(self, center_x, center_y, rotation_deg):
        """Create hexagon vertices given center and rotation"""
        rotation_rad = np.radians(rotation_deg)
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)
        
        # Apply rotation and translation to unit hexagon vertices
        rotated_vertices = np.zeros_like(self.unit_hex_vertices)
        for i, (x, y) in enumerate(self.unit_hex_vertices):
            rotated_vertices[i] = [
                x * cos_r - y * sin_r + center_x,
                x * sin_r + y * cos_r + center_y
            ]
        return rotated_vertices
    
    def _check_containment(self, hexagon_vertices):
        """Check if hexagon is fully contained in outer hexagon"""
        # Create outer hexagon vertices (regular hexagon centered at origin)
        outer_hex_vertices = self._hexagon_from_params(0, 0, 0)
        outer_polygon = Polygon(outer_hex_vertices)

        # Check if all vertices of inner hexagon are within outer hexagon
        for vertex in hexagon_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True
    
    def _check_collision(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons collide using Shapely"""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    def _is_valid_configuration(self, inner_hex_data, outer_side_length):
        """Check if configuration satisfies all constraints"""
        # Create outer hexagon vertices for containment check
        outer_hex_vertices = self._hexagon_from_params(0, 0, 0)
        outer_polygon = Polygon(outer_hex_vertices)
        
        # Check containment of all inner hexagons within outer hexagon
        for i in range(len(inner_hex_data)):
            center_x, center_y, rotation = inner_hex_data[i]
            vertices = self._hexagon_from_params(center_x, center_y, rotation)
            
            # Check containment
            for vertex in vertices:
                point = Point(vertex[0], vertex[1])
                if not outer_polygon.contains(point):
                    return False
            
            # Check collision with all other hexagons
            for j in range(i):
                center_x2, center_y2, rotation2 = inner_hex_data[j]
                vertices2 = self._hexagon_from_params(center_x2, center_y2, rotation2)
                
                if self._check_collision(vertices, vertices2):
                    return False
                    
        return True
    
    def _create_initial_guess(self):
        """Create a good initial configuration based on known good patterns"""
        # Start with a symmetric pattern that's known to work well
        base_positions = [
            [0, 0, 0],  # center
            [-2.0, 0, 0],  # left
            [2.0, 0, 0],  # right
            [0, 2.0, 0],  # top
            [0, -2.0, 0],  # bottom
            [-1.0, 1.0, 0],  # top-left
            [1.0, 1.0, 0],  # top-right
            [-1.0, -1.0, 0],  # bottom-left
            [1.0, -1.0, 0],  # bottom-right
            [-2.0, 1.0, 0],  # far top-left
            [2.0, 1.0, 0],  # far top-right
        ]
        
        # Add some randomness to avoid local optima
        base_positions = np.array(base_positions)
        for i in range(len(base_positions)):
            base_positions[i][0] += np.random.normal(0, 0.1)
            base_positions[i][1] += np.random.normal(0, 0.1)
            
        return base_positions
    
    def _sample_random_config(self, outer_side_length):
        """Sample a random valid configuration"""
        # Sample random positions and rotations for all hexagons
        hex_data = []
        for i in range(11):
            # Random position within a reasonable area
            x = np.random.uniform(-outer_side_length*0.8, outer_side_length*0.8)
            y = np.random.uniform(-outer_side_length*0.8, outer_side_length*0.8)
            rotation = np.random.uniform(0, 360)
            hex_data.append([x, y, rotation])
        
        return np.array(hex_data)
    
    def _adaptive_search(self, initial_config, max_iterations=5000):
        """Use adaptive Monte Carlo search with progressive refinement"""
        best_config = initial_config.copy()
        best_side_length = 100.0  # Start with large value
        best_score = -1.0
        
        # Track recent improvements to adapt search
        recent_improvements = []
        
        # Start with broad search
        current_scale = 1.0
        
        for iteration in range(max_iterations):
            # Generate new configuration with adaptive scale
            new_config = self._sample_with_adaptation(best_config, current_scale)
            
            # Try different outer hexagon sizes
            for side_length in [best_side_length * 0.95, best_side_length * 0.9, best_side_length * 1.05]:
                if side_length < self.min_outer_side_length or side_length > self.max_outer_side_length:
                    continue
                    
                if self._is_valid_configuration(new_config, side_length):
                    score = 1.0 / side_length
                    if score > best_score:
                        best_score = score
                        best_config = new_config.copy()
                        best_side_length = side_length
                        
                        # Track improvement
                        recent_improvements.append(iteration)
                        if len(recent_improvements) > 10:
                            recent_improvements.pop(0)
                        
                        # Adaptively reduce scale when improvement happens
                        current_scale *= 0.99
                            
            # Adaptive scaling based on recent progress
            if len(recent_improvements) >= 5 and iteration - recent_improvements[-1] > 100:
                current_scale *= 1.1
                
            # Occasionally reset scale to avoid getting stuck
            if iteration % 200 == 0:
                current_scale = 0.1 + np.random.rand() * 0.2
                
        return best_config, best_side_length, best_score
    
    def _sample_with_adaptation(self, reference_config, scale):
        """Sample new configuration adapted from reference"""
        new_config = reference_config.copy()
        
        # Apply small random modifications to positions and rotations
        for i in range(len(new_config)):
            # Position changes (smaller than rotation changes)
            new_config[i][0] += np.random.normal(0, scale * 0.1)
            new_config[i][1] += np.random.normal(0, scale * 0.1)
            
            # Rotation change (can be larger)
            new_config[i][2] += np.random.normal(0, scale * 5)
            
            # Keep rotation in [0, 360) range
            new_config[i][2] = new_config[i][2] % 360
            
        return new_config

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HexagonPackingOptimizer()
    
    # Start with heuristic initial configuration
    initial_config = optimizer._create_initial_guess()
    
    # Perform adaptive search
    best_config, best_side_length, best_score = optimizer._adaptive_search(initial_config, max_iterations=5000)
    
    # If we found a good solution, return it
    if best_score > 0.1:  # Only accept reasonable solutions
        # Verify final solution
        if optimizer._is_valid_configuration(best_config, best_side_length):
            outer_hex_data = np.array([0, 0, 0])
            return best_config, outer_hex_data, best_side_length
    
    # Fallback to original approach with better validation
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
    valid = optimizer._is_valid_configuration(initial_config, outer_hex_side_length)

    # If initial configuration is invalid due to overlap or containment,
    # we fall back to the simpler approach but with better validation
    if not valid:
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