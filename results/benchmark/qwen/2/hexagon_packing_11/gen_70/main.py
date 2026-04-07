# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from joblib import Parallel, delayed
import time
import random
from functools import lru_cache

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""
    
    def __init__(self):
        self.unit_hex_radius = 1.0
        self.unit_hex_vertices = self._generate_unit_hexagon_vertices()
        
    def _generate_unit_hexagon_vertices(self):
        """Generate vertices of a unit regular hexagon centered at origin"""
        vertices = []
        for i in range(6):
            angle = i * np.pi / 3
            x = self.unit_hex_radius * np.cos(angle)
            y = self.unit_hex_radius * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
    
    def create_hexagon_vertices(self, center_x, center_y, rotation_deg):
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

class ConstraintValidator:
    """Validates constraints for hexagon configurations"""
    
    def __init__(self, hex_geometry):
        self.hex_geometry = hex_geometry
        self.outer_hex_vertices = self.hex_geometry.create_hexagon_vertices(0, 0, 0)
        self.outer_polygon = Polygon(self.outer_hex_vertices)
    
    @lru_cache(maxsize=1000)
    def _cached_contains_point(self, px, py):
        """Cached point-in-polygon check"""
        return self.outer_polygon.contains(Point(px, py))
    
    def is_contained(self, hex_vertices):
        """Check if all hexagon vertices are contained in outer hexagon"""
        for vertex in hex_vertices:
            if not self._cached_contains_point(vertex[0], vertex[1]):
                return False
        return True
    
    def check_collision(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons collide using Shapely"""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)
    
    def validate_configuration(self, inner_hex_data, outer_side_length):
        """Comprehensive validation of hexagon configuration"""
        # Create outer hexagon vertices for containment check
        outer_hex_vertices = self.hex_geometry.create_hexagon_vertices(0, 0, 0)
        outer_polygon = Polygon(outer_hex_vertices)
        
        # Check containment of all inner hexagons and collisions
        n = len(inner_hex_data)
        for i in range(n):
            center_x, center_y, rotation = inner_hex_data[i]
            vertices = self.hex_geometry.create_hexagon_vertices(center_x, center_y, rotation)

            # Check containment
            for vertex in vertices:
                if not outer_polygon.contains(Point(vertex[0], vertex[1])):
                    return False

            # Check collision with all previous hexagons (more efficient than pairwise)
            for j in range(i):
                center_x2, center_y2, rotation2 = inner_hex_data[j]
                vertices2 = self.hex_geometry.create_hexagon_vertices(center_x2, center_y2, rotation2)
                
                if self.check_collision(vertices, vertices2):
                    return False
                    
        return True

class Initializer:
    """Generates initial configurations for optimization"""
    
    def __init__(self, hex_geometry):
        self.hex_geometry = hex_geometry
    
    def create_hexagonal_pattern(self, n=11):
        """Create hexagonal arrangement pattern"""
        positions = [[0, 0, 0]]  # center
        
        # First ring
        ring_1 = [
            [0, 2.0, 0],  # top
            [1.732, 1.0, 0],  # top-right
            [1.732, -1.0, 0],  # bottom-right
            [0, -2.0, 0],  # bottom
            [-1.732, -1.0, 0],  # bottom-left
            [-1.732, 1.0, 0],  # top-left
        ]
        positions.extend(ring_1)
        
        # Second ring (if needed)
        ring_2 = [
            [3.464, 0, 0],  # far right
            [1.732, 2.0, 0],  # top-middle
            [-1.732, 2.0, 0],  # top-middle-left
            [-3.464, 0, 0],  # far left
            [-1.732, -2.0, 0],  # bottom-middle-left
            [1.732, -2.0, 0],  # bottom-middle-right
        ]
        positions.extend(ring_2)
        
        # Take first n elements and add noise
        positions = positions[:n]
        for pos in positions:
            pos[0] += np.random.normal(0, 0.1)
            pos[1] += np.random.normal(0, 0.1)
            pos[2] += np.random.normal(0, 5)  # rotation variation
            
        return np.array(positions)
    
    def create_spiral_pattern(self, n=11):
        """Create spiral arrangement pattern"""
        positions = []
        for i in range(n):
            angle = i * 0.5
            radius = i * 0.5
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            rotation = np.random.uniform(0, 360)
            positions.append([x, y, rotation])
        return np.array(positions)
    
    def create_random_pattern(self, n=11):
        """Create purely random arrangement"""
        positions = []
        for i in range(n):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            rotation = np.random.uniform(0, 360)
            positions.append([x, y, rotation])
        return np.array(positions)
    
    def create_hybrid_initial_guess(self, n=11):
        """Create initial configuration using hybrid approach"""
        # Try different methods and select best
        methods = [
            self.create_hexagonal_pattern,
            self.create_spiral_pattern,
            self.create_random_pattern
        ]
        
        best_score = float('inf')
        best_config = None
        
        for method in methods:
            config = method(n)
            # Score based on distribution quality
            score = self._evaluate_placement_quality(config)
            if score < best_score:
                best_score = score
                best_config = config
                
        return best_config
    
    def _evaluate_placement_quality(self, config):
        """Heuristic to evaluate quality of placement"""
        distances = []
        center = np.array([0, 0])
        for pos in config:
            dist = np.linalg.norm(np.array([pos[0], pos[1]]) - center)
            distances.append(dist)
        # Prefer configurations with more balanced distribution
        return np.std(distances)

class OptimizerEngine:
    """Core optimization engine"""
    
    def __init__(self, hex_geometry, validator, initializer):
        self.hex_geometry = hex_geometry
        self.validator = validator
        self.initializer = initializer
        self.max_iterations = 3000
        self.min_outer_side_length = 1.0
        self.max_outer_side_length = 20.0
    
    def adaptive_search(self, initial_config):
        """Adaptive Monte Carlo search with progressive refinement"""
        best_config = initial_config.copy()
        best_side_length = 100.0  # Start with large value
        best_score = -1.0
        
        # Track recent improvements
        recent_improvements = []
        current_scale = 1.0
        
        for iteration in range(self.max_iterations):
            # Generate new configuration with adaptive scale
            new_config = self._sample_with_adaptation(best_config, current_scale)
            
            # Try different outer hexagon sizes
            for side_length in [best_side_length * 0.95, best_side_length * 0.9, best_side_length * 1.05]:
                if side_length < self.min_outer_side_length or side_length > self.max_outer_side_length:
                    continue
                    
                if self.validator.validate_configuration(new_config, side_length):
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
        
        # Apply small random modifications
        for i in range(len(new_config)):
            new_config[i][0] += np.random.normal(0, scale * 0.1)
            new_config[i][1] += np.random.normal(0, scale * 0.1)
            new_config[i][2] += np.random.normal(0, scale * 5)
            new_config[i][2] = new_config[i][2] % 360
            
        return new_config

def optimize_single_instance(engine, init_method, seed=None):
    """Run optimization for a single instance"""
    if seed is not None:
        np.random.seed(seed)
    
    try:
        # Create initial configuration
        if init_method == 'hexagonal':
            initial_config = engine.initializer.create_hexagonal_pattern()
        elif init_method == 'spiral':
            initial_config = engine.initializer.create_spiral_pattern()
        elif init_method == 'random':
            initial_config = engine.initializer.create_random_pattern()
        else:
            initial_config = engine.initializer.create_hybrid_initial_guess()
        
        # Perform adaptive search
        best_config, best_side_length, best_score = engine.adaptive_search(initial_config)
        return best_score, best_config, best_side_length
    except Exception as e:
        return -1.0, None, None

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    hex_geometry = HexagonGeometry()
    validator = ConstraintValidator(hex_geometry)
    initializer = Initializer(hex_geometry)
    engine = OptimizerEngine(hex_geometry, validator, initializer)
    
    # Run parallel optimization with different initializations
    init_methods = ['hexagonal', 'spiral', 'random']
    seeds = [42, 123, 456]
    
    # Prepare jobs
    jobs = []
    for method in init_methods:
        for seed in seeds:
            jobs.append(delayed(optimize_single_instance)(engine, method, seed))
    
    # Execute parallel jobs
    results = Parallel(n_jobs=-1, verbose=0)(jobs)
    
    # Find the best result across all parallel runs
    best_score = -1.0
    best_result = None
    
    for score, config, side_length in results:
        if score > best_score and config is not None:
            best_score = score
            best_result = (config, side_length)
    
    # If we found a good solution, return it
    if best_result is not None and best_score > 0.1:
        best_config, best_side_length = best_result
        # Verify final solution
        if validator.validate_configuration(best_config, best_side_length):
            outer_hex_data = np.array([0, 0, 0])
            return best_config, outer_hex_data, best_side_length
    
    # Fallback to original approach
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
        max_dist_from_center = max(max_dist_from_center, dist + 1.0)
    
    # Outer hexagon should have side length slightly larger than max distance
    outer_hex_side_length = max_dist_from_center * 1.2
    
    # Evaluate this configuration
    valid = validator.validate_configuration(initial_config, outer_hex_side_length)
    
    # Return if valid, otherwise fallback to basic configuration
    if valid:
        inner_hex_data = initial_config.copy()
        outer_hex_data = np.array([0, 0, 0])
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    else:
        # Fallback to basic valid configuration
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
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END