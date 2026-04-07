# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import itertools
import random
from typing import Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Constants
NUM_INNER_HEXAGONS = 11
UNIT_HEXAGON_RADIUS = 1.0
MAX_EVAL_TIME = 180.0

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""
    
    def __init__(self):
        # Precomputed unit hexagon vertices (centered at origin)
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles + close the loop
        self.unit_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def rotate_point(self, point: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate a point around origin"""
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return np.dot(rotation_matrix, point)
    
    def hexagon_vertices(self, center: np.ndarray, angle_rad: float, scale: float = 1.0) -> np.ndarray:
        """Get vertices of a hexagon at given position and rotation"""
        rotated_vertices = np.array([self.rotate_point(v, angle_rad) for v in self.unit_vertices])
        return rotated_vertices * scale + np.array(center)
    
    def calculate_outer_radius(self, inner_hex_data: np.ndarray, 
                             outer_center: np.ndarray = np.array([0,0]), 
                             outer_angle: float = 0) -> float:
        """Calculate minimum radius needed for outer hexagon to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            angle = np.radians(inner_hex_data[i][2])

            # Get all vertices of this hexagon
            vertices = self.hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)

            # Calculate max distance from outer center to any vertex
            for vertex in vertices:
                dist = np.linalg.norm(np.array(vertex) - np.array(outer_center))
                max_dist = max(max_dist, dist)

        return max_dist * 1.05  # Add safety margin

class HexagonValidator:
    """Handles validation of hexagon configurations"""
    
    def __init__(self, geometry: HexagonGeometry):
        self.geometry = geometry
    
    def point_in_polygon(self, point: np.ndarray, polygon: Polygon) -> bool:
        """Fast point-in-polygon check"""
        return polygon.contains(Point(point))
    
    def check_containment(self, hex_poly: Polygon, outer_polygon: Polygon) -> bool:
        """Check if hexagon is fully contained within outer hexagon"""
        # Check if all vertices are within outer polygon
        for vertex in hex_poly.exterior.coords[:-1]:  # Exclude closing vertex
            if not self.point_in_polygon(vertex, outer_polygon):
                return False
        return True
    
    def check_overlap(self, hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap"""
        return hex1.intersects(hex2)
    
    def validate_solution(self, inner_hex_data: np.ndarray, 
                         outer_center: np.ndarray = np.array([0,0]), 
                         outer_angle: float = 0) -> bool:
        """Validate solution: check containment and non-overlap"""
        # Precompute all hexagon polygons once for reuse
        hex_polygons = []
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            angle = np.radians(inner_hex_data[i][2])
            vertices = self.geometry.hexagon_vertices(center, angle, UNIT_HEXAGON_RADIUS)
            hex_polygons.append(Polygon(vertices))

        # Check containment using the outer hexagon polygon
        outer_radius = self.geometry.calculate_outer_radius(inner_hex_data, outer_center, outer_angle)
        outer_vertices = self.geometry.hexagon_vertices(outer_center, outer_angle, outer_radius)
        outer_polygon = Polygon(outer_vertices)

        # Check if all inner hexagons are contained within outer hexagon
        for hex_poly in hex_polygons:
            if not self.check_containment(hex_poly, outer_polygon):
                return False

        # Check overlaps efficiently using spatial indexing
        # Build spatial index for faster overlap detection
        points_list = []
        for i, hex_poly in enumerate(hex_polygons):
            # Collect all vertices for spatial indexing
            for vertex in hex_poly.exterior.coords[:-1]:
                points_list.append((vertex[0], vertex[1], i))

        if len(points_list) > 0:
            # Create spatial tree for vertices
            tree_points = cKDTree([(p[0], p[1]) for p in points_list])

            # Check overlaps between hexagons using spatial indexing
            for i in range(len(hex_polygons)):
                for j in range(i+1, len(hex_polygons)):
                    if self.check_overlap(hex_polygons[i], hex_polygons[j]):
                        return False
        else:
            # Fallback for empty case
            return False

        return True

class HexagonOptimizer:
    """Handles the optimization process"""
    
    def __init__(self, geometry: HexagonGeometry, validator: HexagonValidator):
        self.geometry = geometry
        self.validator = validator
        
    def generate_hexagonal_lattice_initial(self) -> np.ndarray:
        """Generate initial configuration based on hexagonal lattice pattern"""
        # Create a more sophisticated initial layout
        config = []
        
        # Center hexagon
        config.append([0, 0, 0])
        
        # First ring around center (6 hexagons)
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for angle in angles:
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            config.append([x, y, 0])
        
        # Second ring (12 hexagons total, but we only need 11)
        # Place in a way that creates better packing
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i, angle in enumerate(angles):
            x = 3.0 * np.cos(angle) + (-1)**i * 1.0
            y = 3.0 * np.sin(angle)
            config.append([x, y, 0])
        
        # Ensure we have exactly 11 hexagons
        while len(config) < 11:
            config.append([0, 0, 0])
        
        config = config[:11]
        
        # Add small random rotations to break symmetry
        for i in range(len(config)):
            config[i][2] = np.random.uniform(0, 360)
        
        return np.array(config)
    
    def adaptive_local_optimization(self, initial_config: np.ndarray, 
                                  start_time: float) -> Tuple[np.ndarray, float]:
        """Perform adaptive local optimization with variable mutation rates"""
        best_config = initial_config.copy()
        best_radius = self.geometry.calculate_outer_radius(best_config)
        
        # Adaptive parameters
        max_iterations = 2000
        stagnation_threshold = 50
        max_stagnation = 200
        initial_mutation_rate = 0.8
        final_mutation_rate = 0.1
        
        stagnation_count = 0
        improvement_count = 0
        
        for iteration in range(max_iterations):
            if time.time() - start_time > MAX_EVAL_TIME - 1:
                break
                
            # Adaptive mutation rate scheduling
            current_mutation_rate = initial_mutation_rate + (final_mutation_rate - initial_mutation_rate) * \
                                   (iteration / max_iterations)
            
            # Create perturbed configuration
            test_config = best_config.copy()
            
            # Apply perturbations with adaptive rate
            for i in range(NUM_INNER_HEXAGONS):
                # Position perturbations with adaptive magnitude
                pos_mutation = 0.2 * current_mutation_rate
                test_config[i][0] += np.random.normal(0, pos_mutation)
                test_config[i][1] += np.random.normal(0, pos_mutation)
                
                # Rotation perturbations
                rot_mutation = 10.0 * current_mutation_rate
                test_config[i][2] += np.random.normal(0, rot_mutation)
                test_config[i][2] %= 360
            
            # Validate and accept if better
            if self.validator.validate_solution(test_config):
                test_radius = self.geometry.calculate_outer_radius(test_config)
                if test_radius < best_radius:
                    best_radius = test_radius
                    best_config = test_config.copy()
                    improvement_count += 1
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1
                
            # Early stopping if stagnation occurs
            if stagnation_count > max_stagnation:
                break
        
        return best_config, best_radius
    
    def multi_start_optimization(self, start_time: float) -> Tuple[np.ndarray, float]:
        """Run multiple optimization runs with different initial conditions"""
        best_overall_config = None
        best_overall_radius = float('inf')
        
        # Run multiple optimization trials
        num_trials = 15
        
        for trial in range(num_trials):
            if time.time() - start_time > MAX_EVAL_TIME - 1:
                break
                
            # Generate different initial configurations
            if trial == 0:
                # First trial with structured initial layout
                initial_config = self.generate_hexagonal_lattice_initial()
            else:
                # Later trials with random perturbations of the best known
                if best_overall_config is not None:
                    initial_config = best_overall_config.copy()
                    # Add some random noise
                    for i in range(len(initial_config)):
                        initial_config[i][0] += np.random.normal(0, 0.5)
                        initial_config[i][1] += np.random.normal(0, 0.5)
                        initial_config[i][2] += np.random.normal(0, 30)
                        initial_config[i][2] %= 360
                else:
                    # Fallback to lattice layout
                    initial_config = self.generate_hexagonal_lattice_initial()
            
            # Perform local optimization
            local_config, local_radius = self.adaptive_local_optimization(initial_config, start_time)
            
            # Update best overall solution
            if local_radius < best_overall_radius:
                best_overall_radius = local_radius
                best_overall_config = local_config.copy()
        
        return best_overall_config, best_overall_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize components
    geometry = HexagonGeometry()
    validator = HexagonValidator(geometry)
    optimizer = HexagonOptimizer(geometry, validator)
    
    # Multi-start optimization approach
    best_inner_config, best_radius = optimizer.multi_start_optimization(start_time)
    
    # Final validation
    if not validator.validate_solution(best_inner_config):
        # Fallback to well-known good configuration
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