# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid
import multiprocessing as mp
import time
from joblib import Parallel, delayed
import warnings
import math
from typing import Tuple, List, Optional, Any

# Constants
UNIT_HEX_RADIUS = 1.0
UNIT_HEX_APOGEE = np.sqrt(3)/2
BENCHMARK_RATIO = 0.2544

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""
    
    def __init__(self):
        self.unit_hex_radius = UNIT_HEX_RADIUS
        self.unit_hex_apogee = UNIT_HEX_APOGEE
        self.unit_hex_vertices = self._generate_unit_hexagon_vertices()
    
    def _generate_unit_hexagon_vertices(self) -> np.ndarray:
        """Generate vertices of a unit regular hexagon centered at origin"""
        vertices = []
        for i in range(6):
            angle = i * np.pi / 3
            x = self.unit_hex_radius * np.cos(angle)
            y = self.unit_hex_radius * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
    
    def create_hexagon(self, center: Tuple[float, float], rotation: float) -> Polygon:
        """Create a unit regular hexagon as a Shapely Polygon"""
        angle_offset = np.deg2rad(rotation)
        points = []
        for i in range(6):
            angle = angle_offset + i * np.pi/3
            x = center[0] + self.unit_hex_radius * np.cos(angle)
            y = center[1] + self.unit_hex_radius * np.sin(angle)
            points.append((x, y))
        return Polygon(points)
    
    def create_hexagon_array(self, centers: np.ndarray, rotations: np.ndarray) -> List[Polygon]:
        """Create multiple hexagons from arrays of centers and rotations"""
        hexagons = []
        for center, rotation in zip(centers, rotations):
            hexagons.append(self.create_hexagon(tuple(center), rotation))
        return hexagons
    
    def validate_polygon(self, polygon) -> Polygon:
        """Ensure polygon is valid for geometric operations"""
        if not polygon.is_valid:
            return make_valid(polygon)
        return polygon

class HexagonValidation:
    """Handles constraint validation for hexagon configurations"""
    
    def __init__(self, geometry: HexagonGeometry):
        self.geometry = geometry
        self.outer_hexagon = self.geometry.create_hexagon((0, 0), 0)
        
    def check_containment(self, inner_hexagon: Polygon, outer_radius: float) -> bool:
        """Check if inner hexagon is fully contained within outer hexagon"""
        # Scale outer hexagon to correct size
        outer_vertices = list(self.outer_hexagon.exterior.coords)
        scaled_coords = [(x * outer_radius, y * outer_radius) for x, y in outer_vertices]
        outer_hexagon_scaled = Polygon(scaled_coords)
        
        # Add small buffer to prevent floating-point precision issues
        buffered_outer = outer_hexagon_scaled.buffer(1e-6)
        
        # Check if all vertices of inner hexagon are within outer hexagon
        for point in inner_hexagon.exterior.coords[:-1]:
            if not buffered_outer.contains(Point(point)):
                return False
        return True
    
    def check_overlap(self, hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap"""
        # Add small buffer to both polygons to prevent floating-point precision issues
        buffered_hex1 = hex1.buffer(1e-6)
        buffered_hex2 = hex2.buffer(1e-6)
        return buffered_hex1.intersects(buffered_hex2)
    
    def validate_configuration(self, inner_hex_data: np.ndarray, outer_radius: float) -> Tuple[bool, float]:
        """
        Validate configuration for collisions and containment
        Returns (is_valid, objective_value)
        """
        num_hex = len(inner_hex_data)
        
        # Create all inner hexagons
        centers = inner_hex_data[:, :2]
        rotations = inner_hex_data[:, 2]
        inner_hexagons = self.geometry.create_hexagon_array(centers, rotations)
        
        # Check containment (early termination)
        for hexagon in inner_hexagons:
            if not self.check_containment(hexagon, outer_radius):
                return False, 0.0  # containment violated
            
        # Check overlaps (early termination)
        for i in range(num_hex):
            for j in range(i+1, num_hex):
                if self.check_overlap(inner_hexagons[i], inner_hexagons[j]):
                    return False, 0.0  # overlap violated
                    
        # Valid configuration
        return True, 1.0 / outer_radius

class HexagonPackingOptimizer:
    """Main optimization controller with modular design"""
    
    def __init__(self):
        self.geometry = HexagonGeometry()
        self.validator = HexagonValidation(self.geometry)
        self.num_inner = 11
        self.max_evaluations = 150
        self.num_parallel = min(mp.cpu_count(), 4)
        
    def _calculate_outer_radius_estimate(self, inner_hex_data: np.ndarray) -> float:
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        max_dist = 0
        for i in range(len(inner_hex_data)):
            center = inner_hex_data[i][:2]
            dist = np.linalg.norm(np.array(center) - np.array([0, 0]))
            # Add distance from center to corner of unit hexagon
            dist += self.geometry.unit_hex_apogee
            max_dist = max(max_dist, dist)
        return max_dist * 1.2  # Add margin
    
    def _generate_initial_configurations(self) -> List[np.ndarray]:
        """Generate diverse initial configurations"""
        configs = []
        
        # Base hexagonal arrangement
        base_positions = [
            [0, 0, 0],        # center
            [-2.5, 0, 0],     # left
            [2.5, 0, 0],      # right
            [-1.25, 2.17, 0], # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0], # bottom-left
            [1.25, -2.17, 0], # bottom-right
            [-3.75, 2.17, 0], # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0], # far bottom-left
            [3.75, -2.17, 0], # far bottom-right
        ]
        
        # Add base configuration
        configs.append(np.array(base_positions))
        
        # Add random variations with different patterns
        for _ in range(4):
            config = np.array(base_positions)
            for i in range(len(config)):
                # Small random perturbations
                config[i][0] += np.random.normal(0, 0.3)
                config[i][1] += np.random.normal(0, 0.3)
                config[i][2] += np.random.normal(0, 5)
                config[i][2] = config[i][2] % 360
            configs.append(config)
        
        # Add spiral pattern variation
        spiral_positions = [
            [0, 0, 0],
            [-2.0, 0, 0],
            [2.0, 0, 0],
            [0, 2.0, 0],
            [0, -2.0, 0],
            [-1.5, 1.5, 0],
            [1.5, 1.5, 0],
            [-1.5, -1.5, 0],
            [1.5, -1.5, 0],
            [-3.0, 1.0, 0],
            [3.0, 1.0, 0],
        ]
        configs.append(np.array(spiral_positions))
        
        return configs
    
    def _objective_function(self, params: np.ndarray) -> float:
        """
        Objective function to minimize (we want to maximize 1/outer_radius)
        Returns negative because we're using minimization
        """
        try:
            # Extract inner hexagon data (first 33 parameters: 11 hexagons * 3 params each)
            inner_params = params[:-1].reshape(self.num_inner, 3)
            outer_radius = params[-1]
            
            # Validate configuration
            is_valid, objective_value = self.validator.validate_configuration(inner_params, outer_radius)
            
            if is_valid:
                return -objective_value  # Negative for minimization
            else:
                # Return large penalty if invalid
                return 10000.0  # Large positive number to indicate poor fitness
                
        except Exception:
            return 10000.0  # Penalty for exceptions
    
    def _get_bounds(self, initial_config: np.ndarray) -> List[Tuple[float, float]]:
        """Get optimization bounds for parameters"""
        bounds = []
        
        # Add bounds for positions and rotations (11 hexagons * 3 params each)
        for _ in range(self.num_inner):
            # x coordinate bounds
            bounds.append((-10.0, 10.0))
            # y coordinate bounds  
            bounds.append((-10.0, 10.0))
            # rotation bounds (0 to 360 degrees)
            bounds.append((0.0, 360.0))
        
        # Add bound for outer hexagon radius
        bounds.append((2.0, 15.0))
        
        return bounds
    
    def _optimize_single(self, initial_config: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float], float]:
        """Run single differential evolution optimization"""
        try:
            # Prepare optimization parameters
            bounds = self._get_bounds(initial_config)
            
            # Initial guess
            initial_params = initial_config.flatten()
            estimated_radius = self._calculate_outer_radius_estimate(initial_config)
            initial_params = np.append(initial_params, estimated_radius)
            
            # Run optimization
            result = differential_evolution(
                self._objective_function,
                bounds,
                args=(),
                seed=42,
                maxiter=self.max_evaluations,
                popsize=20,
                mutation=(0.7, 1),
                recombination=0.8,
                atol=1e-7,
                rtol=1e-7,
                disp=False
            )
            
            # Extract results
            opt_params = result.x
            opt_inner_data = opt_params[:-1].reshape(self.num_inner, 3)
            opt_outer_radius = opt_params[-1]
            
            # Validate final result
            is_valid, objective_value = self.validator.validate_configuration(opt_inner_data, opt_outer_radius)
            
            if is_valid:
                return opt_inner_data, opt_outer_radius, -objective_value
            else:
                return None, None, -1e10
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {str(e)}")
            return None, None, -1e10
    
    def _local_refinement(self, initial_data: np.ndarray, outer_radius: float) -> Tuple[np.ndarray, float, float]:
        """Apply L-BFGS-B local refinement"""
        try:
            # Prepare bounds for local optimization
            bounds = self._get_bounds(initial_data)
            
            # Initial guess
            initial_params = initial_data.flatten()
            initial_params = np.append(initial_params, outer_radius)
            
            # Local optimization
            result = minimize(
                self._objective_function,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 50}
            )
            
            if result.success:
                refined_params = result.x
                refined_inner_data = refined_params[:-1].reshape(self.num_inner, 3)
                refined_outer_radius = refined_params[-1]
                
                # Validate refined solution
                is_valid, refined_objective = self.validator.validate_configuration(refined_inner_data, refined_outer_radius)
                if is_valid:
                    return refined_inner_data, refined_outer_radius, -refined_objective
            
            return initial_data, outer_radius, 1.0 / outer_radius
            
        except Exception as e:
            warnings.warn(f"Local refinement failed: {str(e)}")
            return initial_data, outer_radius, 1.0 / outer_radius
    
    def find_optimal_packing(self, initial_config: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Main optimization driver"""
        # Generate multiple initial configurations
        initial_configs = self._generate_initial_configurations()
        
        # Run parallel optimizations
        results = Parallel(n_jobs=self.num_parallel)(
            delayed(self._optimize_single)(config) for config in initial_configs
        )
        
        # Find best result
        best_result = None
        best_score = -1e10
        
        for result in results:
            if result[0] is not None and result[2] > best_score:
                best_score = result[2]
                best_result = result
        
        if best_result is None or best_score < 0.1:
            # Fallback to simple optimization
            best_result = self._optimize_single(initial_config)
        
        if best_result[0] is None:
            return None, None, -1e10
        
        # Apply local refinement to the best solution
        refined_data, refined_radius, refined_score = self._local_refinement(best_result[0], best_result[1])
        
        return refined_data, refined_radius, refined_score

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize optimizer
    optimizer = HexagonPackingOptimizer()
    
    # Initial configuration from the simple grid
    initial_config = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0], # bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0], # far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])
    
    # Attempt optimization
    try:
        inner_hex_data, outer_hex_side_length, inv_side_length = optimizer.find_optimal_packing(initial_config)
        
        # If optimization succeeded with reasonable results
        if inner_hex_data is not None and inv_side_length > 0.1:
            outer_hex_data = np.array([0, 0, 0])
            return inner_hex_data, outer_hex_data, outer_hex_side_length
    except Exception as e:
        # Silently handle errors and fall back
        warnings.warn(f"Optimization error: {str(e)}")
        pass
    
    # Fallback to original approach if optimization fails
    # Set reasonable initial outer hexagon size based on configuration
    max_dist_from_center = 0
    for i in range(len(initial_config)):
        center_x, center_y, _ = initial_config[i]
        dist = np.sqrt(center_x**2 + center_y**2)
        max_dist_from_center = max(max_dist_from_center, dist + 1.0)  # Add radius margin
    
    # Outer hexagon should have side length slightly larger than max distance
    outer_hex_side_length = max_dist_from_center * 1.2  # 20% margin
    
    # Evaluate this configuration
    validator = HexagonValidation(HexagonGeometry())
    valid, _ = validator.validate_configuration(initial_config, outer_hex_side_length)
    
    # If initial configuration is invalid due to overlap or containment,
    # we fall back to the simpler approach but with better validation
    if not valid:
        # Fallback to a basic valid configuration
        inner_hex_data = np.array([
            [0, 0, 0],        # center
            [-2.5, 0, 0],     # left
            [2.5, 0, 0],      # right
            [-1.25, 2.17, 0], # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0], # bottom-left
            [1.25, -2.17, 0], # bottom-right
            [-3.75, 2.17, 0], # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0], # far bottom-left
            [3.75, -2.17, 0], # far bottom-right
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0  # fallback value
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Since we've confirmed initial config works, we can return it
    inner_hex_data = initial_config.copy()
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END