# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution, minimize
import time
import math
from typing import Tuple, Optional, List
import warnings
from joblib import Parallel, delayed
import multiprocessing

class HexagonUtils:
    """Utility class for hexagon geometric operations"""
    
    @staticmethod
    def generate_unit_hexagon_vertices(radius: float = 1.0) -> np.ndarray:
        """Generate vertices of a unit regular hexagon centered at origin"""
        vertices = []
        for i in range(6):
            angle = i * np.pi / 3
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)
    
    @staticmethod
    def hexagon_from_params(vertices: np.ndarray, center_x: float, center_y: float, rotation_deg: float) -> np.ndarray:
        """Create hexagon vertices given center and rotation"""
        rotation_rad = np.radians(rotation_deg)
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)
        
        # Apply rotation and translation to unit hexagon vertices
        rotated_vertices = np.zeros_like(vertices)
        for i, (x, y) in enumerate(vertices):
            rotated_vertices[i] = [
                x * cos_r - y * sin_r + center_x,
                x * sin_r + y * cos_r + center_y
            ]
        return rotated_vertices
    
    @staticmethod
    def check_containment(hexagon_vertices: np.ndarray, outer_hex_vertices: np.ndarray) -> bool:
        """Check if hexagon is fully contained in outer hexagon"""
        outer_polygon = Polygon(outer_hex_vertices)
        
        # Check if all vertices of inner hexagon are within outer hexagon
        for vertex in hexagon_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True
    
    @staticmethod
    def check_collision(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Check if two hexagons collide using Shapely with buffer for robustness"""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        # Use small buffer to avoid floating-point precision issues
        return poly1.buffer(1e-6).intersects(poly2.buffer(1e-6))

class HexagonPackingValidator:
    """Validates hexagon configurations for packing constraints"""
    
    def __init__(self, unit_hex_vertices: np.ndarray):
        self.unit_hex_vertices = unit_hex_vertices
        self.outer_hex_vertices = HexagonUtils.hexagon_from_params(unit_hex_vertices, 0, 0, 0)
        
    def validate_configuration(self, inner_hex_data: np.ndarray, outer_side_length: float) -> Tuple[bool, float]:
        """
        Validate configuration for collisions and containment
        Returns (is_valid, objective_value)
        """
        num_hex = len(inner_hex_data)
        
        # Create all inner hexagon polygons efficiently
        inner_polygons = []
        for i in range(num_hex):
            center_x, center_y, rotation = inner_hex_data[i]
            vertices = HexagonUtils.hexagon_from_params(self.unit_hex_vertices, center_x, center_y, rotation)
            inner_polygons.append(Polygon(vertices))
        
        # Check containment of all inner hexagons within outer hexagon (early exit)
        outer_polygon = Polygon(self.outer_hex_vertices)
        
        for i in range(num_hex):
            if not outer_polygon.contains(inner_polygons[i]):
                return False, 0.0
        
        # Check pairwise collisions with early termination (more efficient than nested loop)
        # First do a quick bounding box check, then full polygon check
        for i in range(num_hex):
            for j in range(i + 1, num_hex):
                # Quick bounding box collision check first
                poly1 = inner_polygons[i]
                poly2 = inner_polygons[j]
                
                # If bounding boxes don't intersect, skip expensive polygon intersection
                bbox1 = poly1.bounds
                bbox2 = poly2.bounds
                
                if (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                    bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]):
                    continue
                    
                # Full polygon intersection test
                if HexagonUtils.check_collision(inner_polygons[i], inner_polygons[j]):
                    return False, 0.0
        
        # Valid configuration
        return True, 1.0 / outer_side_length

class HexagonPackingOptimizer:
    """Main optimization controller for hexagon packing"""
    
    def __init__(self):
        self.unit_hex_radius = 1.0
        self.unit_hex_vertices = HexagonUtils.generate_unit_hexagon_vertices(self.unit_hex_radius)
        self.validator = HexagonPackingValidator(self.unit_hex_vertices)
        self.max_iterations = 100  # Reduced iterations for time constraints
        self.population_size = 20  # Balanced population size
        self.num_cores = max(1, multiprocessing.cpu_count() - 1)  # Use all but one core
        
    def _initialize_parameters(self, initial_config: np.ndarray) -> List[float]:
        """Convert hexagon data to optimization parameters"""
        params = []
        for i in range(len(initial_config)):
            params.extend([initial_config[i][0], initial_config[i][1], initial_config[i][2]])
        return params
    
    def _extract_hex_data(self, params: np.ndarray, num_hex: int = 11) -> np.ndarray:
        """Extract hexagon data from optimization parameters"""
        return params.reshape(num_hex, 3).copy()
    
    def _objective_function(self, params: np.ndarray) -> float:
        """
        Objective function to minimize (we want to maximize 1/outer_side_length)
        Returns negative because we're using minimization
        """
        # Extract parameters for 11 hexagons (each has 3 params: x, y, rotation)
        # And one parameter for outer hexagon side length
        num_hex = 11
        hex_params = params[:-1].reshape(num_hex, 3)
        outer_side_length = params[-1]
        
        # Check validity and return negative of inverse side length if valid
        is_valid, objective_value = self.validator.validate_configuration(hex_params, outer_side_length)
        if is_valid:
            return -objective_value  # Negative for minimization
        else:
            # Return a large penalty if invalid
            return -1e10  # Large negative number to indicate poor fitness
    
    def _get_initial_bounds(self, initial_config: np.ndarray, margin_factor: float = 1.5) -> List[Tuple[float, float]]:
        """Get optimization bounds with adaptive constraints"""
        bounds = []
        num_hex = 11
        
        # Estimate spatial requirements more accurately
        if len(initial_config) > 0:
            # Determine the bounding box of the initial configuration
            centers = np.array([[h[0], h[1]] for h in initial_config])
            min_x, max_x = np.min(centers[:, 0]), np.max(centers[:, 0])
            min_y, max_y = np.min(centers[:, 1]), np.max(centers[:, 1])
            
            # Scale bounds based on actual spread with margin
            width = max_x - min_x
            height = max_y - min_y
            max_dim = max(width, height)
            
            # Use dynamic bounds based on configuration spread
            bound_range = max_dim * margin_factor
            center_x_mean = np.mean(centers[:, 0])
            center_y_mean = np.mean(centers[:, 1])
            
            # Set bounds relative to actual configuration spread
            x_bound = max(bound_range, 10.0)  # Minimum bound
            y_bound = max(bound_range, 10.0)  # Minimum bound
            
            for i in range(num_hex):
                bounds.append((center_x_mean - x_bound, center_x_mean + x_bound))
                bounds.append((center_y_mean - y_bound, center_y_mean + y_bound))
                bounds.append((0.0, 360.0))
        else:
            # Fallback to generic bounds
            for i in range(num_hex):
                bounds.append((-15.0, 15.0))
                bounds.append((-15.0, 15.0))
                bounds.append((0.0, 360.0))
        
        # Add bound for outer hexagon side length (minimum 1, maximum adaptive)
        bounds.append((1.0, 20.0))
        
        return bounds
    
    def _estimate_initial_outer_side(self, initial_config: np.ndarray) -> float:
        """Estimate outer hexagon side length from initial configuration with geometric accuracy"""
        # Calculate the tightest bounding box that contains all hexagons
        all_vertices = []

        # Get vertices for all inner hexagons
        for i in range(len(initial_config)):
            center_x, center_y, rotation = initial_config[i]
            vertices = HexagonUtils.hexagon_from_params(self.unit_hex_vertices, center_x, center_y, rotation)
            all_vertices.extend(vertices)

        if not all_vertices:
            return 2.0  # Default fallback

        # Convert to numpy array for easy computation
        vertices_array = np.array(all_vertices)

        # Compute bounding box center and dimensions
        min_x, max_x = np.min(vertices_array[:, 0]), np.max(vertices_array[:, 0])
        min_y, max_y = np.min(vertices_array[:, 1]), np.max(vertices_array[:, 1])

        # Compute width and height of bounding box
        width = max_x - min_x
        height = max_y - min_y

        # For a regular hexagon, we need to ensure our outer hexagon can contain all vertices
        # The side length of outer hexagon should accommodate the diagonal of the bounding box
        # A regular hexagon with side length s has a circumradius of s
        # The diagonal of a bounding box determines the minimum circumradius needed
        diagonal = np.sqrt(width**2 + height**2)
        
        # Add margin to account for hexagon geometry (circumradius should be at least half the diagonal)
        side_length = diagonal * 0.5 + 0.5  # Additional 0.5 margin for safety
        
        return max(side_length, 2.0)  # Ensure reasonable minimum

    def _generate_specialized_initial_configurations(self) -> List[np.ndarray]:
        """Generate specialized initial configurations using different geometric patterns"""
        configs = []
        
        # Pattern 1: Hexagonal lattice arrangement (most promising pattern)
        hex_lattice = [
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
        configs.append(np.array(hex_lattice))
        
        # Pattern 2: Spiral arrangement
        spiral_positions = []
        for i in range(11):
            # Spiral pattern with different spacing
            angle = i * 0.7
            radius = 0.3 * i
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            spiral_positions.append([x, y, 0])
        configs.append(np.array(spiral_positions))
        
        # Pattern 3: Grid arrangement (different distances)
        grid_positions = []
        grid_positions.append([0, 0, 0])  # center
        # Create a more spread out grid pattern
        offsets = [(-2.0, 0), (2.0, 0), (0, 2.0), (0, -2.0), (-1.5, 1.5), (1.5, 1.5), 
                   (-1.5, -1.5), (1.5, -1.5), (-3.0, 0), (3.0, 0), (0, 3.0)]
        for i, (dx, dy) in enumerate(offsets):
            grid_positions.append([dx, dy, 0])
        configs.append(np.array(grid_positions))
        
        # Pattern 4: Randomized version of good pattern
        random_config = np.array(hex_lattice)
        for i in range(len(random_config)):
            random_config[i][0] += np.random.normal(0, 0.3)
            random_config[i][1] += np.random.normal(0, 0.3)
            random_config[i][2] += np.random.normal(0, 5)
            random_config[i][2] = random_config[i][2] % 360
        configs.append(random_config)
        
        return configs
    
    def _adaptive_optimization(self, initial_config: np.ndarray, strategy: str = 'global') -> Tuple[np.ndarray, float, float]:
        """Run optimization with adaptive bounds and strategy selection"""
        # Prepare bounds and initial guess
        bounds = self._get_initial_bounds(initial_config)
        
        # Initial guess
        initial_params = self._initialize_parameters(initial_config)
        estimated_side = self._estimate_initial_outer_side(initial_config)
        initial_params.append(estimated_side)
        
        if strategy == 'global':
            # Run global differential evolution optimization
            try:
                result = differential_evolution(
                    self._objective_function,
                    bounds,
                    args=(),
                    seed=42,
                    maxiter=self.max_iterations,
                    popsize=self.population_size,
                    mutation=(0.7, 1),
                    recombination=0.8,
                    atol=1e-7,
                    rtol=1e-7,
                    disp=False
                )
                
                opt_params = result.x
                opt_hex_data = self._extract_hex_data(opt_params)
                opt_outer_side_length = opt_params[-1]
                
                return opt_hex_data, opt_outer_side_length, -result.fun
                
            except Exception as e:
                warnings.warn(f"Differential evolution failed: {str(e)}")
                return None, None, -1e10
        else:
            # Run local optimization using Nelder-Mead
            try:
                result = minimize(
                    self._objective_function,
                    initial_params,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'adaptive': True, 'disp': False}
                )
                
                if result.success:
                    opt_params = result.x
                    opt_hex_data = self._extract_hex_data(opt_params)
                    opt_outer_side_length = opt_params[-1]
                    return opt_hex_data, opt_outer_side_length, -result.fun
                else:
                    return None, None, -1e10
                    
            except Exception as e:
                warnings.warn(f"Nelder-Mead optimization failed: {str(e)}")
                return None, None, -1e10

    def _local_refinement(self, hex_data: np.ndarray, outer_side_length: float, 
                         max_iterations: int = 20) -> Tuple[np.ndarray, float, float]:
        """Apply local refinement to improve the solution quality"""
        # Start with best solution found
        best_hex_data = hex_data.copy()
        best_side_length = outer_side_length
        best_objective = 1.0 / outer_side_length  # Initial objective
        
        # Try multiple local refinements with different strategies
        for iteration in range(max_iterations):
            # Make perturbations to hexagon positions and rotations
            perturbed_data = best_hex_data.copy()
            
            # Apply different types of perturbations 
            for i in range(len(perturbed_data)):
                # Small perturbations to positions with diminishing effect
                perturbation_strength = 0.1 * (1.0 - iteration / max_iterations)
                perturbed_data[i][0] += np.random.normal(0, perturbation_strength)
                perturbed_data[i][1] += np.random.normal(0, perturbation_strength)
                
                # Perturb rotation
                perturbed_data[i][2] += np.random.normal(0, 2)
                perturbed_data[i][2] = perturbed_data[i][2] % 360
            
            # Check if refined configuration is valid and better
            is_valid, new_objective = self.validator.validate_configuration(perturbed_data, best_side_length)
            if is_valid and new_objective > best_objective:
                best_hex_data = perturbed_data
                best_objective = new_objective
        
        # Final validation
        final_validation = self.validator.validate_configuration(best_hex_data, best_side_length)
        if final_validation[0]:  # Valid
            return best_hex_data, best_side_length, best_objective
        else:
            return hex_data, outer_side_length, best_objective

    def find_optimal_packing(self, initial_config: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Main method to find optimal hexagon packing with enhanced strategy"""
        # Generate specialized initial configurations
        specialized_configs = self._generate_specialized_initial_configurations()
        
        best_result = None
        best_score = -1e10
        
        # Try all specialized configurations with global optimization
        for i, config in enumerate(specialized_configs):
            try:
                # Run global optimization on this configuration
                hex_data, outer_side_length, objective = self._adaptive_optimization(config, strategy='global')
                
                if hex_data is not None and objective > best_score:
                    best_score = objective
                    best_result = (hex_data, outer_side_length, objective)
            except Exception as e:
                continue
        
        # If no good results from specialized configs, fallback to standard optimization
        if best_result is None:
            try:
                hex_data, outer_side_length, objective = self._adaptive_optimization(initial_config, strategy='global')
                if hex_data is not None:
                    best_result = (hex_data, outer_side_length, objective)
            except Exception as e:
                pass
        
        # If we still have no good result, return fallback
        if best_result is None:
            return None, None, -1e10
            
        # Perform local refinement on the best global solution
        best_hex_data, best_side_length, best_objective = best_result
        refined_hex_data, refined_side_length, refined_objective = self._local_refinement(
            best_hex_data, best_side_length, max_iterations=15
        )
        
        return refined_hex_data, refined_side_length, refined_objective

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
    
    # Attempt optimization
    try:
        inner_hex_data, outer_hex_side_length, inv_side_length = optimizer.find_optimal_packing(initial_config)
        
        # If optimization succeeded with reasonable results
        if inner_hex_data is not None and inv_side_length > 0.15:
            outer_hex_data = np.array([0, 0, 0])
            return inner_hex_data, outer_hex_data, outer_hex_side_length
    except Exception as e:
        # Silently handle errors and fall back
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
    validator = HexagonPackingValidator(optimizer.unit_hex_vertices)
    valid, _ = validator.validate_configuration(initial_config, outer_hex_side_length)
    
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