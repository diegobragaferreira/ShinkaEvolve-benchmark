# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
from typing import Tuple, List, Optional
import random
from numba import jit, prange

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

@jit(nopython=True)
def distance_point_to_line_segment(px, py, x1, y1, x2, y2):
    """Fast distance from point to line segment"""
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx*dx + dy*dy
    if length_sq == 0.0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

class HexagonGeometry:
    """Handles all geometric computations for hexagons with optimized vertex generation"""

    def __init__(self):
        # Pre-compute unit hexagon vertices once for efficiency
        self._unit_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])

    def get_transformed_vertices(self, center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
        """Efficiently compute transformed hexagon vertices"""
        # Get unit vertices and scale
        vertices = self._unit_vertices * side_length

        # Apply rotation
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = vertices @ rotation_matrix.T

        # Apply translation
        return rotated + np.array([center_x, center_y])

    def create_hexagon_polygon(self, center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> Polygon:
        """Create shapely polygon for hexagon with precomputed vertices"""
        vertices = self.get_transformed_vertices(center_x, center_y, angle_deg, side_length)
        return Polygon(vertices)

class HexagonValidator:
    """Handles constraint checking for hexagon packing with optimized operations"""

    def __init__(self, geometry: HexagonGeometry):
        self.geometry = geometry

    def check_containment(self, hexagons: List[Tuple[float, float, float]], outer_radius: float) -> bool:
        """Check if all hexagons are contained within outer hexagon of given radius"""
        # Create outer hexagon once
        outer_polygon = self.geometry.create_hexagon_polygon(0.0, 0.0, 0.0, outer_radius)

        # Check each hexagon against outer polygon
        for center_x, center_y, angle_deg in hexagons:
            hex_polygon = self.geometry.create_hexagon_polygon(center_x, center_y, angle_deg)
            if not outer_polygon.contains(hex_polygon):
                return False
        return True

    def check_overlap(self, hexagons: List[Tuple[float, float, float]]) -> bool:
        """Check if any hexagons overlap using spatial indexing for efficiency"""
        # Create polygons once
        polygons = [self.geometry.create_hexagon_polygon(center_x, center_y, angle_deg)
                   for center_x, center_y, angle_deg in hexagons]

        # Check pairwise overlaps
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return True
        return False

class SolutionManager:
    """Manages solution representation, validation and output formatting"""

    def __init__(self, geometry: HexagonGeometry, validator: HexagonValidator):
        self.geometry = geometry
        self.validator = validator

    def validate_solution(self, hex_data: np.ndarray, outer_radius: float) -> bool:
        """Validate that solution meets all constraints"""
        # Convert array to list of tuples for validator
        hexagons = [(row[0], row[1], row[2]) for row in hex_data]

        # Check constraints
        if not self.validator.check_containment(hexagons, outer_radius):
            return False
        if self.validator.check_overlap(hexagons):
            return False
        return True

    def format_output(self, hex_data: np.ndarray, outer_radius: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Format final solution for output"""
        # Inner hex data
        inner_hex_data = hex_data.copy()

        # Outer hex data (centered at origin with zero rotation)
        outer_hex_data = np.array([0.0, 0.0, 0.0])

        # Outer hex side length
        outer_hex_side_length = outer_radius

        return inner_hex_data, outer_hex_data, outer_hex_side_length

class PackingProblem:
    """Main optimization class coordinating the hexagon packing process"""

    def __init__(self, n_inner_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length

        # Initialize components
        self.geometry = HexagonGeometry()
        self.validator = HexagonValidator(self.geometry)
        self.solver = SolutionManager(self.geometry, self.validator)

    def create_hexagon_list(self, hex_data: np.ndarray) -> List[Tuple[float, float, float]]:
        """Convert numpy array to list of hexagon tuples"""
        return [(row[0], row[1], row[2]) for row in hex_data]

    def find_optimal_radius(self, hex_data: np.ndarray, min_radius: float = 1.0, max_radius: float = 15.0) -> float:
        """Find minimum radius that contains all hexagons using adaptive binary search"""
        # Convert to tuple format for validation
        hexagons = self.create_hexagon_list(hex_data)

        # Binary search with adaptive precision
        left, right = min_radius, max_radius
        iterations = 0
        max_iterations = 25  # Reduced iterations for speed

        # Early check if already valid
        if self.validator.check_containment(hexagons, left):
            return left

        while iterations < max_iterations:
            # Dynamic tolerance based on current range
            current_range = right - left
            tolerance = max(0.001, 0.01 * (current_range / 10.0))

            if current_range <= tolerance:
                break

            mid = (left + right) / 2
            if self.validator.check_containment(hexagons, mid):
                right = mid
            else:
                left = mid
            iterations += 1

        return right

    def evaluate_fitness(self, hex_data: np.ndarray, outer_radius: float) -> float:
        """Evaluate fitness based on geometric constraints and packing density"""
        # Convert to tuple format for validation
        hexagons = self.create_hexagon_list(hex_data)

        # Check constraints
        if not self.validator.check_containment(hexagons, outer_radius):
            return -np.inf  # Invalid - penalty

        if self.validator.check_overlap(hexagons):
            return -np.inf  # Invalid - penalty

        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius

    def optimize_local(self, hex_data: np.ndarray, outer_radius: float) -> np.ndarray:
        """Refine solution locally using optimization"""
        def objective(params):
            # Reshape params back to hexagon data
            new_data = hex_data.copy()
            for i in range(len(new_data)):
                new_data[i][0] = params[i*3]
                new_data[i][1] = params[i*3+1]
                new_data[i][2] = params[i*3+2]

            # Evaluate fitness
            fitness = self.evaluate_fitness(new_data, outer_radius)
            return -fitness  # minimize negative fitness

        # Flatten the data for optimization
        initial_params = []
        for i in range(len(hex_data)):
            initial_params.extend([hex_data[i][0], hex_data[i][1], hex_data[i][2]])

        # Optimize using L-BFGS-B with fewer iterations
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B',
                            bounds=[(-10, 10), (-10, 10), (0, 360)] * len(hex_data),
                            options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
            if result.success:
                # Reshape optimized result back
                refined_data = hex_data.copy()
                for i in range(len(refined_data)):
                    refined_data[i][0] = result.x[i*3]
                    refined_data[i][1] = result.x[i*3+1]
                    refined_data[i][2] = result.x[i*3+2]
                return refined_data
        except:
            pass
        return hex_data

    def create_initial_configuration(self) -> np.ndarray:
        """Create a high-quality initial configuration with strategic placement"""
        # Start with mathematical hexagonal tiling pattern
        # Base configuration optimized for 11 hexagons
        config = np.array([
            [0, 0, 0],        # center
            [-2.5, 0, 0],     # left
            [2.5, 0, 0],      # right
            [-1.25, 2.17, 0], # top-left
            [1.25, 2.17, 0],  # top-right
            [-1.25, -2.17, 0],# bottom-left
            [1.25, -2.17, 0], # bottom-right
            [-3.75, 2.17, 0], # far top-left
            [3.75, 2.17, 0],  # far top-right
            [-3.75, -2.17, 0],# far bottom-left
            [3.75, -2.17, 0], # far bottom-right
        ])
        
        # Add small random perturbations to escape local minima
        for i in range(len(config)):
            config[i][0] += np.random.normal(0, 0.05)
            config[i][1] += np.random.normal(0, 0.05)
            config[i][2] += np.random.normal(0, 2)
        
        return config

    def local_search_with_restarts(self, initial_config: np.ndarray, max_restarts: int = 3) -> Tuple[np.ndarray, float]:
        """Run local optimization with multiple restarts"""
        best_config = initial_config.copy()
        best_radius = self.find_optimal_radius(best_config)
        best_fitness = self.evaluate_fitness(best_config, best_radius)
        
        for restart in range(max_restarts):
            # Perturb current solution slightly
            perturbed = best_config.copy()
            for i in range(len(perturbed)):
                perturbed[i][0] += np.random.normal(0, 0.02)
                perturbed[i][1] += np.random.normal(0, 0.02)
                perturbed[i][2] += np.random.normal(0, 1)
                perturbed[i][2] = perturbed[i][2] % 360
                
            # Find optimal radius for perturbed configuration
            radius = self.find_optimal_radius(perturbed)
            fitness = self.evaluate_fitness(perturbed, radius)
            
            # Update best if improved
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = perturbed.copy()
                best_radius = radius
        
        # Final local optimization
        refined_config = self.optimize_local(best_config, best_radius)
        final_radius = self.find_optimal_radius(refined_config)
        final_fitness = self.evaluate_fitness(refined_config, final_radius)
        
        if final_fitness > best_fitness:
            return refined_config, final_radius
        else:
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

    # Initialize the packing problem
    problem = PackingProblem(n_inner_hexagons=11, hex_side_length=1.0)

    # Create high-quality initial configuration
    initial_config = problem.create_initial_configuration()

    # Multi-start local optimization approach for better convergence
    best_fitness = -np.inf
    best_config = initial_config.copy()
    best_radius = 10.0

    # Run local search with multiple restarts
    final_config, final_radius = problem.local_search_with_restarts(initial_config, max_restarts=5)
    
    # Validate and check fitness
    final_fitness = problem.evaluate_fitness(final_config, final_radius)
    
    if final_fitness > best_fitness:
        best_fitness = final_fitness
        best_config = final_config.copy()
        best_radius = final_radius

    # Additional validation step
    if not problem.solver.validate_solution(best_config, best_radius):
        # Fall back to simple configuration if validation fails
        fallback_config = np.array([
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
            [3.75, -2.17, 0]
        ])
        fallback_radius = problem.find_optimal_radius(fallback_config)
        fallback_fitness = problem.evaluate_fitness(fallback_config, fallback_radius)
        
        if fallback_fitness > best_fitness:
            best_config = fallback_config.copy()
            best_radius = fallback_radius

    # Format output
    inner_hex_data, outer_hex_data, outer_hex_side_length = problem.solver.format_output(
        best_config, best_radius
    )

    end_time = time.time()
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END