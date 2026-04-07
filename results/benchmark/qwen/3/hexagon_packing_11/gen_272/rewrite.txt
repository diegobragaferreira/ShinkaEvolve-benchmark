# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
import random
import warnings

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class Hexagon:
    """Fast hexagon representation with optimized geometric operations"""
    
    def __init__(self, center_x: float, center_y: float, angle_degrees: float, side_length: float = 1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.angle_degrees = angle_degrees
        self.side_length = side_length
        self.circumradius = side_length * np.sqrt(3) / 2.0  # Distance to vertices from center

    @staticmethod
    @jit(nopython=True)
    def _generate_base_vertices(side_length: float) -> np.ndarray:
        """Generate base vertices of a unit hexagon centered at origin"""
        sqrt3_over_2 = np.sqrt(3) / 2.0
        return np.array([
            [side_length, 0.0],
            [side_length/2.0, sqrt3_over_2 * side_length],
            [-side_length/2.0, sqrt3_over_2 * side_length],
            [-side_length, 0.0],
            [-side_length/2.0, -sqrt3_over_2 * side_length],
            [side_length/2.0, -sqrt3_over_2 * side_length]
        ], dtype=np.float64)

    def get_vertices(self) -> np.ndarray:
        """Get vertices of the hexagon with current transformation"""
        # Get base vertices
        base_vertices = self._generate_base_vertices(self.side_length)

        # Apply rotation
        angle_rad = np.radians(self.angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

        rotated_vertices = base_vertices @ rotation_matrix.T

        # Apply translation
        return rotated_vertices + np.array([self.center_x, self.center_y], dtype=np.float64)

    def to_polygon(self) -> Polygon:
        """Convert hexagon to shapely polygon"""
        return Polygon(self.get_vertices())
    
    def distance_to_point(self, px: float, py: float) -> float:
        """Compute distance from hexagon center to point"""
        return np.sqrt((self.center_x - px)**2 + (self.center_y - py)**2)

class PackingEvaluator:
    """High-performance geometric validation and fitness computation"""
    
    def __init__(self, hex_side_length: float = 1.0):
        self.hex_side_length = hex_side_length
        self.circumradius = hex_side_length * np.sqrt(3) / 2.0

    def check_containment(self, hexagons: list[Hexagon], outer_radius: float) -> bool:
        """Fast containment checking using circumradius"""
        # Check if all hexagons are within outer hexagon
        # A hexagon is contained if its center is within outer hexagon minus its circumradius
        outer_hex = Hexagon(0.0, 0.0, 0.0, outer_radius)
        
        for hexagon in hexagons:
            # Distance from center to origin
            dist_center = hexagon.distance_to_point(0, 0)
            # Hexagon is contained if center plus circumradius < outer radius
            if dist_center + hexagon.circumradius > outer_radius:
                return False
        return True

    def check_overlap(self, hexagons: list[Hexagon]) -> bool:
        """Quick overlap detection using distance thresholds"""
        n = len(hexagons)
        # For unit hexagons, if distance between centers is less than 2, they may overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = hexagons[i].center_x - hexagons[j].center_x
                dy = hexagons[i].center_y - hexagons[j].center_y
                distance_sq = dx*dx + dy*dy
                # Minimum distance for touching hexagons is 2 (sum of radii)
                if distance_sq < 4.0:  # Square of 2
                    # Do detailed check if necessary
                    hex1_polygon = hexagons[i].to_polygon()
                    hex2_polygon = hexagons[j].to_polygon()
                    if hex1_polygon.intersects(hex2_polygon):
                        return True
        return False

    def evaluate_fitness(self, hexagons: list[Hexagon], outer_radius: float) -> float:
        """Fast fitness evaluation"""
        # Check constraints
        if not self.check_containment(hexagons, outer_radius):
            return -np.inf  # Invalid - penalty

        if self.check_overlap(hexagons):
            return -np.inf  # Invalid - penalty

        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius

class AdaptivePackingOptimizer:
    """Main optimizer using hierarchical adaptive strategy"""
    
    def __init__(self, n_inner_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length
        self.evaluator = PackingEvaluator(hex_side_length)
        
    def create_hexagons_from_array(self, hex_data: np.ndarray) -> list[Hexagon]:
        """Convert array data to list of Hexagon objects"""
        return [Hexagon(row[0], row[1], row[2], self.hex_side_length) for row in hex_data]

    def create_array_from_hexagons(self, hexagons: list[Hexagon]) -> np.ndarray:
        """Convert list of Hexagon objects to array data"""
        return np.array([[h.center_x, h.center_y, h.angle_degrees] for h in hexagons])

    def find_optimal_radius(self, hexagons: list[Hexagon], min_radius: float = 1.0, max_radius: float = 15.0) -> float:
        """Find minimum radius that contains all hexagons using binary search"""
        # Binary search with early exit on valid configurations
        left, right = min_radius, max_radius
        for _ in range(20):  # Limit iterations
            mid = (left + right) / 2
            if self.evaluator.check_containment(hexagons, mid):
                right = mid
            else:
                left = mid
        return right

    def _generate_initial_structure(self) -> np.ndarray:
        """Generate highly structured initial configuration"""
        # Start with a geometrically optimized pattern
        # Based on hexagonal lattice principles with symmetric placement
        config = np.zeros((11, 3))
        
        # Central hexagon
        config[0] = [0.0, 0.0, 0.0]
        
        # First ring (6 hexagons) - arranged in hexagonal pattern
        ring1_radius = 2.0  # Distance that allows tight packing
        for i in range(6):
            angle = i * 60  # 60 degrees apart
            rad = np.radians(angle)
            x = ring1_radius * np.cos(rad)
            y = ring1_radius * np.sin(rad)
            config[i+1] = [x, y, 0.0]
        
        # Second ring (4 hexagons) - strategically placed for maximum space utilization
        ring2_radius = 3.5
        ring2_angles = [0, 90, 180, 270]  # 4 cardinal directions
        for i, angle in enumerate(ring2_angles):
            rad = np.radians(angle)
            x = ring2_radius * np.cos(rad)
            y = ring2_radius * np.sin(rad)
            config[i+7] = [x, y, 0.0]
        
        # Add small random perturbations to avoid symmetries
        for i in range(11):
            config[i, 0] += np.random.normal(0, 0.05)
            config[i, 1] += np.random.normal(0, 0.05)
            config[i, 2] += np.random.normal(0, 5)
            config[i, 2] %= 360.0
            
        return config

    def _local_gradient_refinement(self, individual: np.ndarray, outer_radius: float, 
                                 max_iter: int = 100, learning_rate: float = 0.1) -> np.ndarray:
        """Enhanced gradient-based local refinement for hexagon positions"""
        def objective(params):
            # Reshape params back to hexagon data
            new_data = individual.copy()
            for i in range(len(new_data)):
                new_data[i][0] = params[i*3]
                new_data[i][1] = params[i*3+1]
                new_data[i][2] = params[i*3+2]

            # Convert to hexagon objects for evaluation
            hexagons = self.create_hexagons_from_array(new_data)
            fitness = self.evaluator.evaluate_fitness(hexagons, outer_radius)
            return -fitness  # minimize negative fitness

        # Flatten for optimization
        initial_params = individual.flatten()
        
        # Start with a few iterations of adaptive gradient descent
        current_params = initial_params.copy()
        current_fitness = objective(current_params)
        
        for iteration in range(max_iter):
            # Calculate gradient numerically using finite differences
            epsilon = 1e-6
            gradient = np.zeros_like(current_params)
            
            for i in range(len(current_params)):
                # Forward differencing
                params_plus = current_params.copy()
                params_plus[i] += epsilon
                fp = objective(params_plus)
                
                # Backward differencing  
                params_minus = current_params.copy()
                params_minus[i] -= epsilon
                fm = objective(params_minus)
                
                gradient[i] = (fp - fm) / (2 * epsilon)
            
            # Apply gradient descent with adaptive step size
            new_params = current_params - learning_rate * gradient
            
            # Apply bounds for positions and angles
            for i in range(len(new_params)):
                if i % 3 == 0:  # x coordinates
                    new_params[i] = max(-10, min(10, new_params[i]))
                elif i % 3 == 1:  # y coordinates
                    new_params[i] = max(-10, min(10, new_params[i]))
                else:  # angles
                    new_params[i] = new_params[i] % 360
                    
            # Evaluate new fitness
            new_fitness = objective(new_params)
            
            # Accept improvement or reduce learning rate
            if new_fitness < current_fitness:
                current_params = new_params
                current_fitness = new_fitness
                learning_rate = min(1.0, learning_rate * 1.05)  # Increase learning rate if successful
            else:
                learning_rate *= 0.9  # Reduce learning rate if no improvement
                
            # Early stopping condition
            if abs(new_fitness - current_fitness) < 1e-8:
                break
                
        # Reshape back to individual
        refined_individual = individual.copy()
        for i in range(len(refined_individual)):
            refined_individual[i][0] = current_params[i*3]
            refined_individual[i][1] = current_params[i*3+1]
            refined_individual[i][2] = current_params[i*3+2]
            
        return refined_individual

    def _adaptive_search_stage(self, initial_config: np.ndarray, 
                              min_radius: float, max_radius: float,
                              refinement_levels: list) -> tuple[np.ndarray, float]:
        """Multi-stage adaptive search"""
        current_config = initial_config.copy()
        best_radius = max_radius
        best_config = current_config.copy()
        best_fitness = float('-inf')
        
        # Stage 1: Coarse global search
        stage1_radius = (min_radius + max_radius) / 2
        for _ in range(5):
            # Random perturbations with gradually decreasing magnitude
            perturbation_magnitude = 0.3
            temp_config = current_config.copy()
            for i in range(len(temp_config)):
                temp_config[i, 0] += np.random.normal(0, perturbation_magnitude)
                temp_config[i, 1] += np.random.normal(0, perturbation_magnitude)
            # Local refinement on perturbed config
            refined_config = self._local_gradient_refinement(temp_config, stage1_radius, 10, 0.05)
            hexagons = self.create_hexagons_from_array(refined_config)
            radius = self.find_optimal_radius(hexagons)
            fitness = self.evaluator.evaluate_fitness(hexagons, radius)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = refined_config.copy()
                best_radius = radius
                
            # Decrease perturbation magnitude
            perturbation_magnitude *= 0.8
            
        # Stage 2: Fine-tune with progressively smaller steps
        for level in refinement_levels:
            for _ in range(level['iterations']):
                temp_config = best_config.copy()
                perturbation_magnitude = level['magnitude']
                for i in range(len(temp_config)):
                    temp_config[i, 0] += np.random.normal(0, perturbation_magnitude)
                    temp_config[i, 1] += np.random.normal(0, perturbation_magnitude)
                    temp_config[i, 2] += np.random.normal(0, 10 * perturbation_magnitude)  # rotation
                    temp_config[i, 2] %= 360
                # Local refinement
                refined_config = self._local_gradient_refinement(temp_config, best_radius, 20, 0.1)
                hexagons = self.create_hexagons_from_array(refined_config)
                radius = self.find_optimal_radius(hexagons)
                fitness = self.evaluator.evaluate_fitness(hexagons, radius)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_config = refined_config.copy()
                    best_radius = radius
                    
        return best_config, best_radius

    def optimize(self, max_time_seconds: float = 175.0) -> tuple[np.ndarray, float]:
        """Main optimization routine with adaptive strategy"""
        start_time = time.time()
        
        # Initial configuration
        initial_config = self._generate_initial_structure()
        
        # Multi-stage refinement with different tolerances and search intensities
        refinement_stages = [
            {'iterations': 5, 'magnitude': 0.5},
            {'iterations': 10, 'magnitude': 0.2}, 
            {'iterations': 15, 'magnitude': 0.05},
            {'iterations': 20, 'magnitude': 0.02}
        ]
        
        # Initial broad search range
        min_radius = 2.0
        max_radius = 10.0
        
        # Multi-scale approach - start with relaxed search
        best_config, best_radius = self._adaptive_search_stage(
            initial_config, min_radius, max_radius, refinement_stages[:2]
        )
        
        # If we're still early, do more fine-grained search
        if time.time() - start_time < max_time_seconds * 0.7:
            # Refine with tighter search bounds
            fine_min_radius = max(1.0, best_radius * 0.8)
            fine_max_radius = best_radius * 1.2
            if fine_max_radius < 15.0:  # Cap at reasonable value
                best_config, best_radius = self._adaptive_search_stage(
                    best_config, fine_min_radius, fine_max_radius, refinement_stages[2:]
                )
        
        # Final refinement with highest precision
        if time.time() - start_time < max_time_seconds - 5:
            final_config = self._local_gradient_refinement(best_config, best_radius, 30, 0.05)
            final_hexagons = self.create_hexagons_from_array(final_config)
            final_radius = self.find_optimal_radius(final_hexagons)
            final_fitness = self.evaluator.evaluate_fitness(final_hexagons, final_radius)
            
            if final_fitness > self.evaluator.evaluate_fitness(
                self.create_hexagons_from_array(best_config), best_radius
            ):
                best_config = final_config
                best_radius = final_radius
                
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
    
    try:
        # Initialize optimizer
        optimizer = AdaptivePackingOptimizer(n_inner_hexagons=11, hex_side_length=1.0)
        
        # Run adaptive optimization
        best_config, best_radius = optimizer.optimize(max_time_seconds=175.0)
        
        # Validate final solution
        hexagons = optimizer.create_hexagons_from_array(best_config)
        if not optimizer.evaluator.check_containment(hexagons, best_radius):
            warnings.warn("Final solution containment issue")
        if optimizer.evaluator.check_overlap(hexagons):
            warnings.warn("Final solution overlap detected")
            
    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}")
        # Fallback to simple structured configuration
        best_config = np.array([
            [0, 0, 0],           # center
            [-2.0, 0, 0],        # left
            [2.0, 0, 0],         # right
            [0, 3.0, 0],         # top
            [0, -3.0, 0],        # bottom
            [-1.0, 1.5, 0],      # top-left
            [1.0, 1.5, 0],       # top-right
            [-1.0, -1.5, 0],     # bottom-left
            [1.0, -1.5, 0],      # bottom-right
            [-2.0, 3.0, 0],      # top-left far
            [2.0, 3.0, 0],       # top-right far
        ])
        best_radius = 6.0  # Rough estimate for fallback

    # Prepare output
    inner_hex_data = best_config
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = best_radius

    elapsed = time.time() - start_time
    print(f"Eval time: {elapsed:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END