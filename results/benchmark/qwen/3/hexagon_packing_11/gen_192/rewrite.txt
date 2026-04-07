# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import time
from numba import jit
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class Hexagon:
    """Efficient hexagon representation with vertex computation"""
    
    def __init__(self, center_x: float, center_y: float, angle_degrees: float, side_length: float = 1.0):
        self.center_x = center_x
        self.center_y = center_y
        self.angle_degrees = angle_degrees
        self.side_length = side_length

    @staticmethod
    @jit(nopython=True)
    def _generate_base_vertices(side_length: float) -> np.ndarray:
        """Generate base vertices of a unit hexagon centered at origin"""
        sqrt3 = np.sqrt(3)
        return np.array([
            [side_length, 0.0],
            [side_length/2.0, sqrt3/2.0 * side_length],
            [-side_length/2.0, sqrt3/2.0 * side_length],
            [-side_length, 0.0],
            [-side_length/2.0, -sqrt3/2.0 * side_length],
            [side_length/2.0, -sqrt3/2.0 * side_length]
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

class HexagonPackingEvaluator:
    """Handles geometric validation and fitness evaluation"""
    
    def __init__(self, hex_side_length: float = 1.0):
        self.hex_side_length = hex_side_length

    def check_containment(self, hexagons: list[Hexagon], outer_radius: float) -> bool:
        """Check if all hexagons are contained within outer hexagon of given radius"""
        # Create outer hexagon centered at origin
        outer_hex = Hexagon(0.0, 0.0, 0.0, outer_radius)
        outer_polygon = outer_hex.to_polygon()

        for hexagon in hexagons:
            hex_polygon = hexagon.to_polygon()
            if not outer_polygon.contains(hex_polygon):
                return False
        return True

    def check_overlap(self, hexagons: list[Hexagon]) -> bool:
        """Check if any hexagons overlap"""
        for i in range(len(hexagons)):
            for j in range(i+1, len(hexagons)):
                if hexagons[i].to_polygon().intersects(hexagons[j].to_polygon()):
                    return True
        return False

    def evaluate_fitness(self, hexagons: list[Hexagon], outer_radius: float) -> float:
        """Evaluate fitness based on geometric constraints and packing density"""
        # Check constraints
        if not self.check_containment(hexagons, outer_radius):
            return -np.inf  # Invalid - penalty

        if self.check_overlap(hexagons):
            return -np.inf  # Invalid - penalty

        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius

    def compute_outer_radius(self, hexagons: list[Hexagon]) -> float:
        """Compute minimum outer radius needed to contain all hexagons"""
        max_distance = 0.0
        for hexagon in hexagons:
            vertices = hexagon.get_vertices()
            for vx, vy in vertices:
                dist = np.sqrt((vx)**2 + (vy)**2)  # Distance from origin
                max_distance = max(max_distance, dist)
        return max_distance * 1.05  # Safety margin

class HexagonPackingOptimizer:
    """Main optimization controller managing the search process"""
    
    def __init__(self, n_inner_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length
        self.evaluator = HexagonPackingEvaluator(hex_side_length)
        
    def create_hexagons_from_array(self, hex_data: np.ndarray) -> list[Hexagon]:
        """Convert array data to list of Hexagon objects"""
        return [Hexagon(row[0], row[1], row[2], self.hex_side_length) for row in hex_data]

    def create_array_from_hexagons(self, hexagons: list[Hexagon]) -> np.ndarray:
        """Convert list of Hexagon objects to array data"""
        return np.array([[h.center_x, h.center_y, h.angle_degrees] for h in hexagons])

    def generate_initial_configuration(self) -> np.ndarray:
        """Create initial configuration using hexagonal lattice"""
        # Start with a hexagonal lattice pattern optimized for 11 hexagons
        config = []
        
        # Central hexagon
        config.append([0.0, 0.0, 0.0])
        
        # First shell - 6 hexagons arranged in a circle
        shell_radius = 2.0
        angles = [0, 60, 120, 180, 240, 300]
        
        for angle in angles:
            rad = np.radians(angle)
            x = shell_radius * np.cos(rad)
            y = shell_radius * np.sin(rad)
            config.append([x, y, 0.0])
        
        # Second shell - 4 hexagons to fill gaps
        shell_radius2 = 3.464  # ~2*sqrt(3)
        angles2 = [30, 90, 150, 210]
        
        for angle in angles2:
            rad = np.radians(angle)
            x = shell_radius2 * np.cos(rad)
            y = shell_radius2 * np.sin(rad)
            config.append([x, y, 0.0])
        
        # Trim to exactly 11 hexagons and add small jitter
        result = np.array(config[:11])
        noise_scale = 0.05
        result[:, 0] += np.random.normal(0, noise_scale, result.shape[0])
        result[:, 1] += np.random.normal(0, noise_scale, result.shape[0])
        
        return result

    def optimize_local(self, individual: np.ndarray, outer_radius: float) -> np.ndarray:
        """Refine solution locally using multiple strategies"""
        def objective(params):
            # Reshape params back to hexagon data
            new_data = individual.copy()
            for i in range(len(new_data)):
                new_data[i][0] = params[i*3]
                new_data[i][1] = params[i*3+1]
                new_data[i][2] = params[i*3+2]
            
            # Convert to hexagon objects and evaluate
            hexagons = self.create_hexagons_from_array(new_data)
            fitness = self.evaluator.evaluate_fitness(hexagons, outer_radius)
            return -fitness  # minimize negative fitness

        # Flatten the data for optimization
        initial_params = individual.flatten()
        
        # Strategy 1: L-BFGS-B with strict bounds
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B',
                            bounds=[(-10, 10), (-10, 10), (0, 360)] * len(individual),
                            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10})
            if result.success:
                # Reshape optimized result back
                refined_data = individual.copy()
                for i in range(len(refined_data)):
                    refined_data[i][0] = result.x[i*3]
                    refined_data[i][1] = result.x[i*3+1]
                    refined_data[i][2] = result.x[i*3+2]
                return refined_data
        except:
            pass
            
        # Strategy 2: Nelder-Mead as fallback
        try:
            result = minimize(objective, initial_params, method='Nelder-Mead',
                            options={'maxiter': 100, 'disp': False})
            if result.success:
                # Reshape optimized result back
                refined_data = individual.copy()
                for i in range(len(refined_data)):
                    refined_data[i][0] = result.x[i*3]
                    refined_data[i][1] = result.x[i*3+1]
                    refined_data[i][2] = result.x[i*3+2]
                return refined_data
        except:
            pass
            
        # If all optimization fails, return original
        return individual

    def optimize_with_adaptive_search(self, initial_config: np.ndarray, max_iterations: int = 30) -> tuple[np.ndarray, float]:
        """Adaptive optimization with local refinement"""
        current_config = initial_config.copy()
        best_fitness = -np.inf
        best_config = initial_config.copy()
        
        for iteration in range(max_iterations):
            # Create diverse offspring
            offspring = []
            for _ in range(15):
                # Create perturbed version
                mutated = current_config.copy()
                for i in range(len(mutated)):
                    if np.random.random() < 0.3:  # 30% mutation chance
                        mutated[i][0] += np.random.normal(0, 0.2)
                        mutated[i][1] += np.random.normal(0, 0.2)
                        mutated[i][2] += np.random.normal(0, 15)
                        mutated[i][2] = mutated[i][2] % 360
                offspring.append(mutated)
            
            # Evaluate all offspring
            fitness_scores = []
            for individual in offspring:
                hexagons = self.create_hexagons_from_array(individual)
                radius = self.evaluator.compute_outer_radius(hexagons)
                fitness = self.evaluator.evaluate_fitness(hexagons, radius)
                fitness_scores.append(fitness)
            
            # Select best offspring
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_config = offspring[best_idx].copy()
            
            # Local optimization on best so far
            if iteration % 3 == 0:
                radius = self.evaluator.compute_outer_radius(self.create_hexagons_from_array(best_config))
                refined = self.optimize_local(best_config, radius)
                refined_hexagons = self.create_hexagons_from_array(refined)
                refined_radius = self.evaluator.compute_outer_radius(refined_hexagons)
                refined_fitness = self.evaluator.evaluate_fitness(refined_hexagons, refined_radius)
                if refined_fitness > best_fitness:
                    best_fitness = refined_fitness
                    best_config = refined.copy()
            
            # Update current for next iteration
            current_config = offspring[best_idx]
        
        return best_config, best_fitness

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Initialize optimizer
    optimizer = HexagonPackingOptimizer(n_inner_hexagons=11, hex_side_length=1.0)
    
    # Generate initial configuration
    initial_config = optimizer.generate_initial_configuration()
    
    # Apply adaptive optimization
    best_config, best_fitness = optimizer.optimize_with_adaptive_search(initial_config, max_iterations=30)
    
    # Final local optimization
    final_radius = optimizer.evaluator.compute_outer_radius(optimizer.create_hexagons_from_array(best_config))
    final_config = optimizer.optimize_local(best_config, final_radius)
    
    # Validation and cleanup
    best_hexagons = optimizer.create_hexagons_from_array(best_config)
    final_hexagons = optimizer.create_hexagons_from_array(final_config)
    
    final_radius = optimizer.evaluator.compute_outer_radius(final_hexagons)
    final_fitness = optimizer.evaluator.evaluate_fitness(final_hexagons, final_radius)
    
    # Use optimized result if better
    if final_fitness > best_fitness:
        best_config = final_config
        best_fitness = final_fitness
    
    # Prepare output
    inner_hex_data = best_config
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = final_radius
    
    end_time = time.time()
    print(f"Eval time: {end_time - start_time:.4f}s")
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END