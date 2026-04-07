# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
from joblib import Parallel, delayed
import time
from numba import jit
import random
from typing import Tuple, List, Optional, Any
import warnings

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class Hexagon:
    """Represent a hexagon with position, rotation, and size"""
    
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
        """Get vertices with current transformation"""
        base_vertices = self._generate_base_vertices(self.side_length)
        
        angle_rad = np.radians(self.angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        
        rotated_vertices = base_vertices @ rotation_matrix.T
        return rotated_vertices + np.array([self.center_x, self.center_y], dtype=np.float64)
    
    def to_polygon(self) -> Polygon:
        """Convert to shapely polygon"""
        return Polygon(self.get_vertices())

class HexagonValidator:
    """Validates hexagon configurations for geometric constraints"""
    
    def __init__(self, hex_side_length: float = 1.0):
        self.hex_side_length = hex_side_length
    
    def check_containment(self, hexagons: List[Hexagon], outer_radius: float) -> bool:
        """Check if all hexagons are contained within outer hexagon"""
        outer_hex = Hexagon(0.0, 0.0, 0.0, outer_radius)
        outer_polygon = outer_hex.to_polygon()
        
        for hexagon in hexagons:
            hex_polygon = hexagon.to_polygon()
            if not outer_polygon.contains(hex_polygon):
                return False
        return True
    
    def check_overlap(self, hexagons: List[Hexagon]) -> bool:
        """Check if any hexagons overlap"""
        polygons = [h.to_polygon() for h in hexagons]
        
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return True
        return False
    
    def validate_configuration(self, hexagons: List[Hexagon], outer_radius: float) -> bool:
        """Complete validation of hexagon configuration"""
        return (self.check_containment(hexagons, outer_radius) and 
                not self.check_overlap(hexagons))

class FitnessEvaluator:
    """Evaluates fitness of hexagon configurations"""
    
    def __init__(self, hex_side_length: float = 1.0):
        self.hex_side_length = hex_side_length
        self.validator = HexagonValidator(hex_side_length)
    
    def compute_outer_radius(self, hexagons: List[Hexagon]) -> float:
        """Compute minimum outer hexagon radius needed"""
        max_dist = 0
        for hexagon in hexagons:
            vertices = hexagon.get_vertices()
            for vertex in vertices:
                dist = np.sqrt(vertex[0]**2 + vertex[1]**2)
                max_dist = max(max_dist, dist)
        return max_dist + 0.1  # Small buffer
    
    def evaluate_fitness(self, hexagons: List[Hexagon]) -> Tuple[float, float]:
        """Evaluate fitness of hexagon configuration"""
        # Validate configuration
        outer_radius = self.compute_outer_radius(hexagons)
        if not self.validator.validate_configuration(hexagons, outer_radius):
            return -1e10, outer_radius  # Penalize invalid configurations
        
        # Return inverse of radius (higher is better)
        return 1.0 / outer_radius, outer_radius

class PopulationManager:
    """Manages population evolution and optimization"""
    
    def __init__(self, n_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_hexagons = n_hexagons
        self.hex_side_length = hex_side_length
        self.evaluator = FitnessEvaluator(hex_side_length)
    
    def create_hexagons_from_array(self, hex_data: np.ndarray) -> List[Hexagon]:
        """Convert array data to hexagon objects"""
        return [Hexagon(row[0], row[1], row[2], self.hex_side_length) for row in hex_data]
    
    def create_array_from_hexagons(self, hexagons: List[Hexagon]) -> np.ndarray:
        """Convert hexagon objects to array data"""
        return np.array([[h.center_x, h.center_y, h.angle_degrees] for h in hexagons])
    
    def generate_initial_config(self) -> np.ndarray:
        """Generate improved initial configuration using hexagonal packing principles"""
        # Create a strategic arrangement based on hexagonal lattice
        positions = []
        
        # Central hexagon
        positions.append([0.0, 0.0, 0.0])
        
        # First ring - 6 hexagons in regular pattern
        ring_radius = 2.0  # Distance between centers of touching hexagons
        for i in range(6):
            angle = i * 60  # 60 degree increments
            rad = np.radians(angle)
            x = ring_radius * np.cos(rad)
            y = ring_radius * np.sin(rad)
            positions.append([x, y, 0.0])
        
        # Second ring - 4 strategically placed hexagons
        # These positions are chosen to enhance packing density
        second_ring_positions = [
            [-1.0, -1.732, 0.0],   # Bottom left
            [1.0, -1.732, 0.0],    # Bottom right
            [-1.0, 1.732, 0.0],    # Top left
            [1.0, 1.732, 0.0],     # Top right
        ]
        
        positions.extend(second_ring_positions)
        
        # Take first 11 positions
        initial_positions = np.array(positions[:11])
        
        # Apply small random perturbations
        np.random.seed(42)
        for i in range(len(initial_positions)):
            initial_positions[i, 0] += np.random.normal(0, 0.02)
            initial_positions[i, 1] += np.random.normal(0, 0.02)
            initial_positions[i, 2] += np.random.normal(0, 3)
        
        return initial_positions
    
    def evaluate_population_parallel(self, population: List[np.ndarray]) -> List[Tuple[float, float]]:
        """Evaluate population in parallel"""
        def evaluate_single(individual):
            hexagons = self.create_hexagons_from_array(individual)
            return self.evaluator.evaluate_fitness(hexagons)
        
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_single)(individual) for individual in population
        )
        return results
    
    def mutate_individual(self, individual: np.ndarray, generation: int) -> np.ndarray:
        """Apply adaptive mutation"""
        mutated = individual.copy()
        # Decrease mutation rate with generation number
        mutation_rate = 0.2 / (1 + generation * 0.1)
        
        for i in range(len(mutated)):
            for j in range(len(mutated[i])):
                if np.random.rand() < mutation_rate:
                    if j < 2:  # x or y coordinate
                        mutated[i, j] += np.random.normal(0, 0.05)
                    else:  # angle
                        mutated[i, j] += np.random.normal(0, 10)
        return mutated
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Blend crossover between parents"""
        alpha = 0.7
        child = parent1 * alpha + parent2 * (1 - alpha)
        return child
    
    def adaptive_evolutionary_search(self, max_generations: int = 50, 
                                   population_size: int = 30) -> Tuple[np.ndarray, float]:
        """Main evolutionary optimization loop"""
        # Initialize population
        population = [self.generate_initial_config()]
        for _ in range(population_size - 1):
            population.append(self.generate_initial_config())
        
        best_fitness = -np.inf
        best_individual = None
        best_radius = float('inf')
        
        for gen in range(max_generations):
            # Evaluate population
            fitness_results = self.evaluate_population_parallel(population)
            
            # Update best
            for i, (fitness, radius) in enumerate(fitness_results):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = population[i].copy()
                    best_radius = radius
            
            # Select top individuals
            fitness_scores = [result[0] for result in fitness_results]
            sorted_indices = np.argsort(fitness_scores)[::-1][:population_size//2]
            selected = [population[i] for i in sorted_indices]
            
            # Generate new population
            new_population = selected.copy()
            while len(new_population) < population_size:
                parent1 = selected[np.random.randint(len(selected))]
                parent2 = selected[np.random.randint(len(selected))]
                
                child = self.crossover(parent1, parent2)
                child = self.mutate_individual(child, gen)
                new_population.append(child)
            
            population = new_population[:population_size]
        
        return best_individual, best_radius

class LocalOptimizer:
    """Performs local optimization refinements"""
    
    def __init__(self, hex_side_length: float = 1.0):
        self.hex_side_length = hex_side_length
        self.evaluator = FitnessEvaluator(hex_side_length)
    
    def optimize_individual(self, individual: np.ndarray, outer_radius: float) -> np.ndarray:
        """Local optimization with multiple strategies"""
        def objective(params):
            # Reshape parameters
            test_individual = params.reshape(-1, 3)
            hexagons = [Hexagon(row[0], row[1], row[2], self.hex_side_length) 
                       for row in test_individual]
            fitness, _ = self.evaluator.evaluate_fitness(hexagons)
            return -fitness  # Minimize negative fitness
        
        # Flatten for optimization
        initial_params = individual.flatten()
        
        try:
            # Try L-BFGS-B first
            result = minimize(objective, initial_params, method='L-BFGS-B',
                            bounds=[(-10, 10), (-10, 10), (0, 360)] * len(individual),
                            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success:
                refined = result.x.reshape(-1, 3)
                # Validate result
                hexagons = [Hexagon(row[0], row[1], row[2], self.hex_side_length) 
                           for row in refined]
                if self.evaluator.validator.validate_configuration(hexagons, outer_radius):
                    return refined
        except:
            pass
        
        # Fallback to simple coordinate descent
        try:
            current_params = initial_params.copy()
            current_fitness = objective(current_params)
            
            for _ in range(50):
                new_params = current_params.copy()
                for i in range(len(new_params)):
                    if i % 3 < 2:  # x, y coordinates
                        new_params[i] += np.random.normal(0, 0.05)
                    else:  # angle
                        new_params[i] += np.random.normal(0, 5)
                        new_params[i] = new_params[i] % 360
                
                new_fitness = objective(new_params)
                if new_fitness < current_fitness:
                    current_params = new_params
                    current_fitness = new_fitness
            
            refined = current_params.reshape(-1, 3)
            return refined
        except:
            return individual

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
        # Initialize components
        population_manager = PopulationManager(11, 1.0)
        local_optimizer = LocalOptimizer(1.0)
        
        # Evolutionary search
        best_individual, best_radius = population_manager.adaptive_evolutionary_search(
            max_generations=40, population_size=25
        )
        
        # Local optimization refinement
        if best_individual is not None:
            refined_individual = local_optimizer.optimize_individual(best_individual, best_radius)
            
            # Final validation
            hexagons = population_manager.create_hexagons_from_array(refined_individual)
            final_fitness, final_radius = population_manager.evaluator.evaluate_fitness(hexagons)
            
            if final_fitness > population_manager.evaluator.evaluate_fitness(
                population_manager.create_hexagons_from_array(best_individual))[0]:
                best_individual = refined_individual
                best_radius = final_radius
        
        # Prepare final result
        if best_individual is not None:
            inner_hex_data = best_individual
            outer_hex_side_length = best_radius
        else:
            # Fallback to known good configuration if optimization fails
            inner_hex_data = np.array([
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
            outer_hex_side_length = 8.0
        
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}, using fallback")
        inner_hex_data = np.array([
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
        outer_hex_side_length = 8.0
        outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END