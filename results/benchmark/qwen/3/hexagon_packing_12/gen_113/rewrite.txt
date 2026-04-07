# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
from joblib import Parallel, delayed

# Numba for performance optimization
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

class HexagonGeometry:
    """Handles all geometric operations for hexagons with optimizations"""
    
    @staticmethod
    def create_unit_hexagon(center: Tuple[float, float] = (0, 0), rotation: float = 0) -> Polygon:
        """Create a unit regular hexagon with given center and rotation"""
        angle_offset = math.radians(rotation)
        radius = 1
        vertices = []
        for i in range(6):
            angle = angle_offset + i * math.pi / 3
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            vertices.append((x, y))
        return Polygon(vertices)

    @staticmethod
    def get_all_vertices_vectorized(hex_data: np.ndarray) -> np.ndarray:
        """Vectorized extraction of all vertices from all hexagons"""
        if NUMBA_AVAILABLE:
            return _get_vertices_numba(hex_data)
        else:
            # Fallback to pure Python if numba not available
            all_vertices = []
            for i in range(len(hex_data)):
                center = (hex_data[i][0], hex_data[i][1])
                rotation = hex_data[i][2]
                hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
                all_vertices.extend(list(hexagon.exterior.coords))
            return np.array(all_vertices)

@jit(nopython=True, parallel=True) if NUMBA_AVAILABLE else lambda x: x
def _get_vertices_numba(hex_data):
    """Numba-accelerated vertex extraction"""
    n_hex = len(hex_data)
    all_vertices = np.empty((n_hex * 6, 2), dtype=np.float64)
    
    for i in prange(n_hex):
        center_x, center_y, rotation = hex_data[i]
        angle_offset = math.radians(rotation)
        radius = 1
        
        for j in range(6):
            angle = angle_offset + j * math.pi / 3
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            all_vertices[i * 6 + j] = [x, y]
    
    return all_vertices

class HexagonConstraintChecker:
    """Handles constraint checking for hexagon arrangements with optimizations"""
    
    @staticmethod
    def check_overlap_vectorized(hexagons: List[Polygon]) -> bool:
        """Vectorized overlap checking with early termination"""
        n = len(hexagons)
        if n <= 1:
            return False
            
        # Early exit if any pair intersects
        for i in range(n):
            for j in range(i+1, n):
                if hexagons[i].intersects(hexagons[j]):
                    return True
        return False

    @staticmethod
    def compute_overlap_penalty_vectorized(hexagons: List[Polygon]) -> float:
        """Vectorized overlap penalty computation"""
        penalty = 0
        n = len(hexagons)
        if n <= 1:
            return penalty
            
        for i in range(n):
            for j in range(i+1, n):
                if hexagons[i].intersects(hexagons[j]):
                    penalty += 1000
        return penalty

class HexagonPackingEvaluator:
    """Evaluates hexagon packing configurations with optimizations"""
    
    @staticmethod
    def calculate_outer_hex_radius_vectorized(hex_data: np.ndarray) -> float:
        """Vectorized calculation of outer hexagon radius"""
        all_vertices = HexagonGeometry.get_all_vertices_vectorized(hex_data)
        
        if len(all_vertices) == 0:
            return 0.0
            
        # Calculate distances efficiently
        distances_squared = np.sum(all_vertices**2, axis=1)
        max_distance_squared = np.max(distances_squared)
        max_distance = np.sqrt(max_distance_squared)
        
        return max_distance + 0.1
    
    @staticmethod
    def evaluate_configuration_vectorized(hex_data: np.ndarray) -> float:
        """Vectorized evaluation of configuration with early rejection"""
        # Quick validity check - if any hexagon is too far out, reject immediately
        outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius_vectorized(hex_data)
        
        # Create hexagon polygons efficiently
        hexagons = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
            hexagons.append(hexagon)
        
        # Fast overlap check
        if HexagonConstraintChecker.check_overlap_vectorized(hexagons):
            return 1e-10  # Invalid configuration
        
        # If valid configuration, return inverse of outer radius
        return 1.0 / outer_radius

class SymmetryAwareMutation:
    """Handles mutation strategies that preserve geometric symmetries with improvements"""
    
    @staticmethod
    def mutate_symmetrically(hex_data: np.ndarray, mutation_strength: float = 0.2, 
                           generation: int = 0, max_generations: int = 100) -> np.ndarray:
        """Apply symmetric mutation with adaptive strength"""
        mutated_data = hex_data.copy()
        
        # Adaptive mutation strength decay
        decay_factor = max(0.1, 1.0 - (generation / max_generations) * 0.8)
        current_mutation_strength = mutation_strength * decay_factor
        
        # Mutate center hexagon
        mutated_data[0][0] += random.uniform(-current_mutation_strength, current_mutation_strength)
        mutated_data[0][1] += random.uniform(-current_mutation_strength, current_mutation_strength)
        
        # Mutate based on hexagonal symmetry groups
        symmetry_groups = [
            [1, 2, 3, 4, 5, 6],  # First ring (around center)
            [7, 8, 9, 10, 11, 12, 13, 14],  # Second ring corners and edges
        ]
        
        # Mutate hexagons in groups to maintain symmetry
        for group in symmetry_groups:
            if len(group) > 0:
                group_indices = [idx for idx in group if idx < len(mutated_data)]
                if group_indices:
                    # Use the first element of the group as reference for mutation
                    ref_idx = group_indices[0]
                    mutated_data[ref_idx][0] += random.uniform(-current_mutation_strength, current_mutation_strength)
                    mutated_data[ref_idx][1] += random.uniform(-current_mutation_strength, current_mutation_strength)
                    
                    # Apply similar mutation to other members of group
                    for idx in group_indices[1:]:
                        if idx < len(mutated_data):
                            mutated_data[idx][0] += random.uniform(-current_mutation_strength*0.5, current_mutation_strength*0.5)
                            mutated_data[idx][1] += random.uniform(-current_mutation_strength*0.5, current_mutation_strength*0.5)
        
        return mutated_data

class HexagonPackingOptimizer:
    """Main optimizer class that orchestrates the packing process"""
    
    def __init__(self):
        self.best_score = 0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds
    
    def get_initial_configurations(self) -> List[np.ndarray]:
        """Generate improved symmetric configurations to choose from"""
        configs = []
        
        # Configuration 1: Improved hexagonal cluster around center (more optimal)
        config1 = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [0, -2.0, 0],        # bottom  
            [1.732, 1.0, 0],     # top-right
            [-1.732, 1.0, 0],    # top-left
            [1.732, -1.0, 0],    # bottom-right
            [-1.732, -1.0, 0],   # bottom-left
            [3.464, 0, 0],       # far right
            [-3.464, 0, 0],      # far left
            [1.732, 3.0, 0],     # upper right corner
            [-1.732, 3.0, 0],    # upper left corner
            [1.732, -3.0, 0],    # lower right corner
            [-1.732, -3.0, 0],   # lower left corner
        ])
        configs.append(config1[:12])
        
        # Configuration 2: Compact hexagonal arrangement (tighter packing)
        config2 = np.array([
            [0, 0, 0],           # center
            [0, 1.9, 0],         # top
            [0, -1.9, 0],        # bottom
            [1.65, 0.95, 0],     # top-right
            [-1.65, 0.95, 0],    # top-left
            [1.65, -0.95, 0],    # bottom-right
            [-1.65, -0.95, 0],   # bottom-left
            [3.3, 0, 0],         # far right
            [-3.3, 0, 0],        # far left
            [1.65, 2.85, 0],     # upper right corner
            [-1.65, 2.85, 0],    # upper left corner
            [1.65, -2.85, 0],    # lower right corner
            [-1.65, -2.85, 0],   # lower left corner
        ])
        configs.append(config2[:12])
        
        # Configuration 3: Hexagonal ring pattern (better for edge constraints)
        config3 = np.array([
            [0, 0, 0],           # center
            [0, 2.1, 0],         # top
            [1.8, 1.0, 0],       # top-right
            [1.8, -1.0, 0],      # bottom-right
            [0, -2.1, 0],        # bottom
            [-1.8, -1.0, 0],     # bottom-left
            [-1.8, 1.0, 0],      # top-left
            [3.6, 0, 0],         # far right
            [0, 3.6, 0],         # far top
            [-3.6, 0, 0],        # far left
            [0, -3.6, 0],        # far bottom
            [1.8, 2.1, 0],       # upper right corner
            [-1.8, 2.1, 0],      # upper left corner
            [1.8, -2.1, 0],      # lower right corner
            [-1.8, -2.1, 0],     # lower left corner
        ])
        configs.append(config3[:12])
        
        # Configuration 4: Optimized symmetric arrangement
        config4 = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [1.732, 1.0, 0],     # top-right
            [1.732, -1.0, 0],    # bottom-right
            [0, -2.0, 0],        # bottom
            [-1.732, -1.0, 0],   # bottom-left
            [-1.732, 1.0, 0],    # top-left
            [3.464, 0, 0],       # far right
            [0, 3.464, 0],       # far top
            [-3.464, 0, 0],      # far left
            [0, -3.464, 0],      # far bottom
            [1.732, 2.0, 0],     # upper right corner
            [-1.732, 2.0, 0],    # upper left corner
            [1.732, -2.0, 0],    # lower right corner
            [-1.732, -2.0, 0],   # lower left corner
        ])
        configs.append(config4[:12])
        
        return configs

    def evaluate_population_parallel(self, population: List[np.ndarray]) -> List[float]:
        """Parallel evaluation of population for faster processing"""
        def evaluate_individual(individual):
            return HexagonPackingEvaluator.evaluate_configuration_vectorized(individual)
        
        return Parallel(n_jobs=-1, prefer="threads")(
            delayed(evaluate_individual)(individual) for individual in population
        )

    def optimize_with_evolution(self, initial_config: np.ndarray) -> Tuple[np.ndarray, float]:
        """Improved evolutionary optimization with adaptive parameters"""
        # Stage 1: Population initialization with adaptive parameters
        population_size = 25
        generations = 75
        mutation_strength = 0.3
        
        # Start with best configuration
        population = [initial_config.copy() for _ in range(population_size)]
        
        for gen in range(generations):
            # Check timeout
            if time.time() - self.start_time > self.timeout * 0.8:
                break
                
            # Evaluate fitness of entire population in parallel
            fitness_scores = self.evaluate_population_parallel(population)
            
            # Select top performers (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = population_size // 3
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]
            
            # Generate new population through mutation
            new_population = elite.copy()
            
            # Fill remaining slots through mutation of elites
            while len(new_population) < population_size:
                parent = random.choice(elite)
                mutated = SymmetryAwareMutation.mutate_symmetrically(
                    parent, 
                    mutation_strength=mutation_strength, 
                    generation=gen, 
                    max_generations=generations
                )
                new_population.append(mutated)
            
            population = new_population
            
            # Track best overall
            for individual, score in zip(population, fitness_scores):
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = individual.copy()
        
        return self.best_config, self.best_score

    def refine_with_scipy_optimization(self, config: np.ndarray) -> np.ndarray:
        """Refine using scipy optimization with better constraint handling"""
        def objective_func(params):
            positions = params.reshape(-1, 2)
            temp_data = config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius_vectorized(temp_data)
            return outer_radius
        
        def constraint_func(params):
            positions = params.reshape(-1, 2)
            temp_data = config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            
            # Create hexagon polygons
            hexagons = []
            for i in range(12):
                center = (positions[i][0], positions[i][1])
                rotation = config[i][2]
                hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
                hexagons.append(hexagon)
            
            penalty = HexagonConstraintChecker.compute_overlap_penalty_vectorized(hexagons)
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius_vectorized(temp_data)
            
            return penalty
        
        try:
            # Flatten the initial positions for optimization
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()
            
            result = minimize(objective_func, initial_positions, method='L-BFGS-B', 
                             bounds=[(-5, 5) for _ in range(24)], 
                             constraints={'type': 'ineq', 'fun': constraint_func},
                             options={'maxiter': 150})
            
            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]
        except:
            pass  # Fall back to previous best if optimization fails
        
        return config

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete optimized optimization pipeline"""
        # Get multiple symmetric configurations
        configs = self.get_initial_configurations()
        
        # Try multiple configurations and find the best starting point
        best_initial_score = 0
        best_initial_config = None
        
        for config in configs:
            score = HexagonPackingEvaluator.evaluate_configuration_vectorized(config)
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()
        
        # Store the best configuration found so far
        self.best_score = best_initial_score
        self.best_config = best_initial_config.copy()
        
        # Stage 1: Evolutionary optimization with fine-tuned parameters
        print("Stage 1: Evolutionary optimization...")
        evolved_config, evolved_score = self.optimize_with_evolution(best_initial_config)
        
        # Stage 2: Scipy refinement
        print("Stage 2: Scipy refinement...")
        refined_config = self.refine_with_scipy_optimization(evolved_config)
        
        # Final evaluation
        final_score = HexagonPackingEvaluator.evaluate_configuration_vectorized(refined_config)
        final_outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius_vectorized(refined_config)
        outer_hex_side_length = final_outer_radius + 0.2  # Add margin
        
        # Return result
        outer_hex_data = np.array([0, 0, 0])  # centered at origin
        
        return refined_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HexagonPackingOptimizer()
    return optimizer.run_full_optimization()

# EVOLVE-BLOCK-END