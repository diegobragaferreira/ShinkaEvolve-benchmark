# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

@jit(nopython=True)
def hexagon_vertices_jitted(x, y, angle_deg, radius=1.0):
    """Jitted function to compute hexagon vertices"""
    vertices = np.zeros((6, 2))
    angle_rad = np.radians(angle_deg)
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        vertices[i] = [x + radius * np.cos(theta), y + radius * np.sin(theta)]
    return vertices

def create_unit_hexagon(center: Tuple[float, float] = (0, 0), rotation: float = 0) -> Polygon:
    """Create a unit regular hexagon with given center and rotation"""
    x, y = center
    angle_rad = math.radians(rotation)
    radius = 1.0
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        vx = x + radius * math.cos(angle)
        vy = y + radius * math.sin(angle)
        vertices.append((vx, vy))
    return Polygon(vertices)

class HexagonConstraintChecker:
    """Handles constraint checking for hexagon arrangements"""
    
    @staticmethod
    def check_overlap(hex1: Polygon, hex2: Polygon) -> bool:
        """Check if two hexagons overlap with numerical stability"""
        # Add small buffer to handle floating point precision issues
        buffered_hex1 = hex1.buffer(1e-10)
        buffered_hex2 = hex2.buffer(1e-10)
        return buffered_hex1.intersects(buffered_hex2)

    @staticmethod
    def check_containment(inner_hex: Polygon, outer_hex: Polygon) -> bool:
        """Check if inner hexagon is fully contained within outer hexagon"""
        return outer_hex.contains(inner_hex)

    @staticmethod
    def compute_overlap_penalty(hexagons: List[Polygon]) -> float:
        """Compute penalty for overlaps between hexagons"""
        penalty = 0
        n = len(hexagons)
        for i in range(n):
            for j in range(i+1, n):
                if HexagonConstraintChecker.check_overlap(hexagons[i], hexagons[j]):
                    penalty += 1000
        return penalty

class HexagonPackingEvaluator:
    """Evaluates hexagon packing configurations"""
    
    @staticmethod
    def calculate_outer_hex_radius(hex_data: np.ndarray) -> float:
        """Calculate minimum outer hexagon radius needed to contain all inner hexagons"""
        all_vertices = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = create_unit_hexagon(center, rotation)
            all_vertices.extend(list(hexagon.exterior.coords))
        
        if not all_vertices:
            return 0.0
            
        max_distance = 0
        for vertex in all_vertices:
            distance = math.sqrt(vertex[0]**2 + vertex[1]**2)
            max_distance = max(max_distance, distance)
        return max_distance + 0.1

    @staticmethod
    def evaluate_configuration(hex_data: np.ndarray) -> float:
        """Evaluate a configuration and return the inverse radius"""
        outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(hex_data)

        # Create hexagon polygons
        hexagons = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = create_unit_hexagon(center, rotation)
            hexagons.append(hexagon)

        # Compute penalties
        overlap_penalty = HexagonConstraintChecker.compute_overlap_penalty(hexagons)
        total_penalty = overlap_penalty

        # If valid configuration, return inverse of outer radius; otherwise return a very small value
        if total_penalty == 0:
            return 1.0 / outer_radius if outer_radius > 0 else 1e-10
        else:
            # Invalid configuration gets penalized heavily
            return 1e-10

class SymmetryAwareMutation:
    """Handles mutation strategies that preserve geometric symmetries"""
    
    @staticmethod
    def mutate_symmetrically(hex_data: np.ndarray, mutation_strength: float = 0.2, stage: int = 1, generation: int = 0, max_generations: int = 50) -> np.ndarray:
        """Apply symmetric mutation to maintain hexagonal properties with exponential decay"""
        mutated_data = hex_data.copy()

        # Apply exponential decay to mutation strength instead of linear
        decay_rate = 0.95
        mutation_factor = mutation_strength * (decay_rate ** generation) if max_generations > 0 else mutation_strength

        # Different mutation strategies based on optimization stage
        if stage == 1:  # Coarse stage - aggressive mutation for exploration
            effective_mutation_factor = mutation_factor * 2.0
        elif stage == 2:  # Fine stage - moderate mutation
            effective_mutation_factor = mutation_factor * 1.0
        else:  # Final stage - conservative mutation
            effective_mutation_factor = mutation_factor * 0.5

        # Mutate center hexagon with scaled mutation
        mutated_data[0][0] += random.uniform(-effective_mutation_factor, effective_mutation_factor)
        mutated_data[0][1] += random.uniform(-effective_mutation_factor, effective_mutation_factor)

        # Mutate radial positions with scaled mutation while preserving symmetry patterns
        for i in range(1, len(hex_data)):
            mutated_data[i][0] += random.uniform(-effective_mutation_factor, effective_mutation_factor)
            mutated_data[i][1] += random.uniform(-effective_mutation_factor, effective_mutation_factor)

        return mutated_data

    @staticmethod
    def generate_symmetric_configurations() -> List[np.ndarray]:
        """Generate highly symmetric initial configurations based on mathematical principles"""
        configs = []

        # Configuration 1: Optimal 12-hexagon pattern (inspired by mathematical literature)
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

        # Configuration 2: Compact hexagonal arrangement (better for tight packing)
        config2 = np.array([
            [0, 0, 0],           # center
            [0, 1.8, 0],         # top
            [0, -1.8, 0],        # bottom
            [1.55, 0.9, 0],      # top-right
            [-1.55, 0.9, 0],     # top-left
            [1.55, -0.9, 0],     # bottom-right
            [-1.55, -0.9, 0],    # bottom-left
            [3.1, 0, 0],         # far right
            [-3.1, 0, 0],        # far left
            [1.55, 2.7, 0],      # upper right corner
            [-1.55, 2.7, 0],     # upper left corner
            [1.55, -2.7, 0],     # lower right corner
            [-1.55, -2.7, 0],    # lower left corner
        ])
        configs.append(config2[:12])

        # Configuration 3: Ring pattern (good for exploring outer boundaries)
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

        # Configuration 4: Kagome lattice pattern - based on triangular lattice with additional symmetry
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
            [0, -3.464, 0],      # far bottom
            [-3.464, 0, 0],      # far left
            [1.732, 3.0, 0],     # upper right corner
            [-1.732, 3.0, 0],    # upper left corner
            [1.732, -3.0, 0],    # lower right corner
            [-1.732, -3.0, 0],   # lower left corner
        ])
        configs.append(config4[:12])

        # Configuration 5: Hexagonal Close-Packed (HCP) arrangement - maximizes density through efficient packing
        config5 = np.array([
            [0, 0, 0],           # center
            [0, 2.1, 0],         # top
            [0, -2.1, 0],        # bottom
            [1.8, 1.0, 0],       # top-right
            [-1.8, 1.0, 0],      # top-left
            [1.8, -1.0, 0],      # bottom-right
            [-1.8, -1.0, 0],     # bottom-left
            [3.6, 0, 0],         # far right
            [-3.6, 0, 0],        # far left
            [1.8, 2.1, 0],       # upper right corner
            [-1.8, 2.1, 0],      # upper left corner
            [1.8, -2.1, 0],      # lower right corner
            [-1.8, -2.1, 0],     # lower left corner
            [0, 4.2, 0],         # far top
            [0, -4.2, 0],        # far bottom
        ])
        configs.append(config5[:12])

        return configs

class HybridHexagonPackingOptimizer:
    """Main optimizer class that orchestrates the packing process"""
    
    def __init__(self):
        self.best_score = 0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds

    def get_initial_configurations(self) -> List[np.ndarray]:
        """Generate high-quality initial configurations with stochastic variants"""
        configs = []
        
        # Get base symmetric configurations
        base_configs = SymmetryAwareMutation.generate_symmetric_configurations()
        
        # Add stochastic variants to each base configuration
        for base_config in base_configs:
            configs.append(base_config.copy())
            
            # Create 2 additional stochastic variants for each base
            for _ in range(2):
                variant = base_config.copy()
                # Add small random perturbations
                for i in range(len(variant)):
                    if random.random() < 0.7:  # 70% chance to perturb
                        variant[i][0] += random.uniform(-0.1, 0.1)
                        variant[i][1] += random.uniform(-0.1, 0.1)
                configs.append(variant)

        return configs

    def optimize_stage(self, initial_config: np.ndarray, stage: int, max_generations: int = 50) -> Tuple[np.ndarray, float]:
        """Single stage optimization with adaptive temperature control"""
        # Stage-specific parameters
        if stage == 1:  # Coarse stage - high exploration
            population_size = 25
            mutation_strength = 0.3
        elif stage == 2:  # Fine stage - balanced exploration/exploitation
            population_size = 20
            mutation_strength = 0.15
        else:  # Final stage - high exploitation
            population_size = 15
            mutation_strength = 0.05

        # Start with best configuration and add perturbations
        population = [initial_config.copy()]
        for _ in range(population_size - 1):
            variant = initial_config.copy()
            # Add small random perturbations to positions
            for i in range(len(variant)):
                if random.random() < 0.5:
                    variant[i][0] += random.gauss(0, 0.1)
                if random.random() < 0.5:
                    variant[i][1] += random.gauss(0, 0.1)
            population.append(variant)

        for gen in range(max_generations):
            # Check timeout
            if time.time() - self.start_time > self.timeout * 0.8:
                break

            # Adaptive temperature scheduling with dynamic cooling
            base_temperature = 0.8
            cooling_factor = 0.95
            # Dynamic cooling based on progress to improve convergence behavior
            if gen < 20:
                temperature = base_temperature * (cooling_factor ** gen)
            else:
                # Accelerate cooling in later generations for faster convergence
                temperature = base_temperature * (cooling_factor ** (gen * 1.2))

            # Evaluate fitness of entire population
            fitness_scores = []
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
                fitness_scores.append(score)

            # Select top performers (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = population_size // 3
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]

            # Generate new population through mutation with acceptance criterion
            new_population = elite.copy()

            # Fill remaining slots through mutation with probabilistic acceptance
            while len(new_population) < population_size:
                parent = random.choice(elite)
                mutated = SymmetryAwareMutation.mutate_symmetrically(
                    parent, 
                    mutation_strength=mutation_strength, 
                    stage=stage, 
                    generation=gen, 
                    max_generations=max_generations
                )

                # Apply simulated annealing acceptance criterion
                mutated_score = HexagonPackingEvaluator.evaluate_configuration(mutated)
                parent_score = HexagonPackingEvaluator.evaluate_configuration(parent)

                # Accept if better, or with probability based on temperature if worse
                if mutated_score >= parent_score or random.random() < math.exp((mutated_score - parent_score) / temperature):
                    new_population.append(mutated)
                else:
                    new_population.append(parent)

            population = new_population

            # Track best overall
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
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
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(temp_data)
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
                hexagon = create_unit_hexagon(center, rotation)
                hexagons.append(hexagon)

            penalty = HexagonConstraintChecker.compute_overlap_penalty(hexagons)
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(temp_data)

            return penalty

        try:
            # Flatten the initial positions for optimization
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()

            result = minimize(objective_func, initial_positions, method='L-BFGS-B',
                             bounds=[(-5, 5) for _ in range(24)],
                             constraints={'type': 'ineq', 'fun': constraint_func},
                             options={'maxiter': 100, 'ftol': 1e-6})

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]
        except:
            pass  # Fall back to previous best if optimization fails

        return config

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete multi-scale hybrid optimization pipeline"""
        # Get multiple symmetric configurations with variants
        configs = self.get_initial_configurations()

        # Try multiple configurations and find the best starting point
        best_initial_score = 0
        best_initial_config = None

        for config in configs:
            score = HexagonPackingEvaluator.evaluate_configuration(config)
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()

        # Store the best configuration found so far
        self.best_score = best_initial_score
        self.best_config = best_initial_config.copy()

        # Stage 1: Coarse-grained position optimization (fixed rotations)
        print("Stage 1: Coarse position optimization...")
        coarse_config = best_initial_config.copy()
        # Fix rotations for this stage
        for i in range(len(coarse_config)):
            coarse_config[i][2] = 0  # Set all rotations to 0 for coarse optimization

        evolved_config, evolved_score = self.optimize_stage(coarse_config, stage=1, max_generations=30)

        # Stage 2: Fine-grained refinement with rotation awareness
        print("Stage 2: Fine-grained refinement...")
        # Allow rotations to vary but keep positions relatively close to evolved ones
        refined_config = evolved_config.copy()
        # Add small random rotations to improve packing
        for i in range(len(refined_config)):
            # Perturb rotations slightly
            if random.random() < 0.4:
                refined_config[i][2] += random.uniform(-5, 5)

        # Run evolution again with rotations allowed but more constrained
        rotated_config, rotated_score = self.optimize_stage(refined_config, stage=2, max_generations=30)

        # Stage 3: Full optimization with scipy refinement
        print("Stage 3: Full scipy optimization...")
        final_config = self.refine_with_scipy_optimization(rotated_config)

        # Stage 4: Hybrid refinement with continued evolutionary process
        print("Stage 4: Hybrid refinement...")
        # Run one final evolutionary stage with fine parameters
        hybrid_config, _ = self.optimize_stage(final_config, stage=3, max_generations=20)

        # Final evaluation
        final_score = HexagonPackingEvaluator.evaluate_configuration(hybrid_config)
        final_outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(hybrid_config)
        outer_hex_side_length = final_outer_radius + 0.2  # Add margin

        # Return result
        outer_hex_data = np.array([0, 0, 0])  # centered at origin

        return hybrid_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = HybridHexagonPackingOptimizer()
    return optimizer.run_full_optimization()

# EVOLVE-BLOCK-END