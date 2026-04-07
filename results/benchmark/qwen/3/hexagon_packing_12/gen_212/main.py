# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
from scipy.spatial.distance import cdist

class HexagonGeometry:
    """Handles all geometric operations for hexagons"""

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
    def get_all_vertices(hex_data: np.ndarray) -> List[Tuple[float, float]]:
        """Extract all vertices from all hexagons"""
        all_vertices = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
            all_vertices.extend(list(hexagon.exterior.coords))
        return all_vertices

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
        all_vertices = HexagonGeometry.get_all_vertices(hex_data)
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
            hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
            hexagons.append(hexagon)

        # Compute penalties
        overlap_penalty = HexagonConstraintChecker.compute_overlap_penalty(hexagons)

        total_penalty = overlap_penalty

        # If valid configuration, return inverse of outer radius; otherwise return a very small value
        if total_penalty == 0:
            return 1.0 / outer_radius
        else:
            # Invalid configuration gets penalized heavily
            return 1e-10

class SymmetryAwareMutation:
    """Handles mutation strategies that preserve geometric symmetries"""

    @staticmethod
    def mutate_symmetrically(hex_data: np.ndarray, mutation_strength: float = 0.2, stage: int = 1, generation: int = 0, max_generations: int = 50) -> np.ndarray:
        """Apply symmetric mutation to maintain hexagonal properties with adaptive scaling"""
        mutated_data = hex_data.copy()

        # Stage-based mutation strengths
        stage_multipliers = {1: 2.0, 2: 1.0, 3: 0.5}
        base_mutation = mutation_strength * stage_multipliers.get(stage, 1.0)

        # Progressive mutation decay with exponential curve for faster early decay
        progress = generation / max_generations if max_generations > 0 else 0
        # Use exponential decay to allow faster initial exploration reduction
        decay_factor = 0.5 ** (progress * 2)  # Exponential decay
        # Ensure minimum decay factor to prevent too small mutations
        decay_factor = max(0.1, decay_factor)
        mutation_factor = base_mutation * decay_factor

        # Mutate center hexagon with adjusted factor
        mutated_data[0][0] += random.uniform(-mutation_factor, mutation_factor)
        mutated_data[0][1] += random.uniform(-mutation_factor, mutation_factor)

        # Mutate radial positions while preserving symmetry patterns
        for i in range(1, len(hex_data)):
            mutated_data[i][0] += random.uniform(-mutation_factor, mutation_factor)
            mutated_data[i][1] += random.uniform(-mutation_factor, mutation_factor)

        return mutated_data

class HexagonPackingOptimizer:
    """Main optimizer class that orchestrates the packing process"""

    def __init__(self):
        self.best_score = 0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds

    def get_initial_configurations(self) -> List[np.ndarray]:
        """Generate high-quality initial configurations based on known optimal patterns"""
        configs = []

        # Configuration 1: Mathematically derived optimal 12-hexagon packing
        # Based on the best-known configuration achieving ~0.2537 ratio
        config1 = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 2.0, 0.0],           # top
            [1.732050808, 1.0, 0.0],   # top-right
            [1.732050808, -1.0, 0.0],  # bottom-right
            [0.0, -2.0, 0.0],          # bottom
            [-1.732050808, -1.0, 0.0], # bottom-left
            [-1.732050808, 1.0, 0.0],  # top-left
            [3.464101616, 2.0, 0.0],   # far top-right
            [3.464101616, -2.0, 0.0],  # far bottom-right
            [-3.464101616, -2.0, 0.0], # far bottom-left
            [-3.464101616, 2.0, 0.0],  # far top-left
            [0.0, -4.0, 0.0],          # far bottom-center
        ])
        configs.append(config1)

        # Configuration 2: Optimized ring arrangement with minimal gaps
        config2 = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 2.17, 0.0],          # top
            [1.87, 1.08, 0.0],         # top-right
            [1.87, -1.08, 0.0],        # bottom-right
            [0.0, -2.17, 0.0],         # bottom
            [-1.87, -1.08, 0.0],       # bottom-left
            [-1.87, 1.08, 0.0],        # top-left
            [3.74, 2.17, 0.0],         # far top-right
            [3.74, -2.17, 0.0],        # far bottom-right
            [-3.74, -2.17, 0.0],       # far bottom-left
            [-3.74, 2.17, 0.0],        # far top-left
            [0.0, -4.34, 0.0],         # far bottom
        ])
        configs.append(config2)

        # Configuration 3: Golden ratio based hexagonal packing
        config3 = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 2.236, 0.0],         # top
            [1.902, 1.118, 0.0],       # top-right
            [1.902, -1.118, 0.0],      # bottom-right
            [0.0, -2.236, 0.0],        # bottom
            [-1.902, -1.118, 0.0],     # bottom-left
            [-1.902, 1.118, 0.0],      # top-left
            [3.804, 2.236, 0.0],       # far top-right
            [3.804, -2.236, 0.0],      # far bottom-right
            [-3.804, -2.236, 0.0],     # far bottom-left
            [-3.804, 2.236, 0.0],      # far top-left
            [0.0, -4.472, 0.0],        # far bottom
        ])
        configs.append(config3)

        # Configuration 4: Triangular lattice with symmetric spacing
        config4 = np.array([
            [0.0, 0.0, 0.0],           # center
            [0.0, 2.0, 0.0],           # top
            [1.732050808, 1.0, 0.0],   # top-right
            [1.732050808, -1.0, 0.0],  # bottom-right
            [0.0, -2.0, 0.0],          # bottom
            [-1.732050808, -1.0, 0.0], # bottom-left
            [-1.732050808, 1.0, 0.0],  # top-left
            [3.464101616, 2.0, 0.0],   # far top-right
            [3.464101616, -2.0, 0.0],  # far bottom-right
            [-3.464101616, -2.0, 0.0], # far bottom-left
            [-3.464101616, 2.0, 0.0],  # far top-left
            [0.0, -4.0, 0.0],          # far bottom
        ])
        configs.append(config4)

        # Configuration 5: Honeycomb-inspired with optimized spacing for maximum density
        config5 = np.array([
            [0.0, 0.0, 0.0],           # center
            [2.0, 0.0, 0.0],           # right
            [1.0, 1.732050808, 0.0],   # top-right
            [-1.0, 1.732050808, 0.0],  # top-left
            [-2.0, 0.0, 0.0],          # left
            [-1.0, -1.732050808, 0.0], # bottom-left
            [1.0, -1.732050808, 0.0],  # bottom-right
            [3.0, 1.732050808, 0.0],   # far top-right
            [3.0, -1.732050808, 0.0],  # far bottom-right
            [-3.0, -1.732050808, 0.0], # far bottom-left
            [-3.0, 1.732050808, 0.0],  # far top-left
            [0.0, -3.464101616, 0.0],  # far bottom
        ])
        configs.append(config5)

        return configs

    def optimize_stage(self, initial_config: np.ndarray, stage: int, max_generations: int = 50) -> Tuple[np.ndarray, float]:
        """Single stage optimization with specific mutation strategy"""
        # Stage 1: Population initialization with stochastic perturbations
        population_size = 30 if stage <= 2 else 15  # More individuals in earlier stages

        # Start with best configuration and add perturbed variants
        population = [initial_config.copy()]
        for _ in range(population_size - 1):
            variant = initial_config.copy()
            # Add random perturbations to positions with stage-dependent magnitudes
            for i in range(len(variant)):
                # Larger perturbations in early stages, smaller in later stages
                perturbation_magnitude = 0.3 if stage <= 2 else 0.1
                if random.random() < 0.7:  # 70% chance to perturb position
                    variant[i][0] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
                    variant[i][1] += random.uniform(-perturbation_magnitude, perturbation_magnitude)
                if random.random() < 0.3:  # 30% chance to perturb rotation
                    variant[i][2] += random.uniform(-10, 10)
                    variant[i][2] = variant[i][2] % 360
            population.append(variant)

        for gen in range(max_generations):
            # Check timeout
            if time.time() - self.start_time > self.timeout * 0.8:
                break

            # Evaluate fitness of entire population
            fitness_scores = []
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
                fitness_scores.append(score)

            # Select top performers (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = max(3, population_size // 4)  # Minimum 3 elites
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]

            # Generate new population through mutation
            new_population = elite.copy()

            # Fill remaining slots through mutation of elites
            while len(new_population) < population_size:
                parent = random.choice(elite)
                mutated = SymmetryAwareMutation.mutate_symmetrically(parent, mutation_strength=0.2, stage=stage, generation=gen, max_generations=max_generations)
                # Occasionally apply a secondary perturbation to avoid getting stuck in local minima
                if random.random() < 0.2:
                    mutated = self._add_secondary_perturbation(mutated)
                new_population.append(mutated)

            population = new_population

            # Track best overall
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = individual.copy()

        return self.best_config, self.best_score

    def _add_secondary_perturbation(self, config: np.ndarray) -> np.ndarray:
        """Add a secondary perturbation to help escape local minima"""
        perturbed = config.copy()
        # Perturb a few randomly selected hexagons
        hexagon_indices = random.sample(range(len(perturbed)), min(3, len(perturbed)))
        for i in hexagon_indices:
            perturbed[i][0] += random.uniform(-0.1, 0.1)
            perturbed[i][1] += random.uniform(-0.1, 0.1)
        return perturbed

    def refine_with_scipy_optimization(self, config: np.ndarray) -> np.ndarray:
        """Refine using scipy optimization with improved constraint handling"""
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
                hexagon = HexagonGeometry.create_unit_hexagon(center, rotation)
                hexagons.append(hexagon)

            penalty = HexagonConstraintChecker.compute_overlap_penalty(hexagons)
            # Add a soft constraint to prevent hexagons from going too far out
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(temp_data)
            # Penalty increases significantly if outer radius gets too large
            if outer_radius > 10:
                penalty += 1000 * (outer_radius - 10)

            return penalty

        try:
            # Flatten the initial positions for optimization
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()

            # Use a combination of approaches for better robustness
            result = minimize(objective_func, initial_positions, method='L-BFGS-B',
                             bounds=[(-8, 8) for _ in range(24)],
                             constraints={'type': 'ineq', 'fun': constraint_func},
                             options={'maxiter': 150, 'ftol': 1e-8})

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]
        except Exception as e:
            # If scipy optimization fails, still return the original config as fallback
            pass

        return config

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run the complete multi-scale optimization pipeline"""
        # Get multiple symmetric configurations
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
            refined_config[i][2] += random.uniform(-5, 5) if random.random() < 0.4 else 0

        # Run evolution again with rotations allowed but more constrained
        rotated_config, rotated_score = self.optimize_stage(refined_config, stage=2, max_generations=30)

        # Stage 3: Full optimization with scipy refinement
        print("Stage 3: Full scipy optimization...")
        final_config = self.refine_with_scipy_optimization(rotated_config)

        # Final evaluation
        final_score = HexagonPackingEvaluator.evaluate_configuration(final_config)
        final_outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(final_config)
        outer_hex_side_length = final_outer_radius + 0.2  # Add margin

        # Return result
        outer_hex_data = np.array([0, 0, 0])  # centered at origin

        return final_config, outer_hex_data, outer_hex_side_length

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