# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time

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
    def mutate_symmetrically(hex_data: np.ndarray, mutation_strength: float = 0.2, stage: int = 1, temperature: float = 1.0) -> np.ndarray:
        """Apply advanced symmetric mutation to maintain hexagonal properties with temperature control"""
        mutated_data = hex_data.copy()

        # Different mutation strategies based on optimization stage
        if stage == 1:  # Coarse stage - aggressive mutation for exploration
            mutation_factor = mutation_strength * 2.0 * temperature
        elif stage == 2:  # Fine stage - moderate mutation
            mutation_factor = mutation_strength * 1.0 * temperature
        else:  # Final stage - conservative mutation
            mutation_factor = mutation_strength * 0.5 * temperature

        # Apply group-based mutations that respect D6 symmetry (hexagonal symmetry group)

        # 1. Center hexagon mutation (always preserved)
        mutated_data[0][0] += random.gauss(0, mutation_factor * 0.3)
        mutated_data[0][1] += random.gauss(0, mutation_factor * 0.3)

        # 2. First ring mutations (6 hexagons) - use rotational symmetry
        first_ring_indices = [1, 2, 3, 4, 5, 6]  # Top, top-left, top-right, bottom, bottom-left, bottom-right

        # Create a base mutation vector for rotational symmetry
        base_mutation = np.array([random.gauss(0, mutation_factor * 0.8), random.gauss(0, mutation_factor * 0.8)])

        # Apply rotational symmetry operations
        angles = [0, 60, 120, 180, 240, 300]  # 60-degree increments for hexagonal symmetry

        for i, idx in enumerate(first_ring_indices):
            # Rotate the base mutation vector according to position
            angle_rad = math.radians(angles[i])
            rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                                       [math.sin(angle_rad), math.cos(angle_rad)]])

            rotated_mutation = rotation_matrix @ base_mutation

            mutated_data[idx][0] += rotated_mutation[0]
            mutated_data[idx][1] += rotated_mutation[1]

        # 3. Second ring mutations (6 hexagons) - maintain symmetry with respect to both rings
        second_ring_indices = [7, 8, 9, 10, 11]  # Far positions (not including center)

        # Create second ring base mutation
        base_mutation_second = np.array([random.gauss(0, mutation_factor * 0.5), random.gauss(0, mutation_factor * 0.5)])

        # Apply different symmetry operations for second ring
        for i, idx in enumerate(second_ring_indices):
            # Apply a more varied mutation pattern for second ring
            if i % 2 == 0:
                mutated_data[idx][0] += random.gauss(0, mutation_factor * 0.4)
                mutated_data[idx][1] += random.gauss(0, mutation_factor * 0.4)
            else:
                mutated_data[idx][0] += random.gauss(0, mutation_factor * 0.6)
                mutated_data[idx][1] += random.gauss(0, mutation_factor * 0.6)

        # 4. Apply reflectional symmetry (maintain mirror properties)
        # Reflect across vertical axis for symmetry preservation
        # Keep center as is, reflect pairs around y-axis
        for i in range(1, 7):  # First ring hexagons
            if i <= 6:
                # For symmetric pairs, apply consistent changes
                reflection_pair = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}  # Top <-> Bottom, etc.
                if i in reflection_pair:
                    pair_index = reflection_pair[i]
                    # Ensure pair maintains symmetry by applying similar but opposite changes
                    if i < pair_index:  # Only update once per pair
                        x_diff = mutated_data[pair_index][0] - mutated_data[i][0]
                        y_diff = mutated_data[pair_index][1] - mutated_data[i][1]

                        # Apply symmetric adjustment to maintain balance
                        mutated_data[pair_index][0] = mutated_data[i][0] - x_diff
                        mutated_data[pair_index][1] = mutated_data[i][1] - y_diff

        # 5. Add small random rotations to break exact symmetry and promote exploration
        rotation_mutation_strength = mutation_factor * 0.1
        for i in range(len(mutated_data)):
            # Apply small rotations to break exact symmetry but maintain structure
            mutated_data[i][2] += random.gauss(0, rotation_mutation_strength * 20)
            mutated_data[i][2] = mutated_data[i][2] % 360

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

        # Configuration 1: HCP-like arrangement (Hexagonal Close Packed)
        # Based on known dense packing patterns
        config1 = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [1.732, 1.0, 0],     # top-right
            [1.732, -1.0, 0],    # bottom-right
            [0, -2.0, 0],        # bottom
            [-1.732, -1.0, 0],   # bottom-left
            [-1.732, 1.0, 0],    # top-left
            [3.464, 2.0, 0],     # far top-right
            [3.464, -2.0, 0],    # far bottom-right
            [-3.464, -2.0, 0],   # far bottom-left
            [-3.464, 2.0, 0],    # far top-left
            [0, -4.0, 0],        # far bottom
        ])
        configs.append(config1[:12])

        # Configuration 2: Kagome lattice-inspired pattern
        # Combines triangular and hexagonal structures
        config2 = np.array([
            [0, 0, 0],           # center
            [0, 1.8, 0],         # top
            [1.55, 0.9, 0],      # top-right
            [1.55, -0.9, 0],     # bottom-right
            [0, -1.8, 0],        # bottom
            [-1.55, -0.9, 0],    # bottom-left
            [-1.55, 0.9, 0],     # top-left
            [3.1, 0, 0],         # far right
            [0, 3.1, 0],         # far top
            [-3.1, 0, 0],        # far left
            [0, -3.1, 0],        # far bottom
            [1.55, 2.7, 0],      # upper right corner
        ])
        configs.append(config2[:12])

        # Configuration 3: Optimized pattern close to known SOTA
        config3 = np.array([
            [0, 0, 0],           # center
            [0, 2.0, 0],         # top
            [1.732, 1.0, 0],     # top-right
            [1.732, -1.0, 0],    # bottom-right
            [0, -2.0, 0],        # bottom
            [-1.732, -1.0, 0],   # bottom-left
            [-1.732, 1.0, 0],    # top-left
            [3.464, 0, 0],       # far right
            [-3.464, 0, 0],      # far left
            [1.732, 3.0, 0],     # upper right corner
            [-1.732, 3.0, 0],    # upper left corner
            [1.732, -3.0, 0],    # lower right corner
            [-1.732, -3.0, 0],   # lower left corner
        ])
        configs.append(config3[:12])

        # Configuration 4: Perturbed version of known good configuration
        # Adds a bit of randomness to explore neighborhoods
        config4 = config1[:12].copy()
        for i in range(len(config4)):
            config4[i][0] += random.uniform(-0.05, 0.05)
            config4[i][1] += random.uniform(-0.05, 0.05)
            # Slightly randomize rotations
            config4[i][2] += random.uniform(-2, 2)
        configs.append(config4)

        # Configuration 5: Alternating ring pattern
        # Creates a more balanced outer boundary coverage
        config5 = np.array([
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
        ])
        configs.append(config5[:12])

        return configs

    def optimize_stage(self, initial_config: np.ndarray, stage: int, max_generations: int = 50) -> Tuple[np.ndarray, float]:
        """Single stage optimization with specific mutation strategy and temperature control"""
        # Stage 1: Population initialization with stochastic perturbations
        population_size = 25 if stage <= 2 else 15  # Fewer individuals in final stage

        # Start with best configuration and add perturbations
        population = [initial_config.copy()]
        for _ in range(population_size - 1):
            variant = initial_config.copy()
            # Add small random perturbations to positions
            for i in range(len(variant)):
                variant[i][0] += random.gauss(0, 0.1) if random.random() < 0.5 else 0
                variant[i][1] += random.gauss(0, 0.1) if random.random() < 0.5 else 0
            population.append(variant)

        # Temperature decay parameters
        initial_temp = 1.0 if stage <= 2 else 0.5
        final_temp = 0.01 if stage <= 2 else 0.001
        temp_decay_rate = (final_temp / initial_temp) ** (1.0 / max_generations) if max_generations > 0 else 0

        for gen in range(max_generations):
            # Check timeout
            if time.time() - self.start_time > self.timeout * 0.8:
                break

            # Calculate temperature for this generation
            temperature = initial_temp * (temp_decay_rate ** gen) if max_generations > 0 else final_temp

            # Evaluate fitness of entire population
            fitness_scores = []
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
                fitness_scores.append(score)

            # Select top performers (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = population_size // 3
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]

            # Generate new population through mutation
            new_population = elite.copy()

            # Fill remaining slots through mutation of elites
            while len(new_population) < population_size:
                parent = random.choice(elite)
                mutated = SymmetryAwareMutation.mutate_symmetrically(parent, mutation_strength=0.2, stage=stage, temperature=temperature)
                new_population.append(mutated)

            population = new_population

            # Track best overall
            for individual in population:
                score = HexagonPackingEvaluator.evaluate_configuration(individual)
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = individual.copy()

        return self.best_config, self.best_score

    def refine_with_scipy_optimization(self, config: np.ndarray) -> np.ndarray:
        """Refine using scipy optimization"""
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
            outer_radius = HexagonPackingEvaluator.calculate_outer_hex_radius(temp_data)

            return penalty

        try:
            # Flatten the initial positions for optimization
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()

            result = minimize(objective_func, initial_positions, method='L-BFGS-B',
                             bounds=[(-5, 5) for _ in range(24)],
                             constraints={'type': 'ineq', 'fun': constraint_func},
                             options={'maxiter': 100})

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]
        except:
            pass  # Fall back to previous best if optimization fails

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