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
    def group_hexagons_by_symmetry_class(hex_data: np.ndarray) -> List[List[int]]:
        """Group hexagons by their distance from center and angular position to respect hexagonal symmetry"""
        # Classify hexagons into symmetry classes
        symmetry_classes = []

        # Center hexagon
        symmetry_classes.append([0])

        # First ring (6 hexagons at distance 2.0)
        first_ring = []
        for i in range(1, 7):
            first_ring.append(i)
        symmetry_classes.append(first_ring)

        # Second ring (4 hexagons at distance 3.0)
        second_ring = []
        for i in range(7, 11):
            second_ring.append(i)
        symmetry_classes.append(second_ring)

        # Third ring (2 hexagons at distance 3.464)
        third_ring = []
        for i in range(11, 13):
            third_ring.append(i)
        symmetry_classes.append(third_ring)

        return symmetry_classes

    @staticmethod
    def apply_rotation_operation(hex_data: np.ndarray, angle_degrees: float) -> np.ndarray:
        """Apply rotation to all hexagons around the origin"""
        rotated_data = hex_data.copy()
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for i in range(len(rotated_data)):
            x, y = rotated_data[i][0], rotated_data[i][1]
            rotated_data[i][0] = x * cos_a - y * sin_a
            rotated_data[i][1] = x * sin_a + y * cos_a
        return rotated_data

    @staticmethod
    def apply_reflection_operation(hex_data: np.ndarray, axis_angle: float = 0) -> np.ndarray:
        """Apply reflection to all hexagons across specified axis"""
        reflected_data = hex_data.copy()
        axis_rad = math.radians(axis_angle)
        cos_a = math.cos(axis_rad)
        sin_a = math.sin(axis_rad)

        # Reflection matrix for axis angle
        # This reflects across line making angle 'axis_angle' with x-axis
        for i in range(len(reflected_data)):
            x, y = reflected_data[i][0], reflected_data[i][1]
            # Reflect point across line through origin at angle 'axis_angle'
            x_reflected = x * (cos_a * cos_a - sin_a * sin_a) + 2 * y * cos_a * sin_a
            y_reflected = 2 * x * cos_a * sin_a + y * (sin_a * sin_a - cos_a * cos_a)
            reflected_data[i][0] = x_reflected
            reflected_data[i][1] = y_reflected
        return reflected_data

    @staticmethod
    def mutate_symmetrically(hex_data: np.ndarray, mutation_strength: float = 0.2, stage: int = 1) -> np.ndarray:
        """Apply sophisticated symmetry-aware mutation respecting D6 dihedral group properties"""
        mutated_data = hex_data.copy()

        # Scale mutation strength based on optimization stage
        if stage == 1:  # Coarse stage - aggressive mutation for exploration
            current_mutation_strength = mutation_strength * 2.0
        elif stage == 2:  # Fine stage - moderate mutation
            current_mutation_strength = mutation_strength * 1.0
        else:  # Final stage - conservative mutation
            current_mutation_strength = mutation_strength * 0.5

        # Group hexagons by symmetry classes
        symmetry_classes = SymmetryAwareMutation.group_hexagons_by_symmetry_class(hex_data)

        # Mutate each symmetry class separately to preserve structure
        for class_indices in symmetry_classes:
            # For each class, apply coordinated mutation that maintains symmetry relationships
            if len(class_indices) == 1:
                # Single element (center)
                mutated_data[class_indices[0]][0] += random.uniform(-current_mutation_strength, current_mutation_strength)
                mutated_data[class_indices[0]][1] += random.uniform(-current_mutation_strength, current_mutation_strength)
            else:
                # Multiple elements in same orbit
                # Apply coordinated mutation to maintain relative positions in orbits
                base_x = mutated_data[class_indices[0]][0]
                base_y = mutated_data[class_indices[0]][1]

                # Apply random rotation to base position and derive others accordingly
                rotation_angle = random.uniform(-30, 30)  # Small rotation for variety
                cos_r = math.cos(math.radians(rotation_angle))
                sin_r = math.sin(math.radians(rotation_angle))

                # Apply coordinated mutation to all members of this symmetry class
                for idx in class_indices:
                    if idx == 0:  # Center remains unchanged
                        continue

                    # Get original position relative to center
                    rel_x = mutated_data[idx][0] - base_x
                    rel_y = mutated_data[idx][1] - base_y

                    # Rotate relative position
                    new_rel_x = rel_x * cos_r - rel_y * sin_r
                    new_rel_y = rel_x * sin_r + rel_y * cos_r

                    # Apply random displacement
                    displacement_x = random.uniform(-current_mutation_strength * 0.5, current_mutation_strength * 0.5)
                    displacement_y = random.uniform(-current_mutation_strength * 0.5, current_mutation_strength * 0.5)

                    # Update position
                    mutated_data[idx][0] = base_x + new_rel_x + displacement_x
                    mutated_data[idx][1] = base_y + new_rel_y + displacement_y

        # Occasionally apply full symmetry operations (reflection or rotation)
        if random.random() < 0.1:  # 10% chance of full symmetry operation
            if random.random() < 0.5:  # Apply rotation
                angle = random.choice([0, 60, 120, 180, 240, 300])  # Discrete rotations
                mutated_data = SymmetryAwareMutation.apply_rotation_operation(mutated_data, angle)
            else:  # Apply reflection
                axis = random.choice([0, 30, 60, 90, 120, 150])  # Discrete axes
                mutated_data = SymmetryAwareMutation.apply_reflection_operation(mutated_data, axis)

        return mutated_data

class HexagonPackingOptimizer:
    """Main optimizer class that orchestrates the packing process"""

    def __init__(self):
        self.best_score = 0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds

    def get_initial_configurations(self) -> List[np.ndarray]:
        """Generate several symmetric configurations to choose from"""
        configs = []

        # Configuration 1: Hexagonal cluster around center (inspired by mathematical literature)
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

        # Configuration 2: More compact arrangement (optimized for tight packing)
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

        # Configuration 3: Hexagonal ring pattern (good for exploring outer boundaries)
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

        # Configuration 4: Radial pattern (based on advanced packing theory)
        config4 = np.array([
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
        configs.append(config4[:12])

        return configs

    def optimize_stage(self, initial_config: np.ndarray, stage: int, max_generations: int = 50) -> Tuple[np.ndarray, float]:
        """Single stage optimization with specific mutation strategy"""
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
            elite_count = population_size // 3
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]

            # Generate new population through mutation
            new_population = elite.copy()

            # Fill remaining slots through mutation of elites
            while len(new_population) < population_size:
                parent = random.choice(elite)
                mutated = SymmetryAwareMutation.mutate_symmetrically(parent, mutation_strength=0.2, stage=stage)
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