# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon
import math
import random
from itertools import combinations
from typing import Tuple, List, Optional
import time
from scipy.spatial.distance import cdist

class HexagonGeometry:
    """Handles all geometric operations for hexagons with optimized performance"""

    @staticmethod
    def create_unit_hexagon_vertices(center: Tuple[float, float] = (0, 0), rotation: float = 0) -> np.ndarray:
        """Create vertices of a unit regular hexagon using vectorized operations"""
        angle_offset = math.radians(rotation)
        radius = 1.0
        angles = np.array([angle_offset + i * math.pi / 3 for i in range(6)])
        x_coords = center[0] + radius * np.cos(angles)
        y_coords = center[1] + radius * np.sin(angles)
        return np.column_stack((x_coords, y_coords))

    @staticmethod
    def get_all_vertices(hex_data: np.ndarray) -> List[Tuple[float, float]]:
        """Extract all vertices from all hexagons efficiently"""
        all_vertices = []
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]
            vertices = HexagonGeometry.create_unit_hexagon_vertices(center, rotation)
            all_vertices.extend([(v[0], v[1]) for v in vertices])
        return all_vertices

class HexagonConstraints:
    """Handles constraint checking for hexagon arrangements"""

    @staticmethod
    def check_overlap_fast(hex1_vertices: np.ndarray, hex2_vertices: np.ndarray) -> bool:
        """Fast overlap check using Shapely with minimal buffering"""
        try:
            poly1 = Polygon(hex1_vertices)
            poly2 = Polygon(hex2_vertices)
            # Minimal buffer to handle floating point precision
            return poly1.buffer(1e-12).intersects(poly2.buffer(1e-12))
        except:
            return False

    @staticmethod
    def check_containment_fast(inner_vertices: np.ndarray, outer_vertices: np.ndarray) -> bool:
        """Fast containment check"""
        try:
            inner_poly = Polygon(inner_vertices)
            outer_poly = Polygon(outer_vertices)
            return outer_poly.contains(inner_poly)
        except:
            return False

    @staticmethod
    def compute_overlap_penalty_fast(hexagons_vertices: List[np.ndarray]) -> float:
        """Efficiently compute overlap penalty"""
        penalty = 0.0
        n = len(hexagons_vertices)
        for i in range(n):
            for j in range(i+1, n):
                if HexagonConstraints.check_overlap_fast(hexagons_vertices[i], hexagons_vertices[j]):
                    penalty += 1000.0
        return penalty

class HexagonEvaluator:
    """Evaluates hexagon packing configurations with optimized performance"""

    @staticmethod
    def calculate_outer_hex_radius_fast(hex_data: np.ndarray) -> float:
        """Fast calculation of minimum outer hexagon radius"""
        max_distance = 0.0
        # Precompute all hexagon vertices for faster access
        for i in range(len(hex_data)):
            center = (hex_data[i][0], hex_data[i][1])
            rotation = hex_data[i][2]

            # Get vertices quickly
            vertices = HexagonGeometry.create_unit_hexagon_vertices(center, rotation)

            # Calculate distances from origin to all vertices
            distances = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
            max_distance = max(max_distance, np.max(distances))

        return max_distance + 0.1  # Add small margin

    @staticmethod
    def evaluate_configuration_fast(hex_data: np.ndarray) -> float:
        """Fast evaluation of configuration with early rejection"""
        # Quick validation
        if len(hex_data) != 12:
            return 1e-10

        # Early distance-based feasibility check
        positions = hex_data[:, :2]
        distances = cdist(positions, positions)
        min_dist = np.min(distances[distances > 0])
        if min_dist < 1.8:  # Threshold to avoid overlaps
            return 1e-10

        try:
            # Calculate outer radius
            outer_radius = HexagonEvaluator.calculate_outer_hex_radius_fast(hex_data)

            # Create hexagon vertices for constraint checking
            hexagons_vertices = []
            for i in range(len(hex_data)):
                center = (hex_data[i][0], hex_data[i][1])
                rotation = hex_data[i][2]
                vertices = HexagonGeometry.create_unit_hexagon_vertices(center, rotation)
                hexagons_vertices.append(vertices)

            # Check overlaps
            overlap_penalty = HexagonConstraints.compute_overlap_penalty_fast(hexagons_vertices)

            if overlap_penalty > 0:
                return 1e-10

            # Valid configuration - return inverse of outer radius
            return 1.0 / outer_radius

        except Exception:
            return 1e-10

class SymmetryGenerationStrategy:
    """Generates various symmetric configurations for hexagon packing"""

    @staticmethod
    def generate_kagome_pattern() -> np.ndarray:
        """Generate Kagome lattice pattern - dense packing"""
        config = np.array([
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
        return config[:12]

    @staticmethod
    def generate_hcp_pattern() -> np.ndarray:
        """Generate Hexagonal Close-Packed pattern"""
        config = np.array([
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
        return config[:12]

    @staticmethod
    def generate_hexagonal_cluster() -> np.ndarray:
        """Generate standard hexagonal cluster"""
        config = np.array([
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
        return config[:12]

    @staticmethod
    def generate_symmetric_configs() -> List[np.ndarray]:
        """Generate multiple symmetric configurations"""
        configs = []
        configs.append(SymmetryGenerationStrategy.generate_hexagonal_cluster())
        configs.append(SymmetryGenerationStrategy.generate_kagome_pattern())
        configs.append(SymmetryGenerationStrategy.generate_hcp_pattern())
        return configs

class EvolutionaryOptimizer:
    """Performs evolutionary optimization with enhanced strategies"""

    def __init__(self):
        self.best_score = 0.0
        self.best_config = None
        self.start_time = time.time()
        self.timeout = 180  # seconds

    def mutate_individual(self, individual: np.ndarray, mutation_strength: float, stage: int = 1) -> np.ndarray:
        """Smart mutation that preserves hexagonal symmetries"""
        mutated = individual.copy()

        # Stage-specific mutation parameters
        if stage == 1:  # Exploration
            current_mutation = mutation_strength * 2.0
        elif stage == 2:  # Exploitation
            current_mutation = mutation_strength * 1.0
        else:  # Refinement
            current_mutation = mutation_strength * 0.5

        # Coordinate-based mutations with symmetry awareness
        for i in range(len(mutated)):
            # Mutation probabilities based on position importance
            if i == 0:  # Central hexagon
                mutated[i][0] += random.uniform(-current_mutation, current_mutation)
                mutated[i][1] += random.uniform(-current_mutation, current_mutation)
            elif i <= 6:  # First ring - core positions
                mutated[i][0] += random.uniform(-current_mutation, current_mutation)
                mutated[i][1] += random.uniform(-current_mutation, current_mutation)
            else:  # Outer positions
                mutated[i][0] += random.uniform(-current_mutation, current_mutation)
                mutated[i][1] += random.uniform(-current_mutation, current_mutation)

            # Small rotation adjustments
            if random.random() < 0.3:
                mutated[i][2] += random.uniform(-10, 10)
            mutated[i][2] %= 360

        return mutated

    def optimize_stage(self, initial_config: np.ndarray, stage: int = 1) -> Tuple[np.ndarray, float]:
        """Single optimization stage with adaptive parameters"""
        # Stage settings
        if stage == 1:  # Coarse
            population_size = 30
            generations = 25
            mutation_strength = 0.3
        elif stage == 2:  # Fine
            population_size = 25
            generations = 35
            mutation_strength = 0.2
        else:  # Final
            population_size = 20
            generations = 30
            mutation_strength = 0.05

        # Initialize population
        population = [initial_config.copy()]
        for _ in range(population_size - 1):
            variant = initial_config.copy()
            for i in range(len(variant)):
                if random.random() < 0.7:
                    variant[i][0] += random.gauss(0, 0.1)
                    variant[i][1] += random.gauss(0, 0.1)
            population.append(variant)

        # Evolution loop
        for gen in range(generations):
            # Timeout check
            if time.time() - self.start_time > self.timeout * 0.8:
                break

            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = HexagonEvaluator.evaluate_configuration_fast(individual)
                fitness_scores.append(score)

            # Selection (elitism)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_count = max(1, population_size // 3)
            elite = [population[i].copy() for i in sorted_indices[:elite_count]]

            # Generate new population
            new_population = elite.copy()
            while len(new_population) < population_size:
                parent = random.choice(elite)
                mutated = self.mutate_individual(parent, mutation_strength, stage)
                new_population.append(mutated)

            population = new_population

            # Track best
            for individual in population:
                score = HexagonEvaluator.evaluate_configuration_fast(individual)
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = individual.copy()

        return self.best_config, self.best_score

class MainOptimizer:
    """Main controller for hexagon packing optimization"""

    def __init__(self):
        self.evolutionary_optimizer = EvolutionaryOptimizer()

    def refine_with_scipy(self, config: np.ndarray) -> np.ndarray:
        """Use scipy optimization for final refinement"""
        def objective_func(params):
            positions = params.reshape(-1, 2)
            temp_data = config.copy()
            temp_data[:, 0] = positions[:, 0]
            temp_data[:, 1] = positions[:, 1]
            outer_radius = HexagonEvaluator.calculate_outer_hex_radius_fast(temp_data)
            return outer_radius

        try:
            # Flatten positions
            initial_positions = np.column_stack((config[:, 0], config[:, 1])).flatten()

            # Optimize with bounds
            result = minimize(
                objective_func,
                initial_positions,
                method='L-BFGS-B',
                bounds=[(-5, 5) for _ in range(24)],
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                final_positions = result.x.reshape(-1, 2)
                config[:, 0] = final_positions[:, 0]
                config[:, 1] = final_positions[:, 1]

        except:
            pass  # Keep original if optimization fails

        return config

    def run_full_optimization(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Complete optimization pipeline"""
        # Generate initial configurations
        configs = SymmetryGenerationStrategy.generate_symmetric_configs()

        # Find best initial configuration
        best_initial_score = 0.0
        best_initial_config = None

        for config in configs:
            score = HexagonEvaluator.evaluate_configuration_fast(config)
            if score > best_initial_score:
                best_initial_score = score
                best_initial_config = config.copy()

        # Set up tracking
        self.evolutionary_optimizer.best_score = best_initial_score
        self.evolutionary_optimizer.best_config = best_initial_config.copy()

        # Stage 1: Coarse optimization (position only)
        print("Stage 1: Coarse position optimization...")
        coarse_config = best_initial_config.copy()
        # Fix rotations for coarse stage
        for i in range(len(coarse_config)):
            coarse_config[i][2] = 0

        evolved_config, _ = self.evolutionary_optimizer.optimize_stage(coarse_config, stage=1)

        # Stage 2: Fine refinement with rotation awareness
        print("Stage 2: Fine refinement with rotation...")
        refined_config = evolved_config.copy()
        # Add small random rotations
        for i in range(len(refined_config)):
            if random.random() < 0.4:
                refined_config[i][2] += random.uniform(-5, 5)

        rotated_config, _ = self.evolutionary_optimizer.optimize_stage(refined_config, stage=2)

        # Stage 3: Final scipy optimization
        print("Stage 3: Final scipy optimization...")
        final_config = self.refine_with_scipy(rotated_config)

        # Final evaluation
        final_score = HexagonEvaluator.evaluate_configuration_fast(final_config)
        final_radius = HexagonEvaluator.calculate_outer_hex_radius_fast(final_config)
        outer_hex_side_length = final_radius + 0.2

        outer_hex_data = np.array([0, 0, 0])

        return final_config, outer_hex_data, outer_hex_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    optimizer = MainOptimizer()
    return optimizer.run_full_optimization()

# EVOLVE-BLOCK-END