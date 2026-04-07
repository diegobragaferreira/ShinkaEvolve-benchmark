# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import time
from itertools import product
import math
import random

class HexagonGeometry:
    """Handles all geometric computations for hexagons"""

    @staticmethod
    def hexagon_vertices(center_x, center_y, size=1, angle_deg=0):
        """Generate vertices of a regular hexagon given center, size, and rotation."""
        angle_rad = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            angle = angle_rad + i * np.pi / 3
            x = center_x + size * np.cos(angle)
            y = center_y + size * np.sin(angle)
            vertices.append((x, y))
        return np.array(vertices)

    @staticmethod
    def hexagon_area(size):
        """Calculate area of regular hexagon with given size."""
        return (3 * np.sqrt(3) / 2) * size ** 2

class ConstraintValidator:
    """Validates packing constraints efficiently"""

    def __init__(self, outer_center_x=0, outer_center_y=0):
        self.outer_center = np.array([outer_center_x, outer_center_y])

    def check_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely."""
        try:
            poly1 = Polygon(hex1_vertices)
            poly2 = Polygon(hex2_vertices)
            return poly1.intersects(poly2)
        except:
            return self._simple_overlap_check(hex1_vertices, hex2_vertices)

    def _simple_overlap_check(self, hex1_vertices, hex2_vertices):
        """Simple fallback overlap check using distance."""
        centroid1 = np.mean(hex1_vertices, axis=0)
        centroid2 = np.mean(hex2_vertices, axis=0)
        distance = np.linalg.norm(centroid1 - centroid2)
        return distance < 2.0

    def check_containment(self, hex_vertices, outer_radius):
        """Check if all vertices of a hexagon are inside the outer hexagon."""
        try:
            outer_vertices = HexagonGeometry.hexagon_vertices(
                self.outer_center[0], self.outer_center[1], outer_radius, 0
            )
            outer_polygon = Polygon(outer_vertices)

            for vertex in hex_vertices:
                point = Point(vertex[0], vertex[1])
                if not outer_polygon.contains(point):
                    return False
            return True
        except:
            return self._simple_containment_check(hex_vertices, outer_radius)

    def _simple_containment_check(self, hex_vertices, outer_radius):
        """Simple fallback containment check."""
        center = self.outer_center
        for vertex in hex_vertices:
            dist = np.linalg.norm(np.array(vertex) - center)
            if dist > outer_radius:
                return False
        return True

class SymmetryAwareOptimizer:
    """Symmetry-aware optimization approach for hexagon packing"""

    def __init__(self, validator, target_ratio=0.2537):
        self.validator = validator
        self.target_ratio = target_ratio
        self.max_iterations = 500
        self.population_size = 30
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7

    def compute_outer_radius(self, hex_data):
        """Calculate minimum outer radius that can contain all hexagons."""
        max_distance = 0
        for i in range(len(hex_data)):
            cx, cy, _ = hex_data[i]
            # Distance from center plus hexagon radius (1)
            distance = np.sqrt(cx**2 + cy**2) + 1
            max_distance = max(max_distance, distance)
        return max_distance

    def get_bounding_box(self, hex_data):
        """Get bounding box of all hexagons to optimize spatial queries."""
        if len(hex_data) == 0:
            return None, None, None, None

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for i in range(len(hex_data)):
            cx, cy, _ = hex_data[i]
            # We'll check all 6 corners for bounding box
            hex_vertices = HexagonGeometry.hexagon_vertices(cx, cy, 1, 0)
            for vx, vy in hex_vertices:
                min_x = min(min_x, vx)
                max_x = max(max_x, vx)
                min_y = min(min_y, vy)
                max_y = max(max_y, vy)

        return min_x, max_x, min_y, max_y

    def is_valid_configuration(self, hex_data, outer_radius, debug=False):
        """Fast validation of configuration with early exit on violation."""
        # Check containment first (quicker than overlap checking)
        for i in range(len(hex_data)):
            hex_vertices = HexagonGeometry.hexagon_vertices(
                hex_data[i][0], hex_data[i][1], 1, hex_data[i][2]
            )
            if not self.validator.check_containment(hex_vertices, outer_radius):
                if debug:
                    print(f"Containment failure for hex {i}")
                return False, 0.0

        # Check overlaps between pairs (this is the expensive part)
        for i in range(len(hex_data)):
            hex1_vertices = HexagonGeometry.hexagon_vertices(
                hex_data[i][0], hex_data[i][1], 1, hex_data[i][2]
            )
            for j in range(i+1, len(hex_data)):
                hex2_vertices = HexagonGeometry.hexagon_vertices(
                    hex_data[j][0], hex_data[j][1], 1, hex_data[j][2]
                )
                if self.validator.check_overlap(hex1_vertices, hex2_vertices):
                    if debug:
                        print(f"Overlap failure between hex {i} and {j}")
                    return False, 0.0

        return True, 1.0 / outer_radius

    def initialize_population(self, base_config):
        """Initialize population with symmetric variations of base configuration."""
        population = []

        # Add base configuration
        population.append(base_config.copy())

        # Add variations with small random mutations
        for _ in range(self.population_size - 1):
            mutated_config = base_config.copy()
            for i in range(len(mutated_config)):
                # Randomly perturb positions
                if random.random() < 0.7:  # 70% chance to mutate position
                    mutated_config[i][0] += random.uniform(-0.3, 0.3)
                    mutated_config[i][1] += random.uniform(-0.3, 0.3)
                # Randomly change rotation (not too much)
                if random.random() < 0.3:  # 30% chance to mutate rotation
                    mutated_config[i][2] += random.uniform(-15, 15)
                    mutated_config[i][2] = mutated_config[i][2] % 360
            population.append(mutated_config)

        return population

    def evaluate_individual(self, individual, outer_radius):
        """Evaluate a single individual in the population."""
        validity, inv_radius = self.is_valid_configuration(individual, outer_radius)
        if not validity:
            return -1e10  # Penalize invalid solutions
        return inv_radius

    def selection(self, population, fitness_scores):
        """Tournament selection for choosing parents."""
        tournament_size = 3
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_index].copy())
        return selected

    def crossover(self, parent1, parent2):
        """Uniform crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < self.crossover_rate:
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child1[i] = parent2[i].copy()
                    child2[i] = parent1[i].copy()

        return child1, child2

    def mutate_symmetrically(self, individual, generation, max_generations):
        """Mutate an individual with symmetry awareness."""
        mutated = individual.copy()

        # Dynamic mutation rate that decreases over time
        dynamic_mutation_rate = self.mutation_rate * (1 - generation / max_generations)

        for i in range(len(mutated)):
            # Apply position mutation with decreasing strength
            if random.random() < dynamic_mutation_rate:
                # Mutate position with decreased amplitude over time
                amplitude = 0.3 * (1 - generation / max_generations)
                mutated[i][0] += random.uniform(-amplitude, amplitude)
                mutated[i][1] += random.uniform(-amplitude, amplitude)

            # Apply rotation mutation
            if random.random() < dynamic_mutation_rate * 0.5:
                mutated[i][2] += random.uniform(-10, 10)
                mutated[i][2] = mutated[i][2] % 360

        return mutated

    def optimize_with_evolution(self, initial_config):
        """Perform evolutionary optimization with symmetry awareness."""
        # Initialize population
        population = self.initialize_population(initial_config)
        best_individual = initial_config.copy()
        best_fitness = -1e10

        for generation in range(self.max_iterations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            for individual in population:
                outer_radius = self.compute_outer_radius(individual)
                fitness = self.evaluate_individual(individual, outer_radius)
                fitness_scores.append(fitness)

                # Update best solution
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            # Early stopping condition
            if best_fitness > self.target_ratio:
                break

            # Selection
            selected_population = self.selection(population, fitness_scores)

            # Create new population through crossover and mutation
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_individual.copy())

            # Generate rest of population through crossover and mutation
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate_symmetrically(child1, generation, self.max_iterations)
                child2 = self.mutate_symmetrically(child2, generation, self.max_iterations)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

        return best_individual, best_fitness

    def local_refinement(self, config, outer_radius):
        """Apply local refinement to improve a good configuration."""
        # Try small adjustments to each position to improve packing
        best_config = config.copy()
        best_inv_radius = 0.0

        # For each hexagon, try small moves in 8 directions
        directions = [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1),
                     (0.1, 0.1), (-0.1, 0.1), (0.1, -0.1), (-0.1, -0.1)]

        for i in range(len(config)):
            for dx, dy in directions:
                # Create modified config
                test_config = config.copy()
                test_config[i][0] += dx
                test_config[i][1] += dy

                # Check validity
                valid, inv_radius = self.is_valid_configuration(test_config, outer_radius)
                if valid and inv_radius > best_inv_radius:
                    best_inv_radius = inv_radius
                    best_config = test_config.copy()

        return best_config, best_inv_radius

def generate_initial_config():
    """Generate a good initial configuration for 12 hexagons."""
    # Use a known good symmetric configuration that's reasonably close to optimal
    config = np.array([
        [0, 0, 0],           # center
        [-2.0, 0, 0],        # left
        [2.0, 0, 0],         # right
        [-1.0, 1.732, 0],    # top-left
        [1.0, 1.732, 0],     # top-right
        [-1.0, -1.732, 0],   # bottom-left
        [1.0, -1.732, 0],    # bottom-right
        [-3.0, 1.732, 0],    # far top-left
        [3.0, 1.732, 0],     # far top-right
        [-3.0, -1.732, 0],   # far bottom-left
        [3.0, -1.732, 0],    # far bottom-right
        [0, -3.464, 0]       # far bottom-center
    ])
    return config

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initialize components
    validator = ConstraintValidator()
    optimizer = SymmetryAwareOptimizer(validator)

    # Generate initial configuration
    initial_config = generate_initial_config()

    # Multi-stage optimization approach
    # Stage 1: Evolutionary optimization with symmetry awareness
    best_config, best_inv_radius = optimizer.optimize_with_evolution(initial_config)

    # Stage 2: Local refinement to polish the solution
    if best_inv_radius > 0:
        outer_radius = 1.0 / best_inv_radius if best_inv_radius > 0 else 10.0
        refined_config, refined_inv_radius = optimizer.local_refinement(best_config, outer_radius)
        if refined_inv_radius > best_inv_radius:
            best_config = refined_config
            best_inv_radius = refined_inv_radius

    # Final validation and fallback
    outer_radius = 1.0 / best_inv_radius if best_inv_radius > 0 else 10.0
    validity, final_inv_radius = optimizer.is_valid_configuration(best_config, outer_radius)

    if not validity:
        # Fall back to a known working configuration with reasonable parameters
        best_config = np.array([
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
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        final_inv_radius = 1.0 / 8.0  # Conservative estimate
        outer_radius = 8.0

    # Prepare return values
    inner_hex_data = np.array(best_config)
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    outer_hex_side_length = outer_radius * 2  # Side length calculation

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END