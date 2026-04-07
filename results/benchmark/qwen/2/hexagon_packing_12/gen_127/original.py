# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import random
from scipy.spatial.distance import cdist
import time
from math import cos, sin, pi
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import warnings

@dataclass
class HexagonConfig:
    """Data class representing a hexagon configuration."""
    center_x: float
    center_y: float
    rotation_deg: float  # degrees

@dataclass
class PackingResult:
    """Data class representing the final packing result."""
    inner_hex_data: np.ndarray  # shape (12, 3)
    outer_hex_data: np.ndarray  # shape (3,)
    outer_hex_side_length: float

class HexagonGeometry:
    """Handles all geometric operations for hexagons."""

    @staticmethod
    def generate_vertices(center_x: float, center_y: float, side_length: float = 1,
                         rotation_deg: float = 0) -> List[Tuple[float, float]]:
        """Generate vertices of a regular hexagon."""
        angle_step = pi / 3
        rotation_rad = np.radians(rotation_deg)
        vertices = []
        for i in range(6):
            angle = rotation_rad + i * angle_step
            x = center_x + side_length * cos(angle)
            y = center_y + side_length * sin(angle)
            vertices.append((x, y))
        return vertices

    @staticmethod
    def create_polygon(center_x: float, center_y: float, side_length: float = 1,
                      rotation_deg: float = 0) -> Polygon:
        """Create a shapely polygon for a hexagon."""
        vertices = HexagonGeometry.generate_vertices(center_x, center_y, side_length, rotation_deg)
        return Polygon(vertices)

    @staticmethod
    def get_bounding_radius(configs: List[HexagonConfig]) -> float:
        """Calculate the bounding radius needed for all hexagons."""
        if not configs:
            return 1.0

        # Collect all vertices
        all_vertices = []
        for config in configs:
            vertices = HexagonGeometry.generate_vertices(config.center_x, config.center_y, 1, config.rotation_deg)
            all_vertices.extend(vertices)

        if not all_vertices:
            return 1.0

        # Compute centroid
        xs = [v[0] for v in all_vertices]
        ys = [v[1] for v in all_vertices]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        # Find maximum distance to centroid
        max_dist = 0
        for x, y in all_vertices:
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = max(max_dist, dist)

        return max_dist + 0.01  # Add small buffer

class ConstraintChecker:
    """Handles all constraint checking operations."""

    @staticmethod
    def check_containment(hexagon_poly: Polygon, outer_hex_poly: Polygon) -> bool:
        """Check if hexagon is fully contained within outer hexagon."""
        vertices = list(hexagon_poly.exterior.coords)
        for point in vertices:
            if not outer_hex_poly.contains(Point(point[0], point[1])):
                return False
        return True

    @staticmethod
    def check_overlap(hex1_poly: Polygon, hex2_poly: Polygon) -> bool:
        """Check if two hexagons overlap."""
        return hex1_poly.intersects(hex2_poly) and not hex1_poly.touches(hex2_poly)

    @staticmethod
    def count_violations(inner_configs: List[HexagonConfig],
                        outer_side_length: float) -> Tuple[int, int]:
        """Count overlap and containment violations."""
        # Create polygons
        inner_polygons = [HexagonGeometry.create_polygon(
            cfg.center_x, cfg.center_y, 1, cfg.rotation_deg) for cfg in inner_configs]

        outer_polygon = HexagonGeometry.create_polygon(0, 0, outer_side_length, 0)

        overlap_count = 0
        containment_count = 0

        # Check containment
        for poly in inner_polygons:
            if not ConstraintChecker.check_containment(poly, outer_polygon):
                containment_count += 1

        # Check overlaps (optimized by early termination)
        for i in range(len(inner_polygons)):
            for j in range(i+1, len(inner_polygons)):
                if ConstraintChecker.check_overlap(inner_polygons[i], inner_polygons[j]):
                    overlap_count += 1
                    # Early termination if too many violations
                    if overlap_count > 10:  # Reasonable threshold
                        break
            if overlap_count > 10:
                break

        return overlap_count, containment_count

class FitnessEvaluator:
    """Handles fitness computation and optimization logic."""

    PENALTY_WEIGHTS = {
        'overlap': 1000,
        'containment': 1000
    }

    @staticmethod
    def evaluate_fitness(individual: np.ndarray, outer_side_length: float) -> float:
        """Evaluate fitness of individual configuration."""
        # Convert individual to hex data format
        configs = []
        for i in range(12):
            cx = individual[i*3]
            cy = individual[i*3 + 1]
            rot = individual[i*3 + 2]
            configs.append(HexagonConfig(cx, cy, rot))

        # Count violations
        overlap_count, containment_count = ConstraintChecker.count_violations(configs, outer_side_length)

        # Penalty for constraint violations
        penalty = (overlap_count * FitnessEvaluator.PENALTY_WEIGHTS['overlap'] +
                  containment_count * FitnessEvaluator.PENALTY_WEIGHTS['containment'])

        # Fitness is inverse of outer hex side length minus penalties
        if overlap_count > 0 or containment_count > 0:
            return -penalty  # Very bad fitness if constraints violated
        else:
            return 1.0 / outer_side_length

class PopulationManager:
    """Manages population operations and evolutionary processes."""

    def __init__(self, pop_size: int = 100, num_hexagons: int = 12):
        self.pop_size = pop_size
        self.num_hexagons = num_hexagons
        self.best_fitness = -float('inf')
        self.best_individual = None
        self.best_outer_side_length = float('inf')

    def initialize_population(self) -> List[np.ndarray]:
        """Initialize population with diverse configurations."""
        population = []

        # Create base configurations using hexagonal lattice pattern
        base_positions = self._generate_base_positions()

        for _ in range(self.pop_size):
            # Start from base + noise
            individual = np.array(base_positions).flatten() + np.random.normal(0, 0.2, 36)
            # Apply bounds clipping
            individual[0::3] = np.clip(individual[0::3], -10, 10)  # x coords
            individual[1::3] = np.clip(individual[1::3], -10, 10)  # y coords
            population.append(individual)

        return population

    def _generate_base_positions(self) -> List[List[float]]:
        """Generate initial hexagonal arrangement."""
        # More strategic initial arrangement
        positions = []

        # Central hexagon
        positions.append([0, 0, 0])

        # First ring
        angles = np.linspace(0, 2*pi, 6, endpoint=False)
        for angle in angles:
            x = 2.5 * cos(angle)
            y = 2.5 * sin(angle)
            positions.append([x, y, 0])

        # Second ring
        angles = np.linspace(0, 2*pi, 6, endpoint=False)
        for i, angle in enumerate(angles):
            radius = 4.0 if i % 2 == 0 else 4.5
            x = radius * cos(angle)
            y = radius * sin(angle)
            positions.append([x, y, 0])

        # Fill remaining positions strategically
        positions.append([-3.0, 3.0, 0])
        positions.append([3.0, 3.0, 0])
        positions.append([-3.0, -3.0, 0])
        positions.append([3.0, -3.0, 0])
        positions.append([0, -5.0, 0])

        return positions[:12]  # Ensure exactly 12 positions

    def select_parents(self, population: List[np.ndarray],
                      fitness_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Tournament selection for parents."""
        # Tournament selection with k=4
        tournament_size = 4
        parent1_idx = min(random.sample(range(len(population)), tournament_size),
                         key=lambda i: -fitness_scores[i])
        parent2_idx = min(random.sample(range(len(population)), tournament_size),
                         key=lambda i: -fitness_scores[i])
        return population[parent1_idx], population[parent2_idx]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover between two individuals."""
        child = parent1.copy()
        for i in range(len(child)):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Mutate an individual with controlled changes."""
        mutated = individual.copy()

        # Mutate based on probability
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                if i % 3 == 0:  # x coordinate
                    mutated[i] += random.uniform(-0.3, 0.3)
                elif i % 3 == 1:  # y coordinate
                    mutated[i] += random.uniform(-0.3, 0.3)
                else:  # rotation
                    mutated[i] = (mutated[i] + random.uniform(-20, 20)) % 360

        return mutated

def optimize_hexagon_packing() -> PackingResult:
    """Main optimization function with improved architecture."""
    start_time = time.time()

    # Configuration parameters
    pop_size = 100
    num_generations = 500
    elite_size = 10
    initial_mutation_rate = 0.2

    # Initialize components
    population_manager = PopulationManager(pop_size)
    evaluator = FitnessEvaluator()

    # Initialize population
    population = population_manager.initialize_population()

    # Evolution loop
    for generation in range(num_generations):
        # Evaluate fitness for all individuals
        fitness_scores = []

        for individual in population:
            # Determine outer hexagon size based on current configuration
            hex_data = individual.reshape(-1, 3)
            configs = [HexagonConfig(row[0], row[1], row[2]) for row in hex_data]
            outer_side_length = HexagonGeometry.get_bounding_radius(configs)

            # Evaluate fitness with the computed outer size
            fitness = evaluator.evaluate_fitness(individual, outer_side_length)
            fitness_scores.append(fitness)

            # Update best solution
            if fitness > population_manager.best_fitness:
                population_manager.best_fitness = fitness
                population_manager.best_individual = individual.copy()
                population_manager.best_outer_side_length = outer_side_length

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Elitism: keep top individuals
        elites = population[:elite_size]

        # Generate new population
        new_population = elites[:]
        mutation_rate = initial_mutation_rate

        # Create offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Selection
            parent1, parent2 = population_manager.select_parents(population, fitness_scores)

            # Crossover
            child = population_manager.crossover(parent1, parent2)

            # Mutation
            child = population_manager.mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population

        # Adaptive mutation rate
        if generation > 100:
            mutation_rate = max(0.01, mutation_rate * 0.995)

        # Early stopping
        if time.time() - start_time > 170:  # Leave 10 seconds for cleanup
            break

        # Progress tracking
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {population_manager.best_fitness:.6f}, "
                  f"Outer side length = {population_manager.best_outer_side_length:.6f}")

    # Return best solution found
    if population_manager.best_individual is not None:
        hex_data = population_manager.best_individual.reshape(-1, 3)
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = population_manager.best_outer_side_length
        return PackingResult(hex_data, outer_hex_data, outer_hex_side_length)
    else:
        # Fallback to basic configuration
        hex_data = np.array([
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
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return PackingResult(hex_data, outer_hex_data, outer_hex_side_length)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    result = optimize_hexagon_packing()
    return result.inner_hex_data, result.outer_hex_data, result.outer_hex_side_length

# EVOLVE-BLOCK-END