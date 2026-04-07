# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from itertools import combinations
import random
from joblib import Parallel, delayed
from numba import jit, prange
import warnings

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class HexagonGeometry:
    """Handles all geometric computations for hexagons with numba acceleration"""

    @staticmethod
    @jit(nopython=True)
    def get_unit_hexagon_vertices_numba():
        """Return vertices of a unit regular hexagon centered at origin (numba optimized)"""
        vertices = np.empty((6, 2))
        for i in range(6):
            angle = i * np.pi / 3
            vertices[i, 0] = np.cos(angle)
            vertices[i, 1] = np.sin(angle)
        return vertices

    @staticmethod
    @jit(nopython=True)
    def transform_hexagon_vertices_numba(vertices, center_x, center_y, angle_deg):
        """Transform hexagon vertices by translation and rotation (numba optimized)"""
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)

        # Rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Apply rotation and translation
        rotated_vertices = np.empty_like(vertices)
        for i in range(len(vertices)):
            x, y = vertices[i]
            rotated_vertices[i, 0] = x * cos_a - y * sin_a + center_x
            rotated_vertices[i, 1] = x * sin_a + y * cos_a + center_y

        return rotated_vertices

    @staticmethod
    def get_unit_hexagon_vertices():
        """Return vertices of a unit regular hexagon centered at origin."""
        angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 angles, skip last to close the polygon
        vertices = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])
        return vertices

    @staticmethod
    def transform_hexagon_vertices(vertices, center_x, center_y, angle_deg):
        """Transform hexagon vertices by translation and rotation."""
        # Convert angle to radians
        angle_rad = np.radians(angle_deg)

        # Rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Apply rotation and translation
        rotated_vertices = vertices @ rotation_matrix.T
        translated_vertices = rotated_vertices + np.array([center_x, center_y])

        return translated_vertices

    @staticmethod
    def create_hexagon_polygon(center_x, center_y, angle_deg):
        """Create a Shapely polygon representing a unit hexagon at given position and rotation."""
        vertices = HexagonGeometry.transform_hexagon_vertices(
            HexagonGeometry.get_unit_hexagon_vertices(),
            center_x, center_y, angle_deg
        )
        return Polygon(vertices)

class HexagonValidator:
    """Handles constraint checking for hexagon packing"""

    @staticmethod
    def check_containment(hexagons, outer_radius):
        """Check if all hexagons are contained within outer hexagon of given radius"""
        # Create outer hexagon centered at origin
        outer_vertices = HexagonGeometry.transform_hexagon_vertices(
            HexagonGeometry.get_unit_hexagon_vertices(),
            0.0, 0.0, 0.0
        )
        outer_polygon = Polygon(outer_vertices * outer_radius)

        for hexagon in hexagons:
            if not outer_polygon.contains(hexagon):
                return False
        return True

    @staticmethod
    def check_overlap(hexagons):
        """Check if any hexagons overlap"""
        # Check pairwise overlaps with early termination
        for i in range(len(hexagons)):
            for j in range(i+1, len(hexagons)):
                if hexagons[i].intersects(hexagons[j]):
                    return True
        return False

class HexagonPacker:
    """Main class coordinating hexagon packing optimization"""

    def __init__(self, n_inner_hexagons=11, hex_side_length=1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length
        self.validator = HexagonValidator()
        self.geometry = HexagonGeometry()

    def create_hexagons_from_array(self, hex_data):
        """Convert array data to list of Shapely polygon objects"""
        return [self.geometry.create_hexagon_polygon(row[0], row[1], row[2])
                for row in hex_data]

    def evaluate_fitness(self, hexagons, outer_radius):
        """Evaluate fitness based on geometric constraints and packing density"""
        # Check constraints
        if not self.validator.check_containment(hexagons, outer_radius):
            return -np.inf  # Invalid - penalty

        if self.validator.check_overlap(hexagons):
            return -np.inf  # Invalid - penalty

        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius

    def find_optimal_radius(self, hexagons, min_radius=1.0, max_radius=10.0):
        """Find minimum radius that contains all hexagons using binary search"""
        # First check if configuration fits at all
        if self.validator.check_containment(hexagons, min_radius):
            return min_radius

        # Binary search with early termination
        left, right = min_radius, max_radius
        iterations = 0
        max_iterations = 20

        while iterations < max_iterations and abs(right - left) > 0.001:
            mid = (left + right) / 2
            if self.validator.check_containment(hexagons, mid):
                right = mid
            else:
                left = mid
            iterations += 1

        return right

    def generate_initial_population(self, pop_size=50, max_radius=10.0):
        """Generate diverse initial population"""
        population = []
        for _ in range(pop_size):
            # Generate random positions and angles
            individual = np.zeros((self.n_inner_hexagons, 3))
            for i in range(self.n_inner_hexagons):
                # Random positions within reasonable bounds
                x = np.random.uniform(-max_radius/2, max_radius/2)
                y = np.random.uniform(-max_radius/2, max_radius/2)
                angle = np.random.uniform(0, 360)
                individual[i] = [x, y, angle]
            population.append(individual)
        return population

    def mutate_individual(self, individual, mutation_rate=0.1):
        """Apply mutation to an individual"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Mutate position
                mutated[i][0] += np.random.normal(0, 0.5)
                mutated[i][1] += np.random.normal(0, 0.5)
                # Mutate angle
                mutated[i][2] += np.random.normal(0, 30)
                # Keep angle in [0, 360)
                mutated[i][2] = mutated[i][2] % 360
        return mutated

    def crossover_individuals(self, parent1, parent2):
        """Perform uniform crossover between two parents"""
        child = parent1.copy()
        for i in range(len(child)):
            if np.random.random() < 0.5:
                child[i] = parent2[i].copy()
        return child

    def evaluate_individual_fitness(self, individual, max_radius=10.0):
        """Evaluate fitness for a single individual - used for parallel processing"""
        hexagons = self.create_hexagons_from_array(individual)
        radius = self.find_optimal_radius(hexagons, max_radius=max_radius)
        fitness = self.evaluate_fitness(hexagons, radius)
        return fitness, radius

    def parallel_evaluate_population(self, population, n_jobs=-1):
        """Evaluate fitness of all individuals in population in parallel"""
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_individual_fitness)(individual)
            for individual in population
        )

        fitness_scores = [result[0] for result in results]
        radii = [result[1] for result in results]

        return fitness_scores, radii

    def optimize_local(self, individual, outer_radius):
        """Refine solution locally using optimization"""
        def objective(params):
            # Reshape params back to hexagon data
            new_data = individual.copy()
            for i in range(len(new_data)):
                new_data[i][0] = params[i*3]
                new_data[i][1] = params[i*3+1]
                new_data[i][2] = params[i*3+2]

            # Convert to hexagon objects for evaluation
            hexagons = self.create_hexagons_from_array(new_data)

            # Evaluate fitness
            fitness = self.evaluate_fitness(hexagons, outer_radius)
            return -fitness  # minimize negative fitness

        # Flatten the data for optimization
        initial_params = []
        for i in range(len(individual)):
            initial_params.extend([individual[i][0], individual[i][1], individual[i][2]])

        # Optimize using L-BFGS-B
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B',
                            bounds=[(-10, 10), (-10, 10), (0, 360)] * len(individual),
                            options={'maxiter': 100})
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
        return individual

def generate_better_initial_config():
    """
    Generate a better initial configuration for 11 hexagons based on known dense packings
    """
    # This configuration is based on a hexagonal close packing arrangement
    # with strategic placement to achieve better packing density
    initial_positions = [
        # Central hexagon
        [0.0, 0.0, 0.0],
        # First ring (6 hexagons)
        [-2.0, 0.0, 0.0],      # Left
        [2.0, 0.0, 0.0],       # Right
        [0.0, 2.0, 0.0],       # Top
        [0.0, -2.0, 0.0],      # Bottom
        [-1.0, 1.732, 0.0],    # Top-left
        [1.0, 1.732, 0.0],     # Top-right
        # Second ring (4 hexagons)
        [-1.0, -1.732, 0.0],   # Bottom-left
        [1.0, -1.732, 0.0],    # Bottom-right
        [-2.0, 1.0, 0.0],      # Far top-left
        [2.0, 1.0, 0.0],       # Far top-right
        [-2.0, -1.0, 0.0],     # Far bottom-left
        [2.0, -1.0, 0.0],      # Far bottom-right
    ]

    # Keep only first 11 positions (the 11 required hexagons)
    return np.array(initial_positions[:11])

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Initialize packer
    packer = HexagonPacker(n_inner_hexagons=11, hex_side_length=1.0)

    # Start with a good heuristic initial configuration
    # Based on hexagonal tiling pattern from prior research
    initial_config = generate_better_initial_config()

    # Apply evolutionary optimization
    best_fitness = -np.inf
    best_config = None
    best_radius = 10.0

    # Evolutionary process with enhanced parameters
    max_generations = 50
    population_size = 30
    mutation_rate = 0.1

    # Generate initial population
    population = [initial_config]  # Start with our heuristic config
    population.extend(packer.generate_initial_population(population_size - 1))

    for gen in range(max_generations):
        # Evaluate fitness of population in parallel
        fitness_scores, radii = packer.parallel_evaluate_population(population)

        # Update best solution
        max_idx = np.argmax(fitness_scores)
        if fitness_scores[max_idx] > best_fitness:
            best_fitness = fitness_scores[max_idx]
            best_config = population[max_idx].copy()
            best_radius = radii[max_idx]

        # Selection and reproduction
        sorted_indices = np.argsort(fitness_scores)[::-1][:population_size//2]
        selected = [population[i] for i in sorted_indices]

        # Generate new population
        new_population = selected.copy()
        for _ in range(population_size - len(selected)):
            parent1 = selected[np.random.randint(len(selected))]
            parent2 = selected[np.random.randint(len(selected))]
            child = packer.crossover_individuals(parent1, parent2)
            child = packer.mutate_individual(child, mutation_rate)
            new_population.append(child)

        population = new_population

    # Final local optimization
    if best_config is not None:
        refined_config = packer.optimize_local(best_config, best_radius)
        final_radius = packer.find_optimal_radius(packer.create_hexagons_from_array(refined_config))
        # Re-evaluate with final radius
        final_fitness = packer.evaluate_fitness(packer.create_hexagons_from_array(refined_config), final_radius)
        if final_fitness > best_fitness:
            best_config = refined_config
            best_radius = final_radius
            best_fitness = final_fitness

    # Prepare output
    inner_hex_data = best_config if best_config is not None else initial_config
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = best_radius if 'best_radius' in locals() else 8.0

    end_time = time.time()
    eval_time = end_time - start_time

    # Validate solution
    if best_config is not None:
        hexagons = packer.create_hexagons_from_array(best_config)
        if packer.validator.check_overlap(hexagons):
            warnings.warn("Warning: Overlapping hexagons detected!")
        if not packer.validator.check_containment(hexagons, outer_hex_side_length):
            warnings.warn("Warning: Hexagons not contained in outer hexagon!")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END