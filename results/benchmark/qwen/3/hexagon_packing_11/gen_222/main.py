# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union
import time
from numba import jit
import random
from joblib import Parallel, delayed

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class Hexagon:
    """Represents a regular hexagon with center, rotation and side length"""

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
        """Get vertices of the hexagon with current transformation"""
        # Get base vertices
        base_vertices = self._generate_base_vertices(self.side_length)

        # Apply rotation
        angle_rad = np.radians(self.angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

        rotated_vertices = base_vertices @ rotation_matrix.T

        # Apply translation
        return rotated_vertices + np.array([self.center_x, self.center_y], dtype=np.float64)

    def to_polygon(self) -> Polygon:
        """Convert hexagon to shapely polygon"""
        return Polygon(self.get_vertices())

class PackingEvaluator:
    """Handles geometric validation and fitness evaluation"""

    def __init__(self, hex_side_length: float = 1.0):
        self.hex_side_length = hex_side_length

    def check_containment(self, hexagons: list[Hexagon], outer_radius: float) -> bool:
        """Check if all hexagons are contained within outer hexagon of given radius"""
        # Create outer hexagon centered at origin
        outer_hex = Hexagon(0.0, 0.0, 0.0, outer_radius)
        outer_polygon = outer_hex.to_polygon()

        for hexagon in hexagons:
            hex_polygon = hexagon.to_polygon()
            if not outer_polygon.contains(hex_polygon):
                return False
        return True

    def check_overlap(self, hexagons: list[Hexagon]) -> bool:
        """Check if any hexagons overlap"""
        polygons = [h.to_polygon() for h in hexagons]

        # Check pairwise overlaps
        for i in range(len(polygons)):
            for j in range(i+1, len(polygons)):
                if polygons[i].intersects(polygons[j]):
                    return True
        return False

    def evaluate_fitness(self, hexagons: list[Hexagon], outer_radius: float) -> float:
        """Evaluate fitness based on geometric constraints and packing density"""
        # Check constraints
        if not self.check_containment(hexagons, outer_radius):
            return -np.inf  # Invalid - penalty

        if self.check_overlap(hexagons):
            return -np.inf  # Invalid - penalty

        # Valid configuration - maximize 1/outer_radius (minimize outer_radius)
        return 1.0 / outer_radius

class PackingOptimizer:
    """Main optimization class for hexagon packing"""

    def __init__(self, n_inner_hexagons: int = 11, hex_side_length: float = 1.0):
        self.n_inner_hexagons = n_inner_hexagons
        self.hex_side_length = hex_side_length
        self.evaluator = PackingEvaluator(hex_side_length)

    def create_hexagons_from_array(self, hex_data: np.ndarray) -> list[Hexagon]:
        """Convert array data to list of Hexagon objects"""
        return [Hexagon(row[0], row[1], row[2], self.hex_side_length) for row in hex_data]

    def create_array_from_hexagons(self, hexagons: list[Hexagon]) -> np.ndarray:
        """Convert list of Hexagon objects to array data"""
        return np.array([[h.center_x, h.center_y, h.angle_degrees] for h in hexagons])

    def find_optimal_radius(self, hexagons: list[Hexagon], min_radius: float = 1.0, max_radius: float = 10.0) -> float:
        """Find minimum radius that contains all hexagons using binary search"""
        # First check if configuration fits at all
        if self.evaluator.check_containment(hexagons, min_radius):
            return min_radius

        # Binary search
        left, right = min_radius, max_radius
        for _ in range(20):  # Limit iterations
            mid = (left + right) / 2
            if self.evaluator.check_containment(hexagons, mid):
                right = mid
            else:
                left = mid

        return right

    def generate_hierarchical_initial_config(self, max_radius: float = 10.0) -> np.ndarray:
        """Generate a hierarchical initial configuration based on hexagonal tiling pattern"""
        # Create a structured layout that respects geometric constraints
        # Using a more systematic approach for optimal hexagon placement

        # Start with a hexagonal lattice arrangement
        # Hexagon centers form a triangular lattice with spacing of 2 units
        config = []

        # Central hexagon
        config.append([0.0, 0.0, 0.0])

        # First ring (6 hexagons) - spaced at distance 2
        for i in range(6):
            angle = i * 60  # 60 degree increments
            x = 2.0 * np.cos(np.radians(angle))
            y = 2.0 * np.sin(np.radians(angle))
            config.append([x, y, 0.0])

        # Second ring (12 hexagons) - spaced at distance 4
        for i in range(12):
            angle = i * 30  # 30 degree increments for second ring
            distance = 4.0
            x = distance * np.cos(np.radians(angle))
            y = distance * np.sin(np.radians(angle))
            config.append([x, y, 0.0])

        # Third ring (18 hexagons) - spaced at distance 6, but we'll only need 11-1-6=4 more
        # Actually, let's just trim to 11 hexagons and make it more strategic

        # Trim to exactly 11 hexagons using the most effective arrangement
        if len(config) > 11:
            # Keep central, first ring, and some from second ring
            config = [config[0]] + config[1:7] + config[7:11]  # Take first 6 from ring 1, and 4 from ring 2
            config = config[:11]
        else:
            # Fill up to 11 with carefully chosen positions
            while len(config) < 11:
                # Add positions along the outer ring of existing hexagons
                # This ensures good coverage without creating overlaps
                config.append([np.random.uniform(-max_radius/2, max_radius/2),
                              np.random.uniform(-max_radius/2, max_radius/2),
                              np.random.uniform(0, 360)])

        # Add small random perturbations to escape local minima
        result = np.array(config)
        for i in range(len(result)):
            # Add slightly larger perturbations to encourage diversity
            result[i][0] += np.random.normal(0, 0.3)
            result[i][1] += np.random.normal(0, 0.3)
            result[i][2] += np.random.normal(0, 20)  # Larger angle variations
            result[i][2] = result[i][2] % 360

        return result

    def generate_initial_population(self, pop_size: int = 50, max_radius: float = 10.0) -> list[np.ndarray]:
        """Generate diverse initial population with hierarchical initialization"""
        population = []
        # Always include hierarchical initial config as first member
        population.append(self.generate_hierarchical_initial_config(max_radius))

        # Fill remaining population with random configs
        for _ in range(pop_size - 1):
            individual = np.zeros((self.n_inner_hexagons, 3))
            for i in range(self.n_inner_hexagons):
                # Random positions within reasonable bounds
                x = np.random.uniform(-max_radius/2, max_radius/2)
                y = np.random.uniform(-max_radius/2, max_radius/2)
                angle = np.random.uniform(0, 360)
                individual[i] = [x, y, angle]
            population.append(individual)
        return population

    def mutate_individual(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
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

    def crossover_individuals(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform uniform crossover between two parents"""
        child = parent1.copy()
        for i in range(len(child)):
            if np.random.random() < 0.5:
                child[i] = parent2[i].copy()
        return child

    def optimize_local(self, individual: np.ndarray, outer_radius: float) -> np.ndarray:
        """Refine solution locally using multi-stage optimization with adaptive parameters"""
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
            fitness = self.evaluator.evaluate_fitness(hexagons, outer_radius)
            return -fitness  # minimize negative fitness

        # Flatten the data for optimization
        initial_params = []
        for i in range(len(individual)):
            initial_params.extend([individual[i][0], individual[i][1], individual[i][2]])

        # Multi-stage optimization with adaptive parameters
        # Stage 1: Coarse optimization
        try:
            result1 = minimize(objective, initial_params, method='L-BFGS-B',
                             bounds=[(-10, 10), (-10, 10), (0, 360)] * len(individual),
                             options={'maxiter': 50})

            if result1.success:
                # Stage 2: Fine-grained optimization with tighter bounds
                refined_params = result1.x.copy()
                result2 = minimize(objective, refined_params, method='L-BFGS-B',
                                 bounds=[(-5, 5), (-5, 5), (0, 360)] * len(individual),
                                 options={'maxiter': 75})

                if result2.success:
                    # Stage 3: Very fine-grained optimization
                    final_params = result2.x.copy()
                    result3 = minimize(objective, final_params, method='L-BFGS-B',
                                     bounds=[(-2, 2), (-2, 2), (0, 360)] * len(individual),
                                     options={'maxiter': 100})

                    if result3.success:
                        # Reshape optimized result back
                        refined_data = individual.copy()
                        for i in range(len(refined_data)):
                            refined_data[i][0] = result3.x[i*3]
                            refined_data[i][1] = result3.x[i*3+1]
                            refined_data[i][2] = result3.x[i*3+2]
                        return refined_data
                    else:
                        # If stage 3 fails, return stage 2 result
                        refined_data = individual.copy()
                        for i in range(len(refined_data)):
                            refined_data[i][0] = result2.x[i*3]
                            refined_data[i][1] = result2.x[i*3+1]
                            refined_data[i][2] = result2.x[i*3+2]
                        return refined_data
                else:
                    # If stage 2 fails, return stage 1 result
                    refined_data = individual.copy()
                    for i in range(len(refined_data)):
                        refined_data[i][0] = result1.x[i*3]
                        refined_data[i][1] = result1.x[i*3+1]
                        refined_data[i][2] = result1.x[i*3+2]
                    return refined_data
        except:
            pass
        # Fallback to original individual if optimization fails
        return individual

    def adaptive_evolutionary_local_search(self,
                                         initial_config: np.ndarray,
                                         max_generations: int = 50,
                                         population_size: int = 30,
                                         mutation_rate: float = 0.1,
                                         local_optimization_frequency: int = 5) -> tuple[np.ndarray, float]:
        """Adaptive evolutionary-local optimization loop with alternating refinement levels"""
        # Generate initial population
        population = [initial_config]  # Start with our heuristic config
        population.extend(self.generate_initial_population(population_size - 1))

        best_fitness = -np.inf
        best_config = None
        best_radius = 10.0

        for gen in range(max_generations):
            # Evaluate fitness of population with validation
            fitness_scores = []
            valid_individuals = []

            for individual in population:
                hexagons = self.create_hexagons_from_array(individual)

                # Quick geometric validation to filter out obviously invalid configurations
                if self.evaluator.check_overlap(hexagons):
                    fitness_scores.append(-np.inf)  # Penalize overlaps severely
                    continue

                radius = self.find_optimal_radius(hexagons)
                fitness = self.evaluator.evaluate_fitness(hexagons, radius)
                fitness_scores.append(fitness)
                valid_individuals.append(individual)

            # Update best solution
            if len(fitness_scores) > 0:
                max_idx = np.argmax(fitness_scores)
                if fitness_scores[max_idx] > best_fitness:
                    best_fitness = fitness_scores[max_idx]
                    best_config = population[max_idx].copy()
                    best_radius = self.find_optimal_radius(self.create_hexagons_from_array(best_config))

            # Perform local optimization periodically to refine the best solution
            if gen % local_optimization_frequency == 0 and best_config is not None:
                refined_config = self.optimize_local(best_config, best_radius)
                refined_hexagons = self.create_hexagons_from_array(refined_config)
                refined_radius = self.find_optimal_radius(refined_hexagons)
                refined_fitness = self.evaluator.evaluate_fitness(refined_hexagons, refined_radius)

                if refined_fitness > best_fitness:
                    best_fitness = refined_fitness
                    best_config = refined_config
                    best_radius = refined_radius

            # Selection and reproduction
            if len(fitness_scores) > 0:
                sorted_indices = np.argsort(fitness_scores)[::-1][:population_size//2]
                selected = [population[i] for i in sorted_indices if fitness_scores[i] > -np.inf]
            else:
                selected = []

            # Generate new population - keep a mix of elite and new individuals
            new_population = selected.copy()
            for _ in range(population_size - len(selected)):
                if len(selected) > 0:
                    parent1 = selected[np.random.randint(len(selected))]
                    parent2 = selected[np.random.randint(len(selected))]
                    child = self.crossover_individuals(parent1, parent2)
                    child = self.mutate_individual(child, mutation_rate)
                else:
                    # If no valid individuals, create completely new individuals
                    child = self.generate_initial_population(1)[0]
                new_population.append(child)

            population = new_population

        return best_config, best_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    # Initialize optimizer
    optimizer = PackingOptimizer(n_inner_hexagons=11, hex_side_length=1.0)

    # Start with a good heuristic initial configuration
    # Based on hexagonal tiling pattern
    initial_config = np.array([
        [0, 0, 0],        # center
        [-2.5, 0, 0],     # left
        [2.5, 0, 0],      # right
        [-1.25, 2.17, 0], # top-left
        [1.25, 2.17, 0],  # top-right
        [-1.25, -2.17, 0],# bottom-left
        [1.25, -2.17, 0], # bottom-right
        [-3.75, 2.17, 0], # far top-left
        [3.75, 2.17, 0],  # far top-right
        [-3.75, -2.17, 0],# far bottom-left
        [3.75, -2.17, 0], # far bottom-right
    ])

    # Apply adaptive evolutionary-local optimization
    best_config, best_radius = optimizer.adaptive_evolutionary_local_search(
        initial_config=initial_config,
        max_generations=50,
        population_size=30,
        mutation_rate=0.1,
        local_optimization_frequency=3
    )

    # Final local optimization
    if best_config is not None:
        refined_config = optimizer.optimize_local(best_config, best_radius)
        final_radius = optimizer.find_optimal_radius(optimizer.create_hexagons_from_array(refined_config))
        # Re-evaluate with final radius
        final_fitness = optimizer.evaluator.evaluate_fitness(
            optimizer.create_hexagons_from_array(refined_config),
            final_radius
        )
        if final_fitness > optimizer.evaluator.evaluate_fitness(
            optimizer.create_hexagons_from_array(best_config),
            best_radius
        ):
            best_config = refined_config
            best_radius = final_radius

    # Prepare output
    inner_hex_data = best_config if best_config is not None else initial_config
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = best_radius if 'best_radius' in locals() else 8.0

    end_time = time.time()
    eval_time = end_time - start_time

    # Validate solution (optional - for debugging only)
    if best_config is not None:
        hexagons = optimizer.create_hexagons_from_array(best_config)
        if optimizer.evaluator.check_overlap(hexagons):
            print("Warning: Overlapping hexagons detected!")
        if not optimizer.evaluator.check_containment(hexagons, outer_hex_side_length):
            print("Warning: Hexagons not contained in outer hexagon!")

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END