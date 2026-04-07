# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from scipy.optimize import minimize
from joblib import Parallel, delayed
import warnings
from numba import jit, prange
from typing import Tuple, Optional


class HexagonGeometry:
    """Handles all geometric computations for hexagons."""

    def __init__(self, side_length: float = 1.0):
        self.side_length = side_length
        self.radius = side_length  # For unit hexagons, radius equals side length
        self.width = 2 * side_length * np.cos(np.pi/6)
        self.height = 2 * side_length

    @staticmethod
    @jit(nopython=True)
    def _generate_hexagon_vertices_numba(center_x: float, center_y: float,
                                       side_length: float, rotation_rad: float):
        """Fast JIT version of hexagon vertex generation."""
        angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
        vertices_x = np.empty(6)
        vertices_y = np.empty(6)
        for i in range(6):
            vertices_x[i] = center_x + side_length * np.cos(angles[i])
            vertices_y[i] = center_y + side_length * np.sin(angles[i])
        return vertices_x, vertices_y

    def generate_hexagon_vertices(self, center_x: float, center_y: float,
                                rotation_deg: float = 0) -> np.ndarray:
        """Generate vertices of a regular hexagon."""
        rotation_rad = np.radians(rotation_deg)
        angles = np.linspace(0, 2*np.pi, 7) + rotation_rad
        vertices = np.column_stack([
            center_x + self.side_length * np.cos(angles),
            center_y + self.side_length * np.sin(angles)
        ])
        return vertices[:-1]  # Remove duplicate last vertex

    def compute_outer_radius(self, inner_positions: np.ndarray,
                           inner_angles: np.ndarray) -> float:
        """Compute minimal outer hexagon radius."""
        max_dist = 0
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            angle = inner_angles[i]
            # Get vertices of inner hexagon
            vertices = self.generate_hexagon_vertices(cx, cy, angle)
            # Distance from center to each vertex
            distances = np.sqrt((vertices[:, 0])**2 + (vertices[:, 1])**2)
            max_dist = max(max_dist, np.max(distances))
        return max_dist + 0.1  # Small buffer


class HexagonValidator:
    """Validates hexagon configurations for containment and overlap."""

    def __init__(self, geometry: HexagonGeometry):
        self.geometry = geometry

    def check_containment(self, inner_vertices: np.ndarray,
                         outer_vertices: np.ndarray) -> bool:
        """Check if hexagon vertices are contained within outer hexagon."""
        outer_polygon = Polygon(outer_vertices)
        # Early exit if any vertex is outside
        for vertex in inner_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False
        return True

    def check_overlap(self, hex1_vertices: np.ndarray,
                     hex2_vertices: np.ndarray) -> bool:
        """Check if two hexagons overlap using fast bounding box check first."""
        # Fast bounding box check before expensive polygon intersection
        bbox1 = [np.min(hex1_vertices[:, 0]), np.max(hex1_vertices[:, 0]),
                np.min(hex1_vertices[:, 1]), np.max(hex1_vertices[:, 1])]
        bbox2 = [np.min(hex2_vertices[:, 0]), np.max(hex2_vertices[:, 0]),
                np.min(hex2_vertices[:, 1]), np.max(hex2_vertices[:, 1])]

        # Check if bounding boxes intersect first
        if (bbox1[1] < bbox2[0] or bbox2[1] < bbox1[0] or
            bbox1[3] < bbox2[2] or bbox2[3] < bbox1[2]):
            return False

        # Only do expensive polygon intersection if bounding boxes overlap
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)

    def validate_configuration(self, inner_positions: np.ndarray,
                             inner_angles: np.ndarray,
                             outer_radius: float) -> bool:
        """Validate that all hexagons fit properly within the outer hexagon."""
        # Generate outer hexagon vertices
        outer_vertices = self.geometry.generate_hexagon_vertices(0, 0, outer_radius, 0)

        # Validate each inner hexagon
        for i in range(len(inner_positions)):
            center_x, center_y = inner_positions[i]
            angle = inner_angles[i]
            inner_vertices = self.geometry.generate_hexagon_vertices(center_x, center_y, angle)

            # Check containment
            if not self.check_containment(inner_vertices, outer_vertices):
                return False

            # Check overlaps with all other hexagons
            for j in range(i+1, len(inner_positions)):
                center_x2, center_y2 = inner_positions[j]
                angle2 = inner_angles[j]
                inner_vertices2 = self.geometry.generate_hexagon_vertices(center_x2, center_y2, angle2)

                if self.check_overlap(inner_vertices, inner_vertices2):
                    return False

        return True


class HexagonPacker:
    """Main class for hexagon packing optimization with evolved architecture."""

    def __init__(self):
        self.geometry = HexagonGeometry()
        self.validator = HexagonValidator(self.geometry)
        self.max_time_seconds = 170
        self.population_size = 15
        self.n_generations = 30
        self.elite_count = 2
        self.mutation_rate = 0.3
        self.local_optimization_iters = 50
        self.convergence_threshold = 1e-8
        self.convergence_window = 10

    def generate_initial_configuration(self, n_hexagons: int = 11) -> np.ndarray:
        """Generate an initial configuration with strategic placement."""
        # Use a proven hexagonal tiling pattern optimized for 11 hexagons
        # This follows a pattern similar to the best-known configurations
        positions = []

        # Central hexagon - most stable position
        positions.append([0.0, 0.0, 0.0])

        # First ring (6 hexagons) - arranged in a perfect hexagonal pattern
        for i in range(6):
            angle = i * np.pi / 3
            # Distance between centers of touching hexagons is 2 (diameter)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append([x, y, 0.0])

        # Second ring (4 hexagons) - placed to fill gaps in more compact arrangement
        # These are positioned more densely to reduce empty space
        second_ring_positions = [
            [-1.0, -1.732, 0.0],   # Bottom left
            [1.0, -1.732, 0.0],    # Bottom right
            [-1.0, 1.732, 0.0],    # Top left
            [1.0, 1.732, 0.0],     # Top right
        ]

        positions.extend(second_ring_positions)

        # Take only the required number of positions
        initial_positions = np.array(positions[:n_hexagons])

        # Add systematic perturbations based on known good arrangements
        np.random.seed(42)
        for i in range(len(initial_positions)):
            # More controlled perturbations that maintain good spacing
            initial_positions[i, 0] += np.random.uniform(-0.02, 0.02)  # Very small
            initial_positions[i, 1] += np.random.uniform(-0.02, 0.02)
            initial_positions[i, 2] += np.random.uniform(-1, 1)       # Small rotation

        return initial_positions

    def evaluate_fitness_parallel(self, configs: list) -> list:
        """Parallel evaluation of multiple configurations."""
        def evaluate_single(config):
            return self.evaluate_fitness(config)

        results = Parallel(n_jobs=-1)(
            delayed(evaluate_single)(config) for config in configs
        )
        return results

    def evaluate_fitness(self, config: np.ndarray) -> Tuple[float, float]:
        """Evaluate fitness of a configuration - higher is better."""
        # Extract inner hexagon data
        inner_positions = config[:, :2]
        inner_angles = config[:, 2]

        # Compute outer hexagon radius
        outer_radius = self.geometry.compute_outer_radius(inner_positions, inner_angles)

        # Validate configuration
        valid = self.validator.validate_configuration(inner_positions, inner_angles, outer_radius)

        # Return fitness (inverse of radius if valid, very negative otherwise)
        if valid:
            return 1.0 / outer_radius, outer_radius
        else:
            return -1e6, outer_radius

    def local_optimization_step(self, config: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Perform local optimization on a configuration with adaptive iterations."""
        def objective(x):
            # Reshape to proper format
            test_config = x.reshape(-1, 3)
            fitness, _ = self.evaluate_fitness(test_config)
            return -fitness  # Negative because we want to maximize

        # Try multiple optimization strategies for robustness
        strategies = ['L-BFGS-B', 'TNC']  # Use different solvers

        for solver in strategies:
            try:
                result = minimize(objective, config.flatten(), method=solver,
                                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
                if result.success:
                    return result.x.reshape(-1, 3)
            except:
                continue

        # If all solvers fail, return original config
        return config

    def smart_local_optimization(self, individual: np.ndarray, max_iter: int = 200) -> np.ndarray:
        """Apply multi-stage local optimization that progressively refines the solution."""
        refined = individual.copy()

        # Stage 1: Coarse optimization to make large improvements
        refined = self.local_optimization_step(refined, max_iter // 4)

        # Stage 2: Medium optimization for moderate improvements
        refined = self.local_optimization_step(refined, max_iter // 2)

        # Stage 3: Fine optimization for small improvements
        refined = self.local_optimization_step(refined, max_iter // 2)

        return refined

    def adaptive_local_refinement(self, individual: np.ndarray) -> np.ndarray:
        """Apply multi-stage local refinement with intelligent stopping criteria."""
        refined = individual.copy()
        prev_fitness, _ = self.evaluate_fitness(refined)

        # Apply progressive refinement with intelligent stopping
        refinement_stages = [
            {'iterations': 30, 'tolerance': 1e-5},
            {'iterations': 60, 'tolerance': 1e-6},
            {'iterations': 100, 'tolerance': 1e-7}
        ]

        for stage in refinement_stages:
            refined = self.local_optimization_step(refined, stage['iterations'])
            curr_fitness, _ = self.evaluate_fitness(refined)

            # Stop if improvement is below threshold
            if curr_fitness - prev_fitness < stage['tolerance']:
                break
            prev_fitness = curr_fitness

        return refined

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents with improved blending."""
        # Use more sophisticated blend crossover with adaptive weights
        # Use a random alpha between 0.3 and 0.7 for better exploration
        alpha = np.random.uniform(0.3, 0.7)
        child = parent1 * alpha + parent2 * (1 - alpha)

        # Add small random noise to encourage diversity in children
        noise_magnitude = 0.01
        for i in range(len(child)):
            child[i][0] += np.random.normal(0, noise_magnitude)
            child[i][1] += np.random.normal(0, noise_magnitude)
            child[i][2] += np.random.normal(0, 0.5)
            child[i][2] = child[i][2] % 360  # Keep angle in [0, 360)

        return child

    def mutate_individual(self, individual: np.ndarray, generation: int) -> np.ndarray:
        """Mutate an individual with adaptive rate and smarter perturbations."""
        mutated = individual.copy()
        # Decrease mutation rate more aggressively
        mutation_rate = self.mutation_rate / (generation + 1)**0.8

        for i in range(len(mutated)):
            for j in range(len(mutated[i])):
                if np.random.rand() < mutation_rate:
                    if j == 0 or j == 1:  # x or y coordinate
                        # Smaller, more precise mutations for positions
                        mutated[i, j] += np.random.normal(0, 0.02)
                    elif j == 2:  # angle
                        # Slightly larger mutations for angles to allow rotation changes
                        mutated[i, j] += np.random.normal(0, 1)
        return mutated

    def evolutionary_search(self) -> Tuple[np.ndarray, float, float]:
        """Evolutionary algorithm for hexagon packing optimization."""
        # Initialize population with better starting points
        population = []
        for i in range(self.population_size):
            individual = self.generate_initial_configuration(11)
            population.append(individual)

        # Evolution loop with enhanced parameters
        best_fitness = -np.inf
        best_individual = None
        best_radius = float('inf')
        fitness_history = []

        # Better early stopping with adaptive patience
        patience_counter = 0
        max_patience = 15

        start_time = time.time()

        for gen in range(self.n_generations):
            if time.time() - start_time > self.max_time_seconds:
                break

            # Evaluate fitness for all individuals in parallel
            fitness_results = self.evaluate_fitness_parallel(population)

            # Extract fitness scores and update best
            fitness_scores = [result[0] for result in fitness_results]
            radii = [result[1] for result in fitness_results]

            for i, (fitness, radius) in enumerate(zip(fitness_scores, radii)):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = population[i].copy()
                    best_radius = radius
                    patience_counter = 0  # Reset patience when improvement found

            # Track fitness history for convergence detection
            fitness_history.append(best_fitness)
            if len(fitness_history) > self.convergence_window:
                fitness_history.pop(0)

            # Enhanced early stopping with patience mechanism
            if len(fitness_history) >= self.convergence_window:
                recent_improvement = max(fitness_history) - min(fitness_history)
                if recent_improvement < self.convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        break
                else:
                    patience_counter = 0  # Reset if there's significant improvement

            # Create new population through selection and mutation
            # Tournament selection with larger tournaments for better pressure
            new_population = []

            # Elitism: keep top individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
            for i in range(self.elite_count):
                new_population.append(population[sorted_indices[i]])

            # Generate rest by crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents with higher probability of selecting top performers
                parent1_idx = sorted_indices[np.random.randint(0, max(1, int(self.population_size/3)))]
                parent2_idx = sorted_indices[np.random.randint(0, max(1, int(self.population_size/3)))]

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate_individual(child, gen)

                new_population.append(child)

            population = new_population[:self.population_size]

        # Final local refinement on the best individual found with enhanced strategy
        if best_individual is not None:
            # Apply smart local optimization instead of basic adaptive refinement
            best_individual = self.smart_local_optimization(best_individual, 150)
            # Re-evaluate the final refined individual
            final_fitness, final_radius = self.evaluate_fitness(best_individual)
            if final_fitness > best_fitness:
                best_fitness = final_fitness
                best_radius = final_radius

        return best_individual, best_fitness, best_radius


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()

    try:
        # Create packer instance with optimized parameters
        packer = HexagonPacker()

        # Run evolutionary optimization
        best_config, best_fitness, best_radius = packer.evolutionary_search()

        # Ensure we have valid results
        if best_config is None:
            # Fallback to simple configuration
            inner_hex_data = np.array([
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
                [3.75, -2.17, 0]
            ])
            outer_radius = 8.0
        else:
            inner_hex_data = best_config
            outer_radius = best_radius

    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}, using fallback")
        inner_hex_data = np.array([
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
            [3.75, -2.17, 0]
        ])
        outer_radius = 8.0

    # Set up outer hexagon data (centered at origin)
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    # Ensure we have a valid result within time limits
    end_time = time.time()
    if end_time - start_time > 175:  # Leave some buffer
        warnings.warn("Time limit approaching, returning best available result")

    return inner_hex_data, outer_hex_data, outer_radius

# EVOLVE-BLOCK-END