# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.ops import unary_union
import random
import time
from copy import deepcopy
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

class HexagonPackingOptimizer:
    def __init__(self):
        self.hex_vertices_cache = {}

    def hexagon_vertices(self, center_x, center_y, angle_deg, side_length=1):
        """Generate vertices of a regular hexagon with caching for performance"""
        cache_key = (center_x, center_y, angle_deg, side_length)
        if cache_key in self.hex_vertices_cache:
            return self.hex_vertices_cache[cache_key]

        angle_rad = np.radians(angle_deg)
        angles = np.linspace(0, 2*np.pi, 7) + angle_rad
        vertices = np.array([
            [center_x + side_length * np.cos(a), center_y + side_length * np.sin(a)]
            for a in angles
        ])
        self.hex_vertices_cache[cache_key] = vertices
        return vertices

    def check_containment(self, hex_vertices, outer_hex_vertices):
        """Check if hexagon vertices are contained within outer hexagon using Shapely"""
        inner_polygon = Polygon(hex_vertices)
        outer_polygon = Polygon(outer_hex_vertices)
        return outer_polygon.contains(inner_polygon)

    def check_overlap(self, hex1_vertices, hex2_vertices):
        """Check if two hexagons overlap using Shapely"""
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        return poly1.intersects(poly2)

    def compute_outer_hexagon_radius(self, inner_positions, inner_angles, initial_radius_estimate=5.0):
        """Compute minimum outer hexagon radius that contains all inner hexagons with adaptive precision"""
        # Binary search for tightest fit with adaptive precision
        left = initial_radius_estimate
        right = 20.0
        best_radius = right

        # Use adaptive precision that decreases as we converge
        max_iterations = 100
        iterations = 0

        # Track convergence to adapt precision
        prev_diff = float('inf')

        while iterations < max_iterations:
            # Adaptive precision based on current range and convergence
            current_range = right - left
            if iterations > 10:
                # If we're converging well, use stricter precision
                precision_threshold = max(1e-8, current_range * 1e-5)
            else:
                precision_threshold = 1e-6

            # Exit early if precision threshold is met
            if current_range <= precision_threshold:
                break

            mid = (left + right) / 2.0
            outer_vertices = self.hexagon_vertices(0, 0, 0, mid)
            valid = True

            # Check all inner hexagons
            for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
                hex_vertices = self.hexagon_vertices(pos[0], pos[1], angle)
                if not self.check_containment(hex_vertices, outer_vertices):
                    valid = False
                    break

            if valid:
                best_radius = mid
                right = mid
            else:
                left = mid
            iterations += 1

            # Check for convergence stagnation
            if iterations > 5:
                diff = abs(right - left)
                if abs(diff - prev_diff) < precision_threshold * 10:
                    # Converging slowly, decrease precision threshold further
                    precision_threshold *= 0.5
                prev_diff = diff

        return best_radius

    def evaluate_fitness(self, inner_positions, inner_angles, max_radius=20.0):
        """Evaluate fitness: higher is better, maximize 1/radius"""
        # Create outer hexagon vertices
        outer_radius = self.compute_outer_hexagon_radius(inner_positions, inner_angles)

        # Check all constraints
        total_penalty = 0

        # Check containment for all inner hexagons
        outer_vertices = self.hexagon_vertices(0, 0, 0, outer_radius)
        for i, (pos, angle) in enumerate(zip(inner_positions, inner_angles)):
            hex_vertices = self.hexagon_vertices(pos[0], pos[1], angle)
            if not self.check_containment(hex_vertices, outer_vertices):
                total_penalty += 10000  # Large penalty for containment violation

        # Check overlaps between all pairs of inner hexagons
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                hex1_vertices = self.hexagon_vertices(inner_positions[i][0], inner_positions[i][1], inner_angles[i])
                hex2_vertices = self.hexagon_vertices(inner_positions[j][0], inner_positions[j][1], inner_angles[j])
                if self.check_overlap(hex1_vertices, hex2_vertices):
                    total_penalty += 10000  # Large penalty for overlap violation

        # Fitness is negative of the radius plus penalties
        # We want to minimize radius, so fitness = -radius
        fitness = -outer_radius - total_penalty

        return fitness, outer_radius

    def evaluate_fitness_parallel(self, individuals):
        """Parallel fitness evaluation for a batch of individuals"""
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(self.evaluate_fitness)(individual[:, :2], individual[:, 2])
            for individual in individuals
        )
        fitnesses = [r[0] for r in results]
        outer_radii = [r[1] for r in results]
        return fitnesses, outer_radii

class EvolutionaryHexagonPacker:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def mutate_individual(self, individual, mutation_rate=0.1, max_displacement=0.5):
        """Mutate individual with position and rotation changes"""
        mutated = deepcopy(individual)
        n = len(mutated)

        for i in range(n):
            # Mutate position
            if random.random() < mutation_rate:
                mutated[i][0] += random.uniform(-max_displacement, max_displacement)
                mutated[i][1] += random.uniform(-max_displacement, max_displacement)

            # Mutate rotation
            if random.random() < mutation_rate:
                mutated[i][2] += random.uniform(-30, 30)
                mutated[i][2] = mutated[i][2] % 360

        return mutated

    def crossover(self, parent1, parent2, crossover_rate=0.8):
        """Single-point crossover for hexagon packing"""
        if random.random() > crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)

        # Create offspring by combining parent genes
        n = len(parent1)
        crossover_point = random.randint(1, n-1)

        child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

        return child1, child2

    def create_hierarchical_initial_config(self):
        """Create a hierarchical hexagon layout that starts with a proven good configuration"""
        # Start with a 3x4 hexagonal tiling pattern that respects geometric relationships
        # This creates a balanced structure with minimal overlap risk

        # Base positions arranged in hexagonal lattice pattern
        base_positions = [
            # Central row
            [0, 0, 0],           # center
            [-2.0, 0, 0],        # left
            [2.0, 0, 0],         # right

            # Top row
            [-1.0, 1.732, 0],   # top-left
            [1.0, 1.732, 0],    # top-right

            # Bottom row
            [-1.0, -1.732, 0],  # bottom-left
            [1.0, -1.732, 0],   # bottom-right

            # Far top row
            [-2.0, 3.464, 0],   # far top-left
            [2.0, 3.464, 0],    # far top-right

            # Far bottom row
            [-2.0, -3.464, 0],  # far bottom-left
            [2.0, -3.464, 0],   # far bottom-right
        ]

        individual = np.array(base_positions)

        # Add noise that respects geometric constraints
        for i in range(len(individual)):
            # Apply smaller noise to avoid extreme movements that might cause overlaps
            individual[i][0] += random.uniform(-0.15, 0.15)
            individual[i][1] += random.uniform(-0.15, 0.15)
            individual[i][2] += random.uniform(-10, 10)
            individual[i][2] = individual[i][2] % 360

        return individual

    def create_diverse_initial_config(self):
        """Create a more scattered initial individual"""
        individual = np.zeros((11, 3))

        # Place center hexagon
        individual[0] = [0, 0, 0]

        # Place others with more spread
        positions = [
            [-3.0, 0], [3.0, 0],  # left and right
            [-1.5, 2.6], [1.5, 2.6],  # top
            [-1.5, -2.6], [1.5, -2.6],  # bottom
            [-4.0, 2.6], [4.0, 2.6],  # far top
            [-4.0, -2.6], [4.0, -2.6],  # far bottom
        ]

        # Fill in positions with randomness
        for i in range(1, 11):
            individual[i] = [
                positions[i-1][0] + random.uniform(-0.5, 0.5),
                positions[i-1][1] + random.uniform(-0.5, 0.5),
                random.uniform(0, 360)
            ]

        return individual

    def initialize_population(self, population_size):
        """Initialize population with diverse individuals using multi-phase approach"""
        population = []

        # Use 50% phase 1, 50% phase 2 for diversity
        half_size = population_size // 2
        for i in range(half_size):
            individual = self.create_hierarchical_initial_config()
            population.append(individual)

        for i in range(half_size, population_size):
            individual = self.create_diverse_initial_config()
            population.append(individual)

        return population

    def select_parents(self, population, fitnesses, tournament_size=3):
        """Tournament selection with pressure adjustment"""
        # Use a slightly higher tournament size but less pressure for better exploration
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            # Select with some probability of selecting lower fitness to maintain diversity
            if random.random() < 0.3:
                # Select worst performer with small probability
                winner_idx = tournament_indices[np.argmin(tournament_fitnesses)]
            else:
                # Normal selection
                winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(deepcopy(population[winner_idx]))
        return selected

    def local_optimization_step(self, individual, max_iter=100):
        """Enhanced local optimization with gradient guidance and smart perturbation"""
        best_individual = deepcopy(individual)
        best_fitness, _ = self.optimizer.evaluate_fitness(best_individual[:, :2], best_individual[:, 2])

        # Enhanced stages with adaptive perturbation sizes
        stages = [
            {"iterations": max_iter // 4, "displacement": 0.3, "smart_perturb": True},
            {"iterations": max_iter // 4, "displacement": 0.15, "smart_perturb": True},
            {"iterations": max_iter // 4, "displacement": 0.05, "smart_perturb": True},
            {"iterations": max_iter // 4, "displacement": 0.01, "smart_perturb": True}
        ]

        for stage in stages:
            for _ in range(stage["iterations"]):
                mutated = deepcopy(best_individual)
                idx = random.randint(0, len(mutated)-1)

                # Smart perturbation based on fitness landscape
                if stage["smart_perturb"]:
                    # If we're near a good solution, make smaller moves
                    # If we're exploring, make larger moves
                    displacement = stage["displacement"]

                    # Perturb position
                    mutated[idx][0] += random.uniform(-displacement, displacement)
                    mutated[idx][1] += random.uniform(-displacement, displacement)

                    # Perturb rotation with appropriate scaling
                    mutated[idx][2] += random.uniform(-10, 10) if stage["displacement"] > 0.05 else random.uniform(-3, 3)
                    mutated[idx][2] = mutated[idx][2] % 360
                else:
                    # Regular perturbation
                    mutated[idx][0] += random.uniform(-stage["displacement"], stage["displacement"])
                    mutated[idx][1] += random.uniform(-stage["displacement"], stage["displacement"])
                    mutated[idx][2] += random.uniform(-5, 5)
                    mutated[idx][2] = mutated[idx][2] % 360

                mutated_fitness, _ = self.optimizer.evaluate_fitness(mutated[:, :2], mutated[:, 2])
                if mutated_fitness > best_fitness:
                    best_individual = mutated
                    best_fitness = mutated_fitness

        # Gradient-guided refinement phase
        # Use finite differences to estimate gradients for position improvements
        gradient_individual = deepcopy(best_individual)
        eps = 1e-4  # Small epsilon for finite difference
        grad_pos = np.zeros((len(gradient_individual), 2))  # Position gradients

        # Estimate position gradients for each hexagon
        for i in range(len(gradient_individual)):
            # Estimate partial derivatives for x and y coordinates
            orig_x, orig_y = gradient_individual[i][0], gradient_individual[i][1]

            # Perturb x coordinate
            test_individual_x = deepcopy(gradient_individual)
            test_individual_x[i][0] = orig_x + eps
            fitness_x_plus, _ = self.optimizer.evaluate_fitness(test_individual_x[:, :2], test_individual_x[:, 2])

            test_individual_x[i][0] = orig_x - eps
            fitness_x_minus, _ = self.optimizer.evaluate_fitness(test_individual_x[:, :2], test_individual_x[:, 2])

            grad_x = (fitness_x_plus - fitness_x_minus) / (2 * eps)
            grad_pos[i][0] = grad_x

            # Perturb y coordinate
            test_individual_y = deepcopy(gradient_individual)
            test_individual_y[i][1] = orig_y + eps
            fitness_y_plus, _ = self.optimizer.evaluate_fitness(test_individual_y[:, :2], test_individual_y[:, 2])

            test_individual_y[i][1] = orig_y - eps
            fitness_y_minus, _ = self.optimizer.evaluate_fitness(test_individual_y[:, :2], test_individual_y[:, 2])

            grad_y = (fitness_y_plus - fitness_y_minus) / (2 * eps)
            grad_pos[i][1] = grad_y

        # Apply gradient-based updates if they improve fitness
        updated_individual = deepcopy(gradient_individual)
        learning_rate = 0.1
        for i in range(len(updated_individual)):
            # Update positions based on estimated gradients
            new_x = updated_individual[i][0] + learning_rate * grad_pos[i][0]
            new_y = updated_individual[i][1] + learning_rate * grad_pos[i][1]

            # Ensure new positions remain valid
            updated_individual[i][0] = new_x
            updated_individual[i][1] = new_y

        # Final validation of gradient update
        try:
            updated_fitness, _ = self.optimizer.evaluate_fitness(updated_individual[:, :2], updated_individual[:, 2])
            if updated_fitness > best_fitness:
                best_individual = updated_individual
        except:
            pass  # If gradient update fails, keep original

        return best_individual

    def optimize(self, max_time_seconds=170):
        """Main optimization loop"""
        # Parameters
        population_size = 70  # Slightly larger for better diversity
        generations = 130     # More generations for better convergence
        initial_mutation_rate = 0.25  # Higher initial mutation for exploration
        crossover_rate = 0.85
        elitism_rate = 0.18  # More elitism to preserve good solutions

        start_time = time.time()

        # Initialize population
        population = self.initialize_population(population_size)

        best_fitness_history = []
        improvement_window = 25
        window_fitness = []

        for gen in range(generations):
            if time.time() - start_time > max_time_seconds:
                break

            # Adaptive mutation rate that decreases over generations
            mutation_rate = max(0.05, initial_mutation_rate * (1 - gen / generations))

            # Evaluate fitness for all individuals
            fitnesses, _ = self.optimizer.evaluate_fitness_parallel(population)

            # Track best
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            best_fitness_history.append(best_fitness)

            # Track window to detect convergence
            window_fitness.append(best_fitness)
            if len(window_fitness) > improvement_window:
                window_fitness.pop(0)

            # Adjust parameters based on recent performance
            if len(window_fitness) == improvement_window:
                improvement_rate = window_fitness[-1] - window_fitness[0]
                if improvement_rate < 1e-5:
                    # Converged, reduce mutation and increase exploitation
                    mutation_rate *= 0.6
                    if gen % 4 == 0:
                        # Increase elitism during convergence
                        elitism_rate = min(0.25, elitism_rate + 0.015)

            # Local optimization on best individual every 3 generations
            if gen % 3 == 0:
                population[best_idx] = self.local_optimization_step(population[best_idx])

            # Elitism - keep best individuals
            elite_count = int(elitism_rate * population_size)
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            elites = [deepcopy(population[i]) for i in elite_indices]

            # Selection
            parents = self.select_parents(population, fitnesses)

            # Crossover and mutation
            new_population = elites.copy()

            while len(new_population) < population_size:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                child1, child2 = self.crossover(parent1, parent2, crossover_rate)

                child1 = self.mutate_individual(child1, mutation_rate, max_displacement=0.35)
                child2 = self.mutate_individual(child2, mutation_rate, max_displacement=0.35)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:population_size]

        # Final evaluation
        final_fitnesses, _ = self.optimizer.evaluate_fitness_parallel(population)

        best_idx = np.argmax(final_fitnesses)
        best_individual = population[best_idx]

        # Final optimization with more intensive search
        best_individual = self.local_optimization_step(best_individual, max_iter=150)

        # Get final results
        final_fitness, outer_radius = self.optimizer.evaluate_fitness(best_individual[:, :2], best_individual[:, 2])

        return best_individual, outer_radius

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Create optimizer and packer instances
    optimizer = HexagonPackingOptimizer()
    packer = EvolutionaryHexagonPacker(optimizer)

    # Run improved evolutionary optimization
    inner_hex_data, outer_hex_side_length = packer.optimize()

    # Format output as required
    outer_hex_data = np.array([0, 0, 0])  # centered at origin

    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END