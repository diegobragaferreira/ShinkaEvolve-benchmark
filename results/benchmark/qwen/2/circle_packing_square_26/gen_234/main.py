# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from collections import Counter
import time
from typing import Tuple, List, Optional

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

class CircleValidator:
    """Validates circle configurations and checks all constraints efficiently."""

    def __init__(self):
        self.tree_cache = {}

    def is_valid(self, circles: np.ndarray) -> bool:
        """Check if the configuration satisfies all constraints."""
        if len(circles) == 0:
            return False

        # Check containment constraints
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]

        # Vectorized containment check
        containment_check = (
            (radii <= x_coords) &
            (radii <= y_coords) &
            (radii <= 1 - x_coords) &
            (radii <= 1 - y_coords)
        )

        if not np.all(containment_check):
            return False

        # Check overlap constraints using spatial indexing for efficiency
        n = len(circles)
        if n <= 1:
            return True

        # Use cached KDTree if available (for repeated validation)
        points = circles[:, :2]
        tree = cKDTree(points)

        # Query for neighbors within double the maximum radius
        for i in range(n):
            x, y, r = circles[i]
            # Find nearby circles (within 2*r distance) using spatial indexing
            indices = tree.query_ball_point([x, y], 2*r)
            # Check overlap with all nearby circles
            for j in indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < r + r2:
                        return False

        return True

class FitnessEvaluator:
    """Evaluates the fitness of circle configurations."""

    @staticmethod
    def calculate_sum_radii(circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])

class PopulationInitializer:
    """Creates initial population with valid configurations using advanced initialization."""

    def __init__(self, n_circles: int):
        self.n_circles = n_circles

    def create_hexagonal_initialization(self) -> np.ndarray:
        """Initialize circles using a hexagonal grid pattern for better spatial distribution."""
        circles = np.zeros((self.n_circles, 3))

        # Create hexagonal grid pattern
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))

        spacing = 1.0 / (max(rows, cols) + 2)
        hex_height = spacing * np.sqrt(3) / 2

        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= self.n_circles:
                    break
                # Offset every other row for hexagonal packing
                x = (j + 0.5 + (i % 2) * 0.5) * spacing
                y = (i + 0.5) * hex_height

                # Ensure within bounds
                x = np.clip(x, spacing/2, 1 - spacing/2)
                y = np.clip(y, spacing/2, 1 - spacing/2)

                # Estimate radius based on proximity to boundaries
                max_radius = min(x, y, 1-x, 1-y)
                r = min(max_radius * 0.3, spacing/3)

                circles[count] = [x, y, r]
                count += 1

            if count >= self.n_circles:
                break

        return circles

    def create_multi_scale_grid_initialization(self) -> np.ndarray:
        """Create initial circles using multi-scale grid placement for diversity."""
        circles = np.zeros((self.n_circles, 3))

        # Try multiple grid resolutions
        scales = [10, 8, 6, 5]
        best_layout = None
        best_fitness = -np.inf

        for scale in scales:
            # Create grid layout
            grid_rows = scale
            grid_cols = scale
            spacing_x = 1.0 / (grid_cols + 1)
            spacing_y = 1.0 / (grid_rows + 1)

            temp_circles = np.zeros((self.n_circles, 3))
            count = 0

            for i in range(grid_rows):
                for j in range(grid_cols):
                    if count >= self.n_circles:
                        break
                    x = (j + 1) * spacing_x
                    y = (i + 1) * spacing_y

                    # Add controlled perturbation
                    perturbation = 0.1 * min(spacing_x, spacing_y)
                    x += np.random.uniform(-perturbation, perturbation)
                    y += np.random.uniform(-perturbation, perturbation)

                    # Bound the coordinates
                    x = np.clip(x, spacing_x/2, 1 - spacing_x/2)
                    y = np.clip(y, spacing_y/2, 1 - spacing_y/2)

                    max_radius = min(x, 1-x, y, 1-y)
                    r = min(max_radius * 0.3, spacing_x/3)

                    temp_circles[count] = [x, y, r]
                    count += 1

                if count >= self.n_circles:
                    break

            if count >= self.n_circles:
                # Validate and measure fitness
                if self._validate_and_adjust(temp_circles):
                    fitness = FitnessEvaluator.calculate_sum_radii(temp_circles)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_layout = temp_circles.copy()

        # Fallback to hexagonal if nothing works
        if best_layout is None:
            return self.create_hexagonal_initialization()

        return best_layout

    def _validate_and_adjust(self, circles: np.ndarray) -> bool:
        """Validate and adjust circles to make them feasible."""
        # Adjust positions and radii to ensure containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Adjust radius
            max_radius = min(x, 1-x, y, 1-y)
            if r > max_radius:
                circles[i, 2] = max_radius * 0.95

            # Adjust position
            x = np.clip(x, circles[i, 2], 1 - circles[i, 2])
            y = np.clip(y, circles[i, 2], 1 - circles[i, 2])
            circles[i, 0] = x
            circles[i, 1] = y

        return True

    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Initialize population with diverse and valid configurations."""
        population = []

        # Create diverse initial configurations with better initialization
        for i in range(pop_size):
            if i == 0:
                # First: hexagonal initialization
                circles = self.create_hexagonal_initialization()
            elif i < pop_size // 3:
                # Second third: multi-scale grid
                circles = self.create_multi_scale_grid_initialization()
            else:
                # Remaining: random with constraint checking
                circles = self._create_constrained_random()

            # Apply local refinement to improve fitness
            circles = self._local_refinement(circles)

            population.append(circles)

        return population

    def _create_constrained_random(self) -> np.ndarray:
        """Create a constrained random initialization."""
        circles = np.zeros((self.n_circles, 3))

        for i in range(self.n_circles):
            attempts = 0
            while attempts < 100:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)

                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/3)

                overlap = False
                for j in range(i):
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < r + existing_r:
                        overlap = True
                        break

                if not overlap:
                    circles[i] = [x, y, r]
                    break
                attempts += 1

            if attempts >= 100:
                # Fallback to simple initialization
                grid_size = int(np.ceil(np.sqrt(self.n_circles)))
                spacing = 1.0 / (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 5
                circles[i] = [x, y, r]

        return circles

    def _local_refinement(self, circles: np.ndarray) -> np.ndarray:
        """Apply local refinement to improve initial configuration."""
        circles = circles.copy()
        n = len(circles)

        # Try to increase radii while respecting constraints
        for _ in range(50):  # Limited iterations for performance
            improved = False

            for i in range(n):
                x, y, r = circles[i]

                # Calculate maximum possible radius
                max_radius = min(x, 1-x, y, 1-y)

                if max_radius <= r:
                    continue

                # Try to increase radius
                new_r = min(r + 0.005, max_radius)

                # Check if we can actually increase it without violating constraints
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < new_r + r2:
                            valid = False
                            break

                if valid and new_r > r:
                    circles[i, 2] = new_r
                    improved = True

            if not improved:
                break

        return circles

class LocalOptimizer:
    """Performs local optimization to improve circle placements with hierarchy."""

    @staticmethod
    def optimize_placement(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply hierarchical local optimization to improve placement."""
        circles = circles.copy()
        n = len(circles)

        # Calculate overlap severity for intelligent optimization
        overlap_severity = LocalOptimizer._calculate_overlap_severity(circles)

        # Apply optimized refinement based on overlap severity
        if overlap_severity < 5:  # Low overlap - light refinement
            return LocalOptimizer._light_refinement(circles, max_iter)
        elif overlap_severity < 15:  # Medium overlap - moderate refinement
            return LocalOptimizer._moderate_refinement(circles, max_iter)
        else:  # High overlap - intensive refinement
            return LocalOptimizer._intensive_refinement(circles, max_iter)

    @staticmethod
    def _calculate_overlap_severity(circles: np.ndarray) -> int:
        """Calculate number of overlap violations."""
        n = len(circles)
        overlaps = 0

        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < r1 + r2:
                    overlaps += 1

        return overlaps

    @staticmethod
    def _light_refinement(circles: np.ndarray, max_iter: int) -> np.ndarray:
        """Light refinement for configurations with few overlaps."""
        circles = circles.copy()

        for _ in range(max_iter):
            improved = False

            for i in range(len(circles)):
                x, y, r = circles[i]
                max_r = min(x, 1-x, y, 1-y)

                if max_r <= r:
                    continue

                new_r = min(r + 0.002, max_r)

                # Quick constraint check
                valid = True
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < new_r + r2:
                            valid = False
                            break

                if valid and new_r > r:
                    circles[i, 2] = new_r
                    improved = True

            if not improved:
                break

        return circles

    @staticmethod
    def _moderate_refinement(circles: np.ndarray, max_iter: int) -> np.ndarray:
        """Moderate refinement for configurations with moderate overlaps."""
        circles = circles.copy()

        # Start with radius expansion
        for i in range(len(circles)):
            x, y, r = circles[i]
            max_r = min(x, 1-x, y, 1-y)

            if max_r <= r:
                continue

            new_r = min(r + 0.005, max_r)

            # Check constraint
            valid = True
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < new_r + r2:
                        valid = False
                        break

            if valid and new_r > r:
                circles[i, 2] = new_r

        # Then apply geometric refinement
        for _ in range(max_iter):
            improved = False

            for i in range(len(circles)):
                x, y, r = circles[i]
                max_r = min(x, 1-x, y, 1-y)

                if max_r <= r:
                    continue

                new_r = min(r + 0.003, max_r)

                # Check constraint
                valid = True
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < new_r + r2:
                            valid = False
                            break

                if valid and new_r > r:
                    circles[i, 2] = new_r
                    improved = True

            if not improved:
                break

        return circles

    @staticmethod
    def _intensive_refinement(circles: np.ndarray, max_iter: int) -> np.ndarray:
        """Intensive refinement for configurations with many overlaps."""
        circles = circles.copy()

        # Physics-inspired force-based relaxation
        n = len(circles)

        # Apply multiple rounds of force relaxation
        for round_num in range(3):
            # Initialize forces
            forces = np.zeros((n, 2))

            # Calculate forces between all pairs of circles
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist > 0 and dist < r1 + r2:
                        # Overlapping - apply repulsive force
                        force_magnitude = (r1 + r2 - dist) / dist
                        forces[i, 0] += dx * force_magnitude * 0.02
                        forces[i, 1] += dy * force_magnitude * 0.02
                        forces[j, 0] -= dx * force_magnitude * 0.02
                        forces[j, 1] -= dy * force_magnitude * 0.02
                    elif dist > 0:
                        # Not overlapping - apply weak attractive force to encourage packing
                        force_magnitude = 1.0 / (dist * dist * 0.01)
                        forces[i, 0] -= dx * force_magnitude * 0.001
                        forces[i, 1] -= dy * force_magnitude * 0.001
                        forces[j, 0] += dx * force_magnitude * 0.001
                        forces[j, 1] += dy * force_magnitude * 0.001

            # Apply forces to move circles
            for i in range(n):
                x, y, r = circles[i]

                # Apply forces with damping
                new_x = x + forces[i, 0] * 0.5
                new_y = y + forces[i, 1] * 0.5

                # Boundary constraints
                new_x = np.clip(new_x, r, 1-r)
                new_y = np.clip(new_y, r, 1-r)

                circles[i, 0] = new_x
                circles[i, 1] = new_y

            # Attempt to expand radii after force relaxation
            for i in range(n):
                x, y, r = circles[i]
                max_r = min(x, 1-x, y, 1-y)

                if max_r <= r:
                    continue

                # Try to increase radius
                new_r = min(r + 0.005, max_r)

                # Check constraint with all other circles
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < new_r + r2:
                            valid = False
                            break

                if valid and new_r > r:
                    circles[i, 2] = new_r

        return circles

class EvolutionaryOperator:
    """Handles evolutionary operators with constraint awareness."""

    def __init__(self, n_circles: int):
        self.n_circles = n_circles

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform constraint-aware crossover with overlap probability weighting."""
        if np.random.random() > 0.85:  # Allow some direct copying
            return parent1.copy(), parent2.copy()

        # Weighted crossover based on overlap risk
        child1 = parent1.copy()
        child2 = parent2.copy()

        # For each circle, assess overlap risk with the other parent
        for i in range(self.n_circles):
            # Calculate average distance to other circles in both parents
            dist1_to_others = 0
            dist2_to_others = 0

            # Calculate distance to others in parent1
            for j in range(self.n_circles):
                if i != j:
                    x1, y1, r1 = parent1[i]
                    x2, y2, r2 = parent1[j]
                    dist1_to_others += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # Calculate distance to others in parent2
            for j in range(self.n_circles):
                if i != j:
                    x1, y1, r1 = parent2[i]
                    x2, y2, r2 = parent2[j]
                    dist2_to_others += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            # Use lower distance to make safer decisions (less overlap risk)
            risk_factor = min(dist1_to_others, dist2_to_others) / (self.n_circles - 1)

            # Adjust crossover probability based on risk
            crossover_prob = 0.8 if risk_factor > 1.0 else 0.3

            if np.random.random() < crossover_prob:
                child1[i], child2[i] = child2[i], child1[i]

        # Ensure children are valid through local optimization
        child1 = LocalOptimizer.optimize_placement(child1)
        child2 = LocalOptimizer.optimize_placement(child2)

        return child1, child2

    def mutate(self, circles: np.ndarray, mutation_rate: float = 0.15) -> np.ndarray:
        """Apply adaptive mutation with multiple strategies."""
        mutated = circles.copy()

        # Different mutation strategies based on circle properties
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Determine mutation type
                mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])

                if mutation_type == 'position':
                    # Mutate position with adaptive magnitude based on proximity to boundaries
                    x, y, r = mutated[i]
                    bound_distance = min(x, 1-x, y, 1-y)
                    max_mutation = min(0.03, bound_distance * 0.5)

                    mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, max_mutation * 0.5), 0, 1)
                    mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, max_mutation * 0.5), 0, 1)
                else:
                    # Mutate radius with bounded adaptation
                    old_r = mutated[i, 2]
                    max_delta = min(old_r, 0.5 - old_r) * 0.8
                    mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, max_delta * 0.5), 0.001, 0.5)

        # Apply constraint repair and local optimization
        mutated = LocalOptimizer.optimize_placement(mutated)

        return mutated

class TournamentSelector:
    """Implements adaptive tournament selection for evolutionary algorithm."""

    @staticmethod
    def select(population: List[np.ndarray], fitnesses: List[float],
               diversity: float, generation: int, max_generations: int) -> int:
        """Select an individual using adaptive tournament selection."""
        # Dynamic tournament size based on generation and diversity
        base_tournament_size = 3
        diversity_factor = 1.0 if diversity > 0.05 else 0.5
        generation_factor = 1.0 + (generation / max_generations) * 2.0

        tournament_size = int(max(base_tournament_size,
                                 base_tournament_size * diversity_factor * generation_factor))
        tournament_size = min(tournament_size, len(population))

        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

class CirclePackingOptimizer:
    """Main optimizer class that orchestrates the evolutionary process."""

    def __init__(self):
        self.n_circles = 26
        self.validator = CircleValidator()
        self.fitness_evaluator = FitnessEvaluator()
        self.initializer = PopulationInitializer(self.n_circles)
        self.evolution_operator = EvolutionaryOperator(self.n_circles)
        self.tournament_selector = TournamentSelector()

    def compute_fitness(self, circles: np.ndarray) -> float:
        """Compute fitness with penalty for invalid configurations."""
        if self.validator.is_valid(circles):
            return self.fitness_evaluator.calculate_sum_radii(circles)
        else:
            # Invalid configurations get very low fitness
            return -10000.0

    def run_evolution(self) -> np.ndarray:
        """Run the enhanced evolutionary algorithm."""
        # Initialize population
        population = self.initializer.initialize_population(100)

        if not population:
            # Fallback to simple initialization
            return self.initializer._create_constrained_random()

        best_solution = None
        best_fitness = -np.inf
        generation_times = []

        for generation in range(300):
            start_time = time.time()

            # Evaluate fitness for all individuals
            fitnesses = [self.compute_fitness(circles) for circles in population]

            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Calculate population diversity
            diversity = self._calculate_diversity(population)

            # Create new population
            new_population = []

            # Elitism: keep best individuals
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:10]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < 100:
                # Tournament selection with adaptive parameters
                parent1_idx = self.tournament_selector.select(
                    population, fitnesses, diversity, generation, 300)
                parent2_idx = self.tournament_selector.select(
                    population, fitnesses, diversity, generation, 300)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child1, child2 = self.evolution_operator.crossover(parent1, parent2)

                # Adaptive mutation rate scheduling
                mutation_rate = self._get_adaptive_mutation_rate(generation, 300)
                child1 = self.evolution_operator.mutate(child1, mutation_rate)
                child2 = self.evolution_operator.mutate(child2, mutation_rate)

                # Add children to new population
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:100]

            end_time = time.time()
            generation_times.append(end_time - start_time)

            # Print progress
            if generation % 30 == 0:
                avg_fitness = np.mean(fitnesses)
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, "
                      f"Avg fitness = {avg_fitness:.6f}, Time = {end_time - start_time:.4f}s")

        # Return the best solution found
        if best_solution is None:
            # Fallback to a simple configuration if nothing worked
            return self.initializer._create_constrained_random()

        return best_solution

    def _calculate_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity based on radius variation."""
        if len(population) == 0:
            return 0.0

        radii = np.array([circle[2] for individual in population for circle in individual])
        return np.std(radii) if len(radii) > 0 else 0.0

    def _get_adaptive_mutation_rate(self, generation: int, max_generations: int) -> float:
        """Adaptive mutation rate with phased scheduling."""
        # Three-phase scheduling
        if generation < 100:
            # Phase 1: High exploration
            return 0.15 * (1 - generation / 100)
        elif generation < 200:
            # Phase 2: Balanced exploration/exploitation
            return 0.05 * (1 - (generation - 100) / 100)
        else:
            # Phase 3: Fine-tuning exploitation
            return 0.015 * (1 - (generation - 200) / 100)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END