# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional
import random
from copy import deepcopy
import time

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE_INITIAL = 0.15
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class CirclePackingOptimizer:
    def __init__(self):
        self.n_circles = 26

    def is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Check if the configuration satisfies all constraints."""
        if len(circles) != self.n_circles:
            return False

        # Check containment constraints using vectorized operations
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]

        # Check if any radius violates containment
        containment_check = (
            (radii <= x_coords) &
            (radii <= y_coords) &
            (radii <= 1 - x_coords) &
            (radii <= 1 - y_coords)
        )

        if not np.all(containment_check):
            return False

        # Check overlap constraints using pairwise distance matrix
        if self.n_circles > 1:
            distances = cdist(circles[:, :2], circles[:, :2])
            # Create upper triangular mask to avoid duplicate comparisons
            mask = np.triu(np.ones((self.n_circles, self.n_circles), dtype=bool), k=1)

            # Calculate minimum required distance
            min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask

            # Check for overlaps
            overlaps = distances < min_distances
            if np.any(overlaps):
                return False

        return True

    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])

    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Initialize population with valid configurations."""
        population = []

        # Generate diverse initial configurations
        for _ in range(pop_size):
            circles = self._create_diverse_initialization()
            if self.is_valid_configuration(circles):
                population.append(circles.copy())
            else:
                # Fallback to valid configuration
                circles = self._create_simple_initialization()
                if self.is_valid_configuration(circles):
                    population.append(circles.copy())

        return population

    def _create_simple_initialization(self) -> np.ndarray:
        """Create a simple but valid initial configuration."""
        circles = np.zeros((self.n_circles, 3))

        # Place in a simple grid pattern
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)

        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4  # Conservative radius
                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def _create_diverse_initialization(self) -> np.ndarray:
        """Create an initialization with better spatial distribution."""
        circles = np.zeros((self.n_circles, 3))

        # Multi-scale approach: create different layers of grid
        if self.n_circles <= 9:
            # Small number: use tight grid
            grid_size = int(np.ceil(np.sqrt(self.n_circles)))
            spacing = 1.0 / (grid_size + 1)
            idx = 0
            for row in range(grid_size):
                for col in range(grid_size):
                    if idx >= self.n_circles:
                        break
                    x = (col + 1) * spacing
                    y = (row + 1) * spacing
                    r = spacing / 3
                    # Add small randomness
                    x += np.random.uniform(-spacing/8, spacing/8)
                    y += np.random.uniform(-spacing/8, spacing/8)
                    r = max(0.01, min(r, x, y, 1-x, 1-y))
                    circles[idx] = [x, y, r]
                    idx += 1
        elif self.n_circles <= 16:
            # Medium number: use two concentric grids
            outer_grid_size = 4
            inner_grid_size = 2

            # Outer grid
            outer_spacing = 1.0 / (outer_grid_size + 1)
            idx = 0
            for row in range(outer_grid_size):
                for col in range(outer_grid_size):
                    if idx >= self.n_circles:
                        break
                    x = (col + 1) * outer_spacing
                    y = (row + 1) * outer_spacing
                    r = outer_spacing / 4
                    # Add randomness
                    x += np.random.uniform(-outer_spacing/6, outer_spacing/6)
                    y += np.random.uniform(-outer_spacing/6, outer_spacing/6)
                    r = max(0.01, min(r, x, y, 1-x, 1-y))
                    circles[idx] = [x, y, r]
                    idx += 1
        else:
            # Larger number: use strategic placement
            # First place some key circles at corners and center
            key_positions = [
                (0.1, 0.1, 0.05),      # bottom-left
                (0.9, 0.1, 0.05),      # bottom-right
                (0.1, 0.9, 0.05),      # top-left
                (0.9, 0.9, 0.05),      # top-right
                (0.5, 0.5, 0.1),       # center
            ]

            # Fill remaining positions with grid
            grid_size = int(np.ceil(np.sqrt(self.n_circles - len(key_positions))))
            spacing = 1.0 / (grid_size + 1)

            idx = 0
            for pos in key_positions:
                if idx >= self.n_circles:
                    break
                circles[idx] = list(pos)
                idx += 1

            # Fill remaining with grid
            remaining_count = self.n_circles - idx
            for i in range(remaining_count):
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 3
                # Add randomness
                x += np.random.uniform(-spacing/8, spacing/8)
                y += np.random.uniform(-spacing/8, spacing/8)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def optimize_placement(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply advanced local optimization to improve placement."""
        n = len(circles)
        circles = circles.copy()

        # Gradient-based local optimization
        for iteration in range(max_iter):
            improved = False

            # Try to increase radii while respecting constraints
            for i in range(n):
                original_radius = circles[i][2]
                original_x, original_y = circles[i][0], circles[i][1]

                # Calculate maximum possible radius
                max_radius = min(
                    original_x,
                    original_y,
                    1 - original_x,
                    1 - original_y
                )

                # Try to increase the radius
                new_radius = min(original_radius + 0.005, max_radius)

                if new_radius > original_radius:
                    # Temporarily update radius
                    circles[i][2] = new_radius

                    # Check if valid configuration
                    if self.is_valid_configuration(circles):
                        improved = True
                    else:
                        # Revert if invalid
                        circles[i][2] = original_radius

            if not improved:
                break

        return circles

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        n = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Uniform crossover for positions and radii
        for i in range(n):
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]

        # Ensure children are valid
        child1 = self.optimize_placement(child1)
        child2 = self.optimize_placement(child2)

        return child1, child2

    def mutate(self, circles: np.ndarray, mutation_rate: float = MUTATION_RATE_INITIAL) -> np.ndarray:
        """Apply mutation to a configuration."""
        mutated = circles.copy()
        n = len(mutated)

        for i in range(n):
            if np.random.random() < mutation_rate:
                # Mutate either position or radius
                if np.random.random() < 0.7:  # 70% chance to mutate position
                    # Mutate position
                    mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, 0.03), 0, 1)
                    mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, 0.03), 0, 1)
                else:
                    # Mutate radius
                    mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.01), 0.01, 0.5)

        # Optimize the mutated configuration
        mutated = self.optimize_placement(mutated)

        return mutated

    def select_tournament(self, population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = TOURNAMENT_SIZE) -> int:
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

    def compute_fitness(self, circles: np.ndarray) -> float:
        """Compute fitness with penalty for invalid configurations."""
        if self.is_valid_configuration(circles):
            return self.calculate_sum_radii(circles)
        else:
            # Invalid configurations get very low fitness
            return 0.0

    def run_evolution(self) -> np.ndarray:
        """Run the complete evolutionary algorithm."""
        # Initialize population
        population = self.initialize_population(POPULATION_SIZE)

        if not population:
            # Fallback to simple initialization
            return self._create_simple_initialization()

        best_solution = None
        best_fitness = -1

        for generation in range(GENERATIONS):
            # Adjust mutation rate based on generation (adaptive)
            mutation_rate = max(MUTATION_RATE_INITIAL * (1 - generation / GENERATIONS), 0.01)

            # Evaluate fitness for all individuals (can be parallelized)
            fitnesses = [self.compute_fitness(circles) for circles in population]

            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Create new population
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Tournament selection
                parent1_idx = self.select_tournament(population, fitnesses)
                parent2_idx = self.select_tournament(population, fitnesses)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)

                # Add children to new population
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:POPULATION_SIZE]

        # Return the best solution found
        if best_solution is None:
            # Fallback to a simple configuration if nothing worked
            return self._create_simple_initialization()

        return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END