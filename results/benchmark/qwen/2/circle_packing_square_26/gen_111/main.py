# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
import time
from typing import Tuple, List, Optional
import math

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePackingOptimizer:
    def __init__(self, n_circles: int = 26, pop_size: int = 50, max_generations: int = 1000):
        self.n_circles = n_circles
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.best_fitness_history = []

    def validate_circles(self, circles: np.ndarray) -> bool:
        """Validate that circles are within bounds and non-overlapping."""
        if len(circles) != self.n_circles:
            return False

        # Check containment constraints
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1 - r or y < r or y > 1 - r:
                return False

        # Check overlap constraints using KDTree for efficiency
        points = circles[:, :2]
        tree = KDTree(points)

        for i in range(self.n_circles):
            x, y, r = circles[i]
            # Find nearby circles (within 2*r distance)
            nearby = tree.query_ball_point([x, y], 2 * r)
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance < r + r2:
                        return False

        return True

    def calculate_fitness(self, circles: np.ndarray) -> float:
        """Calculate total radius sum as fitness."""
        return np.sum(circles[:, 2])

    def initialize_population(self) -> List[np.ndarray]:
        """Generate initial population of circle arrangements."""
        population = []

        # Use a more systematic initialization strategy inspired by the DEAP approach
        for _ in range(self.pop_size):
            circles = self._create_valid_circles()
            population.append(circles)

        return population

    def _create_valid_circles(self) -> np.ndarray:
        """Create a valid configuration of circles."""
        circles = np.zeros((self.n_circles, 3))

        # Start with a structured approach: place circles in a grid-like pattern
        # with diminishing sizes to allow better packing
        grid_size = int(math.ceil(math.sqrt(self.n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= self.n_circles:
                    break

                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                # Initial radius based on grid spacing
                r = min(spacing_x, spacing_y) * 0.3

                # Add some randomness to avoid perfect grid
                r = max(0.005, r * np.random.uniform(0.8, 1.2))
                x = max(r, min(1-r, x + np.random.uniform(-spacing_x*0.1, spacing_x*0.1)))
                y = max(r, min(1-r, y + np.random.uniform(-spacing_y*0.1, spacing_y*0.1)))

                circles[idx] = [x, y, r]
                idx += 1

        # If we still have unfilled circles, place them randomly
        for i in range(idx, self.n_circles):
            max_attempts = 1000
            placed = False
            attempts = 0

            while not placed and attempts < max_attempts:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = np.random.uniform(0.01, 0.1)

                # Check if valid placement
                valid_placement = True
                if r <= x <= 1 - r and r <= y <= 1 - r:
                    # Check overlap with existing circles
                    for j in range(i):
                        existing_x, existing_y, existing_r = circles[j]
                        distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if distance < r + existing_r:
                            valid_placement = False
                            break
                else:
                    valid_placement = False

                if valid_placement:
                    circles[i] = [x, y, r]
                    placed = True
                attempts += 1

            # If failed to place, use minimum radius
            if not placed:
                circles[i] = [0.5, 0.5, 0.01]

        return circles

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Perform crossover between two parents."""
        child = np.copy(parent1)

        # Perform uniform crossover - inspired by DEAP's approach
        mask = np.random.rand(self.n_circles) > 0.5
        child[mask] = parent2[mask]

        # Ensure child remains valid by repositioning circles if necessary
        for i in range(self.n_circles):
            x, y, r = child[i]
            # Adjust position if out of bounds
            if x < r:
                x = r
            elif x > 1 - r:
                x = 1 - r
            if y < r:
                y = r
            elif y > 1 - r:
                y = 1 - r
            child[i] = [x, y, r]

        # Recheck for overlaps and fix them
        return self._repair_overlaps(child)

    def _repair_overlaps(self, circles: np.ndarray) -> np.ndarray:
        """Repair overlapping circles by adjusting positions and radii."""
        # Try several iterations to resolve overlaps
        for _ in range(5):
            if self.validate_circles(circles):
                return circles

            # Simple overlap resolution: reduce radii and adjust positions slightly
            for i in range(self.n_circles):
                x, y, r = circles[i]
                # Reduce radius
                circles[i] = [x, y, max(0.001, r * 0.95)]

        # If still invalid, do one final major adjustment
        for i in range(self.n_circles):
            x, y, r = circles[i]
            # If still invalid, just make it safe
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]

        return circles

    def mutate(self, circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Apply mutations to circles."""
        mutated = np.copy(circles)

        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Mutate either position or radius - balanced approach
                if np.random.random() < 0.5:
                    # Mutate position
                    x, y, r = mutated[i]
                    # Small random perturbation
                    x += np.random.normal(0, 0.01)
                    y += np.random.normal(0, 0.01)
                    # Keep within bounds
                    x = max(r, min(1 - r, x))
                    y = max(r, min(1 - r, y))
                    mutated[i] = [x, y, r]
                else:
                    # Mutate radius
                    x, y, r = mutated[i]
                    # Small random change in radius (log-normal to avoid negative values)
                    r *= np.random.lognormal(0, 0.1)  # More controlled than normal distribution
                    # Keep positive
                    r = max(0.001, r)
                    mutated[i] = [x, y, r]

        # Ensure validity after mutation
        return self._repair_overlaps(mutated)

    def evaluate_population(self, population: List[np.ndarray]) -> List[float]:
        """Evaluate fitness of entire population."""
        return [self.calculate_fitness(circles) for circles in population]

    def select_parents(self, population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Select two parents using tournament selection."""
        # Tournament selection with smaller tournament size for more pressure
        tournament_size = min(3, len(population))
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitnesses[i] for i in tournament_indices]
        winner1_idx = tournament_indices[np.argmax(tournament_fitness)]

        # Tournament selection for second parent (different from first)
        tournament_indices2 = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness2 = [fitnesses[i] for i in tournament_indices2]
        winner2_idx = tournament_indices2[np.argmax(tournament_fitness2)]

        return population[winner1_idx], population[winner2_idx]

    def adaptive_mutation_rate(self, generation: int, diversity: float) -> float:
        """Adaptively adjust mutation rate based on generation and population diversity."""
        base_rate = 0.1
        # Decrease mutation rate over time
        time_factor = 1.0 - (generation / self.max_generations)
        # Increase mutation rate if diversity is low (to escape local optima)
        diversity_factor = max(0.5, 1.0 - diversity) if diversity < 0.5 else 1.0

        return base_rate * time_factor * diversity_factor

    def get_population_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity based on spread of radii."""
        if len(population) < 2:
            return 0.0

        all_radii = np.concatenate([circles[:, 2] for circles in population])
        return np.std(all_radii) / (np.mean(all_radii) + 1e-8)  # Avoid division by zero

    def local_search_improve(self, circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Apply local search to improve a given solution by iteratively adjusting positions/radii."""
        current = circles.copy()
        current_fitness = self.calculate_fitness(current)

        for iteration in range(max_iterations):
            improved = False
            # Try to improve each circle individually
            for i in range(self.n_circles):
                original_x, original_y, original_r = current[i]
                best_x, best_y, best_r = original_x, original_y, original_r
                best_fitness = current_fitness

                # Try small adjustments to position and radius
                step_sizes = [0.005, 0.01, 0.02]
                for step in step_sizes:
                    # Test position changes
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            new_x = original_x + dx
                            new_y = original_y + dy

                            # Ensure new position is within bounds
                            if new_x - original_r >= 0 and new_x + original_r <= 1 and \
                               new_y - original_r >= 0 and new_y + original_r <= 1:

                                # Create temporary configuration
                                temp_circles = current.copy()
                                temp_circles[i] = [new_x, new_y, original_r]

                                # Check if this improves overall fitness
                                if self.validate_circles(temp_circles):
                                    new_fitness = self.calculate_fitness(temp_circles)
                                    if new_fitness > best_fitness:
                                        best_fitness = new_fitness
                                        best_x, best_y, best_r = new_x, new_y, original_r
                                        improved = True

                    # Test radius changes
                    for dr in [-step, 0, step]:
                        new_r = original_r + dr
                        if new_r > 0.001 and new_r < 0.5:  # Reasonable bounds
                            # Ensure new radius allows for valid positioning
                            new_r = min(new_r, original_x, 1-original_x, original_y, 1-original_y)
                            if new_r > 0.001:
                                # Create temporary configuration
                                temp_circles = current.copy()
                                temp_circles[i] = [original_x, original_y, new_r]

                                # Check if this improves overall fitness
                                if self.validate_circles(temp_circles):
                                    new_fitness = self.calculate_fitness(temp_circles)
                                    if new_fitness > best_fitness:
                                        best_fitness = new_fitness
                                        best_x, best_y, best_r = original_x, original_y, new_r
                                        improved = True

                # Update if we found a better configuration
                if improved:
                    current[i] = [best_x, best_y, best_r]
                    current_fitness = best_fitness

            # Stop if no improvement was made in this iteration
            if not improved:
                break

        return current

    def evolve(self) -> np.ndarray:
        """Main evolutionary algorithm for circle packing."""
        # Initialize population
        population = self.initialize_population()

        # Evolution loop
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitnesses = self.evaluate_population(population)

            # Track best fitness
            best_fitness = max(fitnesses)
            self.best_fitness_history.append(best_fitness)

            # Print progress every 100 generations
            if generation % 100 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

            # Create new population
            new_population = []

            # Elitism: keep the best individual
            best_idx = np.argmax(fitnesses)
            best_individual = population[best_idx]

            # Apply local search to the best individual before keeping it
            improved_best = self.local_search_improve(best_individual)
            new_population.append(improved_best)

            # Calculate population diversity for adaptive parameters
            diversity = self.get_population_diversity(population)

            # Generate offspring
            while len(new_population) < self.pop_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitnesses)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation with adaptive rate
                mutation_rate = self.adaptive_mutation_rate(generation, diversity)
                child = self.mutate(child, mutation_rate)

                # Apply local search to offspring for fine-tuning
                child = self.local_search_improve(child)

                new_population.append(child)

            population = new_population[:self.pop_size]  # Ensure exact population size

            # Early stopping if fitness improves very little
            if len(self.best_fitness_history) > 10:
                recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-10]
                if recent_improvement < 1e-6:
                    break

        # Return best solution
        final_fitnesses = self.evaluate_population(population)
        best_idx = np.argmax(final_fitnesses)
        return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()

    try:
        # Create optimizer instance
        optimizer = CirclePackingOptimizer(n_circles=26, pop_size=50, max_generations=1000)

        # Run evolution
        circles = optimizer.evolve()

        # Validate result
        if not optimizer.validate_circles(circles):
            # If validation fails, try a simpler approach
            print("Validation failed on evolved solution, using fallback...")
            circles = np.zeros((26, 3))
            # Use a simple heuristic: distribute evenly with decreasing radii
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]

        end_time = time.time()
        eval_time = end_time - start_time
        print(f"Evolution completed in {eval_time:.2f} seconds")

    except Exception as e:
        print(f"Error during evolution: {e}")
        # Fallback to simple initialization
        circles = np.zeros((26, 3))
        print("Using fallback solution due to error")

    return circles

# EVOLVE-BLOCK-END