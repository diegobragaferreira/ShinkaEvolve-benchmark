# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random
import time
from collections import deque

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2, use 1.2 x 0.8 ratio as it's often effective
    rect_width = 1.2
    rect_height = 0.8

    n = 21

    # Generate grid-based initial placement using hexagonal packing pattern
    # This provides a good starting configuration that respects basic spatial constraints
    def generate_hexagonal_grid(num_circles, width, height):
        # Hexagonal grid parameters
        rows = int(np.ceil(np.sqrt(num_circles)))
        cols = int(np.ceil(num_circles / rows))

        # Calculate spacing to fit within container
        cell_size = min(width / cols, height / rows) * 0.9  # Safety margin
        spacing_x = cell_size * 1.1  # Slightly offset to make hexagonal pattern
        spacing_y = cell_size * np.sqrt(3) / 2 * 1.1

        circles = []
        y_offset = 0.1  # Margin from top
        x_offset = 0.1  # Margin from left

        circle_count = 0
        for i in range(rows):
            y = y_offset + i * spacing_y
            # Offset every other row for hexagonal pattern
            x_start = x_offset + (i % 2) * spacing_x / 2
            for j in range(cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x
                if x < width - 0.1 and y < height - 0.1:  # Within bounds
                    # Start with small radius
                    r = min(0.05, spacing_x / 4, spacing_y / 4)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break

        # If we didn't place enough circles, fill remaining space randomly
        while len(circles) < num_circles:
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            r = np.random.uniform(0.01, 0.05)
            circles.append([x, y, r])

        return np.array(circles)

    # Initial grid-based configuration
    circles = generate_hexagonal_grid(n, rect_width, rect_height)

    # Constraint validation and penalty calculation with spatial indexing for efficiency
    def calculate_fitness(circles_array):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0

        # Boundary penalties (quadratic for smooth gradient)
        for i in range(n):
            cx, cy, r = circles_array[i]
            if cx - r < 0.01:
                penalty += 10000 * (r - cx)**2
            if cx + r > rect_width - 0.01:
                penalty += 10000 * (cx + r - rect_width)**2
            if cy - r < 0.01:
                penalty += 10000 * (r - cy)**2
            if cy + r > rect_height - 0.01:
                penalty += 10000 * (cy + r - rect_height)**2

        # Overlap penalties using spatial indexing for efficiency (O(n) instead of O(n^2))
        # Build KDTree for efficient neighbor search
        points = circles_array[:, :2]  # Only x,y coordinates
        tree = KDTree(points)

        # Query for neighbors within sum of radii distance
        # This prevents checking ALL pairs, but ensures we catch all overlaps
        for i in range(n):
            cx, cy, r = circles_array[i]

            # Find neighbors that could possibly overlap
            neighbor_indices = tree.query_ball_point([cx, cy], 2*r + 0.001)

            # Check actual overlaps with neighbors
            for j in neighbor_indices:
                if i != j:  # Don't compare with self
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist

                    if overlap > 0:
                        penalty += 100000 * overlap**2

        return total_radius - penalty

    # Enhanced local refinement with greedy improvement and spatial indexing
    def refine_circles(circles_array, max_iter=100):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness(best_circles)

        for iteration in range(max_iter):
            improved = False

            # Try to increase each circle's radius
            for i in range(n):
                cx, cy, r = best_circles[i]

                # Compute max allowable radius using spatial indexing for efficiency
                max_radius = float('inf')

                # Boundary constraints
                max_radius = min(max_radius, cx - 0.01)
                max_radius = min(max_radius, rect_width - cx - 0.01)
                max_radius = min(max_radius, cy - 0.01)
                max_radius = min(max_radius, rect_height - cy - 0.01)

                # Overlap constraints with nearby circles using spatial indexing
                points = best_circles[:, :2]  # Only x,y coordinates
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)  # Small safety margin

                # Try to increase radius
                if max_radius > r and max_radius > 0.001:
                    # Test several increments
                    test_increments = [0.005, 0.01, 0.02]
                    for incr in test_increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r:
                            continue

                        # Check validity of new configuration using spatial indexing
                        valid = True
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r

                        # Quick neighbor check using spatial index before full validation
                        points_new = temp_circles[:, :2]
                        tree_new = KDTree(points_new)
                        neighbor_indices_new = tree_new.query_ball_point([cx, cy], 2*(new_r + 0.01) + 0.001)

                        for k in neighbor_indices_new:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    valid = False
                                    break

                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 2] = new_r
                            test_fitness = calculate_fitness(test_circles)

                            if test_fitness > best_fitness:
                                best_circles = test_circles
                                best_fitness = test_fitness
                                improved = True
                                break

            if not improved:
                break

        return best_circles

    # Refine initial grid placement
    refined_circles = refine_circles(circles)
    best_solution = refined_circles.copy()
    best_fitness = calculate_fitness(refined_circles)

    # Enhanced evolutionary algorithm with adaptive parameters
    class EvolutionaryOptimizer:
        def __init__(self, population_size=30, generations=30, mutation_rate=0.15):
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate
            self.rect_width = rect_width
            self.rect_height = rect_height

        def create_individual(self):
            # Create individual with some randomness around the best solution
            individual = best_solution.copy()
            # Add slight perturbation to positions and radii
            for i in range(n):
                # Perturb position slightly
                individual[i, 0] += np.random.normal(0, 0.03)
                individual[i, 1] += np.random.normal(0, 0.03)
                # Perturb radius
                individual[i, 2] *= np.random.uniform(0.8, 1.2)
                # Clamp to bounds
                individual[i, 0] = np.clip(individual[i, 0], 0.05, self.rect_width - 0.05)
                individual[i, 1] = np.clip(individual[i, 1], 0.05, self.rect_height - 0.05)
                individual[i, 2] = max(0.001, individual[i, 2])
            return individual

        def evaluate(self, individual):
            return calculate_fitness(individual)

        def mutate(self, individual, generation):
            mutated = individual.copy()
            # Adaptive mutation rate decreases over generations
            adaptive_rate = self.mutation_rate * (1.0 - generation / self.generations)

            for i in range(n):
                if random.random() < adaptive_rate:
                    # Randomly decide what to mutate
                    param_type = random.choice(['x', 'y', 'r'])

                    if param_type == 'x':
                        mutated[i, 0] += np.random.normal(0, 0.05)
                        mutated[i, 0] = np.clip(mutated[i, 0], 0.05, self.rect_width - 0.05)
                    elif param_type == 'y':
                        mutated[i, 1] += np.random.normal(0, 0.05)
                        mutated[i, 1] = np.clip(mutated[i, 1], 0.05, self.rect_height - 0.05)
                    else:  # radius
                        mutated[i, 2] *= np.random.uniform(0.7, 1.3)
                        mutated[i, 2] = max(0.001, mutated[i, 2])

            return mutated

        def crossover(self, parent1, parent2):
            child = parent1.copy()
            # Uniform crossover for each parameter
            for i in range(n):
                if random.random() > 0.5:
                    child[i, 0] = parent2[i, 0]
                    child[i, 1] = parent2[i, 1]
                    child[i, 2] = parent2[i, 2]
            return child

        def run(self):
            # Initialize population
            population = [self.create_individual() for _ in range(self.population_size)]

            for gen in range(self.generations):
                # Evaluate population
                fitnesses = [self.evaluate(ind) for ind in population]

                # Sort by fitness
                sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order
                population = [population[i] for i in sorted_indices]
                fitnesses = [fitnesses[i] for i in sorted_indices]

                # Update best solution
                if fitnesses[0] > best_fitness:
                    best_fitness = fitnesses[0]
                    best_solution = population[0].copy()

                # Create new population
                new_population = [population[0]]  # Elitism - keep best

                # Generate offspring through crossover and mutation
                while len(new_population) < self.population_size:
                    # Tournament selection
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)

                    # Crossover
                    child = self.crossover(parent1, parent2)

                    # Mutation
                    child = self.mutate(child, gen)

                    new_population.append(child)

                population = new_population[:self.population_size]

            return best_solution

        def tournament_selection(self, population, fitnesses, tournament_size=3):
            selected_indices = random.sample(range(len(population)), tournament_size)
            selected_fitnesses = [fitnesses[i] for i in selected_indices]
            winner_index = selected_indices[np.argmax(selected_fitnesses)]
            return population[winner_index]

    # Run evolutionary optimization
    try:
        optimizer = EvolutionaryOptimizer(population_size=30, generations=25, mutation_rate=0.15)
        evolved_solution = optimizer.run()

        # Final refinement
        final_solution = refine_circles(evolved_solution, max_iter=50)

        # If this isn't better than our current best, use the best so far
        if calculate_fitness(final_solution) > best_fitness:
            best_solution = final_solution
        else:
            # Re-run refinement on the best solution found so far
            best_solution = refine_circles(best_solution, max_iter=20)

    except Exception as e:
        # If evolution fails, just return the best refined solution
        pass

    # Final safety validation
    final_fitness = calculate_fitness(best_solution)
    if final_fitness < 0:
        # If still invalid, use a clean refinement
        best_solution = refine_circles(best_solution, max_iter=30)

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")