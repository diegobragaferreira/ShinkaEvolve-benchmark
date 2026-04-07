# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
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

    # Rectangle dimensions: width + height = 2, optimized ratio
    rect_width = 1.2
    rect_height = 0.8

    n = 21

    # Generate adaptive grid-based initial placement
    # Uses better spacing calculations and more uniform distribution
    def generate_adaptive_grid(num_circles, width, height):
        # Calculate optimal grid dimensions based on container area and circle count
        # Use a more sophisticated approach to achieve better packing density

        # First estimate the area required per circle
        container_area = width * height
        # Leave 10% margin for better packing efficiency
        available_area = container_area * 0.9
        area_per_circle = available_area / num_circles

        # Estimate the average radius
        avg_radius = np.sqrt(area_per_circle / np.pi)

        # Calculate grid dimensions
        cols = max(1, int(np.sqrt(num_circles * (width / height))))
        rows = max(1, int(np.ceil(num_circles / cols)))

        # Ensure grid fits within container with margins
        max_cols = max(1, int(width / (2 * avg_radius)))
        max_rows = max(1, int(height / (2 * avg_radius)))

        cols = min(cols, max_cols)
        rows = min(rows, max_rows)

        # Adjust to ensure sufficient circles
        if cols * rows < num_circles:
            cols = max(1, int(np.ceil(np.sqrt(num_circles * (width / height)))))
            rows = max(1, int(np.ceil(num_circles / cols)))
            cols = min(cols, max_cols)
            rows = min(rows, max_rows)

        # Calculate actual spacing
        spacing_x = width / cols if cols > 0 else width
        spacing_y = height / rows if rows > 0 else height

        # Apply slight hexagonal offset for better packing
        hex_offset_x = spacing_x * 0.5
        hex_offset_y = spacing_y * np.sqrt(3) / 2

        circles = []
        margin_x = 0.1
        margin_y = 0.1

        circle_count = 0
        for i in range(rows):
            y = margin_y + i * spacing_y
            x_start = margin_x + (i % 2) * hex_offset_x  # Hexagonal offset
            for j in range(cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x
                if x < width - margin_x and y < height - margin_y:
                    # Use radius based on spacing but cap it
                    r = min(avg_radius * 0.8, spacing_x / 4, spacing_y / 4)
                    r = max(0.005, min(r, 0.1))  # Clamp to reasonable range
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break

        # Fill remaining spots with random placements that are more strategically positioned
        while len(circles) < num_circles:
            # Place in central region to maximize room for expansion
            x = np.random.uniform(margin_x + avg_radius, width - margin_x - avg_radius)
            y = np.random.uniform(margin_y + avg_radius, height - margin_y - avg_radius)
            # Smaller radius for random placements
            r = np.random.uniform(0.01, avg_radius * 0.7)
            r = max(0.005, min(r, 0.1))
            circles.append([x, y, r])

        return np.array(circles)

    # Initial grid-based configuration
    circles = generate_adaptive_grid(n, rect_width, rect_height)

    # Efficient constraint validation with spatial indexing
    def calculate_fitness_with_spatial_indexing(circles_array):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0

        # Boundary penalties (quadratic for smooth gradient)
        for i in range(n):
            cx, cy, r = circles_array[i]
            if cx - r < 0.01:
                penalty += 50000 * (r - cx)**2
            if cx + r > rect_width - 0.01:
                penalty += 50000 * (cx + r - rect_width)**2
            if cy - r < 0.01:
                penalty += 50000 * (r - cy)**2
            if cy + r > rect_height - 0.01:
                penalty += 50000 * (cy + r - rect_height)**2

        # Overlap penalties using spatial indexing for efficiency
        points = circles_array[:, :2]
        tree = KDTree(points)

        # Query for all possible overlapping pairs efficiently
        for i in range(n):
            cx, cy, r = circles_array[i]

            # Find neighbors within 2*(r + safety_margin) distance
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist

                    if overlap > 0:
                        penalty += 200000 * overlap**2

        return total_radius - penalty

    # Enhanced local refinement with smarter approach and better spatial indexing
    def refine_circles_smart(circles_array, max_iter=100):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness_with_spatial_indexing(best_circles)

        improvement_history = []

        for iteration in range(max_iter):
            improved = False

            # Process circles in random order for unbiased optimization
            indices = list(range(n))
            random.shuffle(indices)

            # Try to increase each circle's radius
            for i in indices:
                cx, cy, r = best_circles[i]

                # Compute max allowable radius using spatial indexing for overlap constraints
                max_radius = float('inf')

                # Boundary constraints
                max_radius = min(max_radius, cx - 0.01)
                max_radius = min(max_radius, rect_width - cx - 0.01)
                max_radius = min(max_radius, cy - 0.01)
                max_radius = min(max_radius, rect_height - cy - 0.01)

                # Overlap constraints using spatial indexing
                points = best_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)  # Small safety margin

                # Try to increase radius aggressively
                if max_radius > r and max_radius > 0.001:
                    # Use a more aggressive step size
                    # Try multiple increments to see which works best
                    increments = [0.002, 0.005, 0.01, 0.015, 0.02]
                    best_new_r = r
                    best_test_fitness = best_fitness

                    # Test increasing radius systematically
                    for incr in increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r + 0.0001:  # Skip tiny increases
                            continue

                        # Quick validation with spatial indexing
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r

                        # Check for overlap using spatial indexing
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([cx, cy], 2*(new_r + 0.01) + 0.001)

                        for k in temp_neighbor_indices:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    valid = False
                                    break

                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 2] = new_r
                            test_fitness = calculate_fitness_with_spatial_indexing(test_circles)

                            if test_fitness > best_test_fitness:
                                best_test_fitness = test_fitness
                                best_new_r = new_r
                                improved = True

                    if improved:
                        best_circles = test_circles
                        best_fitness = best_test_fitness

            # Track improvement
            improvement_history.append(best_fitness)
            if len(improvement_history) > 10:
                improvement_history.pop(0)

            # Early stopping if no significant improvement
            if len(improvement_history) >= 10:
                recent_improvements = [
                    improvement_history[i] - improvement_history[i-1]
                    for i in range(1, len(improvement_history))
                ]
                if all(improvement < 1e-6 for improvement in recent_improvements):
                    break

        return best_circles

    # Refine initial grid placement
    refined_circles = refine_circles_smart(circles)
    best_solution = refined_circles.copy()
    best_fitness = calculate_fitness_with_spatial_indexing(refined_circles)

    # Enhanced evolutionary algorithm with improved parameters and structure
    class EvolutionaryOptimizer:
        def __init__(self, population_size=25, generations=20, mutation_rate=0.2):
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate
            self.rect_width = rect_width
            self.rect_height = rect_height

        def create_individual(self):
            # Create individual with more intelligent perturbations around the best solution
            individual = best_solution.copy()
            # Add more substantial perturbations for better exploration
            for i in range(n):
                # Perturb position more significantly
                individual[i, 0] += np.random.normal(0, 0.05)
                individual[i, 1] += np.random.normal(0, 0.05)
                # Perturb radius with wider range
                individual[i, 2] *= np.random.uniform(0.7, 1.3)
                # Clamp to bounds
                individual[i, 0] = np.clip(individual[i, 0], 0.05, self.rect_width - 0.05)
                individual[i, 1] = np.clip(individual[i, 1], 0.05, self.rect_height - 0.05)
                individual[i, 2] = max(0.001, individual[i, 2])
            return individual

        def evaluate(self, individual):
            return calculate_fitness_with_spatial_indexing(individual)

        def mutate(self, individual, generation):
            mutated = individual.copy()
            # Adaptive mutation rate decreases over generations
            adaptive_rate = self.mutation_rate * (1.0 - generation / self.generations)

            for i in range(n):
                if random.random() < adaptive_rate:
                    # Randomly decide what to mutate with higher probability for positions
                    param_type = random.choices(['x', 'y', 'r'], weights=[0.4, 0.4, 0.2])[0]

                    if param_type == 'x':
                        mutated[i, 0] += np.random.normal(0, 0.08)
                        mutated[i, 0] = np.clip(mutated[i, 0], 0.05, self.rect_width - 0.05)
                    elif param_type == 'y':
                        mutated[i, 1] += np.random.normal(0, 0.08)
                        mutated[i, 1] = np.clip(mutated[i, 1], 0.05, self.rect_height - 0.05)
                    else:  # radius
                        mutated[i, 2] *= np.random.uniform(0.8, 1.2)
                        mutated[i, 2] = max(0.001, mutated[i, 2])

            return mutated

        def crossover(self, parent1, parent2):
            child = parent1.copy()
            # Uniform crossover but with preference for keeping good features
            for i in range(n):
                if random.random() > 0.6:  # Slight bias towards parent1
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
                    # Tournament selection with larger tournament size
                    parent1 = self.tournament_selection(population, fitnesses, tournament_size=4)
                    parent2 = self.tournament_selection(population, fitnesses, tournament_size=4)

                    # Crossover
                    child = self.crossover(parent1, parent2)

                    # Mutation
                    child = self.mutate(child, gen)

                    new_population.append(child)

                population = new_population[:self.population_size]

            return best_solution

        def tournament_selection(self, population, fitnesses, tournament_size=4):
            selected_indices = random.sample(range(len(population)), tournament_size)
            selected_fitnesses = [fitnesses[i] for i in selected_indices]
            winner_index = selected_indices[np.argmax(selected_fitnesses)]
            return population[winner_index]

    # Run evolutionary optimization with tuned parameters
    try:
        optimizer = EvolutionaryOptimizer(population_size=25, generations=20, mutation_rate=0.2)
        evolved_solution = optimizer.run()

        # Final refinement with spatial indexing
        final_solution = refine_circles_smart(evolved_solution, max_iter=40)

        # If this isn't better than our current best, use the best so far
        if calculate_fitness_with_spatial_indexing(final_solution) > best_fitness:
            best_solution = final_solution
        else:
            # Re-run refinement on the best solution found so far with better spatial indexing
            best_solution = refine_circles_smart(best_solution, max_iter=20)

    except Exception as e:
        # If evolution fails, just return the best refined solution
        pass

    # Final safety validation with spatial indexing
    final_fitness = calculate_fitness_with_spatial_indexing(best_solution)
    if final_fitness < 0:
        # If still invalid, use a clean refinement with spatial indexing
        best_solution = refine_circles_smart(best_solution, max_iter=30)

    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")