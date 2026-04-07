# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
import math
from itertools import combinations
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    rect_width = 1.2
    rect_height = 0.8

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    def compute_circle_radius(point, points, rect_width, rect_height):
        """Compute maximum possible radius for a circle centered at 'point'"""
        center_x, center_y = point
        # Distance to rectangle edges
        dist_to_edges = [
            center_x,                    # distance to left edge
            rect_width - center_x,       # distance to right edge
            center_y,                    # distance to bottom edge
            rect_height - center_y       # distance to top edge
        ]

        # Distance to other circles (excluding self)
        min_dist_to_others = float('inf')
        for i, other_point in enumerate(points):
            if not (abs(other_point[0] - center_x) < 1e-10 and abs(other_point[1] - center_y) < 1e-10):
                dist = distance.euclidean(point, other_point)
                min_dist_to_others = min(min_dist_to_others, dist)

        # Maximum radius is limited by both edges and other circles
        max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
        return max(0.001, max_radius)

    def compute_gradient(point, points, rect_width, rect_height):
        """Compute approximate gradient of radius function at given point"""
        epsilon = 1e-6
        base_radius = compute_circle_radius(point, points, rect_width, rect_height)

        grad = np.zeros(2)
        for dim in range(2):
            perturbed = point.copy()
            perturbed[dim] += epsilon
            perturbed_radius = compute_circle_radius(perturbed, points, rect_width, rect_height)
            grad[dim] = (perturbed_radius - base_radius) / epsilon

        return grad

    def adaptive_local_search(points, rect_width, rect_height, max_iterations=500):
        """Perform adaptive local search with gradient information"""
        current_points = points.copy()

        # Progressive refinement with decreasing steps
        for iteration in range(max_iterations):
            step_size = max(0.005, 0.1 * (1.0 - iteration/max_iterations))
            updated = False

            # For each point, try to find better position
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

                # Sample neighborhood with adaptive step
                step_range = min(0.1, step_size * 5)
                steps = max(3, int(step_range / step_size))
                search_space = np.linspace(-step_range, step_range, steps)

                # Try gradient-based direction first
                try:
                    grad = compute_gradient(current_point, current_points, rect_width, rect_height)
                    grad_magnitude = np.linalg.norm(grad)
                    if grad_magnitude > 1e-8:
                        # Move in gradient direction (scaled by step size)
                        scaled_grad = grad / grad_magnitude * step_size * 0.5
                        new_x = current_point[0] + scaled_grad[0]
                        new_y = current_point[1] + scaled_grad[1]

                        # Keep within bounds
                        if (0.05 <= new_x <= rect_width - 0.05 and
                            0.05 <= new_y <= rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = compute_circle_radius(test_point, current_points, rect_width, rect_height)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point
                                updated = True
                except:
                    pass

                # If no gradient improvement, do structured search
                if not updated:
                    for dx in search_space:
                        for dy in search_space:
                            new_x = current_point[0] + dx
                            new_y = current_point[1] + dy

                            # Keep within bounds
                            if (0.05 <= new_x <= rect_width - 0.05 and
                                0.05 <= new_y <= rect_height - 0.05):

                                test_point = np.array([new_x, new_y])
                                test_radius = compute_circle_radius(test_point, current_points, rect_width, rect_height)

                                if test_radius > best_radius:
                                    best_radius = test_radius
                                    best_point = test_point
                                    updated = True

                current_points[i] = best_point

            # Early stopping if no significant improvement
            if not updated and iteration > 100:
                break

        return current_points

    def evaluate_configuration(points, rect_width, rect_height):
        """Evaluate a configuration by computing sum of radii"""
        total_radius = 0
        circles = []
        for point in points:
            radius = compute_circle_radius(point, points, rect_width, rect_height)
            circles.append([point[0], point[1], radius])
            total_radius += radius
        return total_radius, np.array(circles)

    def validate_and_correct_configuration(points, rect_width, rect_height):
        """Ensure configuration is valid and correct any boundary issues"""
        corrected_points = []
        for point in points:
            x, y = point
            # Keep within bounds
            x = max(0.05, min(rect_width - 0.05, x))
            y = max(0.05, min(rect_height - 0.05, y))
            corrected_points.append([x, y])
        return np.array(corrected_points)

    def generate_initial_population(pop_size, rect_width, rect_height):
        """Generate diverse initial population using Voronoi-based seeding"""
        population = []
        for _ in range(pop_size):
            # Generate points using different seeding strategies
            points = []

            # Strategy 1: Corner placements
            corner_positions = [
                (rect_width * 0.1, rect_height * 0.1),
                (rect_width * 0.9, rect_height * 0.1),
                (rect_width * 0.1, rect_height * 0.9),
                (rect_width * 0.9, rect_height * 0.9),
                (rect_width / 2, rect_height / 2)
            ]

            # Add corners with slight perturbations
            for x, y in corner_positions:
                pert_x = np.random.normal(0, 0.05)
                pert_y = np.random.normal(0, 0.05)
                points.append([x + pert_x, y + pert_y])

            # Strategy 2: Grid placement
            grid_size = 3
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < 21:
                        x = rect_width * (0.2 + i * 0.3)
                        y = rect_height * (0.2 + j * 0.3)
                        pert_x = np.random.normal(0, 0.03)
                        pert_y = np.random.normal(0, 0.03)
                        points.append([x + pert_x, y + pert_y])

            # Strategy 3: Random points to fill
            while len(points) < 21:
                x = np.random.uniform(0.05, rect_width - 0.05)
                y = np.random.uniform(0.05, rect_height - 0.05)
                points.append([x, y])

            points = points[:21]
            population.append(np.array(points))

        return population

    def crossover(parent1, parent2):
        """Create offspring via uniform crossover"""
        offspring = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        return np.array(offspring)

    def mutate(individual, mutation_rate=0.1):
        """Apply random mutations to individual"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = mutated[i] + np.random.normal(0, 0.05, 2)
                # Keep within bounds
                mutated[i][0] = np.clip(mutated[i][0], 0.05, rect_width - 0.05)
                mutated[i][1] = np.clip(mutated[i][1], 0.05, rect_height - 0.05)
        return mutated

    def smart_crossover(parent1, parent2, rect_width, rect_height):
        """Smart crossover that tries to preserve good structural properties"""
        offspring = []
        # Create a Voronoi diagram to understand spatial relationships
        try:
            # For simplicity, use a combination approach
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
        except:
            # Fallback to standard crossover
            offspring = []
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    offspring.append(parent1[i])
                else:
                    offspring.append(parent2[i])
        return np.array(offspring)

    def tournament_selection(population, fitnesses, tournament_size=3):
        """Select best individual from tournament"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

    # Evolutionary algorithm parameters
    pop_size = 30
    num_generations = 100
    elite_size = 5

    # Generate initial population
    population = generate_initial_population(pop_size, rect_width, rect_height)

    best_solution = None
    best_fitness = 0

    for generation in range(num_generations):
        # Adaptive mutation rate that decreases over time
        adaptive_mutation_rate = max(0.05, 0.1 * (1.0 - generation / num_generations))

        # Evaluate fitness of entire population
        fitness_scores = []
        population_circles = []

        for individual in population:
            # Ensure individual is valid
            valid_individual = validate_and_correct_configuration(individual, rect_width, rect_height)
            fitness, circles = evaluate_configuration(valid_individual, rect_width, rect_height)
            fitness_scores.append(fitness)
            population_circles.append(circles)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = circles.copy()

        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_circles = [population_circles[i] for i in sorted_indices]

        # Keep elite individuals
        new_population = sorted_population[:elite_size]

        # Generate offspring through crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(sorted_population, fitness_scores)
            parent2 = tournament_selection(sorted_population, fitness_scores)

            # Smart crossover
            offspring = smart_crossover(parent1, parent2, rect_width, rect_height)

            # Mutation with adaptive rate
            offspring = mutate(offspring, adaptive_mutation_rate)

            # Local refinement for offspring (helps maintain good solutions)
            valid_offspring = validate_and_correct_configuration(offspring, rect_width, rect_height)
            _, refined_circles = evaluate_configuration(valid_offspring, rect_width, rect_height)

            # Only keep refined version if it's significantly better
            refined_fitness = np.sum(refined_circles[:, 2])
            if refined_fitness > np.sum(offspring[:, 2]) * 1.01:  # 1% improvement threshold
                new_population.append(valid_offspring)
            else:
                new_population.append(offspring)

        population = new_population

    # Final refinement using local search with Voronoi analysis
    if best_solution is not None:
        final_points = []
        for i in range(21):
            final_points.append([best_solution[i, 0], best_solution[i, 1]])
        final_points = np.array(final_points)

        # Perform intensive local search using gradient-based approach
        refined_points = adaptive_local_search(final_points, rect_width, rect_height, max_iterations=800)

        # Recalculate final configuration
        final_fitness, final_circles = evaluate_configuration(refined_points, rect_width, rect_height)

        # Additional local optimization on the final solution
        # Try a more focused local search on the best solution
        for _ in range(50):
            improved = False
            for i in range(len(refined_points)):
                # Try to improve each circle's radius and position
                original_point = refined_points[i]
                original_radius = compute_circle_radius(original_point, refined_points, rect_width, rect_height)

                # Try small perturbations to see if we can improve
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        new_x = original_point[0] + dx
                        new_y = original_point[1] + dy
                        if (0.05 <= new_x <= rect_width - 0.05 and
                            0.05 <= new_y <= rect_height - 0.05):
                            test_point = np.array([new_x, new_y])
                            test_radius = compute_circle_radius(test_point, refined_points, rect_width, rect_height)
                            if test_radius > original_radius:
                                refined_points[i] = test_point
                                improved = True
            if not improved:
                break

        # Final recalculation
        final_fitness, final_circles = evaluate_configuration(refined_points, rect_width, rect_height)

        return final_circles

    # Fallback to simple grid pattern
    circles = np.zeros((21, 3))
    row_size = int(np.ceil(np.sqrt(21)))
    col_size = int(np.ceil(21 / row_size))

    spacing_x = rect_width / (col_size + 1)
    spacing_y = rect_height / (row_size + 1)

    count = 0
    for i in range(row_size):
        for j in range(col_size):
            if count < 21:
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                # Set radius to be proportional to available space
                radius = min(x, rect_width - x, y, rect_height - y) * 0.4
                circles[count] = [x, y, max(radius, 0.001)]
                count += 1

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")