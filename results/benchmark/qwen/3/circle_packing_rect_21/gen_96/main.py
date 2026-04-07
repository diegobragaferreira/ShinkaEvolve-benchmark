# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
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

    def multi_scale_optimization(initial_points, rect_width, rect_height):
        """Multi-scale optimization with progressive refinement"""
        current_points = initial_points.copy()

        # Scale 1: Coarse grid search (larger steps)
        for _ in range(50):
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

                # Large step size for coarse search
                for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
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

                current_points[i] = best_point

        # Scale 2: Medium local search (medium steps)
        for _ in range(100):
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

                # Medium step size
                for dx in [-0.05, -0.025, 0, 0.025, 0.05]:
                    for dy in [-0.05, -0.025, 0, 0.025, 0.05]:
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

                current_points[i] = best_point

        # Scale 3: Fine local search (small steps)
        for _ in range(200):
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

                # Fine step size
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
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

                current_points[i] = best_point

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

    def generate_initial_population(pop_size, rect_width, rect_height):
        """Generate diverse initial population using Voronoi-based seeding"""
        population = []
        for _ in range(pop_size):
            # Generate points using improved seeding strategies
            points = []

            # Strategy 1: Corner placements with better distribution
            corner_positions = [
                (rect_width * 0.1, rect_height * 0.1),
                (rect_width * 0.9, rect_height * 0.1),
                (rect_width * 0.1, rect_height * 0.9),
                (rect_width * 0.9, rect_height * 0.9),
                (rect_width / 2, rect_height / 2)
            ]

            # Add corners with more systematic perturbations
            for x, y in corner_positions:
                # Smaller perturbations for corners to avoid extreme positions
                pert_x = np.random.normal(0, 0.03)
                pert_y = np.random.normal(0, 0.03)
                points.append([x + pert_x, y + pert_y])

            # Strategy 2: More evenly distributed edge positions
            edge_positions = [
                (rect_width/2, rect_height * 0.1),  # top center
                (rect_width/2, rect_height * 0.9),  # bottom center
                (rect_width * 0.1, rect_height/2),  # left center
                (rect_width * 0.9, rect_height/2),  # right center
                (rect_width * 0.3, rect_height * 0.3),  # diagonal
                (rect_width * 0.7, rect_height * 0.7),  # diagonal
                (rect_width * 0.3, rect_height * 0.7),  # diagonal
                (rect_width * 0.7, rect_height * 0.3),  # diagonal
            ]

            for x, y in edge_positions:
                pert_x = np.random.normal(0, 0.02)
                pert_y = np.random.normal(0, 0.02)
                points.append([x + pert_x, y + pert_y])

            # Strategy 3: Grid placement with proper spacing
            grid_size = 4  # Increase grid size for better distribution
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < 21:
                        x = rect_width * (0.15 + i * 0.25)
                        y = rect_height * (0.15 + j * 0.25)
                        pert_x = np.random.normal(0, 0.02)
                        pert_y = np.random.normal(0, 0.02)
                        points.append([x + pert_x, y + pert_y])

            # Strategy 4: Random points to fill remaining slots
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
        # Evaluate fitness of entire population
        fitness_scores = []
        population_circles = []

        for individual in population:
            fitness, circles = evaluate_configuration(individual, rect_width, rect_height)
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

            # Crossover
            offspring = crossover(parent1, parent2)

            # Mutation
            offspring = mutate(offspring)

            new_population.append(offspring)

        population = new_population

    # Apply multi-scale optimization instead of single local search
    if best_solution is not None:
        final_points = []
        for i in range(21):
            final_points.append([best_solution[i, 0], best_solution[i, 1]])
        final_points = np.array(final_points)

        # Use multi-scale optimization for better results
        refined_points = multi_scale_optimization(final_points, rect_width, rect_height)

        # Recalculate final configuration
        final_fitness, final_circles = evaluate_configuration(refined_points, rect_width, rect_height)

        return final_circles

    # Enhanced refinement using gradient-based local search
    if best_solution is not None:
        final_points = []
        for i in range(21):
            final_points.append([best_solution[i, 0], best_solution[i, 1]])
        final_points = np.array(final_points)

        # Use gradient-based local search for final optimization
        refined_points = gradient_based_local_search(final_points, rect_width, rect_height)

        # Recalculate final configuration
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

def gradient_based_local_search(points, rect_width, rect_height, max_iterations=500):
    """Enhanced local search using numerical gradient computation"""
    current_points = points.copy()

    # Define epsilon for numerical gradient computation
    eps = 1e-5

    for iteration in range(max_iterations):
        updated = False
        new_points = current_points.copy()

        for i in range(len(current_points)):
            current_point = current_points[i]
            current_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

            # Compute numerical gradient
            grad_x = 0.0
            grad_y = 0.0

            # Compute partial derivatives w.r.t. x and y
            test_point_x_pos = current_point.copy()
            test_point_x_neg = current_point.copy()
            test_point_y_pos = current_point.copy()
            test_point_y_neg = current_point.copy()

            test_point_x_pos[0] += eps
            test_point_x_neg[0] -= eps
            test_point_y_pos[1] += eps
            test_point_y_neg[1] -= eps

            # Ensure points are within bounds
            test_point_x_pos[0] = np.clip(test_point_x_pos[0], 0.05, rect_width - 0.05)
            test_point_x_neg[0] = np.clip(test_point_x_neg[0], 0.05, rect_width - 0.05)
            test_point_y_pos[1] = np.clip(test_point_y_pos[1], 0.05, rect_height - 0.05)
            test_point_y_neg[1] = np.clip(test_point_y_neg[1], 0.05, rect_height - 0.05)

            # Calculate finite differences
            radius_x_pos = compute_circle_radius(test_point_x_pos, current_points, rect_width, rect_height)
            radius_x_neg = compute_circle_radius(test_point_x_neg, current_points, rect_width, rect_height)
            radius_y_pos = compute_circle_radius(test_point_y_pos, current_points, rect_width, rect_height)
            radius_y_neg = compute_circle_radius(test_point_y_neg, current_points, rect_width, rect_height)

            grad_x = (radius_x_pos - radius_x_neg) / (2 * eps)
            grad_y = (radius_y_pos - radius_y_neg) / (2 * eps)

            # Update using gradient ascent (since we want to maximize radius)
            step_size = 0.01
            new_x = current_point[0] + step_size * grad_x
            new_y = current_point[1] + step_size * grad_y

            # Keep within bounds
            new_x = np.clip(new_x, 0.05, rect_width - 0.05)
            new_y = np.clip(new_y, 0.05, rect_height - 0.05)

            # Test the new position
            test_point = np.array([new_x, new_y])
            test_radius = compute_circle_radius(test_point, current_points, rect_width, rect_height)

            # Accept the move if it improves the radius
            if test_radius > current_radius:
                new_points[i] = test_point
                updated = True

        current_points = new_points

        # Early stopping if no significant improvement
        if not updated and iteration > 100:
            break

    return current_points

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")