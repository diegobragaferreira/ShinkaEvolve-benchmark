# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple, List, Optional
import time

class CirclePackingOptimizer:
    def __init__(self, rect_width: float = 1.2, rect_height: float = 0.8, seed: int = 42):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def compute_circle_radius_with_tolerance(self, point: np.ndarray, points: np.ndarray,
                                           overlap_tolerance: float = 1.0) -> float:
        """Compute maximum possible radius for a circle centered at 'point' with adaptive overlap tolerance"""
        center_x, center_y = point
        # Distance to rectangle edges
        dist_to_edges = [
            center_x,                    # distance to left edge
            self.rect_width - center_x,  # distance to right edge
            center_y,                    # distance to bottom edge
            self.rect_height - center_y  # distance to top edge
        ]

        # Distance to other circles (excluding self)
        min_dist_to_others = float('inf')
        if len(points) > 1:
            # Use cKDTree for efficient nearest neighbor search
            tree = cKDTree(points)
            distances, _ = tree.query(point, k=2)  # Get 2 nearest (including self)
            if len(distances) >= 2:
                min_dist_to_others = distances[1]  # Exclude self-distance

        # Apply overlap tolerance factor to make constraints less strict initially
        adjusted_min_dist = min_dist_to_others / (2.0 * overlap_tolerance)

        # Maximum radius is limited by both edges and other circles
        max_radius = min(min(dist_to_edges), adjusted_min_dist)
        return max(0.001, max_radius)

    def compute_circle_radius(self, point: np.ndarray, points: np.ndarray) -> float:
        """Compute maximum possible radius for a circle centered at 'point'"""
        center_x, center_y = point
        # Distance to rectangle edges
        dist_to_edges = [
            center_x,                    # distance to left edge
            self.rect_width - center_x,  # distance to right edge
            center_y,                    # distance to bottom edge
            self.rect_height - center_y  # distance to top edge
        ]

        # Distance to other circles (excluding self)
        min_dist_to_others = float('inf')
        if len(points) > 1:
            # Use cKDTree for efficient nearest neighbor search
            tree = cKDTree(points)
            distances, _ = tree.query(point, k=2)  # Get 2 nearest (including self)
            if len(distances) >= 2:
                min_dist_to_others = distances[1]  # Exclude self-distance

        # Maximum radius is limited by both edges and other circles
        max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
        return max(0.001, max_radius)

    def evaluate_configuration(self, points: np.ndarray, overlap_tolerance: float = 1.0) -> Tuple[float, np.ndarray]:
        """Evaluate a configuration by computing sum of radii"""
        total_radius = 0.0
        circles = []
        for point in points:
            radius = self.compute_circle_radius_with_tolerance(point, points, overlap_tolerance)
            circles.append([point[0], point[1], radius])
            total_radius += radius
        return total_radius, np.array(circles)

    def generate_initial_config(self) -> np.ndarray:
        """Generate highly optimized initial configuration using Voronoi-distributed points"""
        points = []

        # Strategy 1: Corner placements with strategic perturbation
        corner_positions = [
            (self.rect_width * 0.1, self.rect_height * 0.1),
            (self.rect_width * 0.9, self.rect_height * 0.1),
            (self.rect_width * 0.1, self.rect_height * 0.9),
            (self.rect_width * 0.9, self.rect_height * 0.9),
            (self.rect_width / 2, self.rect_height / 2)
        ]

        for x, y in corner_positions:
            pert_x = np.random.normal(0, 0.03)
            pert_y = np.random.normal(0, 0.03)
            points.append([x + pert_x, y + pert_y])

        # Strategy 2: Edge placements
        edge_positions = [
            (self.rect_width/2, self.rect_height * 0.1),  # top center
            (self.rect_width/2, self.rect_height * 0.9),  # bottom center
            (self.rect_width * 0.1, self.rect_height/2),  # left center
            (self.rect_width * 0.9, self.rect_height/2),  # right center
        ]

        for x, y in edge_positions:
            pert_x = np.random.normal(0, 0.02)
            pert_y = np.random.normal(0, 0.02)
            points.append([x + pert_x, y + pert_y])

        # Strategy 3: Grid placements in interior (more evenly spaced)
        grid_x = np.linspace(self.rect_width * 0.15, self.rect_width * 0.85, 4)
        grid_y = np.linspace(self.rect_height * 0.15, self.rect_height * 0.85, 4)

        for x in grid_x:
            for y in grid_y:
                if len(points) < 21:
                    pert_x = np.random.normal(0, 0.02)
                    pert_y = np.random.normal(0, 0.02)
                    points.append([x + pert_x, y + pert_y])

        # Strategy 4: Fill remaining with more structured distribution
        while len(points) < 21:
            # Use a better distribution that avoids clustering
            x = np.random.triangular(0.05, self.rect_width/2, self.rect_width - 0.05)
            y = np.random.triangular(0.05, self.rect_height/2, self.rect_height - 0.05)
            points.append([x, y])

        return np.array(points[:21])

    def evolve_population(self, population: List[np.ndarray],
                         fitness_scores: List[float]) -> List[np.ndarray]:
        """Evolve population through selection, crossover, and mutation"""
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]

        # Keep elite individuals (more aggressive than before)
        elite_size = max(4, len(population) // 4)
        new_population = sorted_population[:elite_size]

        # Generate offspring through crossover and mutation
        while len(new_population) < len(population):
            # Tournament selection
            parent1 = self.tournament_selection(sorted_population, fitness_scores)
            parent2 = self.tournament_selection(sorted_population, fitness_scores)

            # Crossover
            offspring = self.crossover(parent1, parent2)

            # Mutation with higher probability for exploration
            offspring = self.mutate(offspring, mutation_rate=0.15)

            new_population.append(offspring)

        return new_population

    def tournament_selection(self, population: List[np.ndarray],
                           fitnesses: List[float], tournament_size: int = 3) -> np.ndarray:
        """Select best individual from tournament"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring via uniform crossover"""
        offspring = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])
        return np.array(offspring)

    def mutate(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Apply random mutations to individual"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Larger perturbations early, smaller ones later
                mutation_magnitude = 0.05
                mutated[i] = mutated[i] + np.random.normal(0, mutation_magnitude, 2)
                # Keep within bounds
                mutated[i][0] = np.clip(mutated[i][0], 0.05, self.rect_width - 0.05)
                mutated[i][1] = np.clip(mutated[i][1], 0.05, self.rect_height - 0.05)
        return mutated

    def gradient_based_local_search(self, points: np.ndarray, max_iterations: int = 500,
                                  initial_tolerance: float = 2.0, final_tolerance: float = 1.0) -> np.ndarray:
        """Advanced local search using numerical gradient computation with adaptive tolerance"""
        current_points = points.copy()
        eps = 1e-5  # Small epsilon for gradient calculation

        for iteration in range(max_iterations):
            # Adaptive tolerance: start relaxed, then tighten constraints
            current_tolerance = initial_tolerance + (final_tolerance - initial_tolerance) * (iteration / max_iterations)

            updated = False
            new_points = current_points.copy()

            # For each point, compute gradient and move in direction of steepest ascent
            for i in range(len(current_points)):
                current_point = current_points[i]
                current_radius = self.compute_circle_radius_with_tolerance(current_point, current_points, current_tolerance)

                # Compute numerical gradient using finite differences
                grad_x = 0.0
                grad_y = 0.0

                # Calculate partial derivative w.r.t. x
                test_point_x_pos = current_point.copy()
                test_point_x_neg = current_point.copy()
                test_point_x_pos[0] += eps
                test_point_x_neg[0] -= eps

                # Ensure points are within bounds
                test_point_x_pos[0] = np.clip(test_point_x_pos[0], 0.05, self.rect_width - 0.05)
                test_point_x_neg[0] = np.clip(test_point_x_neg[0], 0.05, self.rect_width - 0.05)

                radius_x_pos = self.compute_circle_radius_with_tolerance(test_point_x_pos, current_points, current_tolerance)
                radius_x_neg = self.compute_circle_radius_with_tolerance(test_point_x_neg, current_points, current_tolerance)
                grad_x = (radius_x_pos - radius_x_neg) / (2 * eps)

                # Calculate partial derivative w.r.t. y
                test_point_y_pos = current_point.copy()
                test_point_y_neg = current_point.copy()
                test_point_y_pos[1] += eps
                test_point_y_neg[1] -= eps

                # Ensure points are within bounds
                test_point_y_pos[1] = np.clip(test_point_y_pos[1], 0.05, self.rect_height - 0.05)
                test_point_y_neg[1] = np.clip(test_point_y_neg[1], 0.05, self.rect_height - 0.05)

                radius_y_pos = self.compute_circle_radius_with_tolerance(test_point_y_pos, current_points, current_tolerance)
                radius_y_neg = self.compute_circle_radius_with_tolerance(test_point_y_neg, current_points, current_tolerance)
                grad_y = (radius_y_pos - radius_y_neg) / (2 * eps)

                # Update using gradient ascent (since we want to maximize radius)
                step_size = 0.02

                # Adaptive step size based on gradient magnitude
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                if grad_magnitude > 1e-8:
                    step_size = min(0.05, 0.05 / grad_magnitude)

                new_x = current_point[0] + step_size * grad_x
                new_y = current_point[1] + step_size * grad_y

                # Keep within bounds
                new_x = np.clip(new_x, 0.05, self.rect_width - 0.05)
                new_y = np.clip(new_y, 0.05, self.rect_height - 0.05)

                # Test the new position
                test_point = np.array([new_x, new_y])
                test_radius = self.compute_circle_radius_with_tolerance(test_point, current_points, current_tolerance)

                # Accept the move if it improves the radius
                if test_radius > current_radius:
                    new_points[i] = test_point
                    updated = True

            current_points = new_points

            # Early stopping if no significant improvement
            if not updated and iteration > 100:
                break

        return current_points

    def multi_scale_optimization(self, points: np.ndarray,
                               max_iterations: int = 300,
                               initial_tolerance: float = 2.0,
                               final_tolerance: float = 1.0) -> np.ndarray:
        """Multi-scale optimization with aggressive initial search and adaptive tolerance"""
        current_points = points.copy()

        # Scale 1: Very coarse search (largest steps) for global exploration
        for iteration in range(30):
            # Adaptive tolerance for this scale
            current_tolerance = initial_tolerance + (final_tolerance - initial_tolerance) * (iteration / 30)
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius_with_tolerance(current_point, current_points, current_tolerance)

                # Very coarse step size
                for dx in [-0.15, -0.1, 0, 0.1, 0.15]:
                    for dy in [-0.15, -0.1, 0, 0.1, 0.15]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius_with_tolerance(test_point, current_points, current_tolerance)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        # Scale 2: Coarse search for medium exploration
        for iteration in range(50):
            # Adaptive tolerance for this scale
            current_tolerance = initial_tolerance + (final_tolerance - initial_tolerance) * (iteration / 50)
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius_with_tolerance(current_point, current_points, current_tolerance)

                # Coarse step size
                for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius_with_tolerance(test_point, current_points, current_tolerance)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        # Scale 3: Medium search with moderate steps
        for iteration in range(80):
            # Adaptive tolerance for this scale
            current_tolerance = initial_tolerance + (final_tolerance - initial_tolerance) * (iteration / 80)
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius_with_tolerance(current_point, current_points, current_tolerance)

                # Medium step size
                for dx in [-0.05, -0.025, 0, 0.025, 0.05]:
                    for dy in [-0.05, -0.025, 0, 0.025, 0.05]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius_with_tolerance(test_point, current_points, current_tolerance)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        # Scale 4: Fine search for final refinement
        for iteration in range(100):
            # Adaptive tolerance for this scale
            current_tolerance = initial_tolerance + (final_tolerance - initial_tolerance) * (iteration / 100)
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius_with_tolerance(current_point, current_points, current_tolerance)

                # Fine step size
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius_with_tolerance(test_point, current_points, current_tolerance)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        return current_points

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = CirclePackingOptimizer()

    # Generate initial configuration
    points = optimizer.generate_initial_config()

    # Stage 1: Multi-scale global optimization for broad exploration
    points = optimizer.multi_scale_optimization(points, max_iterations=200,
                                              initial_tolerance=2.0, final_tolerance=1.0)

    # Stage 2: Gradient-based local search for fast convergence
    points = optimizer.gradient_based_local_search(points, max_iterations=300,
                                                 initial_tolerance=2.0, final_tolerance=1.0)

    # Stage 3: Evolutionary refinement with aggressive population management
    pop_size = 25
    num_generations = 60
    elite_size = 6

    # Initial population with more diversity
    population = [optimizer.generate_initial_config() for _ in range(pop_size)]

    best_fitness = 0
    best_solution = None

    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = []
        population_circles = []

        for individual in population:
            fitness, circles = optimizer.evaluate_configuration(individual)
            fitness_scores.append(fitness)
            population_circles.append(circles)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = circles.copy()

        # Evolve population
        population = optimizer.evolve_population(population, fitness_scores)

    # Stage 4: Final comprehensive refinement
    if best_solution is not None:
        final_points = []
        for i in range(21):
            final_points.append([best_solution[i, 0], best_solution[i, 1]])
        final_points = np.array(final_points)

        # Multiple refinement passes
        final_points = optimizer.multi_scale_optimization(final_points, max_iterations=150,
                                                        initial_tolerance=2.0, final_tolerance=1.0)
        final_points = optimizer.gradient_based_local_search(final_points, max_iterations=200,
                                                           initial_tolerance=2.0, final_tolerance=1.0)

        # Final evaluation
        _, circles = optimizer.evaluate_configuration(final_points)
        return circles

    # Fallback to simple grid pattern
    circles = np.zeros((21, 3))
    row_size = int(np.ceil(np.sqrt(21)))
    col_size = int(np.ceil(21 / row_size))

    spacing_x = optimizer.rect_width / (col_size + 1)
    spacing_y = optimizer.rect_height / (row_size + 1)

    count = 0
    for i in range(row_size):
        for j in range(col_size):
            if count < 21:
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                # Set radius to be proportional to available space
                radius = min(x, optimizer.rect_width - x, y, optimizer.rect_height - y) * 0.4
                circles[count] = [x, y, max(radius, 0.001)]
                count += 1

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")