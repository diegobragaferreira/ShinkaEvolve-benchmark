# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi, distance
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple, List, Optional
import time

class CirclePackOptimizer:
    def __init__(self, rect_width: float = 1.2, rect_height: float = 0.8, seed: int = 42):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def compute_circle_radius(self, point: np.ndarray, points: np.ndarray) -> float:
        """Compute maximum possible radius for a circle centered at 'point' using cKDTree for efficiency"""
        center_x, center_y = point
        # Distance to rectangle edges
        dist_to_edges = [
            center_x,                    # distance to left edge
            self.rect_width - center_x,  # distance to right edge
            center_y,                    # distance to bottom edge
            self.rect_height - center_y  # distance to top edge
        ]

        # Distance to other circles (excluding self) using cKDTree for efficiency
        min_dist_to_others = float('inf')
        if len(points) > 1:
            tree = cKDTree(points)
            distances, indices = tree.query(point, k=2)  # Get 2 nearest (including self)
            if len(distances) >= 2:
                min_dist_to_others = distances[1]  # Exclude self-distance

        # Maximum radius is limited by both edges and other circles
        max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
        return max(0.001, max_radius)

    def evaluate_configuration(self, points: np.ndarray) -> Tuple[float, np.ndarray]:
        """Evaluate a configuration by computing sum of radii"""
        total_radius = 0.0
        circles = []
        for point in points:
            radius = self.compute_circle_radius(point, points)
            circles.append([point[0], point[1], radius])
            total_radius += radius
        return total_radius, np.array(circles)

    def generate_voronoi_initialization(self) -> np.ndarray:
        """Generate initial points using Voronoi-based seeding with strategic placement"""
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

        # Strategy 2: More evenly distributed edge positions
        edge_positions = [
            (self.rect_width/2, self.rect_height * 0.1),  # top center
            (self.rect_width/2, self.rect_height * 0.9),  # bottom center
            (self.rect_width * 0.1, self.rect_height/2),  # left center
            (self.rect_width * 0.9, self.rect_height/2),  # right center
            (self.rect_width * 0.3, self.rect_height * 0.3),  # diagonal
            (self.rect_width * 0.7, self.rect_height * 0.7),  # diagonal
            (self.rect_width * 0.3, self.rect_height * 0.7),  # diagonal
            (self.rect_width * 0.7, self.rect_height * 0.3),  # diagonal
        ]

        for x, y in edge_positions:
            pert_x = np.random.normal(0, 0.02)
            pert_y = np.random.normal(0, 0.02)
            points.append([x + pert_x, y + pert_y])

        # Strategy 3: Grid placements in interior with better spacing
        grid_x = np.linspace(self.rect_width * 0.15, self.rect_width * 0.85, 4)
        grid_y = np.linspace(self.rect_height * 0.15, self.rect_height * 0.85, 4)

        for x in grid_x:
            for y in grid_y:
                if len(points) < 21:
                    pert_x = np.random.normal(0, 0.02)
                    pert_y = np.random.normal(0, 0.02)
                    points.append([x + pert_x, y + pert_y])

        # Strategy 4: Fill remaining with triangular distribution to avoid clustering
        while len(points) < 21:
            x = np.random.triangular(0.05, self.rect_width/2, self.rect_width - 0.05)
            y = np.random.triangular(0.05, self.rect_height/2, self.rect_height - 0.05)
            points.append([x, y])

        return np.array(points[:21])

    def generate_voronoi_seed_population(self, pop_size: int = 30) -> List[np.ndarray]:
        """Generate a diverse population using Voronoi-seeded initialization"""
        population = []
        for _ in range(pop_size):
            # Start with Voronoi-based initialization
            points = self.generate_voronoi_initialization()

            # Add some variation to make population diverse
            noise_magnitude = 0.05
            for i in range(len(points)):
                if random.random() < 0.3:  # 30% chance to add noise
                    points[i] = points[i] + np.random.normal(0, noise_magnitude, 2)
                    # Clamp to bounds
                    points[i][0] = np.clip(points[i][0], 0.05, self.rect_width - 0.05)
                    points[i][1] = np.clip(points[i][1], 0.05, self.rect_height - 0.05)

            population.append(points)
        return population

    def voronoi_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring using Voronoi-based crossover"""
        offspring = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])

        # Add some randomness to encourage exploration
        for i in range(len(offspring)):
            if random.random() < 0.1:  # 10% chance to add noise
                offspring[i] = offspring[i] + np.random.normal(0, 0.02, 2)
                # Clamp to bounds
                offspring[i][0] = np.clip(offspring[i][0], 0.05, self.rect_width - 0.05)
                offspring[i][1] = np.clip(offspring[i][1], 0.05, self.rect_height - 0.05)

        return np.array(offspring)

    def voronoi_mutate(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Mutate individual using Voronoi-guided mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Apply more sophisticated mutation with spatial awareness
                # First check if this point is near another point (high density region)
                distances = [distance.euclidean(mutated[i], other) for other in mutated if not np.array_equal(mutated[i], other)]
                if distances:
                    avg_dist = np.mean(distances)
                    # Adjust mutation strength based on local density
                    mutation_strength = 0.05 if avg_dist > 0.1 else 0.02
                else:
                    mutation_strength = 0.05

                mutated[i] = mutated[i] + np.random.normal(0, mutation_strength, 2)
                # Keep within bounds
                mutated[i][0] = np.clip(mutated[i][0], 0.05, self.rect_width - 0.05)
                mutated[i][1] = np.clip(mutated[i][1], 0.05, self.rect_height - 0.05)
        return mutated

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float],
                           tournament_size: int = 3) -> np.ndarray:
        """Select best individual from tournament"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

    def multi_scale_optimization(self, points: np.ndarray, max_iterations: int = 300) -> np.ndarray:
        """Multi-scale optimization with progressive refinement"""
        current_points = points.copy()

        # Scale 1: Very coarse search (largest steps) for global exploration
        for _ in range(30):
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius(current_point, current_points)

                # Very coarse step size
                for dx in [-0.15, -0.1, 0, 0.1, 0.15]:
                    for dy in [-0.15, -0.1, 0, 0.1, 0.15]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius(test_point, current_points)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        # Scale 2: Coarse search for medium exploration
        for _ in range(50):
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius(current_point, current_points)

                # Coarse step size
                for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius(test_point, current_points)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        # Scale 3: Medium search with moderate steps
        for _ in range(80):
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius(current_point, current_points)

                # Medium step size
                for dx in [-0.05, -0.025, 0, 0.025, 0.05]:
                    for dy in [-0.05, -0.025, 0, 0.025, 0.05]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius(test_point, current_points)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        # Scale 4: Fine search for final refinement
        for _ in range(100):
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = self.compute_circle_radius(current_point, current_points)

                # Fine step size
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= new_x <= self.rect_width - 0.05 and
                            0.05 <= new_y <= self.rect_height - 0.05):

                            test_point = np.array([new_x, new_y])
                            test_radius = self.compute_circle_radius(test_point, current_points)

                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point

                current_points[i] = best_point

        return current_points

    def gradient_based_local_search(self, points: np.ndarray, max_iterations: int = 500) -> np.ndarray:
        """Advanced local search using numerical gradient computation with improved adaptive step sizing"""
        current_points = points.copy()
        eps = 1e-5  # Small epsilon for gradient calculation

        # Track improvement history for adaptive step sizing
        recent_improvements = []
        max_recent_history = 10

        for iteration in range(max_iterations):
            updated = False
            new_points = current_points.copy()

            # For each point, compute gradient and move in direction of steepest ascent
            for i in range(len(current_points)):
                current_point = current_points[i]
                current_radius = self.compute_circle_radius(current_point, current_points)

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

                radius_x_pos = self.compute_circle_radius(test_point_x_pos, current_points)
                radius_x_neg = self.compute_circle_radius(test_point_x_neg, current_points)
                grad_x = (radius_x_pos - radius_x_neg) / (2 * eps)

                # Calculate partial derivative w.r.t. y
                test_point_y_pos = current_point.copy()
                test_point_y_neg = current_point.copy()
                test_point_y_pos[1] += eps
                test_point_y_neg[1] -= eps

                # Ensure points are within bounds
                test_point_y_pos[1] = np.clip(test_point_y_pos[1], 0.05, self.rect_height - 0.05)
                test_point_y_neg[1] = np.clip(test_point_y_neg[1], 0.05, self.rect_height - 0.05)

                radius_y_pos = self.compute_circle_radius(test_point_y_pos, current_points)
                radius_y_neg = self.compute_circle_radius(test_point_y_neg, current_points)
                grad_y = (radius_y_pos - radius_y_neg) / (2 * eps)

                # Update using gradient ascent (since we want to maximize radius)
                # Improved step size calculation considering recent improvements
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                # Base step size
                base_step_size = 0.02

                # Adaptive component based on gradient magnitude and recent improvements
                if grad_magnitude > 1e-8:
                    # Normalize step size by gradient magnitude, but cap it
                    adaptive_step = min(0.05, base_step_size / grad_magnitude)

                    # Further modify based on recent improvement trends
                    if len(recent_improvements) > 0:
                        avg_improvement = np.mean(recent_improvements[-min(len(recent_improvements), max_recent_history):])
                        if avg_improvement > 0.001:  # Significant recent improvement
                            adaptive_step *= 1.2  # Boost step size
                        elif avg_improvement < 0.0001:  # Little recent improvement
                            adaptive_step *= 0.8  # Reduce step size
                    step_size = adaptive_step
                else:
                    step_size = base_step_size

                new_x = current_point[0] + step_size * grad_x
                new_y = current_point[1] + step_size * grad_y

                # Keep within bounds
                new_x = np.clip(new_x, 0.05, self.rect_width - 0.05)
                new_y = np.clip(new_y, 0.05, self.rect_height - 0.05)

                # Test the new position
                test_point = np.array([new_x, new_y])
                test_radius = self.compute_circle_radius(test_point, current_points)

                # Accept the move if it improves the radius
                if test_radius > current_radius:
                    new_points[i] = test_point
                    updated = True

                    # Track improvement for adaptive control
                    improvement = test_radius - current_radius
                    recent_improvements.append(improvement)
                    if len(recent_improvements) > max_recent_history:
                        recent_improvements.pop(0)

            current_points = new_points

            # Early stopping if no significant improvement
            if not updated and iteration > 100:
                break

        return current_points

    def run_evolutionary_search(self, max_generations: int = 50) -> Tuple[float, np.ndarray]:
        """Execute the evolutionary search with Voronoi-guided operators"""
        # Evolutionary algorithm parameters
        pop_size = 30
        elite_size = 5

        # Generate initial population using Voronoi-based seeding
        population = self.generate_voronoi_seed_population(pop_size)

        best_solution = None
        best_fitness = 0

        for generation in range(max_generations):
            # Evaluate fitness of entire population
            fitness_scores = []
            population_circles = []

            for individual in population:
                fitness, circles = self.evaluate_configuration(individual)
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

            # Generate offspring through Voronoi-guided crossover and mutation
            while len(new_population) < pop_size:
                # Tournament selection
                parent1 = self.tournament_selection(sorted_population, fitness_scores)
                parent2 = self.tournament_selection(sorted_population, fitness_scores)

                # Crossover
                offspring = self.voronoi_crossover(parent1, parent2)

                # Mutation
                offspring = self.voronoi_mutate(offspring)

                new_population.append(offspring)

            population = new_population

        return best_fitness, best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize optimizer
    optimizer = CirclePackOptimizer()

    # Stage 1: Voronoi-based initial seeding
    points = optimizer.generate_voronoi_initialization()

    # Stage 2: Evaluate initial configuration
    initial_fitness, initial_circles = optimizer.evaluate_configuration(points)

    # Stage 3: Run Voronoi-guided evolutionary search
    best_fitness, best_solution = optimizer.run_evolutionary_search(max_generations=40)

    # Stage 4: Final refinement
    final_points = []
    if best_solution is not None:
        for i in range(21):
            final_points.append([best_solution[i, 0], best_solution[i, 1]])
        final_points = np.array(final_points)

        # Multiple refinement passes for better results
        refined_points = optimizer.multi_scale_optimization(final_points, max_iterations=150)
        refined_points = optimizer.gradient_based_local_search(refined_points, max_iterations=200)

        # Recalculate final configuration
        _, final_circles = optimizer.evaluate_configuration(refined_points)
        return final_circles

    # Fallback to initial solution if evolutionary search failed
    return initial_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")