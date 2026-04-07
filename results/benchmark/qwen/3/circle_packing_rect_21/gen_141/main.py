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

# Configuration constants
POPULATION_SIZE = 30
ELITE_SIZE = 5
MAX_GENERATIONS = 40
INITIAL_REFINEMENT_ITERATIONS = 150
FINAL_REFINEMENT_ITERATIONS = 200

def compute_circle_radius(point: np.ndarray, points: np.ndarray, 
                         rect_width: float, rect_height: float) -> float:
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
    if len(points) > 1:
        # Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(points)
        distances, _ = tree.query(point, k=2)  # Get 2 nearest (including self)
        if len(distances) >= 2:
            min_dist_to_others = distances[1]  # Exclude self-distance

    # Maximum radius is limited by both edges and other circles
    max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
    return max(0.001, max_radius)

def evaluate_configuration(points: np.ndarray, rect_width: float, 
                          rect_height: float) -> Tuple[float, np.ndarray]:
    """Evaluate a configuration by computing sum of radii"""
    total_radius = 0.0
    circles = []
    for point in points:
        radius = compute_circle_radius(point, points, rect_width, rect_height)
        circles.append([point[0], point[1], radius])
        total_radius += radius
    return total_radius, np.array(circles)

def generate_initial_config(rect_width: float = 1.2, rect_height: float = 0.8) -> np.ndarray:
    """Generate highly optimized initial configuration using Voronoi-distributed points"""
    points = []

    # Strategy 1: Corner placements with strategic perturbation
    corner_positions = [
        (rect_width * 0.1, rect_height * 0.1),
        (rect_width * 0.9, rect_height * 0.1),
        (rect_width * 0.1, rect_height * 0.9),
        (rect_width * 0.9, rect_height * 0.9),
        (rect_width / 2, rect_height / 2)
    ]

    for x, y in corner_positions:
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

    # Strategy 3: Grid placements in interior with better spacing
    grid_x = np.linspace(rect_width * 0.15, rect_width * 0.85, 4)
    grid_y = np.linspace(rect_height * 0.15, rect_height * 0.85, 4)

    for x in grid_x:
        for y in grid_y:
            if len(points) < 21:
                pert_x = np.random.normal(0, 0.02)
                pert_y = np.random.normal(0, 0.02)
                points.append([x + pert_x, y + pert_y])

    # Strategy 4: Fill remaining with triangular distribution to avoid clustering
    while len(points) < 21:
        x = np.random.triangular(0.05, rect_width/2, rect_width - 0.05)
        y = np.random.triangular(0.05, rect_height/2, rect_height - 0.05)
        points.append([x, y])

    return np.array(points[:21])

def generate_seed_population(pop_size: int, rect_width: float, rect_height: float) -> List[np.ndarray]:
    """Generate diverse population using strategic initialization"""
    population = []
    for _ in range(pop_size):
        # Start with good initialization
        points = generate_initial_config(rect_width, rect_height)
        
        # Add some variation to make population diverse
        noise_magnitude = 0.05
        for i in range(len(points)):
            if random.random() < 0.3:  # 30% chance to add noise
                points[i] = points[i] + np.random.normal(0, noise_magnitude, 2)
                # Clamp to bounds
                points[i][0] = np.clip(points[i][0], 0.05, rect_width - 0.05)
                points[i][1] = np.clip(points[i][1], 0.05, rect_height - 0.05)
        
        population.append(points)
    return population

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], 
                         tournament_size: int = 3) -> np.ndarray:
    """Select best individual from tournament"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def voronoi_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
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
            offspring[i][0] = np.clip(offspring[i][0], 0.05, 1.2 - 0.05)
            offspring[i][1] = np.clip(offspring[i][1], 0.05, 0.8 - 0.05)
    
    return np.array(offspring)

def voronoi_mutate(individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate individual using Voronoi-guided mutation"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Apply more sophisticated mutation with spatial awareness
            # First check if this point is near another point (high density region)
            distances = [np.linalg.norm(mutated[i] - other) for other in mutated if not np.array_equal(mutated[i], other)]
            if distances:
                avg_dist = np.mean(distances)
                # Adjust mutation strength based on local density
                mutation_strength = 0.05 if avg_dist > 0.1 else 0.02
            else:
                mutation_strength = 0.05
            
            mutated[i] = mutated[i] + np.random.normal(0, mutation_strength, 2)
            # Keep within bounds
            mutated[i][0] = np.clip(mutated[i][0], 0.05, 1.2 - 0.05)
            mutated[i][1] = np.clip(mutated[i][1], 0.05, 0.8 - 0.05)
    return mutated

def multi_scale_optimization(points: np.ndarray, rect_width: float, 
                            rect_height: float, max_iterations: int = 300) -> np.ndarray:
    """Multi-scale optimization with progressive refinement"""
    current_points = points.copy()

    # Scale 1: Very coarse search (largest steps) for global exploration
    for _ in range(30):
        for i in range(len(current_points)):
            current_point = current_points[i]
            best_point = current_point.copy()
            best_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

            # Very coarse step size
            for dx in [-0.15, -0.1, 0, 0.1, 0.15]:
                for dy in [-0.15, -0.1, 0, 0.1, 0.15]:
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

    # Scale 2: Coarse search for medium exploration
    for _ in range(50):
        for i in range(len(current_points)):
            current_point = current_points[i]
            best_point = current_point.copy()
            best_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

            # Coarse step size
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

    # Scale 3: Medium search with moderate steps
    for _ in range(80):
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

    # Scale 4: Fine search for final refinement
    for _ in range(100):
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

def gradient_based_local_search(points: np.ndarray, rect_width: float, 
                               rect_height: float, max_iterations: int = 500) -> np.ndarray:
    """Advanced local search using numerical gradient computation"""
    current_points = points.copy()
    eps = 1e-5  # Small epsilon for gradient calculation

    for iteration in range(max_iterations):
        updated = False
        new_points = current_points.copy()

        # For each point, compute gradient and move in direction of steepest ascent
        for i in range(len(current_points)):
            current_point = current_points[i]
            current_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)

            # Compute numerical gradient using finite differences
            grad_x = 0.0
            grad_y = 0.0

            # Calculate partial derivative w.r.t. x
            test_point_x_pos = current_point.copy()
            test_point_x_neg = current_point.copy()
            test_point_x_pos[0] += eps
            test_point_x_neg[0] -= eps

            # Ensure points are within bounds
            test_point_x_pos[0] = np.clip(test_point_x_pos[0], 0.05, rect_width - 0.05)
            test_point_x_neg[0] = np.clip(test_point_x_neg[0], 0.05, rect_width - 0.05)

            radius_x_pos = compute_circle_radius(test_point_x_pos, current_points, rect_width, rect_height)
            radius_x_neg = compute_circle_radius(test_point_x_neg, current_points, rect_width, rect_height)
            grad_x = (radius_x_pos - radius_x_neg) / (2 * eps)

            # Calculate partial derivative w.r.t. y
            test_point_y_pos = current_point.copy()
            test_point_y_neg = current_point.copy()
            test_point_y_pos[1] += eps
            test_point_y_neg[1] -= eps

            # Ensure points are within bounds
            test_point_y_pos[1] = np.clip(test_point_y_pos[1], 0.05, rect_height - 0.05)
            test_point_y_neg[1] = np.clip(test_point_y_neg[1], 0.05, rect_height - 0.05)

            radius_y_pos = compute_circle_radius(test_point_y_pos, current_points, rect_width, rect_height)
            radius_y_neg = compute_circle_radius(test_point_y_neg, current_points, rect_width, rect_height)
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

def run_evolutionary_search(initial_points: np.ndarray, 
                           rect_width: float, rect_height: float) -> Tuple[float, np.ndarray]:
    """Execute the evolutionary search with Voronoi-guided operators"""
    # Evolutionary algorithm parameters
    pop_size = POPULATION_SIZE
    elite_size = ELITE_SIZE
    
    # Generate initial population using strategic seeding
    population = generate_seed_population(pop_size, rect_width, rect_height)
    
    best_solution = None
    best_fitness = 0
    
    for generation in range(MAX_GENERATIONS):
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
        
        # Generate offspring through Voronoi-guided crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(sorted_population, fitness_scores)
            parent2 = tournament_selection(sorted_population, fitness_scores)
            
            # Crossover
            offspring = voronoi_crossover(parent1, parent2)
            
            # Mutation
            offspring = voronoi_mutate(offspring)
            
            new_population.append(offspring)
        
        population = new_population
    
    return best_fitness, best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Fixed rectangle dimensions for perimeter = 4 (width + height = 2)
    rect_width = 1.2
    rect_height = 0.8
    
    # Stage 1: Strategic initial seeding 
    points = generate_initial_config(rect_width, rect_height)
    
    # Stage 2: Evaluate initial configuration
    initial_fitness, initial_circles = evaluate_configuration(points, rect_width, rect_height)
    
    # Stage 3: Run evolutionary search
    best_fitness, best_solution = run_evolutionary_search(points, rect_width, rect_height)
    
    # Stage 4: Final refinement 
    final_points = []
    if best_solution is not None:
        for i in range(21):
            final_points.append([best_solution[i, 0], best_solution[i, 1]])
        final_points = np.array(final_points)
        
        # Multiple refinement passes for better results
        refined_points = multi_scale_optimization(final_points, rect_width, rect_height, INITIAL_REFINEMENT_ITERATIONS)
        refined_points = gradient_based_local_search(refined_points, rect_width, rect_height, FINAL_REFINEMENT_ITERATIONS)
        
        # Recalculate final configuration
        _, final_circles = evaluate_configuration(refined_points, rect_width, rect_height)
        return final_circles
    
    # Fallback to initial solution if evolutionary search failed
    return initial_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")