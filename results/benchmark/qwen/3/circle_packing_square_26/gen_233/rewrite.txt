# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 150
NUM_GENERATIONS = 600
TOURNAMENT_SIZE = 7
MUTATION_RATE_START = 0.2
MUTATION_RATE_END = 0.02
CROSSOVER_RATE = 0.85
BOUNDARY_PENALTY_WEIGHT_START = 10000.0
OVERLAP_PENALTY_WEIGHT_START = 100000.0

class SpatialIndexer:
    """Efficient spatial indexing for overlap detection with adaptive grid"""
    
    def __init__(self, grid_size: int = 15):
        self.grid_size = grid_size
        self.grid_cells = {}
        
    def build_index(self, circles: np.ndarray) -> dict:
        """Build spatial grid index"""
        self.grid_cells.clear()
        for i, (x, y, r) in enumerate(circles):
            cell_x = int(x * self.grid_size)
            cell_y = int(y * self.grid_size)
            cell = (cell_x, cell_y)
            if cell not in self.grid_cells:
                self.grid_cells[cell] = []
            self.grid_cells[cell].append(i)
        return self.grid_cells
    
    def get_candidates(self, x: float, y: float, radius: float) -> List[int]:
        """Get candidate circles within neighborhood"""
        candidates = []
        cell_x = int(x * self.grid_size)
        cell_y = int(y * self.grid_size)
        
        # Check nearby cells in 3x3 grid
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (cell_x + dx, cell_y + dy)
                if cell in self.grid_cells:
                    candidates.extend(self.grid_cells[cell])
        return candidates

def is_valid(circles: np.ndarray, spatial_indexer: SpatialIndexer = None) -> bool:
    """Check if all circles are within bounds and non-overlapping"""
    n = len(circles)

    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Use spatial indexing for efficient overlap checking
    if n > 1 and spatial_indexer is not None:
        try:
            # Build index once
            spatial_indexer.build_index(circles)
            
            # Query candidates efficiently
            for i in range(n):
                x, y, r = circles[i]
                candidates = spatial_indexer.get_candidates(x, y, r)
                
                for j in candidates:
                    if i < j:  # Avoid duplicate checking
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if distance < r1 + r2 - 1e-8:
                            return False
        except Exception:
            # Fallback to brute force if tree fails
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2 - 1e-8:
                        return False
    elif n > 1:
        # Brute force fallback for small populations
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2 - 1e-8:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def evaluate_fitness(circles: np.ndarray, generation: int = 0, total_generations: int = 0) -> float:
    """Evaluate fitness of a solution, higher is better."""
    if not is_valid(circles):
        # Apply penalty for constraint violations
        penalty = 0

        # Boundary penalty - scaled by generation for adaptive penalty
        boundary_violations = 0
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0:
                boundary_violations += (r - x)**2
            if x + r > 1:
                boundary_violations += (x + r - 1)**2
            if y - r < 0:
                boundary_violations += (r - y)**2
            if y + r > 1:
                boundary_violations += (y + r - 1)**2

        # Dynamic penalty weights that decrease over time
        dynamic_boundary_weight = BOUNDARY_PENALTY_WEIGHT_START * (1 - generation / total_generations)
        penalty += dynamic_boundary_weight * boundary_violations

        # Overlap penalty - compute based on actual overlap amounts
        overlap_penalty = 0
        n = len(circles)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2 - 1e-8:
                    overlap = (r1 + r2 - distance)
                    overlap_penalty += overlap**2

        # Dynamic overlap penalty weight
        dynamic_overlap_weight = OVERLAP_PENALTY_WEIGHT_START * (1 - generation / total_generations)
        penalty += dynamic_overlap_weight * overlap_penalty

        return -penalty

    return calculate_sum_radii(circles)

def generate_poisson_disk_points(n_points: int, min_distance: float = 0.1) -> List[Tuple[float, float]]:
    """Generate poisson disk distributed points"""
    points = []
    attempts = 0
    max_attempts = 10000

    while len(points) < n_points and attempts < max_attempts:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)

        # Check minimum distance to existing points
        valid = True
        for px, py in points:
            distance = math.sqrt((x - px)**2 + (y - py)**2)
            if distance < min_distance:
                valid = False
                break

        if valid:
            points.append((x, y))
        attempts += 1

    return points

def initialize_population(size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with improved Voronoi-inspired placement using Poisson disk sampling"""
    population = []
    spatial_indexer = SpatialIndexer()

    # Generate seed points using poisson disk sampling for better distribution
    seed_points = generate_poisson_disk_points(n_circles, min_distance=0.12)
    
    # Pad to required number if needed
    while len(seed_points) < n_circles:
        seed_points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))

    # Take first n_circles points
    seed_points = seed_points[:n_circles]

    for _ in range(size):
        circles = np.zeros((n_circles, 3))

        # Assign positions from seed points with noise
        for i, (x, y) in enumerate(seed_points):
            # Add small random noise
            x += random.uniform(-0.015, 0.015)
            y += random.uniform(-0.015, 0.015)

            # Clamp to valid range
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))

            # Initial radius - varied but reasonable
            r = 0.01 + random.uniform(0.015, 0.045)

            circles[i] = [x, y, r]

        # Use spatial indexing for initial refinement
        circles = refine_circles(circles, spatial_indexer)
        population.append(circles)

    return population

def refine_circles(circles: np.ndarray, spatial_indexer: SpatialIndexer = None) -> np.ndarray:
    """Refine circles to ensure valid constraints with spatial indexing optimization"""
    refined = circles.copy()

    # First enforce boundary constraints
    for i in range(len(refined)):
        x, y, r = refined[i]
        # Ensure circle fits in unit square
        max_radius = min(x, 1-x, y, 1-y)
        refined[i, 2] = min(r, max_radius)
        # Clamp coordinates
        refined[i, 0] = max(refined[i, 2], min(1-refined[i, 2], x))
        refined[i, 1] = max(refined[i, 2], min(1-refined[i, 2], y))

    # Resolve overlaps through iterative adjustment
    n = len(refined)
    max_iterations = 15
    for _ in range(max_iterations):
        any_changed = False
        positions = [(x, y) for x, y, r in refined]
        if len(positions) > 1:
            try:
                # Use spatial indexing when available
                if spatial_indexer is not None:
                    spatial_indexer.build_index(refined)
                    # Query candidates efficiently
                    for i in range(n):
                        x, y, r = refined[i]
                        candidates = spatial_indexer.get_candidates(x, y, r)
                        
                        for j in candidates:
                            if i < j:
                                x1, y1, r1 = refined[i]
                                x2, y2, r2 = refined[j]
                                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                                if distance < r1 + r2 - 1e-8:
                                    # Adjust positions to separate circles
                                    if distance > 0.001:
                                        dx = (x2 - x1) / distance
                                        dy = (y2 - y1) / distance
                                        move_dist = (r1 + r2 - distance) * 0.5

                                        # Apply adjustment with small damping
                                        refined[i, 0] -= dx * move_dist * 0.2
                                        refined[i, 1] -= dy * move_dist * 0.2
                                        refined[j, 0] += dx * move_dist * 0.2
                                        refined[j, 1] += dy * move_dist * 0.2
                                        any_changed = True
                else:
                    # Fallback to standard KDTree
                    tree = cKDTree(positions)
                    pairs = tree.query_pairs(r=0.01, output_type='ndarray')

                    for i, j in pairs:
                        if i < j:
                            x1, y1, r1 = refined[i]
                            x2, y2, r2 = refined[j]
                            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                            if distance < r1 + r2 - 1e-8:
                                # Adjust positions to separate circles
                                if distance > 0.001:
                                    dx = (x2 - x1) / distance
                                    dy = (y2 - y1) / distance
                                    move_dist = (r1 + r2 - distance) * 0.5

                                    # Apply adjustment with small damping
                                    refined[i, 0] -= dx * move_dist * 0.2
                                    refined[i, 1] -= dy * move_dist * 0.2
                                    refined[j, 0] += dx * move_dist * 0.2
                                    refined[j, 1] += dy * move_dist * 0.2
                                    any_changed = True
            except Exception:
                pass

        if not any_changed:
            break

    return refined

def tournament_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Select an individual using tournament selection."""
    # Increase tournament size for better selection pressure
    tournament_size = max(5, TOURNAMENT_SIZE + int(len(population) * 0.02))
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Multi-point crossover for better exploration
    crossover_points = sorted(random.sample(range(1, n), min(7, n//3)))
    crossover_points = [0] + crossover_points + [n]

    # Alternate segments between parents
    for i in range(len(crossover_points) - 1):
        start = crossover_points[i]
        end = crossover_points[i + 1]
        if i % 2 == 0:
            child1[start:end] = parent2[start:end].copy()
            child2[start:end] = parent1[start:end].copy()
        else:
            child1[start:end] = parent1[start:end].copy()
            child2[start:end] = parent2[start:end].copy()

    return child1, child2

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive rate and dual strategy."""
    mutated = circles.copy()
    n = len(mutated)

    # Adaptive mutation rate that decreases over generations
    mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * (
        1 / (1 + np.exp(10 * (generation / total_generations - 0.5)))
    )

    # Determine adaptive mutation scales based on generation phase
    progress = generation / total_generations
    
    # Early generations: exploration-focused
    if progress < 0.3:
        pos_scale = 0.06
        rad_scale = 0.04
    # Middle generations: balance
    elif progress < 0.7:
        pos_scale = 0.03
        rad_scale = 0.025
    # Late generations: exploitation-focused
    else:
        pos_scale = 0.015
        rad_scale = 0.015

    # Mutate each circle with adaptive probability
    for i in range(n):
        if random.random() < mutation_rate:
            # Choose which component to mutate with preference to position
            component = random.choices([0, 1, 2], weights=[0.45, 0.45, 0.1])[0]

            if component == 0:  # x coordinate
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, pos_scale)))
            elif component == 1:  # y coordinate
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, pos_scale)))
            else:  # radius
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, rad_scale)))

    # Apply post-mutation refinement to fix constraint violations
    spatial_indexer = SpatialIndexer()
    mutated = refine_circles(mutated, spatial_indexer)
    return mutated

def get_best_individual(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Get the individual with highest fitness."""
    best_idx = fitnesses.index(max(fitnesses))
    return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 26
    population = initialize_population(POPULATION_SIZE, n)

    best_fitness_history = []
    spatial_indexer = SpatialIndexer()

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual, generation, NUM_GENERATIONS) for individual in population]

        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")

        # Create new population
        new_population = []

        # Elitism: keep best individual
        best_individual = get_best_individual(population, fitnesses)
        new_population.append(best_individual)

        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, generation, NUM_GENERATIONS)
            child2 = mutate(child2, generation, NUM_GENERATIONS)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]

    # Get final best solution
    final_fitnesses = [evaluate_fitness(individual, NUM_GENERATIONS, NUM_GENERATIONS) for individual in population]
    best_solution = get_best_individual(population, final_fitnesses)

    # Final validation and repair if needed
    if not is_valid(best_solution, spatial_indexer):
        # Apply final refinement
        best_solution = refine_circles(best_solution, spatial_indexer)

    # Ensure everything is within bounds
    for i in range(len(best_solution)):
        x, y, r = best_solution[i]
        # Ensure it stays within bounds
        best_solution[i, 0] = max(r, min(1-r, x))
        best_solution[i, 1] = max(r, min(1-r, y))
        best_solution[i, 2] = max(0.001, min(0.49, r))

    # Return the best solution found
    return best_solution


# EVOLVE-BLOCK-END