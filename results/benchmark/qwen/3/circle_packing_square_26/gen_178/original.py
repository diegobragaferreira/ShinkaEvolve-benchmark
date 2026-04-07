# EVOLVE-BLOCK-START
import numpy as np
import random
from copy import deepcopy
from typing import Tuple, List
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time

class OptimizedSpatialGrid:
    """Optimized spatial grid with adaptive resolution and caching for overlap detection"""

    def __init__(self):
        self.grids = {}
        self.resolutions = [10, 20, 40]  # Different resolutions for different scales
        self.cached_queries = {}  # Cache for expensive queries

    def _get_cell_coords(self, x: float, y: float, resolution: int) -> Tuple[int, int]:
        """Get grid cell coordinates for a point"""
        cell_size = 1.0 / resolution
        return (int(x / cell_size), int(y / cell_size))

    def clear(self):
        """Clear all grids and cache"""
        self.grids.clear()
        self.cached_queries.clear()

    def add_circle(self, idx: int, x: float, y: float, radius: float):
        """Add a circle to all relevant grids"""
        for res in self.resolutions:
            if res not in self.grids:
                self.grids[res] = {}
            cell_coords = self._get_cell_coords(x, y, res)
            if cell_coords not in self.grids[res]:
                self.grids[res][cell_coords] = []
            self.grids[res][cell_coords].append((idx, x, y, radius))

    def _get_neighbors_cached(self, x: float, y: float, radius: float,
                            target_resolution: int = 20) -> List[Tuple[int, float, float, float]]:
        """Get neighbors with caching for repeated queries"""
        cache_key = (hash((x, y, radius)), target_resolution)
        if cache_key in self.cached_queries:
            return self.cached_queries[cache_key]

        if target_resolution not in self.grids:
            result = []
            self.cached_queries[cache_key] = result
            return result

        cell_x, cell_y = self._get_cell_coords(x, y, target_resolution)
        neighbors = []

        # Check surrounding cells (3x3 grid around main cell)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grids[target_resolution]:
                    for idx, nx, ny, nr in self.grids[target_resolution][neighbor_cell]:
                        # Skip self
                        if idx == -1:
                            continue
                        neighbors.append((idx, nx, ny, nr))

        self.cached_queries[cache_key] = neighbors
        return neighbors

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n_circles = 26
    max_generations = 1000
    population_size = 50

    # Global optimized spatial grid instance for validation
    spatial_grid = OptimizedSpatialGrid()

    def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Initialize population with enhanced strategies"""
        population = []

        # Pre-computed Voronoi-like distribution points for better starting configurations
        voronoi_points = []
        n_voronoi = int(np.ceil(np.sqrt(n_circles)) * 1.2)
        for i in range(n_voronoi):
            for j in range(n_voronoi):
                if len(voronoi_points) < n_circles * 2:  # Generate extra points
                    x = (i + 0.5) / n_voronoi
                    y = (j + 0.5) / n_voronoi
                    # Add gentle jitter to avoid perfect grid
                    x += np.random.normal(0, 0.02)
                    y += np.random.normal(0, 0.02)
                    # Clamp to valid range
                    x = max(0.02, min(0.98, x))
                    y = max(0.02, min(0.98, y))
                    voronoi_points.append((x, y))

        voronoi_points = voronoi_points[:n_circles]

        for _ in range(pop_size):
            circles = np.zeros((n_circles, 3))

            # Strategy 1: Start with Voronoi-like points
            points = voronoi_points.copy()

            # Strategy 2: Perturb points slightly for diversity
            for i in range(len(points)):
                points[i] = (
                    max(0.02, min(0.98, points[i][0] + np.random.normal(0, 0.03))),
                    max(0.02, min(0.98, points[i][1] + np.random.normal(0, 0.03)))
                )

            # Compute proper radii for each circle based on proximity to others
            for i in range(n_circles):
                x, y = points[i]

                # Calculate minimum distance to all other centers
                min_dist = float('inf')
                for j in range(n_circles):
                    if i != j:
                        other_x, other_y = points[j]
                        dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                        min_dist = min(min_dist, dist)

                # Set radius to be safe from boundaries and neighbors
                radius = min(0.05, x, 1-x, y, 1-y, min_dist/2)
                if radius <= 0:
                    radius = 0.01

                circles[i] = [x, y, radius]

            # Strategy 3: Apply small randomized adjustments for variety
            for i in range(n_circles):
                if random.random() < 0.3:
                    circles[i, 0] += np.random.normal(0, 0.01)
                    circles[i, 1] += np.random.normal(0, 0.01)
                    circles[i, 0] = max(0.01, min(0.99, circles[i, 0]))
                    circles[i, 1] = max(0.01, min(0.99, circles[i, 1]))

            population.append(circles)
        return population

    def is_valid(circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping"""
        n = len(circles)

        # Early exit if any circle violates boundaries
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

        # Clear spatial grid
        spatial_grid.clear()

        # Add all circles to spatial grid for overlap checking
        for i in range(n):
            x, y, r = circles[i]
            spatial_grid.add_circle(i, x, y, r)

        # Fast overlap detection using spatial grid with smart resolution selection
        for i in range(n):
            x1, y1, r1 = circles[i]

            # Select resolution based on circle size for optimal performance
            if r1 < 0.03:
                resolution = 40  # Very fine gridding for small circles
            elif r1 < 0.07:
                resolution = 20  # Medium gridding
            else:
                resolution = 10  # Coarse gridding for large circles

            candidates = spatial_grid._get_neighbors_cached(x1, y1, r1, resolution)

            # Check actual overlaps
            for _, x2, y2, r2 in candidates:
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False

        return True

    def evaluate_fitness(circles: np.ndarray, generation: int = 0) -> float:
        """Evaluate fitness of a solution with optimizations"""
        if not is_valid(circles):
            # Progressive constraint relaxation
            penalty_weight = max(0.2, 1.0 - (generation / max_generations) * 0.8)

            # Fast penalty calculation with early termination
            total_penalty = 0

            # Boundary violations (fast vectorized check)
            boundary_penalties = np.maximum(0, -circles[:, 0] + circles[:, 2]) + \
                               np.maximum(0, circles[:, 0] - circles[:, 2] - 1) + \
                               np.maximum(0, -circles[:, 1] + circles[:, 2]) + \
                               np.maximum(0, circles[:, 1] - circles[:, 2] - 1)
            total_penalty += np.sum(boundary_penalties) * 1000 * penalty_weight

            # Overlap violations (vectorized approach)
            n = len(circles)
            if n > 1:
                coords = circles[:, :2]
                radii = circles[:, 2]

                # Vectorized distance computation
                try:
                    # Fast pairwise distance calculation for small populations
                    distances = cdist(coords, coords)
                    np.fill_diagonal(distances, np.inf)

                    # Check overlaps efficiently
                    overlap_matrix = (radii[:, None] + radii[None, :] - distances) > 0
                    np.fill_diagonal(overlap_matrix, False)  # No self-overlap

                    # Sum up all overlaps
                    overlap_distances = np.where(overlap_matrix,
                                               radii[:, None] + radii[None, :] - distances, 0)
                    total_penalty += np.sum(overlap_distances) * 10000 * penalty_weight

                except Exception:
                    # Fallback to slower approach if vectorization fails
                    for i in range(n):
                        for j in range(i+1, n):
                            x1, y1, r1 = circles[i]
                            x2, y2, r2 = circles[j]
                            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            overlap = max(0, r1 + r2 - distance)
                            total_penalty += overlap * 10000 * penalty_weight

            return -total_penalty

        # Valid configuration: maximize sum of radii
        return np.sum(circles[:, 2])

    def mutate(circles: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Enhanced mutation with multiple strategies"""
        mutated = deepcopy(circles)
        # Adaptive mutation rate
        mutation_rate = 0.15 - (0.13 * generation / max_generations)

        # Different mutation strategies
        mutation_types = ['position', 'radius', 'global_shift']
        weights = [0.6, 0.25, 0.15]  # Weighted probabilities

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutation_type = random.choices(mutation_types, weights=weights)[0]

                if mutation_type == 'position':
                    # Position mutation with adaptive step sizes
                    step_size = 0.02 * (1.0 - generation/max_generations * 0.8)  # Decrease over time
                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))

                elif mutation_type == 'radius':
                    # Radius mutation with careful bounds
                    mutated[i, 2] += np.random.normal(0, 0.008)
                    mutated[i, 2] = max(0.001, mutated[i, 2])

                elif mutation_type == 'global_shift':
                    # Large shift for global exploration
                    mutated[i, 0] += np.random.uniform(-0.05, 0.05)
                    mutated[i, 1] += np.random.uniform(-0.05, 0.05)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))

        return mutated

    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Enhanced crossover with better mixing"""
        child = deepcopy(parent1)
        n = len(parent1)

        # Use uniform crossover with more strategic mixing
        for i in range(n):
            if random.random() < 0.5:
                child[i] = parent2[i].copy()

        return child

    def tournament_selection(population: List[np.ndarray], k: int = 3) -> np.ndarray:
        """Tournament selection with early termination"""
        selected = random.sample(population, k)
        # Sort by fitness (descending) and return best
        return max(selected, key=lambda x: evaluate_fitness(x))

    # Initialize population
    population = initialize_population(population_size, n_circles)

    # Evolve with timing
    start_time = time.time()
    best_fitness = float('-inf')
    best_solution = None

    for generation in range(max_generations):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual, generation) for individual in population]

        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])

        # Early termination check
        if time.time() - start_time > 55:  # Leave 5 seconds for cleanup
            break

        # Elitism: keep top 15%
        elite_count = max(1, population_size // 7)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite = [population[i] for i in sorted_indices]

        # Create new population
        new_population = deepcopy(elite)

        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            mutated_child = mutate(child, generation, max_generations)

            new_population.append(mutated_child)

        population = new_population[:population_size]

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to first individual if nothing was found
        return population[0]

# EVOLVE-BLOCK-END