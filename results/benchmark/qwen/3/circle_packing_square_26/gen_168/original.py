# EVOLVE-BLOCK-START
import numpy as np
import random
from copy import deepcopy
from typing import Tuple, List
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time

class SpatialGrid:
    """Efficient spatial grid for fast overlap detection"""

    def __init__(self, resolution: int = 20):
        self.resolution = resolution
        self.grid = {}
        self.cell_size = 1.0 / resolution
        self.neighbors_cache = {}

    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid cell coordinates for a point"""
        return (int(x / self.cell_size), int(y / self.cell_size))

    def clear(self):
        """Clear the spatial grid"""
        self.grid.clear()
        self.neighbors_cache.clear()

    def add_circle(self, idx: int, x: float, y: float, radius: float):
        """Add a circle to the spatial grid"""
        cell_coords = self._get_cell_coords(x, y)
        if cell_coords not in self.grid:
            self.grid[cell_coords] = []
        self.grid[cell_coords].append((idx, x, y, radius))

    def get_neighbors(self, x: float, y: float, radius: float, use_cache: bool = True) -> List[Tuple[int, float, float, float]]:
        """Get all circles in neighboring cells that could potentially overlap"""
        cache_key = (x, y, radius)
        if use_cache and cache_key in self.neighbors_cache:
            return self.neighbors_cache[cache_key]

        neighbors = []
        cell_x, cell_y = self._get_cell_coords(x, y)

        # Check surrounding cells (3x3 grid around main cell)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grid:
                    for idx, nx, ny, nr in self.grid[neighbor_cell]:
                        # Skip self
                        if idx == -1:  # Placeholder for self-check
                            continue
                        neighbors.append((idx, nx, ny, nr))

        if use_cache:
            self.neighbors_cache[cache_key] = neighbors

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

    # Global spatial grid instance for validation
    spatial_grid = SpatialGrid(resolution=20)

    def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Initialize population with force-directed relaxation"""
        population = []

        def force_directed_initialization(n_circles: int) -> np.ndarray:
            """Initialize circles using force-directed relaxation"""
            circles = np.zeros((n_circles, 3))

            # Start with random positions
            for i in range(n_circles):
                circles[i] = [
                    np.random.uniform(0.05, 0.95),
                    np.random.uniform(0.05, 0.95),
                    0.02  # Initial small radius
                ]

            # Simple force-directed relaxation
            for _ in range(500):  # Relaxation iterations
                forces = np.zeros((n_circles, 2))  # Force on each circle

                # Compute repulsive forces between all pairs
                for i in range(n_circles):
                    x1, y1, r1 = circles[i]

                    # Repulsive force from boundaries
                    fx = 0.0
                    fy = 0.0

                    # Left boundary
                    if x1 - r1 < 0:
                        fx += (r1 - x1) * 0.1
                    # Right boundary
                    if x1 + r1 > 1:
                        fx += (1 - r1 - x1) * 0.1

                    # Bottom boundary
                    if y1 - r1 < 0:
                        fy += (r1 - y1) * 0.1
                    # Top boundary
                    if y1 + r1 > 1:
                        fy += (1 - r1 - y1) * 0.1

                    forces[i] = [fx, fy]

                    # Repulsive forces from other circles
                    for j in range(i+1, n_circles):
                        x2, y2, r2 = circles[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = np.sqrt(dx*dx + dy*dy)

                        if distance > 0 and distance < r1 + r2:
                            # Overlapping - strong repulsive force
                            force_magnitude = 5.0 / (distance * distance + 0.01)
                            fx += dx / distance * force_magnitude
                            fy += dy / distance * force_magnitude
                        elif distance < 1.5 * (r1 + r2):  # Near neighbors
                            # Mild repulsive force
                            force_magnitude = 1.0 / (distance * distance + 0.01)
                            fx += dx / distance * force_magnitude
                            fy += dy / distance * force_magnitude

                # Apply forces (with damping)
                for i in range(n_circles):
                    step_size = 0.001
                    circles[i, 0] += forces[i, 0] * step_size
                    circles[i, 1] += forces[i, 1] * step_size

                    # Clamp to boundaries
                    circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
                    circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))

            # Assign initial radii based on spacing to neighbors
            coords = circles[:, :2]
            try:
                tree = cKDTree(coords)
                distances, indices = tree.query(coords, k=min(5, n_circles), p=2)

                for i in range(n_circles):
                    # Radius constrained by boundaries
                    boundary_radius = min(
                        circles[i, 0],  # left boundary
                        1 - circles[i, 0],  # right boundary
                        circles[i, 1],  # bottom boundary
                        1 - circles[i, 1]  # top boundary
                    )

                    # Radius constrained by neighbors
                    neighbor_min_dist = np.inf
                    for j in range(len(indices[i])):
                        if indices[i][j] != i:  # Not self
                            dist = distances[i][j]
                            if dist > 0:  # Not identical points
                                neighbor_min_dist = min(neighbor_min_dist, dist)

                    neighbor_radius = neighbor_min_dist / 2.0 if neighbor_min_dist != np.inf else 0.05
                    circles[i, 2] = min(boundary_radius, neighbor_radius, 0.05)

            except Exception:
                # Fallback to brute force method if KDTree fails
                distances = cdist(coords, coords)
                np.fill_diagonal(distances, float('inf'))

                for i in range(n_circles):
                    # Radius constrained by boundaries
                    boundary_radius = min(
                        circles[i, 0],  # left boundary
                        1 - circles[i, 0],  # right boundary
                        circles[i, 1],  # bottom boundary
                        1 - circles[i, 1]  # top boundary
                    )

                    # Radius constrained by neighbors
                    neighbor_min_dist = np.min(distances[i]) if n_circles > 1 else float('inf')
                    neighbor_radius = neighbor_min_dist / 2.0 if neighbor_min_dist != float('inf') else 0.05
                    circles[i, 2] = min(boundary_radius, neighbor_radius, 0.05)

            return circles

        for _ in range(pop_size):
            circles = force_directed_initialization(n_circles)

            # Add some diversity through small random perturbations
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

        # Clear spatial grid
        spatial_grid.clear()

        # First check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

            # Add to spatial grid for overlap checking
            spatial_grid.add_circle(i, x, y, r)

        # Then check overlap constraints using spatial grid
        for i in range(n):
            x1, y1, r1 = circles[i]

            # Get potential overlapping candidates from neighbors
            candidates = spatial_grid.get_neighbors(x1, y1, r1, use_cache=True)

            # Check actual overlaps
            for _, x2, y2, r2 in candidates:
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False

        return True

    def evaluate_fitness(circles: np.ndarray, generation: int = 0) -> float:
        """Enhanced fitness evaluation with progressive relaxation and quality awareness"""
        if not is_valid(circles):
            # Use progressive relaxation based on solution quality level
            # Early generations: focus on avoiding major constraint violations
            # Later generations: allow more flexibility to escape local optima

            penalty_weight = max(0.3, 1.0 - (generation / max_generations) * 0.7)

            # Dynamic penalty based on constraint violations
            total_penalty = 0

            # Boundary violations
            for i in range(len(circles)):
                x, y, r = circles[i]
                boundary_violation = 0
                if x - r < 0:
                    boundary_violation += abs(x - r) * 500  # Reduced penalty initially
                if x + r > 1:
                    boundary_violation += abs(x + r - 1) * 500  # Reduced penalty initially
                if y - r < 0:
                    boundary_violation += abs(y - r) * 500  # Reduced penalty initially
                if y + r > 1:
                    boundary_violation += abs(y + r - 1) * 500  # Reduced penalty initially
                total_penalty += boundary_violation * penalty_weight

            # Overlap violations - these are more serious
            # Use spatial grid for efficient overlap detection
            spatial_grid.clear()
            for i in range(len(circles)):
                x, y, r = circles[i]
                spatial_grid.add_circle(i, x, y, r)

            # Detect overlaps with early termination
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                candidates = spatial_grid.get_neighbors(x1, y1, r1, use_cache=False)
                for _, x2, y2, r2 in candidates:
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < r1 + r2:
                        overlap = max(0, r1 + r2 - distance)
                        total_penalty += overlap * 20000 * penalty_weight  # Increased penalty

            return -total_penalty

        # Valid configuration: maximize sum of radii with quality bonus
        sum_radii = np.sum(circles[:, 2])

        # Add bonus for geometric efficiency (density)
        total_area = np.sum(np.pi * circles[:, 2]**2)
        density_bonus = total_area  # Since area of square is 1
        bonus_scale = 0.1  # Small bonus scale

        return sum_radii + density_bonus * bonus_scale

    def mutate(circles: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply enhanced mutation with adaptive strategy"""
        mutated = deepcopy(circles)

        # Adaptive mutation rate with early aggressive exploration
        gen_progress = generation / max_generations
        mutation_rate = 0.2 - (0.15 * gen_progress)

        # Apply different mutation strategies
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Choose mutation type based on progress
                mutation_type = random.choices(
                    ['position', 'radius', 'both'],
                    weights=[0.5, 0.3, 0.2]
                )[0]

                if mutation_type == 'position':
                    # Position mutation with larger steps in early generations
                    step_size = 0.03 * (1 - gen_progress * 0.8)
                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))

                elif mutation_type == 'radius':
                    # Radius mutation with careful bounds
                    mutated[i, 2] += np.random.normal(0, 0.01)
                    mutated[i, 2] = max(0.001, mutated[i, 2])

                elif mutation_type == 'both':
                    # Combined position and radius mutation
                    step_size = 0.02 * (1 - gen_progress * 0.6)
                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)
                    mutated[i, 2] += np.random.normal(0, 0.005)
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))
                    mutated[i, 2] = max(0.001, mutated[i, 2])

        return mutated

    def constraint_aware_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring via crossover that preserves constraint quality"""
        child = deepcopy(parent1)
        n = len(parent1)

        # Perform uniform crossover
        for i in range(n):
            if random.random() < 0.5:
                child[i] = parent2[i].copy()

        # Apply constraint-aware refinement to maintain validity
        # First ensure boundary constraints are satisfied
        for i in range(n):
            x, y, r = child[i]
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            child[i] = [x, y, r]

        return child

    def tournament_selection(population: List[np.ndarray], k: int = 3, generation: int = 0) -> np.ndarray:
        """Select individual using tournament selection with fitness awareness"""
        selected = random.sample(population, k)
        # Use tournament selection with fitness evaluation adjusted for generation
        return max(selected, key=lambda x: evaluate_fitness(x, generation))

    def refine_and_optimize(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
        """Local optimization through gradient-descent-like adjustment"""
        refined = deepcopy(circles)

        # Simple local optimization for a few iterations
        for iteration in range(max_iterations):
            improved = False

            # Try local adjustments to each circle
            for i in range(len(refined)):
                old_x, old_y, old_r = refined[i]
                best_x, best_y, best_r = old_x, old_y, old_r
                best_fitness = evaluate_fitness(refined)

                # Test small position adjustments
                for dx in [-0.005, 0, 0.005]:
                    for dy in [-0.005, 0, 0.005]:
                        test_x = old_x + dx
                        test_y = old_y + dy
                        # Maintain boundary checks
                        test_x = max(old_r, min(1-old_r, test_x))
                        test_y = max(old_r, min(1-old_r, test_y))

                        # Temporarily update
                        refined[i] = [test_x, test_y, old_r]

                        if is_valid(refined):
                            new_fitness = evaluate_fitness(refined)
                            if new_fitness > best_fitness:
                                best_fitness = new_fitness
                                best_x, best_y = test_x, test_y
                                improved = True

                        # Restore
                        refined[i] = [old_x, old_y, old_r]

                # Test small radius adjustments
                for dr in [-0.002, 0, 0.002]:
                    test_r = old_r + dr
                    test_r = max(0.001, test_r)

                    # Temporarily update
                    refined[i] = [old_x, old_y, test_r]

                    if is_valid(refined):
                        new_fitness = evaluate_fitness(refined)
                        if new_fitness > best_fitness:
                            best_fitness = new_fitness
                            best_r = test_r
                            improved = True

                    # Restore
                    refined[i] = [old_x, old_y, old_r]

                # Apply best improvement
                refined[i] = [best_x, best_y, best_r]

            # Stop if no improvements were made
            if not improved:
                break

        return refined

    # Initialize population
    population = initialize_population(population_size, n_circles)

    # Evolve
    best_fitness = float('-inf')
    best_solution = None
    start_time = time.time()

    for generation in range(max_generations):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual, generation) for individual in population]

        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])
            # print(f"Generation {generation}: New best fitness = {best_fitness:.6f}")

        # Early termination check
        if time.time() - start_time > 55:  # Leave 5 seconds for cleanup
            break

        # Elitism: keep top 15%
        elite_count = max(1, population_size // 7)
        sorted_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elite = [population[i] for i in sorted_indices]

        # Create new population
        new_population = deepcopy(elite)

        # Generate offspring through constraint-aware crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = tournament_selection(population, generation=generation)
            parent2 = tournament_selection(population, generation=generation)

            # Constraint-aware crossover
            child = constraint_aware_crossover(parent1, parent2)

            # Mutation
            mutated_child = mutate(child, generation, max_generations)

            # Local optimization
            optimized_child = refine_and_optimize(mutated_child)

            new_population.append(optimized_child)

        population = new_population[:population_size]

        # Progress reporting
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to first individual if nothing was found
        return population[0]

# EVOLVE-BLOCK-END