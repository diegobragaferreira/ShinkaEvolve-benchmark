# EVOLVE-BLOCK-STARTimport numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class SpatialIndexer:
    """Efficient spatial indexing for circle collision detection"""

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.grid_cells = {}

    def _get_grid_key(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to grid cell indices"""
        return (int(x * self.grid_size), int(y * self.grid_size))

    def build_index(self, circles: np.ndarray) -> dict:
        """Build spatial grid index for efficient neighbor queries"""
        self.grid_cells.clear()
        for i, (x, y, r) in enumerate(circles):
            cell = self._get_grid_key(x, y)
            if cell not in self.grid_cells:
                self.grid_cells[cell] = []
            self.grid_cells[cell].append(i)
        return self.grid_cells

    def get_neighbors(self, x: float, y: float, radius: float) -> List[int]:
        """Get candidate neighbors within a search radius"""
        neighbors = []
        center_cell = self._get_grid_key(x, y)

        # Check nearby cells in a 3x3 grid around center
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.grid_cells:
                    neighbors.extend(self.grid_cells[cell])

        return neighbors

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained within the unit square."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def check_overlap_fast(circles: np.ndarray, spatial_indexer: SpatialIndexer) -> bool:
    """Fast overlap checking using spatial indexing."""
    n = len(circles)
    if n <= 1:
        return True

    # Build index for efficient neighbor lookup
    spatial_indexer.build_index(circles)

    # Check each circle against its neighbors
    for i, (xi, yi, ri) in enumerate(circles):
        # Get nearby candidates using spatial index
        candidates = spatial_indexer.get_neighbors(xi, yi, ri)

        for j in candidates:
            if i != j:
                xj, yj, rj = circles[j]
                distance = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if distance < ri + rj - 1e-8:  # Validity threshold
                    return False

    return True

def check_overlap(circles: np.ndarray) -> bool:
    """Check if any circles overlap (slower but more robust version)."""
    n = len(circles)
    # Calculate pairwise distances
    positions = circles[:, :2]
    radii = circles[:, 2]

    # Create distance matrix
    distances = cdist(positions, positions)

    # Check for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            if dist < radii[i] + radii[j]:
                return False
    return True

def fitness(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with better starting configurations."""
    population = []

    # Start with more structured initialization
    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = rows
        spacing_x = 0.8 / cols
        spacing_y = 0.8 / rows
        base_radius = 0.02

        # Fill with hexagonal pattern
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                # Offset odd rows for hexagonal pattern
                x_base = 0.1 + (j + 0.5 * (i % 2)) * spacing_x
                y_base = 0.1 + i * spacing_y

                x = max(0.01, min(0.99, x_base + np.random.uniform(-spacing_x/4, spacing_x/4)))
                y = max(0.01, min(0.99, y_base + np.random.uniform(-spacing_y/4, spacing_y/4)))
                r = base_radius + np.random.uniform(0, 0.02)

                circles[idx] = [x, y, r]
                idx += 1

        # Enforce boundary constraints
        for i in range(n_circles):
            x, y, r = circles[i]
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            circles[i] = [x, y, r]

        population.append(circles)

    return population

def mutate(circles: np.ndarray, generation: int, total_generations: int, base_mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate a circle configuration with adaptive mutation rate."""
    mutated = circles.copy()
    n = len(mutated)

    # Adaptive mutation rate that decreases over time
    mutation_rate = base_mutation_rate * (1 - generation / total_generations) + 0.01

    for i in range(n):
        if random.random() < mutation_rate:
            # Choose which component to mutate with probabilities favoring position over radius
            choice = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]

            if choice == 0:  # X coordinate - larger mutation for exploration
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, 0.03),
                                      mutated[i, 2], 1 - mutated[i, 2])
            elif choice == 1:  # Y coordinate - larger mutation for exploration
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, 0.03),
                                      mutated[i, 2], 1 - mutated[i, 2])
            else:  # Radius - smaller mutation for fine-tuning
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, 0.015), 0.001, 0.4)

    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parent solutions."""
    # Simple uniform crossover
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Crossover points for each circle
    for i in range(len(parent1)):
        if random.random() < 0.5:
            # Swap position and radius between parents
            child1[i, 0], child2[i, 0] = child2[i, 0], child1[i, 0]
            child1[i, 1], child2[i, 1] = child2[i, 1], child1[i, 1]
            child1[i, 2], child2[i, 2] = child2[i, 2], child1[i, 2]

    return child1, child2

def select_parents(population: List[np.ndarray], fitnesses: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Select two parents using tournament selection."""
    tournament_size = 5  # Larger tournament for stronger selection pressure
    # Select first parent
    idx1 = random.randint(0, len(population)-1)
    best_idx = idx1
    best_fit = fitnesses[idx1]
    for _ in range(tournament_size - 1):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    parent1 = population[best_idx]

    # Select second parent
    idx2 = random.randint(0, len(population)-1)
    best_idx = idx2
    best_fit = fitnesses[idx2]
    for _ in range(tournament_size - 1):
        idx = random.randint(0, len(population)-1)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    parent2 = population[best_idx]

    return parent1, parent2

def optimize_circles_evolutionary(max_generations: int = 1000, pop_size: int = 50) -> np.ndarray:
    """Evolutionary optimization for circle packing with spatial indexing."""
    n = 26
    spatial_indexer = SpatialIndexer()

    # Initialize population
    population = initialize_population(pop_size, n)
    best_solution = None
    best_fitness = -float('inf')

    for generation in range(max_generations):
        # Evaluate fitness for all individuals
        fitnesses = []
        for circles in population:
            if check_containment(circles) and check_overlap_fast(circles, spatial_indexer):
                fit = fitness(circles)
                fitnesses.append(fit)
            else:
                fitnesses.append(-1000)  # Penalize invalid solutions

        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Print progress every 100 generations
        if generation % 100 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")

        # Create new population through selection, crossover, and mutation
        new_population = []

        # Keep best individuals (elitism)
        sorted_indices = np.argsort(fitnesses)[::-1][:pop_size//4]
        for idx in sorted_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1, parent2 = select_parents(population, fitnesses)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, generation, max_generations)
            child2 = mutate(child2, generation, max_generations)

            # Ensure validity
            if check_containment(child1) and check_overlap_fast(child1, spatial_indexer):
                new_population.append(child1)
            else:
                # Try to fix if invalid
                new_population.append(parent1.copy())  # Fallback to parent

            if len(new_population) < pop_size and check_containment(child2) and check_overlap_fast(child2, spatial_indexer):
                new_population.append(child2)
            elif len(new_population) < pop_size:
                new_population.append(parent2.copy())  # Fallback to parent

        population = new_population[:pop_size]

    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run evolutionary optimization
    circles = optimize_circles_evolutionary(max_generations=500, pop_size=30)

    # Final validation
    if circles is None or not check_containment(circles) or not check_overlap_fast(circles, SpatialIndexer()):
        # Fallback to a simple arrangement if optimization failed
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.3

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1

        # Adjust last few circles to fit
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.01]

    return circles


# EVOLVE-BLOCK-END