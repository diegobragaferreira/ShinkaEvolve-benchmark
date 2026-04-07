# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import math

class SpatialIndexer:
    """Efficient spatial indexing for circle collision detection"""

    def __init__(self, grid_size: int = 15):
        self.grid_size = grid_size
        self.grid_cells = {}
        self.kdtree = None
        self.circle_positions = None

    def _get_grid_key(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates to grid cell indices"""
        return (int(x * self.grid_size), int(y * self.grid_size))

    def build_index(self, circles: np.ndarray) -> dict:
        """Build spatial grid index for efficient neighbor queries"""
        self.grid_cells.clear()
        self.circle_positions = [(x, y) for x, y, r in circles]
        for i, (x, y, r) in enumerate(circles):
            cell = self._get_grid_key(x, y)
            if cell not in self.grid_cells:
                self.grid_cells[cell] = []
            self.grid_cells[cell].append(i)

        # Also build KDTree for more efficient nearest neighbor searches
        if len(self.circle_positions) > 0:
            try:
                self.kdtree = cKDTree(self.circle_positions)
            except:
                self.kdtree = None
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

        # Additional optimization: use KDTree for radius-based queries when available
        if self.kdtree is not None:
            try:
                query_radius = 2 * (radius + 0.01)
                indices = self.kdtree.query_ball_point([x, y], query_radius)
                # Merge with grid-based neighbors but avoid duplicates
                neighbor_set = set(neighbors)
                for idx in indices:
                    if idx not in neighbor_set:
                        neighbors.append(idx)
            except:
                pass  # Fall back to grid-based approach if KDTree fails

        return neighbors

class ConstraintValidator:
    """Handles all constraint validation and enforcement"""

    @staticmethod
    def validate_containment(circles: np.ndarray) -> bool:
        """Check if all circles are fully contained in the unit square"""
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        return True

    @staticmethod
    def validate_overlap(circles: np.ndarray, spatial_indexer: Optional[SpatialIndexer] = None) -> bool:
        """Check for circle overlaps using spatial indexing for efficiency"""
        if len(circles) <= 1:
            return True

        # Use spatial indexing for efficiency
        if spatial_indexer is not None:
            positions = [(x, y) for x, y, r in circles]
            try:
                tree = cKDTree(positions)
                # Check each circle against its neighbors
                for i, (xi, yi, ri) in enumerate(circles):
                    # Query nearby points with search radius
                    indices = tree.query_ball_point([xi, yi], 2 * (ri + 0.01))
                    for j in indices:
                        if i != j:
                            xj, yj, rj = circles[j]
                            distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                            if distance < (ri + rj - 1e-8):
                                return False
            except:
                # Fallback to brute force for edge cases
                for i in range(len(circles)):
                    for j in range(i+1, len(circles)):
                        xi, yi, ri = circles[i]
                        xj, yj, rj = circles[j]
                        distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                        if distance < (ri + rj - 1e-8):
                            return False
        else:
            # Fallback to brute force for small populations
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    if distance < (ri + rj - 1e-8):
                        return False

        return True

    @staticmethod
    def enforce_bounds(circles: np.ndarray) -> np.ndarray:
        """Enforce boundary constraints by adjusting positions and radii"""
        result = circles.copy()

        for i in range(len(result)):
            x, y, r = result[i]

            # Ensure circle fits in the unit square
            max_radius = min(x, 1-x, y, 1-y)
            r = min(r, max_radius)
            r = max(0.001, min(0.49, r))

            # Clamp coordinates to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))

            result[i] = [x, y, r]

        return result

class FitnessEvaluator:
    """Evaluates fitness with adaptive penalty system"""

    def __init__(self, boundary_weight: float = 5000.0, overlap_weight: float = 50000.0):
        self.boundary_weight = boundary_weight
        self.overlap_weight = overlap_weight

    def evaluate(self, circles: np.ndarray, spatial_indexer: Optional[SpatialIndexer] = None, 
                generation: int = 0, total_generations: int = 100) -> float:
        """Evaluate fitness with adaptive penalty scaling"""
        # Progressive penalty scaling - become stricter in later generations
        progress = min(generation / total_generations, 1.0)
        penalty_scale = 1.0 + progress * 4.0  # Scale from 1 to 5

        # Check constraints
        if not ConstraintValidator.validate_containment(circles):
            penalty = self._compute_boundary_penalty(circles, penalty_scale)
            return -penalty

        if not ConstraintValidator.validate_overlap(circles, spatial_indexer):
            penalty = self._compute_overlap_penalty(circles, penalty_scale)
            return -penalty

        # Valid solution - return sum of radii
        return float(np.sum(circles[:, 2]))

    def _compute_boundary_penalty(self, circles: np.ndarray, penalty_scale: float = 1.0) -> float:
        """Compute penalty based on boundary violations"""
        penalty = 0.0

        for x, y, r in circles:
            # Calculate boundary violations with squared penalties for stronger effect
            left_violation = max(0, r - x)
            right_violation = max(0, r - (1 - x))
            bottom_violation = max(0, r - y)
            top_violation = max(0, r - (1 - y))
            
            # Apply penalty with squared violations for strong penalties
            penalty += self.boundary_weight * penalty_scale * (
                left_violation**2 + right_violation**2 + 
                bottom_violation**2 + top_violation**2
            )

        return penalty

    def _compute_overlap_penalty(self, circles: np.ndarray, penalty_scale: float = 1.0) -> float:
        """Compute penalty based on overlap violations"""
        penalty = 0.0

        # Compute actual overlap amounts for more accurate penalty
        n = len(circles)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                if distance < (r1 + r2):
                    overlap = (r1 + r2 - distance)
                    penalty += overlap * self.overlap_weight * penalty_scale

        return penalty

class VoronoiInitialization:
    """Implements Voronoi-like initialization with better spatial distribution"""

    @staticmethod
    def generate_voronoi_distribution(n: int) -> np.ndarray:
        """Generate high-quality Voronoi-like distribution"""
        circles = np.zeros((n, 3))

        # Create a more sophisticated Voronoi-like pattern
        # Use a hexagonal grid for better spatial distribution
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        # Hexagonal grid spacing
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows

        # Generate hexagonal grid with offsets
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) < n:
                    # Offset odd rows for hexagonal pattern
                    x_base = 0.05 + (j + 0.5 * (i % 2)) * spacing_x
                    y_base = 0.05 + i * spacing_y
                    
                    # Add randomness to avoid perfect grid
                    x = max(0.01, min(0.99, x_base + random.uniform(-spacing_x/6, spacing_x/6)))
                    y = max(0.01, min(0.99, y_base + random.uniform(-spacing_y/6, spacing_y/6)))
                    
                    points.append((x, y))

        # Fill remaining points with random distribution
        while len(points) < n:
            points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))

        # Assign circles with varying radii
        for i in range(n):
            x, y = points[i]
            # Initial radius - vary more significantly
            r = 0.015 + random.uniform(0, 0.06)
            circles[i] = [x, y, r]

        return circles

    @staticmethod
    def generate_grid_distribution(n: int) -> np.ndarray:
        """Generate grid-based distribution with better spacing"""
        circles = np.zeros((n, 3))

        # Create a grid with better spacing and randomness
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        spacing_x = 0.85 / (cols + 1)
        spacing_y = 0.85 / (rows + 1)

        # Generate grid points with better distribution
        for i in range(n):
            row = i // cols
            col = i % cols
            
            # Add randomness to spacing to improve distribution
            x = 0.075 + (col + 1) * spacing_x + random.uniform(-spacing_x/4, spacing_x/4)
            y = 0.075 + (row + 1) * spacing_y + random.uniform(-spacing_y/4, spacing_y/4)
            
            # Ensure within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Radius with variation
            r = 0.02 + random.uniform(0, 0.05)
            circles[i] = [x, y, r]

        return circles

class EvolutionaryOptimizer:
    """Main evolutionary optimization class with enhanced strategies"""

    def __init__(self, population_size: int = 120, generations: int = 250):
        self.population_size = population_size
        self.generations = generations
        self.spatial_indexer = SpatialIndexer()
        self.validator = ConstraintValidator()
        self.evaluator = FitnessEvaluator()
        self.init_strategy = VoronoiInitialization()

    def initialize_population(self, n: int) -> np.ndarray:
        """Initialize population with hybrid approach"""
        population = []

        # Use hybrid initialization - mix of Voronoi-like and grid approaches
        for i in range(self.population_size):
            # Alternate between initialization methods with different probabilities
            rand_val = random.random()
            if rand_val < 0.6:  # 60% chance for Voronoi-like
                circles = self.init_strategy.generate_voronoi_distribution(n)
            elif rand_val < 0.9:  # 30% chance for grid
                circles = self.init_strategy.generate_grid_distribution(n)
            else:  # 10% chance for random with better bounds
                circles = np.zeros((n, 3))
                for j in range(n):
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                    r = random.uniform(0.01, 0.08)
                    circles[j] = [x, y, r]

            # Apply constraint enforcement
            circles = self.validator.enforce_bounds(circles)
            
            # Add noise to ensure diversity in initial population
            for j in range(n):
                if random.random() < 0.15:  # 15% chance to perturb
                    circles[j, 0] = max(0.01, min(0.99, circles[j, 0] + random.gauss(0, 0.015)))
                    circles[j, 1] = max(0.01, min(0.99, circles[j, 1] + random.gauss(0, 0.015)))
                    circles[j, 2] = max(0.001, min(0.49, circles[j, 2] + random.gauss(0, 0.008)))

            population.append(circles)

        return np.array(population)

    def mutate(self, circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
        """Enhanced mutation with adaptive rates and specialized operators"""
        mutated = circles.copy()

        # Adaptive mutation rate based on generation progress
        # Start high and decrease gradually
        progress = generation / total_generations
        mutation_rate = 0.25 * (1 - progress * 0.8) + 0.05

        n = len(mutated)

        # Mutate circles with adaptive rate
        for i in range(n):
            if random.random() < mutation_rate:
                # Choose which component to mutate with probabilities
                choice = random.choices([0, 1, 2], weights=[0.45, 0.45, 0.1])[0]

                if choice == 0:  # X coordinate - medium mutation
                    mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, 0.025)))
                elif choice == 1:  # Y coordinate - medium mutation
                    mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, 0.025)))
                else:  # Radius - smaller mutation for fine-tuning
                    mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.015)))

        # Apply refinement steps
        mutated = self._refine_after_mutation(mutated)
        mutated = self.validator.enforce_bounds(mutated)

        return mutated

    def _refine_after_mutation(self, circles: np.ndarray) -> np.ndarray:
        """Refine mutated individuals to resolve potential issues"""
        # Build spatial index for efficient overlap checking
        self.spatial_indexer.build_index(circles)

        # Resolve overlaps using iterative approach with early termination
        for iteration in range(20):  # Slightly more iterations for better resolution
            resolved = False
            for i in range(len(circles)):
                x, y, r = circles[i]

                # Find overlapping circles using spatial indexing
                neighbors = self.spatial_indexer.get_neighbors(x, y, r)

                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x - x2)**2 + (y - y2)**2)

                        # If overlap exists, adjust positions
                        if distance < (r + r2 - 1e-8):
                            # Move circles apart along displacement vector
                            dx = x2 - x
                            dy = y2 - y
                            dist = max(1e-8, distance)

                            # Normalize and move apart
                            dx /= dist
                            dy /= dist

                            # Compute overlap amount and move apart
                            move_amount = (r + r2 - dist) * 0.5

                            # Dynamic damping factor based on iteration count
                            damping = 0.3 * (1 - iteration / 20.0) + 0.1

                            # Apply movement with damping
                            circles[i, 0] -= dx * move_amount * damping * 0.3
                            circles[i, 1] -= dy * move_amount * damping * 0.3
                            circles[j, 0] += dx * move_amount * damping * 0.3
                            circles[j, 1] += dy * move_amount * damping * 0.3
                            resolved = True

            # If no changes made, stop iteration
            if not resolved:
                break

        return circles

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Improved crossover with adaptive strategy"""
        # Lower crossover probability for more diversity
        if random.random() > 0.8:
            # Return one parent if crossover doesn't happen
            return parent1.copy() if random.random() < 0.5 else parent2.copy()

        n = len(parent1)
        child = np.zeros_like(parent1)

        # Multi-point crossover with more sophisticated segmentation
        crossover_points = sorted(random.sample(range(1, n), min(6, n//3)))
        crossover_points = [0] + crossover_points + [n]

        # Alternate between parents for segments
        for i in range(len(crossover_points) - 1):
            start = crossover_points[i]
            end = crossover_points[i + 1]
            if i % 2 == 0:
                child[start:end] = parent1[start:end].copy()
            else:
                child[start:end] = parent2[start:end].copy()

        # Apply refinement to ensure validity
        return self._refine_after_crossover(child)

    def _refine_after_crossover(self, child: np.ndarray) -> np.ndarray:
        """Refine offspring after crossover"""
        # Force boundary enforcement
        child = self.validator.enforce_bounds(child)

        # Quick overlap resolution with more comprehensive approach
        self.spatial_indexer.build_index(child)
        if not ConstraintValidator.validate_overlap(child, self.spatial_indexer):
            # More thorough resolution for immediate fixes
            for i in range(len(child)):
                for j in range(i+1, len(child)):
                    x1, y1, r1 = child[i]
                    x2, y2, r2 = child[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    if distance < (r1 + r2 - 1e-8):
                        # Advanced adjustment with geometric considerations
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = max(1e-8, distance)

                        dx /= dist
                        dy /= dist

                        move_amount = (r1 + r2 - dist) / 2.0

                        # Apply movement with damping
                        damping = 0.35
                        child[i, 0] -= dx * move_amount * damping
                        child[i, 1] -= dy * move_amount * damping
                        child[j, 0] += dx * move_amount * damping
                        child[j, 1] += dy * move_amount * damping

        return child

    def evolve(self, n: int) -> np.ndarray:
        """Main evolution loop with improvements"""
        # Initialize population
        population = self.initialize_population(n)

        # Track best fitness
        best_fitness_history = []

        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = []
            for individual in population:
                fitness = self.evaluator.evaluate(individual, self.spatial_indexer, gen, self.generations)
                fitnesses.append(fitness)

            # Track best
            best_fitness = max(fitnesses)
            best_fitness_history.append(best_fitness)

            # Print progress every 25 generations
            if gen % 25 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness:.6f}")

            # Selection, crossover, and mutation
            new_population = []

            # Elitism: keep best individuals (stronger elitism for better convergence)
            sorted_indices = np.argsort(fitnesses)[::-1][:self.population_size // 3]
            for idx in sorted_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection (larger tournament for more pressure)
                parent1 = self._tournament_select(population, fitnesses, tournament_size=8)
                parent2 = self._tournament_select(population, fitnesses, tournament_size=8)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child, gen, self.generations)

                new_population.append(child)

            # Trim to exact population size
            population = new_population[:self.population_size]

        # Return the best individual
        final_fitnesses = []
        for individual in population:
            fitness = self.evaluator.evaluate(individual, self.spatial_indexer, self.generations, self.generations)
            final_fitnesses.append(fitness)

        best_index = np.argmax(final_fitnesses)
        best_solution = population[best_index]

        return best_solution

    def _tournament_select(self, population: np.ndarray, fitnesses: List[float], tournament_size: int = 8) -> np.ndarray:
        """Tournament selection with larger tournament size for better selection pressure"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    optimizer = EvolutionaryOptimizer(population_size=120, generations=250)
    circles = optimizer.evolve(26)

    return circles

# EVOLVE-BLOCK-END