# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass
from enum import Enum

@dataclass
class Circle:
    """Data class representing a circle with x, y coordinates and radius"""
    x: float
    y: float
    r: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.r])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Circle':
        return cls(arr[0], arr[1], arr[2])

class ValidationStrategy(Enum):
    """Enumeration of validation strategies"""
    FAST = "fast"
    ACCURATE = "accurate"

class CircleValidator:
    """Validates circle configurations for containment and overlap constraints"""

    @staticmethod
    def validate_placement(circles: List[Circle], strategy: ValidationStrategy = ValidationStrategy.FAST) -> bool:
        """
        Validate that all circles are within bounds and don't overlap

        Args:
            circles: List of Circle objects
            strategy: Validation approach (FAST for performance, ACCURATE for thoroughness)

        Returns:
            bool: True if valid, False otherwise
        """
        if not circles:
            return True

        n = len(circles)

        # Fast containment check using vectorized operations
        positions = np.array([[c.x, c.y] for c in circles])
        radii = np.array([c.r for c in circles])

        # Vectorized containment check
        if np.any(radii <= 0) or np.any(positions[:, 0] < radii) or np.any(positions[:, 0] > 1 - radii) or \
           np.any(positions[:, 1] < radii) or np.any(positions[:, 1] > 1 - radii):
            return False

        # Use a unified approach for overlap checking that's efficient
        tree = cKDTree(positions)

        # For performance, we'll do batch processing rather than individual checks
        if strategy == ValidationStrategy.ACCURATE:
            # More thorough validation using batch queries for better performance
            indices_list = []
            for i in range(n):
                x, y, r = circles[i].x, circles[i].y, circles[i].r
                # Find nearby circles with a reasonable radius threshold
                indices = tree.query_ball_point([x, y], 2*r)
                indices_list.append(indices)

            # Check overlaps in batches
            for i in range(n):
                indices = indices_list[i]
                x, y, r = circles[i].x, circles[i].y, circles[i].r
                for j in indices:
                    if i != j:
                        x2, y2, r2 = circles[j].x, circles[j].y, circles[j].r
                        distance_sq = (x - x2)**2 + (y - y2)**2
                        if distance_sq < (r + r2)**2:
                            return False
        else:
            # Fast validation - only check a subset of potential collisions
            # Use a more efficient approach: precompute all pairs that might collide
            max_radius = np.max(radii)
            if max_radius > 0:
                # Query all pairs within a reasonable distance
                query_radius = max_radius * 2.5
                try:
                    pairs = tree.query_pairs(query_radius)
                    for i, j in pairs:
                        x1, y1, r1 = circles[i].x, circles[i].y, circles[i].r
                        x2, y2, r2 = circles[j].x, circles[j].y, circles[j].r
                        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if distance < r1 + r2:
                            return False
                except Exception:
                    # Fallback to direct pairwise checking for safety
                    for i in range(n):
                        x, y, r = circles[i].x, circles[i].y, circles[i].r
                        indices = tree.query_ball_point([x, y], 3*r)
                        for j in indices:
                            if i != j:
                                x2, y2, r2 = circles[j].x, circles[j].y, circles[j].r
                                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                                if distance < r + r2:
                                    return False

        return True

class CircleInitializer:
    """Initializes populations of circle configurations"""

    @staticmethod
    def create_multi_scale_grid_initialization(num_circles: int) -> List[Circle]:
        """
        Create a multi-scale grid-based initialization for circles

        Args:
            num_circles: Number of circles to initialize

        Returns:
            List[Circle]: Initialized circles
        """
        circles = []

        # Try different grid configurations to find a good initial setup
        configs = [
            (int(np.ceil(np.sqrt(num_circles))), int(np.ceil(num_circles / np.ceil(np.sqrt(num_circles))))),
            (5, 6),
            (6, 5),
            (4, 7),
            (7, 4)
        ]

        best_config = None
        best_score = -np.inf

        for rows, cols in configs:
            if rows * cols >= num_circles:
                # Create positions
                grid_positions = []
                for i in range(rows):
                    for j in range(cols):
                        if len(grid_positions) >= num_circles:
                            break
                        x = (j + 0.5) / cols
                        y = (i + 0.5) / rows
                        grid_positions.append((x, y))

                if len(grid_positions) >= num_circles:
                    # Calculate score for this configuration
                    score = 0
                    temp_circles = []
                    for i in range(num_circles):
                        x, y = grid_positions[i]
                        # Add small random perturbation
                        x += (random.random() - 0.5) * 0.03
                        y += (random.random() - 0.5) * 0.03
                        r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                        temp_circles.append(Circle(x, y, r))
                        score += r

                    if score > best_score:
                        best_score = score
                        best_config = (grid_positions, rows, cols)

        if best_config:
            grid_positions, rows, cols = best_config
            for i in range(num_circles):
                x, y = grid_positions[i]
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles.append(Circle(x, y, r))
        else:
            # Fallback to random initialization
            for i in range(num_circles):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles.append(Circle(x, y, r))

        return circles

    @classmethod
    def create_initial_population(cls, pop_size: int, num_circles: int) -> List[List[Circle]]:
        """
        Create initial population with enhanced initialization

        Args:
            pop_size: Size of population to generate
            num_circles: Number of circles per individual

        Returns:
            List[List[Circle]]: Population of circle configurations
        """
        population = []

        for _ in range(pop_size):
            # Create initial configuration
            circles = cls.create_multi_scale_grid_initialization(num_circles)

            # Apply local optimization to improve initial placement
            optimized = CircleOptimizer.optimize_local(circles)

            # Store valid configurations
            if CircleValidator.validate_placement(optimized):
                population.append(optimized)
            else:
                # Fallback to simple initialization if needed
                fallback_circles = []
                for i in range(num_circles):
                    x = random.uniform(0.05, 0.95)
                    y = random.uniform(0.05, 0.95)
                    r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                    fallback_circles.append(Circle(x, y, r))
                population.append(fallback_circles)

        return population

class CircleOptimizer:
    """Performs local optimization to maximize sum of radii"""

    @staticmethod
    def optimize_local(circles: List[Circle]) -> List[Circle]:
        """
        Apply enhanced local improvement to maximize sum of radii

        Args:
            circles: List of Circle objects

        Returns:
            List[Circle]: Optimized circles
        """
        # Convert to numpy array for performance and use more sophisticated optimization
        circles_copy = [Circle(c.x, c.y, c.r) for c in circles]  # Deep copy

        # Use a more aggressive optimization approach with better heuristics
        max_iterations = 300  # Increased for more thorough search
        improvement_count = 0

        # Pre-compute spatial structure for faster neighbor lookups
        positions = np.array([[c.x, c.y] for c in circles_copy])
        radii = np.array([c.r for c in circles_copy])
        tree = cKDTree(positions)

        for iteration in range(max_iterations):
            improved = False

            # Process circles in a hybrid order: first random, then systematic
            if iteration < 100:  # Most iterations: random order for exploration
                circle_order = list(range(len(circles_copy)))
                random.shuffle(circle_order)
            else:  # Later iterations: systematic for exploitation
                circle_order = list(range(len(circles_copy)))

            for i in circle_order:
                c = circles_copy[i]
                x, y, r = c.x, c.y, c.r

                # Calculate maximum possible radius at current position
                max_r = min(x, 1-x, y, 1-y)

                # Try to increase radius as much as possible while avoiding overlaps
                if max_r > r + 1e-6:  # Use a slightly larger tolerance
                    # Try to increase radius aggressively
                    new_r = max_r

                    # Quick check with fast spatial indexing
                    indices = tree.query_ball_point([x, y], 2*new_r)
                    valid_radius = True
                    for j in indices:
                        if i != j:
                            other_c = circles_copy[j]
                            x2, y2, r2 = other_c.x, other_c.y, other_c.r
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < new_r + r2:
                                valid_radius = False
                                break

                    if valid_radius:
                        circles_copy[i].r = new_r
                        improved = True
                        # Update KDTree for next iteration since we changed a radius
                        positions[i] = [x, y]
                        tree = cKDTree(positions)
                    else:
                        # Try a more strategic approach for smaller increases
                        step_size = 0.001
                        test_r = min(r + step_size, max_r)
                        while test_r > r + 1e-6 and not valid_radius:
                            # Check again using spatial indexing
                            indices = tree.query_ball_point([x, y], 2*test_r)
                            valid_radius = True
                            for j in indices:
                                if i != j:
                                    other_c = circles_copy[j]
                                    x2, y2, r2 = other_c.x, other_c.y, other_c.r
                                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                                    if distance < test_r + r2:
                                        valid_radius = False
                                        break

                            if valid_radius:
                                circles_copy[i].r = test_r
                                improved = True
                                positions[i] = [x, y]  # Update position for tree rebuild
                                tree = cKDTree(positions)
                                break
                            else:
                                test_r -= step_size

                # Strategic position adjustment
                if iteration % 3 == 0 and improved:
                    # Look for better positions using neighborhood search
                    original_x, original_y = x, y
                    original_r = r

                    # Try nearby positions systematically
                    best_x, best_y = x, y
                    best_r = r
                    best_score = r  # Just the radius for now

                    # Generate candidate positions around current center
                    candidates = []
                    step = 0.01  # Smaller steps for better precision
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            candidates.append((x + dx, y + dy))

                    # Also consider moving in direction of maximum space
                    # Calculate repulsive forces from neighbors (but more efficiently)
                    force_x, force_y = 0.0, 0.0
                    indices = tree.query_ball_point([x, y], 3*original_r)
                    for j in indices:
                        if i != j:
                            other_c = circles_copy[j]
                            x2, y2, r2 = other_c.x, other_c.y, other_c.r
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if distance < original_r + r2:
                                # Repulsive force
                                dx = x - x2
                                dy = y - y2
                                dist = np.sqrt(dx*dx + dy*dy)
                                if dist > 0:
                                    force_x += dx / dist * (original_r + r2 - distance) * 0.01
                                    force_y += dy / dist * (original_r + r2 - distance) * 0.01

                    # If there's a strong repulsive force, move in the opposite direction
                    force_magnitude = np.sqrt(force_x**2 + force_y**2)
                    if force_magnitude > 0.001:
                        # Add force-directed movement to candidates
                        candidates.append((x - force_x * 0.05, y - force_y * 0.05))

                    # Evaluate each candidate
                    for new_x, new_y in candidates:
                        # Ensure it's within bounds
                        if new_x >= original_r and new_x <= 1 - original_r and \
                           new_y >= original_r and new_y <= 1 - original_r:

                            # Quick overlap check with neighbors using spatial indexing
                            indices = tree.query_ball_point([new_x, new_y], 2*original_r)
                            valid_pos = True
                            for j in indices:
                                if i != j:
                                    other_c = circles_copy[j]
                                    x2, y2, r2 = other_c.x, other_c.y, other_c.r
                                    distance = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                                    if distance < original_r + r2:
                                        valid_pos = False
                                        break

                            if valid_pos:
                                # This position is valid, we'll evaluate it
                                # For simplicity, just evaluate the radius increase potential
                                # We're optimizing for sum of radii, so we can increase radius here too
                                new_max_r = min(new_x, 1-new_x, new_y, 1-new_y)
                                if new_max_r > best_r:
                                    best_r = new_max_r
                                    best_x = new_x
                                    best_y = new_y
                                    best_score = new_max_r  # Score based on increased radius potential

                    # Apply the best improvement
                    if best_x != x or best_y != y or best_r > r:
                        circles_copy[i].x = best_x
                        circles_copy[i].y = best_y
                        circles_copy[i].r = best_r
                        improved = True
                        # Update the spatial index
                        positions[i] = [best_x, best_y]
                        tree = cKDTree(positions)

            # Early termination conditions
            if not improved:
                improvement_count += 1
                if improvement_count > 20:  # Early termination if no improvement for 20 iterations
                    break
            else:
                improvement_count = 0

        return circles_copy

class EvolutionEngine:
    """Handles evolutionary operations like selection, crossover, and mutation"""

    @staticmethod
    def tournament_selection(population: List[List[Circle]], fitnesses: List[float],
                             tournament_size: int) -> List[Circle]:
        """
        Select individual using tournament selection

        Args:
            population: List of circle configurations
            fitnesses: Corresponding fitness values
            tournament_size: Number of individuals to compete

        Returns:
            List[Circle]: Selected individual
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_index].copy()

    @staticmethod
    def uniform_crossover(parent1: List[Circle], parent2: List[Circle]) -> List[Circle]:
        """
        Perform uniform crossover between two parents

        Args:
            parent1: First parent configuration
            parent2: Second parent configuration

        Returns:
            List[Circle]: Offspring configuration
        """
        child = []
        n = len(parent1)

        # Use vectorized operations for crossover
        mask = np.random.random(n) < 0.5

        for i in range(n):
            if mask[i]:
                child.append(Circle(parent1[i].x, parent1[i].y, parent1[i].r))
            else:
                child.append(Circle(parent2[i].x, parent2[i].y, parent2[i].r))

        return child

    @staticmethod
    def adaptive_mutation(individual: List[Circle], generation: int,
                         max_generations: int) -> List[Circle]:
        """
        Apply adaptive mutation to an individual

        Args:
            individual: Current individual to mutate
            generation: Current generation number
            max_generations: Total number of generations

        Returns:
            List[Circle]: Mutated individual
        """
        mutated = [Circle(c.x, c.y, c.r) for c in individual]  # Deep copy
        n = len(mutated)

        # Adaptive mutation rate based on generation progress
        mutation_rate_start = 0.15
        mutation_rate_end = 0.015
        mutation_rate = mutation_rate_start - (mutation_rate_start - mutation_rate_end) * (generation / max_generations)

        # Vectorized mutation
        mutation_mask = np.random.random(n) < mutation_rate

        # Mutate positions and radii separately
        for i in range(n):
            if mutation_mask[i]:
                # Mutate either position or radius
                if random.random() < 0.5:
                    # Mutate position
                    mutated[i].x += (random.random() - 0.5) * 0.1
                    mutated[i].y += (random.random() - 0.5) * 0.1

                    # Keep within bounds
                    mutated[i].x = np.clip(mutated[i].x, 0.01, 0.99)
                    mutated[i].y = np.clip(mutated[i].y, 0.01, 0.99)
                else:
                    # Mutate radius
                    mutated[i].r += (random.random() - 0.5) * 0.05

                    # Ensure positive radius
                    mutated[i].r = max(0.001, mutated[i].r)

        # Repair any constraint violations
        repaired = CircleRepair.repair_constraints(mutated)
        return repaired

class CircleRepair:
    """Repairs constraint violations in circle configurations"""

    @staticmethod
    def repair_constraints(circles: List[Circle]) -> List[Circle]:
        """
        Repair any constraint violations

        Args:
            circles: List of Circle objects

        Returns:
            List[Circle]: Repaired circles
        """
        repaired = [Circle(c.x, c.y, c.r) for c in circles]  # Deep copy
        n = len(repaired)

        # Ensure all circles are within bounds and have positive radius
        for i in range(n):
            c = repaired[i]
            c.r = max(0.001, c.r)
            c.x = np.clip(c.x, c.r, 1-c.r)
            c.y = np.clip(c.y, c.r, 1-c.r)

        # Apply constraint repair with early termination
        for _ in range(10):  # Reduced iterations for performance
            any_changes = False
            for i in range(n):
                c = repaired[i]
                x, y, r = c.x, c.y, c.r
                # Check overlaps and adjust if needed
                for j in range(n):
                    if i != j:
                        other_c = repaired[j]
                        x2, y2, r2 = other_c.x, other_c.y, other_c.r
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        min_distance = r + r2
                        if distance < min_distance:
                            # Move circle away from overlapping one
                            dx = x2 - x
                            dy = y2 - y
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > 0:
                                factor = (min_distance - distance) / dist * 0.1
                                x += dx * factor
                                y += dy * factor
                                any_changes = True

                # Keep within bounds
                r = max(0.001, r)
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                repaired[i] = Circle(x, y, r)

            if not any_changes:
                break

        return repaired

class CirclePack26:
    """Main controller for the circle packing optimization process"""

    def __init__(self):
        self.best_solution = None
        self.best_fitness = -np.inf
        self.start_time = time.time()
        self.max_time = 60  # seconds
        self.population_size = 100
        self.generations = 200
        self.tournament_size = 3
        self.elitism_count = 5
        self.max_attempts = 1000

    def evaluate_fitness(self, individual: List[Circle]) -> float:
        """
        Evaluate fitness of an individual

        Args:
            individual: Circle configuration

        Returns:
            float: Fitness value (sum of radii if valid, penalty otherwise)
        """
        if CircleValidator.validate_placement(individual):
            return sum(c.r for c in individual)
        else:
            return -1000000  # Penalty for invalid placements

    def should_terminate(self) -> bool:
        """Check if we should terminate due to time limit"""
        return time.time() - self.start_time > self.max_time * 0.95  # Leave some buffer

    def evolve(self) -> List[Circle]:
        """
        Main evolutionary loop with early termination

        Returns:
            List[Circle]: Best solution found
        """
        # Create initial population with enhanced initialization
        population = CircleInitializer.create_initial_population(self.population_size, 26)

        # Track improvement for early termination
        last_best = -np.inf
        no_improvement_count = 0
        max_no_improvement = 30  # Increased for more patience

        # Add a parameter to control when to switch to more intensive local optimization
        local_optimization_generation_threshold = 50

        for generation in range(self.generations):
            # Check early termination
            if self.should_terminate():
                break

            # Evaluate fitness of each individual
            fitnesses = [self.evaluate_fitness(individual) for individual in population]

            # Track best solution so far
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_solution = population[max_fitness_idx].copy()
                last_best = self.best_fitness
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early termination if no improvement for too long
            if no_improvement_count > max_no_improvement:
                break

            # Elitism: keep best individuals - but make sure we preserve at least one good solution
            elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
            elites = [population[i].copy() for i in elite_indices]

            # Create new population
            new_population = elites.copy()

            # Generate offspring through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1 = EvolutionEngine.tournament_selection(population, fitnesses, self.tournament_size)
                parent2 = EvolutionEngine.tournament_selection(population, fitnesses, self.tournament_size)

                # Crossover
                child = EvolutionEngine.uniform_crossover(parent1, parent2)

                # Mutation with adaptive parameters
                child = EvolutionEngine.adaptive_mutation(child, generation, self.generations)

                # Apply enhanced local optimization to offspring based on generation
                if generation >= local_optimization_generation_threshold:
                    # Apply more thorough local optimization
                    child = CircleOptimizer.optimize_local(child)

                # Add to new population
                new_population.append(child)

            population = new_population[:self.population_size]

        # Final refinement of the best solution
        if self.best_solution is not None:
            # Apply the most thorough optimization to the final best solution
            refined_best = CircleOptimizer.optimize_local(self.best_solution)
            # Validate and return the best of the two
            if CircleValidator.validate_placement(refined_best) and \
               sum(c.r for c in refined_best) > self.best_fitness:
                self.best_solution = refined_best
                self.best_fitness = sum(c.r for c in refined_best)

        # Return the best solution found
        if self.best_solution is not None:
            return self.best_solution
        else:
            # Fallback to final population if no valid solution was found
            return population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    packer = CirclePack26()
    result = packer.evolve()
    return np.array([[c.x, c.y, c.r] for c in result])

# EVOLVE-BLOCK-END