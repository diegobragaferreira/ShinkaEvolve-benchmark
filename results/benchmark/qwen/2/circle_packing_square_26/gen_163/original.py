# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from typing import Tuple, List, Optional
import random
from copy import deepcopy
import time

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE_INITIAL = 0.15
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class EfficientCirclePackingOptimizer:
    def __init__(self):
        self.n_circles = 26

    def is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Check if the configuration satisfies all constraints efficiently."""
        if len(circles) != self.n_circles:
            return False

        # Check containment constraints using vectorized operations
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]

        # Check if any radius violates containment
        containment_check = (
            (radii <= x_coords) &
            (radii <= y_coords) &
            (radii <= 1 - x_coords) &
            (radii <= 1 - y_coords)
        )

        if not np.all(containment_check):
            return False

        # Optimized overlap check using spatial indexing
        if self.n_circles > 1:
            # Build KD tree for fast neighbor search
            tree = KDTree(circles[:, :2])

            # Find potential neighbors within 2*(max_radius) distance
            max_radius = np.max(radii)
            # For each circle, find neighbors that could potentially overlap
            for i in range(self.n_circles):
                # Only check against circles that could realistically overlap
                potential_neighbors = tree.query_ball_point(circles[i, :2], 2 * max_radius)
                # Skip self
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]

                for j in potential_neighbors:
                    # Calculate actual distance
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    min_dist = radii[i] + radii[j]
                    if dist < min_dist:
                        return False

        return True

    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])

    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Initialize population with valid configurations using enhanced multi-phase approach."""
        population = []

        # Generate diverse initial configurations with enhanced strategies
        for i in range(pop_size):
            if i == 0:
                # Phase 1: Strategic corner placement
                circles = self._create_strategic_initialization()
            elif i == 1:
                # Phase 2: Grid-based with slight perturbations
                circles = self._create_grid_initialization()
            elif i == 2:
                # Phase 3: Random with overlap avoidance
                circles = self._create_random_initialization()
            else:
                # Phase 4: Hybrid approach
                circles = self._create_hybrid_initialization()

            # Ensure validity and optimize
            if self.is_valid_configuration(circles):
                population.append(circles.copy())
            else:
                # Fallback to valid configuration
                circles = self._create_simple_initialization()
                if self.is_valid_configuration(circles):
                    population.append(circles.copy())

        return population

    def _create_simple_initialization(self) -> np.ndarray:
        """Create a simple but valid initial configuration."""
        circles = np.zeros((self.n_circles, 3))

        # Place in a simple grid pattern
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)

        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4  # Conservative radius
                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def _create_strategic_initialization(self) -> np.ndarray:
        """Create initialization with strategic corner placement."""
        circles = np.zeros((self.n_circles, 3))

        # Place key circles at strategic positions
        key_positions = [
            (0.1, 0.1, 0.05),      # bottom-left
            (0.9, 0.1, 0.05),      # bottom-right
            (0.1, 0.9, 0.05),      # top-left
            (0.9, 0.9, 0.05),      # top-right
            (0.5, 0.5, 0.1),       # center
        ]

        # Fill remaining positions with grid or random placement
        grid_size = int(np.ceil(np.sqrt(self.n_circles - len(key_positions))))
        spacing = 1.0 / (grid_size + 1)

        idx = 0
        for pos in key_positions:
            if idx >= self.n_circles:
                break
            circles[idx] = list(pos)
            idx += 1

        # Fill remaining positions with grid pattern
        remaining_count = self.n_circles - idx
        for i in range(remaining_count):
            row = i // grid_size
            col = i % grid_size
            x = (col + 1) * spacing
            y = (row + 1) * spacing
            r = spacing / 3
            # Add randomness
            x += np.random.uniform(-spacing/8, spacing/8)
            y += np.random.uniform(-spacing/8, spacing/8)
            r = max(0.01, min(r, x, y, 1-x, 1-y))
            circles[idx] = [x, y, r]
            idx += 1

        return circles

    def _create_grid_initialization(self) -> np.ndarray:
        """Create grid-based initialization with perturbations."""
        circles = np.zeros((self.n_circles, 3))

        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)

        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 3
                # Add small randomness to spread out
                x += np.random.uniform(-spacing/10, spacing/10)
                y += np.random.uniform(-spacing/10, spacing/10)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def _create_random_initialization(self) -> np.ndarray:
        """Create random initialization with overlap avoidance."""
        circles = np.zeros((self.n_circles, 3))

        for i in range(self.n_circles):
            attempts = 0
            while attempts < 100:
                # Random placement in unit square
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)

                # Radius based on distance to closest boundary
                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/2)

                # Check if it overlaps with existing circles
                overlap = False
                for j in range(i):
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < r + existing_r:
                        overlap = True
                        break

                if not overlap:
                    circles[i] = [x, y, r]
                    break
                attempts += 1

            if attempts >= 100:
                # Fallback to simple grid
                grid_size = int(np.ceil(np.sqrt(self.n_circles)))
                spacing = 1.0 / (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4
                circles[i] = [x, y, r]

        return circles

    def _create_hybrid_initialization(self) -> np.ndarray:
        """Create hybrid initialization combining multiple strategies."""
        circles = np.zeros((self.n_circles, 3))

        # Start with strategic positions
        key_positions = [
            (0.1, 0.1, 0.05),
            (0.9, 0.1, 0.05),
            (0.1, 0.9, 0.05),
            (0.9, 0.9, 0.05),
        ]

        idx = 0
        for pos in key_positions:
            if idx >= self.n_circles:
                break
            circles[idx] = list(pos)
            idx += 1

        # Fill remaining with grid pattern with small randomness
        remaining_count = self.n_circles - idx
        if remaining_count > 0:
            grid_size = int(np.ceil(np.sqrt(remaining_count)))
            spacing = 1.0 / (grid_size + 1)

            for i in range(remaining_count):
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 3
                # Add small randomness
                x += np.random.uniform(-spacing/12, spacing/12)
                y += np.random.uniform(-spacing/12, spacing/12)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def optimize_placement(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply efficient local optimization to improve placement."""
        n = len(circles)
        circles = circles.copy()

        # Calculate overlap count for severity classification
        def count_overlaps(config):
            if n <= 1:
                return 0
            # Use KDTree for efficient overlap detection
            tree = KDTree(config[:, :2])
            overlap_count = 0
            max_radius = np.max(config[:, 2])

            for i in range(n):
                # Find neighbors that could potentially overlap
                potential_neighbors = tree.query_ball_point(config[i, :2], 2 * max_radius)
                # Skip self
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]

                for j in potential_neighbors:
                    dist = np.sqrt((config[i, 0] - config[j, 0])**2 + (config[i, 1] - config[j, 1])**2)
                    min_dist = config[i, 2] + config[j, 2]
                    if dist < min_dist:
                        overlap_count += 1
            return overlap_count // 2  # Each overlap counted twice

        # Classify solution based on overlap severity
        overlap_count = count_overlaps(circles)
        if overlap_count == 0:
            # No overlaps - apply aggressive radius expansion
            max_refinement_iter = max_iter // 2
        elif overlap_count <= 3:
            # Low overlap - moderate refinement
            max_refinement_iter = max_iter // 3
        elif overlap_count <= 10:
            # Medium overlap - intensive refinement
            max_refinement_iter = max_iter // 2
        else:
            # High overlap - full refinement
            max_refinement_iter = max_iter

        # Optimized local refinement with early termination
        for iteration in range(max_refinement_iter):
            improved = False

            # Strategy 1: Try to expand radii with priority to less constrained circles
            # Sort by how much space they have left (ascending = more constrained)
            space_left = np.min([
                circles[:, 0],
                circles[:, 1],
                1 - circles[:, 0],
                1 - circles[:, 1]
            ], axis=0)

            # Process circles in order of least constrained first to maximize impact
            sorted_indices = np.argsort(space_left)

            for i in sorted_indices:
                original_radius = circles[i][2]
                original_x, original_y = circles[i][0], circles[i][1]

                # Calculate maximum possible radius for this circle
                max_radius = min(
                    original_x,
                    original_y,
                    1 - original_x,
                    1 - original_y
                )

                # Try to increase radius with more careful increment
                if overlap_count <= 3:
                    # Aggressive expansion for low overlap cases
                    increment = min(0.01, (max_radius - original_radius) / 2)
                else:
                    # Conservative expansion for high overlap cases
                    increment = min(0.002, (max_radius - original_radius) / 4)

                new_radius = min(original_radius + increment, max_radius)

                if new_radius > original_radius:
                    circles[i][2] = new_radius

                    # Check if valid configuration
                    if self.is_valid_configuration(circles):
                        improved = True
                    else:
                        # Revert if invalid
                        circles[i][2] = original_radius

            # Strategy 2: Position adjustments to resolve conflicts (only if we made progress)
            if improved or overlap_count > 3:
                # Apply position adjustments to resolve overlaps more systematically
                adjustment_multiplier = 1.0 if overlap_count <= 3 else 2.0
                adjustments = [
                    (0.001 * adjustment_multiplier, 0),
                    (-0.001 * adjustment_multiplier, 0),
                    (0, 0.001 * adjustment_multiplier),
                    (0, -0.001 * adjustment_multiplier),
                    (0.0005 * adjustment_multiplier, 0.0005 * adjustment_multiplier),
                    (-0.0005 * adjustment_multiplier, -0.0005 * adjustment_multiplier)
                ]

                # Process in reverse order of space constraint for more flexibility
                reversed_sorted_indices = sorted_indices[::-1]

                for i in reversed_sorted_indices:
                    original_x, original_y = circles[i][0], circles[i][1]

                    # Try adjustments to resolve overlaps
                    for dx, dy in adjustments:
                        new_x = np.clip(original_x + dx, 0, 1)
                        new_y = np.clip(original_y + dy, 0, 1)

                        if new_x != original_x or new_y != original_y:
                            circles[i][0] = new_x
                            circles[i][1] = new_y

                            if self.is_valid_configuration(circles):
                                improved = True
                                break
                            else:
                                # Revert if invalid
                                circles[i][0] = original_x
                                circles[i][1] = original_y

            # Early termination if no improvement
            if not improved:
                break

        return circles

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform constraint-aware crossover with overlap probability weighting."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        n = len(parent1)
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Calculate overlap probabilities for constraint-aware crossover
        def calculate_overlap_probability(circle1, circle2):
            """Estimate probability of overlap between two circles."""
            dist = np.sqrt((circle1[0] - circle2[0])**2 + (circle1[1] - circle2[1])**2)
            required_dist = circle1[2] + circle2[2]
            if dist < required_dist * 0.5:  # Very close - high probability
                return 1.0
            elif dist < required_dist * 1.5:  # Moderately close - moderate probability
                return 0.7
            else:
                return 0.3  # Far apart - low probability

        # Perform crossover with overlap awareness
        for i in range(n):
            # Weighted crossover based on distance between circles
            prob_overlap = calculate_overlap_probability(parent1[i], parent2[i])

            # Higher probability of swapping if circles are far apart
            if np.random.random() < (1 - prob_overlap) * 0.8:
                # Swap genes with higher probability for distant pairs
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()
            else:
                # Normal uniform crossover
                if np.random.random() < 0.5:
                    child1[i] = parent2[i].copy()
                    child2[i] = parent1[i].copy()
                else:
                    child1[i] = parent1[i].copy()
                    child2[i] = parent2[i].copy()

        # Ensure children are valid
        child1 = self.optimize_placement(child1)
        child2 = self.optimize_placement(child2)

        return child1, child2

    def mutate(self, circles: np.ndarray, mutation_rate: float = MUTATION_RATE_INITIAL) -> np.ndarray:
        """Apply adaptive mutation with improved strategies."""
        mutated = circles.copy()
        n = len(mutated)

        # Calculate diversity metric for adaptive mutation
        radii = mutated[:, 2]
        diversity = np.std(radii) if len(radii) > 1 else 0.0

        # Adaptive mutation rate based on diversity and generation
        adaptive_rate = mutation_rate
        if diversity > 0.1:
            # High diversity - maintain higher mutation rate
            adaptive_rate *= 1.2
        elif diversity < 0.05:
            # Low diversity - reduce mutation rate to exploit
            adaptive_rate *= 0.8

        for i in range(n):
            if np.random.random() < adaptive_rate:
                # Mutate either position or radius with adaptive probabilities
                if np.random.random() < 0.7:  # 70% chance to mutate position
                    # Mutate position with adaptive magnitude
                    mutation_magnitude = 0.03 * (1 - adaptive_rate)
                    mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, mutation_magnitude), 0, 1)
                    mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, mutation_magnitude), 0, 1)
                else:
                    # Mutate radius with smaller magnitude
                    mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.01), 0.01, 0.5)

        # Optimize the mutated configuration
        mutated = self.optimize_placement(mutated)

        return mutated

    def select_tournament(self, population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = TOURNAMENT_SIZE) -> int:
        """Select an individual using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        return winner_index

    def compute_fitness(self, circles: np.ndarray) -> float:
        """Compute fitness with penalty for invalid configurations."""
        if self.is_valid_configuration(circles):
            return self.calculate_sum_radii(circles)
        else:
            # Invalid configurations get very low fitness
            return 0.0

    def run_evolution(self) -> np.ndarray:
        """Run the complete evolutionary algorithm."""
        # Initialize population
        population = self.initialize_population(POPULATION_SIZE)

        if not population:
            # Fallback to simple initialization
            return self._create_simple_initialization()

        best_solution = None
        best_fitness = -1

        for generation in range(GENERATIONS):
            # Adjust mutation rate based on generation (adaptive)
            mutation_rate = max(MUTATION_RATE_INITIAL * (1 - generation / GENERATIONS), 0.01)

            # Evaluate fitness for all individuals (can be parallelized)
            fitnesses = [self.compute_fitness(circles) for circles in population]

            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Create new population
            new_population = []

            # Elitism: keep best individual
            new_population.append(best_solution.copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                # Tournament selection
                parent1_idx = self.select_tournament(population, fitnesses)
                parent2_idx = self.select_tournament(population, fitnesses)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)

                # Add children to new population
                new_population.extend([child1, child2])

            # Trim population to exact size
            population = new_population[:POPULATION_SIZE]

        # Return the best solution found
        if best_solution is None:
            # Fallback to a simple configuration if nothing worked
            return self._create_simple_initialization()

        return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = EfficientCirclePackingOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END