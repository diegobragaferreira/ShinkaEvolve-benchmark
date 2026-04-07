# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from typing import Tuple, List, Optional
import random
from copy import deepcopy
import time

# Global constants
POPULATION_SIZE = 120  # Optimized population size
GENERATIONS = 120      # Balanced generations
MUTATION_RATE_INITIAL = 0.15  # Controlled initial mutation
CROSSOVER_RATE = 0.85  # High crossover for diversity
TOURNAMENT_SIZE = 6    # Balanced tournament size
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class EfficientCirclePacker:
    def __init__(self):
        self.n_circles = 26
        self.epsilon = 1e-8

    def is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Check if the configuration satisfies all constraints efficiently."""
        if len(circles) != self.n_circles:
            return False

        # Vectorized containment check
        radii = circles[:, 2]
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]

        # Check containment constraints
        containment_check = (
            (radii <= x_coords) &
            (radii <= y_coords) &
            (radii <= 1 - x_coords) &
            (radii <= 1 - y_coords)
        )

        if not np.all(containment_check):
            return False

        # Optimized overlap check using KDTree with early termination
        if self.n_circles > 1:
            tree = KDTree(circles[:, :2])
            max_radius = np.max(radii) if len(radii) > 0 else 0
            
            # Process each circle once with early termination
            for i in range(self.n_circles):
                # Find potential overlapping neighbors within 2*max_radius distance
                potential_neighbors = tree.query_ball_point(circles[i, :2], 2 * max_radius)
                # Skip self
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]

                if len(potential_neighbors) == 0:
                    continue

                # Check actual overlaps with squared distances
                for j in potential_neighbors:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist_sq = (radii[i] + radii[j]) * (radii[i] + radii[j])

                    if dist_sq < min_dist_sq:
                        return False
                        
        return True

    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])

    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Initialize population with enhanced multi-phase approach."""
        population = []

        # Multi-phase initialization
        for i in range(pop_size):
            if i == 0:
                circles = self._create_strategic_initialization()  # Best starting point
            elif i == 1:
                circles = self._create_hexagonal_initialization()
            elif i == 2:
                circles = self._create_spiral_initialization()
            elif i == 3:
                circles = self._create_grid_initialization()
            elif i == 4:
                circles = self._create_random_initialization()
            else:
                # Hybrid with more refinement
                circles = self._create_hybrid_initialization_with_refinement()

            # Ensure validity
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
        grid_size = int(np.ceil(np.sqrt(self.n_circles)))
        spacing = 1.0 / (grid_size + 1)
        
        idx = 0
        for row in range(grid_size):
            for col in range(grid_size):
                if idx >= self.n_circles:
                    break
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 4
                circles[idx] = [x, y, r]
                idx += 1
        return circles

    def _create_strategic_initialization(self) -> np.ndarray:
        """Create initialization with strategic corner and center placement."""
        circles = np.zeros((self.n_circles, 3))
        
        # Key positions with increased spacing
        key_positions = [
            (0.1, 0.1, 0.07),
            (0.9, 0.1, 0.07),
            (0.1, 0.9, 0.07),
            (0.9, 0.9, 0.07),
            (0.5, 0.5, 0.13),
        ]
        
        # Additional positions
        additional_positions = [
            (0.25, 0.25, 0.05),
            (0.75, 0.25, 0.05),
            (0.25, 0.75, 0.05),
            (0.75, 0.75, 0.05),
            (0.5, 0.25, 0.04),
            (0.5, 0.75, 0.04),
            (0.25, 0.5, 0.04),
            (0.75, 0.5, 0.04),
        ]
        
        idx = 0
        positions_to_place = key_positions + additional_positions
        
        for pos in positions_to_place:
            if idx >= self.n_circles:
                break
            circles[idx] = list(pos)
            idx += 1
            
        # Fill remaining with grid
        remaining_count = self.n_circles - idx
        if remaining_count > 0:
            grid_rows = int(np.ceil(np.sqrt(remaining_count)))
            grid_cols = int(np.ceil(remaining_count / grid_rows))
            spacing_x = 0.8 / (grid_cols + 1)
            spacing_y = 0.8 / (grid_rows + 1)
            
            for i in range(remaining_count):
                row = i // grid_cols
                col = i % grid_cols
                x = 0.1 + (col + 1) * spacing_x
                y = 0.1 + (row + 1) * spacing_y
                r = min(spacing_x, spacing_y) * 0.35
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1
                
        return circles

    def _create_hexagonal_initialization(self) -> np.ndarray:
        """Create hexagonal packing for dense initial configuration."""
        circles = np.zeros((self.n_circles, 3))
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))
        hex_radius = 0.15
        width = 2 * hex_radius
        height = hex_radius * np.sqrt(3)
        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if count >= self.n_circles:
                    break
                x_offset = (i % 2) * (width / 2)
                x = x_offset + j * width + random.uniform(-0.01, 0.01)
                y = i * height + random.uniform(-0.01, 0.01)
                x = (x / (cols * width)) * 0.9 + 0.05
                y = (y / (rows * height)) * 0.9 + 0.05
                max_radius = min(x, 1-x, y, 1-y) * 0.8
                r = max(0.01, min(max_radius, random.uniform(0.02, 0.1)))
                circles[count] = [x, y, r]
                count += 1
        return circles

    def _create_spiral_initialization(self) -> np.ndarray:
        """Create a spiral arrangement for even spatial coverage."""
        circles = np.zeros((self.n_circles, 3))
        angle_step = 2 * np.pi / 5
        radius_step = 0.4 / self.n_circles
        
        for i in range(self.n_circles):
            angle = i * angle_step + random.uniform(-0.1, 0.1)
            radius = i * radius_step + 0.1 + random.uniform(-0.02, 0.02)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            distance_from_center = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            max_radius = min(x, 1-x, y, 1-y) * 0.8
            r = max(0.01, min(max_radius, random.uniform(0.02, 0.1)))
            circles[i] = [x, y, r]
        return circles

    def _create_grid_initialization(self) -> np.ndarray:
        """Create grid-based initialization."""
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
                r = spacing / 2.5
                x += np.random.uniform(-spacing/8, spacing/8)
                y += np.random.uniform(-spacing/8, spacing/8)
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
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/1.5)
                
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
                # Fallback to grid
                grid_size = int(np.ceil(np.sqrt(self.n_circles)))
                spacing = 1.0 / (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 3.5
                circles[i] = [x, y, r]
                
        return circles

    def _create_hybrid_initialization_with_refinement(self) -> np.ndarray:
        """Create hybrid initialization with additional optimization."""
        circles = np.zeros((self.n_circles, 3))
        
        # Start with strategic positions
        key_positions = [
            (0.1, 0.1, 0.06),
            (0.9, 0.1, 0.06),
            (0.1, 0.9, 0.06),
            (0.9, 0.9, 0.06),
            (0.5, 0.5, 0.12),
        ]
        
        idx = 0
        for pos in key_positions:
            if idx >= self.n_circles:
                break
            circles[idx] = list(pos)
            idx += 1
            
        # Fill remaining with grid pattern
        remaining_count = self.n_circles - idx
        if remaining_count > 0:
            grid_size = int(np.ceil(np.sqrt(remaining_count)))
            spacing = 1.0 / (grid_size + 1)
            
            for i in range(remaining_count):
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 2.5
                x += np.random.uniform(-spacing/10, spacing/10)
                y += np.random.uniform(-spacing/10, spacing/10)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1
                
        return circles

    def optimize_placement(self, circles: np.ndarray, max_iter: int = 120) -> np.ndarray:
        """Apply efficient local optimization to improve placement."""
        n = len(circles)
        circles = circles.copy()

        # Count overlaps for priority determination
        def count_overlaps(config):
            if n <= 1:
                return 0
            tree = KDTree(config[:, :2])
            overlap_count = 0
            max_radius = np.max(config[:, 2])
            
            for i in range(n):
                potential_neighbors = tree.query_ball_point(config[i, :2], 2 * max_radius)
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]
                
                for j in potential_neighbors:
                    dx = config[i, 0] - config[j, 0]
                    dy = config[i, 1] - config[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist_sq = (config[i, 2] + config[j, 2]) * (config[i, 2] + config[j, 2])
                    if dist_sq < min_dist_sq:
                        overlap_count += 1
            return overlap_count // 2

        overlap_count = count_overlaps(circles)
        
        # Set refinement iterations based on overlap severity
        if overlap_count == 0:
            max_refinement_iter = max_iter
        elif overlap_count <= 5:
            max_refinement_iter = max_iter * 0.8
        elif overlap_count <= 10:
            max_refinement_iter = max_iter * 0.6
        else:
            max_refinement_iter = max_iter * 0.4

        # Local refinement loop
        for iteration in range(int(max_refinement_iter)):
            improved = False
            
            # Strategy 1: Expand radii for under-constrained circles
            # Calculate space constraints for each circle
            space_constraints = np.column_stack([
                circles[:, 0],
                circles[:, 1],
                1 - circles[:, 0],
                1 - circles[:, 1]
            ])
            space_min = np.min(space_constraints, axis=1)
            sorted_by_constraint = np.argsort(space_min)  # Most constrained first
            
            # Process circles in order of constraint level
            for i in sorted_by_constraint:
                original_radius = circles[i][2]
                original_x, original_y = circles[i][0], circles[i][1]
                max_radius = min(original_x, original_y, 1 - original_x, 1 - original_y)
                
                # Adaptive increment based on overlap count
                if overlap_count <= 2:
                    increment = min(0.02, (max_radius - original_radius) / 1.2)
                elif overlap_count <= 5:
                    increment = min(0.015, (max_radius - original_radius) / 1.5)
                elif overlap_count <= 10:
                    increment = min(0.01, (max_radius - original_radius) / 2)
                else:
                    increment = min(0.005, (max_radius - original_radius) / 3)
                    
                # Test incremental improvements
                test_increments = [increment, increment*0.7, increment*0.4, increment*0.2]
                best_increment = 0
                best_radius = original_radius

                for inc in test_increments:
                    new_radius = min(original_radius + inc, max_radius)
                    if new_radius > best_radius:
                        circles[i][2] = new_radius
                        if self.is_valid_configuration(circles):
                            best_radius = new_radius
                            best_increment = inc
                        else:
                            circles[i][2] = original_radius

                if best_increment > 0:
                    circles[i][2] = best_radius
                    improved = True

            # Strategy 2: Positional adjustments to resolve overlaps
            if improved or overlap_count > 2:
                adjustment_multiplier = 1.0 if overlap_count <= 5 else 2.0
                
                adjustments = [
                    (0.003 * adjustment_multiplier, 0),
                    (-0.003 * adjustment_multiplier, 0),
                    (0, 0.003 * adjustment_multiplier),
                    (0, -0.003 * adjustment_multiplier),
                    (0.002 * adjustment_multiplier, 0.002 * adjustment_multiplier),
                    (-0.002 * adjustment_multiplier, -0.002 * adjustment_multiplier),
                    (0.002 * adjustment_multiplier, -0.002 * adjustment_multiplier),
                    (-0.002 * adjustment_multiplier, 0.002 * adjustment_multiplier),
                ]

                # Process in order of importance (most constrained first)
                for i in sorted_by_constraint:
                    original_x, original_y = circles[i][0], circles[i][1]
                    
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
                                circles[i][0] = original_x
                                circles[i][1] = original_y

            if not improved:
                break
                
        return circles

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform constraint-aware crossover."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        n = len(parent1)
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Weighted crossover based on separation distance
        for i in range(n):
            dist = np.sqrt((parent1[i, 0] - parent2[i, 0])**2 + (parent1[i, 1] - parent2[i, 1])**2)
            required_dist = parent1[i, 2] + parent2[i, 2]
            
            # Higher probability of swapping for far-apart circles
            swap_prob = 0.5 + 0.3 * max(0, (required_dist - dist) / (required_dist + self.epsilon))
            if np.random.random() < swap_prob:
                child1[i] = parent2[i].copy()
                child2[i] = parent1[i].copy()
            else:
                child1[i] = parent1[i].copy()
                child2[i] = parent2[i].copy()

        # Ensure valid configurations
        child1 = self.optimize_placement(child1)
        child2 = self.optimize_placement(child2)

        return child1, child2

    def mutate(self, circles: np.ndarray, mutation_rate: float = MUTATION_RATE_INITIAL) -> np.ndarray:
        """Apply adaptive constraint-aware mutation."""
        mutated = circles.copy()
        n = len(mutated)

        # Identify overlapping circles
        overlapping_pairs = []
        if n > 1:
            tree = KDTree(mutated[:, :2])
            max_radius = np.max(mutated[:, 2]) if len(mutated[:, 2]) > 0 else 0

            for i in range(n):
                potential_neighbors = tree.query_ball_point(mutated[i, :2], 2 * max_radius)
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]

                for j in potential_neighbors:
                    dx = mutated[i, 0] - mutated[j, 0]
                    dy = mutated[i, 1] - mutated[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist_sq = (mutated[i, 2] + mutated[j, 2]) * (mutated[i, 2] + mutated[j, 2])

                    if dist_sq < min_dist_sq:
                        overlapping_pairs.append((i, j))

        # Count overlaps per circle
        overlap_count = np.zeros(n)
        for i, j in overlapping_pairs:
            overlap_count[i] += 1
            overlap_count[j] += 1

        # Apply mutation
        for i in range(n):
            if np.random.random() < mutation_rate:
                overlap_factor = min(1.0, overlap_count[i] * 0.3)
                mutation_prob = 0.75 * (1 - overlap_factor) + 0.25 * overlap_factor

                if np.random.random() < mutation_prob:  # Mutate position
                    mutation_magnitude = 0.04 * (1 - mutation_rate) * (1 + overlap_factor * 0.5)
                    mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, mutation_magnitude), 0, 1)
                    mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, mutation_magnitude), 0, 1)
                else:  # Mutate radius
                    radius_mutation_magnitude = 0.015 * (1 + overlap_factor * 0.5)
                    mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, radius_mutation_magnitude), 0.01, 0.5)

        # Optimize mutated configuration
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
            return 0.0

    def run_evolution(self) -> np.ndarray:
        """Run the complete evolutionary algorithm."""
        # Initialize population
        population = self.initialize_population(POPULATION_SIZE)

        if not population:
            return self._create_simple_initialization()

        best_solution = None
        best_fitness = -1
        stagnant_generations = 0

        for generation in range(GENERATIONS):
            # Adaptive mutation rate
            mutation_rate = max(MUTATION_RATE_INITIAL * (1 - generation / GENERATIONS) ** 0.8, 0.02)

            # Evaluate fitness
            fitnesses = [self.compute_fitness(circles) for circles in population]

            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            # Early termination
            if stagnant_generations > 15:
                break

            # Create new population with elitism
            new_population = []
            elite_size = max(10, POPULATION_SIZE // 8)
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < POPULATION_SIZE:
                parent1_idx = self.select_tournament(population, fitnesses)
                parent2_idx = self.select_tournament(population, fitnesses)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)

                new_population.extend([child1, child2])

            # Trim to exact size
            population = new_population[:POPULATION_SIZE]

        # Return best solution
        if best_solution is None:
            return self._create_simple_initialization()

        return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    optimizer = EfficientCirclePacker()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END