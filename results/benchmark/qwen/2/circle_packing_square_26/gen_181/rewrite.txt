# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from typing import Tuple, List, Optional
import random
from copy import deepcopy
import time

# Global constants
POPULATION_SIZE = 150  # Increased population size for better diversity
GENERATIONS = 100     # More generations for better convergence
MUTATION_RATE_INITIAL = 0.2  # Higher initial mutation for better exploration
CROSSOVER_RATE = 0.9  # Higher crossover rate for more recombination
TOURNAMENT_SIZE = 7   # Larger tournament for better selection pressure
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class AdaptiveMultiPhaseOptimizer:
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

        # Optimized overlap check using spatial indexing with early termination and squared distances
        if self.n_circles > 1:
            # Use KDTree for efficient neighbor search
            tree = KDTree(circles[:, :2])

            # Precompute max radius to avoid repeated calls
            max_radius = np.max(radii) if len(radii) > 0 else 0
            max_radius_sq = max_radius * max_radius

            # Pre-compute all squared distances for better performance
            for i in range(self.n_circles):
                # Find neighbors that could potentially overlap
                potential_neighbors = tree.query_ball_point(circles[i, :2], 2 * max_radius)
                # Skip self
                potential_neighbors = [idx for idx in potential_neighbors if idx != i]

                # Early termination: if no potential neighbors, skip detailed check
                if len(potential_neighbors) == 0:
                    continue

                # Check overlaps with potential neighbors using squared distances
                for j in potential_neighbors:
                    # Use squared distances to avoid expensive sqrt operations
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
        """Initialize population with valid configurations using enhanced multi-phase approach."""
        population = []

        # Generate diverse initial configurations with enhanced strategies
        for i in range(pop_size):
            if i == 0:
                # Phase 1: Hexagonal packing for dense initial configuration
                circles = self._create_hexagonal_initialization()
            elif i == 1:
                # Phase 2: Spiral arrangement for good coverage
                circles = self._create_spiral_initialization()
            elif i == 2:
                # Phase 3: Strategic corner placement with more positions
                circles = self._create_strategic_initialization()
            elif i == 3:
                # Phase 4: Grid with enhanced perturbation
                circles = self._create_grid_initialization()
            elif i == 4:
                # Phase 5: Random with improved overlap avoidance
                circles = self._create_random_initialization()
            else:
                # Phase 6: Hybrid approach with multiple strategies
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

    def _create_hexagonal_initialization(self) -> np.ndarray:
        """Create hexagonal packing for dense initial configuration."""
        circles = np.zeros((self.n_circles, 3))
        
        # Create a hexagonal grid pattern
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))
        
        # Hexagon parameters
        hex_radius = 0.15  # Adjust for better packing
        width = 2 * hex_radius
        height = hex_radius * np.sqrt(3)
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= self.n_circles:
                    break
                    
                # Offset every other row for better packing
                x_offset = (i % 2) * (width / 2)
                x = x_offset + j * width + random.uniform(-0.01, 0.01)
                y = i * height + random.uniform(-0.01, 0.01)
                
                # Scale to fit within unit square
                x = (x / (cols * width)) * 0.9 + 0.05
                y = (y / (rows * height)) * 0.9 + 0.05
                
                # Ensure it fits within bounds
                max_radius = min(x, 1-x, y, 1-y) * 0.8
                r = max(0.01, min(max_radius, random.uniform(0.02, 0.1)))
                
                circles[count] = [x, y, r]
                count += 1
                
        return circles

    def _create_spiral_initialization(self) -> np.ndarray:
        """Create a spiral arrangement for even spatial coverage."""
        circles = np.zeros((self.n_circles, 3))
        
        # Create spiral pattern
        angle_step = 2 * np.pi / 5
        radius_step = 0.4 / self.n_circles
        
        for i in range(self.n_circles):
            angle = i * angle_step + random.uniform(-0.1, 0.1)
            radius = i * radius_step + 0.1 + random.uniform(-0.02, 0.02)
            
            # Convert to Cartesian coordinates
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Keep within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Radius based on distance from center
            distance_from_center = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            max_radius = min(x, 1-x, y, 1-y) * 0.8
            r = max(0.01, min(max_radius, random.uniform(0.02, 0.1)))
            
            circles[i] = [x, y, r]
            
        return circles

    def _create_strategic_initialization(self) -> np.ndarray:
        """Create initialization with strategic corner placement and better spacing."""
        circles = np.zeros((self.n_circles, 3))

        # Place key circles at strategic positions with better separation
        key_positions = [
            (0.1, 0.1, 0.06),      # bottom-left
            (0.9, 0.1, 0.06),      # bottom-right
            (0.1, 0.9, 0.06),      # top-left
            (0.9, 0.9, 0.06),      # top-right
            (0.5, 0.5, 0.12),       # center
        ]

        # Additional strategic positions to improve distribution
        additional_positions = [
            (0.25, 0.25, 0.05),
            (0.75, 0.25, 0.05),
            (0.25, 0.75, 0.05),
            (0.75, 0.75, 0.05),
            (0.5, 0.25, 0.04),
            (0.5, 0.75, 0.04),
            (0.25, 0.5, 0.04),
            (0.75, 0.5, 0.04),
            (0.3, 0.3, 0.03),
            (0.7, 0.3, 0.03),
            (0.3, 0.7, 0.03),
            (0.7, 0.7, 0.03),
        ]

        # Fill positions with strategic placement
        idx = 0
        positions_to_place = key_positions + additional_positions

        for pos in positions_to_place:
            if idx >= self.n_circles:
                break
            circles[idx] = list(pos)
            idx += 1

        # Fill remaining positions with better grid spacing
        remaining_count = self.n_circles - idx
        if remaining_count > 0:
            # Use a more sophisticated grid approach
            grid_rows = int(np.ceil(np.sqrt(remaining_count)))
            grid_cols = int(np.ceil(remaining_count / grid_rows))

            # Adjust spacing to ensure better distribution
            spacing_x = 0.8 / (grid_cols + 1)
            spacing_y = 0.8 / (grid_rows + 1)

            for i in range(remaining_count):
                row = i // grid_cols
                col = i % grid_cols
                x = 0.1 + (col + 1) * spacing_x
                y = 0.1 + (row + 1) * spacing_y
                # Reduce radius slightly to account for potential overlap with other circles
                r = min(spacing_x, spacing_y) * 0.35
                # Add randomness to break perfect symmetry
                x += np.random.uniform(-spacing_x/8, spacing_x/8)
                y += np.random.uniform(-spacing_y/8, spacing_y/8)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def _create_grid_initialization(self) -> np.ndarray:
        """Create grid-based initialization with enhanced perturbations."""
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
                r = spacing / 2.5  # Slightly larger radius
                # Add medium randomness to spread out
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
                # Random placement in unit square
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)

                # Radius based on distance to closest boundary
                min_dist = min(x, y, 1-x, 1-y)
                r = np.random.uniform(0.01, min_dist/1.5)  # Tighter constraint for better initial fit

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
                r = spacing / 3.5
                circles[i] = [x, y, r]

        return circles

    def _create_hybrid_initialization(self) -> np.ndarray:
        """Create hybrid initialization combining multiple strategies."""
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

        # Fill remaining with better grid pattern with small randomness
        remaining_count = self.n_circles - idx
        if remaining_count > 0:
            grid_size = int(np.ceil(np.sqrt(remaining_count)))
            spacing = 1.0 / (grid_size + 1)

            for i in range(remaining_count):
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                r = spacing / 2.5  # Larger radius for better initial packing
                # Add randomness
                x += np.random.uniform(-spacing/10, spacing/10)
                y += np.random.uniform(-spacing/10, spacing/10)
                r = max(0.01, min(r, x, y, 1-x, 1-y))
                circles[idx] = [x, y, r]
                idx += 1

        return circles

    def optimize_placement(self, circles: np.ndarray, max_iter: int = 150) -> np.ndarray:
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
                    # Use squared distances for performance
                    dx = config[i, 0] - config[j, 0]
                    dy = config[i, 1] - config[j, 1]
                    dist_sq = dx * dx + dy * dy
                    min_dist_sq = (config[i, 2] + config[j, 2]) * (config[i, 2] + config[j, 2])
                    if dist_sq < min_dist_sq:
                        overlap_count += 1
            return overlap_count // 2  # Each overlap counted twice

        # Classify solution based on overlap severity
        overlap_count = count_overlaps(circles)
        
        # Adjust refinement iterations based on overlap severity
        if overlap_count == 0:
            # No overlaps - apply aggressive radius expansion with more iterations
            max_refinement_iter = max_iter
        elif overlap_count <= 3:
            # Low overlap - moderate refinement 
            max_refinement_iter = max_iter * 0.8
        elif overlap_count <= 10:
            # Medium overlap - intensive refinement
            max_refinement_iter = max_iter * 0.6
        else:
            # High overlap - focused refinement
            max_refinement_iter = max_iter * 0.4

        # Optimized local refinement with early termination and improved strategies
        for iteration in range(int(max_refinement_iter)):
            improved = False

            # Strategy 1: Try to expand radii with improved prioritization
            # Calculate space constraints for each circle
            space_constraints = np.column_stack([
                circles[:, 0],
                circles[:, 1],
                1 - circles[:, 0],
                1 - circles[:, 1]
            ])

            # How much space each circle has in each direction (smaller is more constrained)
            space_min = np.min(space_constraints, axis=1)

            # Process circles ordered by constraint level (most constrained first)
            # This prioritizes fixing the circles that are most likely to cause issues
            sorted_indices = np.argsort(space_min)

            # Also consider overlap severity for each circle
            overlap_severity = np.zeros(n)
            if overlap_count > 0:
                tree = KDTree(circles[:, :2])
                max_radius = np.max(circles[:, 2])
                for i in range(n):
                    potential_neighbors = tree.query_ball_point(circles[i, :2], 2 * max_radius)
                    potential_neighbors = [idx for idx in potential_neighbors if idx != i]
                    overlap_severity[i] = len(potential_neighbors)

            # Combine constraints and overlap severity for better prioritization
            # Lower priority = more constrained or more problematic
            priority_scores = space_min * 0.6 + overlap_severity * 0.4

            # Process circles with highest priority (lowest score) first
            sorted_by_priority = np.argsort(priority_scores)

            # Process circles with adaptive increments based on overlap count
            for i in sorted_by_priority:
                original_radius = circles[i][2]
                original_x, original_y = circles[i][0], circles[i][1]

                # Calculate maximum possible radius for this circle
                max_radius = min(
                    original_x,
                    original_y,
                    1 - original_x,
                    1 - original_y
                )

                # Adaptive increment based on overlap count and constraint level
                if overlap_count <= 2:
                    # Very low overlap - aggressive expansion
                    increment = min(0.02, (max_radius - original_radius) / 1.2)
                elif overlap_count <= 5:
                    # Low overlap - moderate expansion
                    increment = min(0.015, (max_radius - original_radius) / 1.5)
                elif overlap_count <= 10:
                    # Moderate overlap - conservative expansion
                    increment = min(0.01, (max_radius - original_radius) / 2)
                else:
                    # High overlap - very cautious expansion
                    increment = min(0.005, (max_radius - original_radius) / 3)

                # Try various increments to see what works best
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
                            circles[i][2] = original_radius  # Revert

                if best_increment > 0:
                    circles[i][2] = best_radius
                    improved = True

            # Strategy 2: Position adjustments with more systematic approach
            if improved or overlap_count > 2:
                # Apply position adjustments to resolve overlaps more systematically
                adjustment_multiplier = 1.0 if overlap_count <= 5 else 2.0

                # Comprehensive adjustment patterns
                adjustments = [
                    (0.003 * adjustment_multiplier, 0),
                    (-0.003 * adjustment_multiplier, 0),
                    (0, 0.003 * adjustment_multiplier),
                    (0, -0.003 * adjustment_multiplier),
                    (0.002 * adjustment_multiplier, 0.002 * adjustment_multiplier),
                    (-0.002 * adjustment_multiplier, -0.002 * adjustment_multiplier),
                    (0.002 * adjustment_multiplier, -0.002 * adjustment_multiplier),
                    (-0.002 * adjustment_multiplier, 0.002 * adjustment_multiplier),
                    (0.001 * adjustment_multiplier, 0.001 * adjustment_multiplier),
                    (-0.001 * adjustment_multiplier, -0.001 * adjustment_multiplier),
                ]

                # For high overlap cases, process circles that are most likely to be causing issues
                if overlap_count > 5:
                    # Focus on circles with many neighbors
                    tree = KDTree(circles[:, :2])
                    max_radius = np.max(circles[:, 2])
                    high_impact_indices = []
                    for i in range(n):
                        potential_neighbors = tree.query_ball_point(circles[i, :2], 2 * max_radius)
                        potential_neighbors = [idx for idx in potential_neighbors if idx != i]
                        if len(potential_neighbors) > 3:  # Many neighbors - more problematic
                            high_impact_indices.append(i)

                    # Process high-impact circles first
                    if len(high_impact_indices) > 0:
                        process_indices = high_impact_indices
                    else:
                        process_indices = sorted_by_priority
                else:
                    process_indices = sorted_by_priority

                # Process with more thorough adjustment tries
                for i in process_indices:
                    original_x, original_y = circles[i][0], circles[i][1]

                    # Try multiple adjustment patterns
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
            if dist < required_dist * 0.3:  # Very close - high probability
                return 1.0
            elif dist < required_dist * 1.2:  # Moderately close - moderate probability
                return 0.8
            else:
                return 0.4  # Far apart - lower probability

        # Perform crossover with overlap awareness
        for i in range(n):
            # Weighted crossover based on distance between circles
            prob_overlap = calculate_overlap_probability(parent1[i], parent2[i])

            # Higher probability of swapping if circles are far apart
            if np.random.random() < (1 - prob_overlap) * 0.9:
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
        if diversity > 0.15:
            # High diversity - maintain higher mutation rate
            adaptive_rate *= 1.1
        elif diversity < 0.05:
            # Low diversity - reduce mutation rate to exploit
            adaptive_rate *= 0.9

        for i in range(n):
            if np.random.random() < adaptive_rate:
                # Mutate either position or radius with adaptive probabilities
                if np.random.random() < 0.75:  # 75% chance to mutate position
                    # Mutate position with adaptive magnitude
                    mutation_magnitude = 0.04 * (1 - adaptive_rate)
                    mutated[i][0] = np.clip(mutated[i][0] + np.random.normal(0, mutation_magnitude), 0, 1)
                    mutated[i][1] = np.clip(mutated[i][1] + np.random.normal(0, mutation_magnitude), 0, 1)
                else:
                    # Mutate radius with smaller magnitude
                    mutated[i][2] = np.clip(mutated[i][2] + np.random.normal(0, 0.015), 0.01, 0.5)

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

        # Track progress for early termination
        stagnant_generations = 0
        previous_best = -1

        for generation in range(GENERATIONS):
            # Adjust mutation rate based on generation (adaptive)
            mutation_rate = max(MUTATION_RATE_INITIAL * (1 - generation / GENERATIONS) ** 0.8, 0.02)

            # Evaluate fitness for all individuals (can be parallelized)
            fitnesses = [self.compute_fitness(circles) for circles in population]

            # Track best solution
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()
                stagnant_generations = 0  # Reset stagnation counter
            else:
                stagnant_generations += 1

            # Early termination if no improvement for too long
            if stagnant_generations > 20:
                break

            # Create new population
            new_population = []

            # Elitism: keep best individuals (more generous elitism)
            elite_size = max(10, POPULATION_SIZE // 10)
            elite_indices = np.argsort(fitnesses)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

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
    optimizer = AdaptiveMultiPhaseOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END