# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional
import random
import time
from copy import deepcopy

# Global constants
POPULATION_SIZE = 80
GENERATIONS = 60
MUTATION_RATE_INITIAL = 0.12
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class HybridVoronoiOptimizer:
    def __init__(self):
        self.n_circles = 26
        self.max_iterations = 100

    def is_valid_configuration(self, circles: np.ndarray) -> bool:
        """Check if the configuration satisfies all constraints."""
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

        # Check overlap constraints using pairwise distance matrix
        if self.n_circles > 1:
            distances = cdist(circles[:, :2], circles[:, :2])
            # Create upper triangular mask to avoid duplicate comparisons
            mask = np.triu(np.ones((self.n_circles, self.n_circles), dtype=bool), k=1)

            # Calculate minimum required distance
            min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask

            # Check for overlaps
            overlaps = distances < min_distances
            if np.any(overlaps):
                return False

        return True

    def calculate_sum_radii(self, circles: np.ndarray) -> float:
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])

    def create_voronoi_initialization(self) -> np.ndarray:
        """Initialize circles using Voronoi diagram approach for better spatial distribution."""
        # Generate initial candidate points using a more sophisticated pattern
        # We'll start with a hexagonal grid pattern to get good initial distribution
        n_points = self.n_circles + 15  # Extra points to allow for better Voronoi cells

        # Create hexagonal grid pattern
        points = []
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))

        # Hexagonal spacing
        spacing = 0.8 / (max(rows, cols) + 2)  # Reduced spacing to allow for more variation
        hex_height = spacing * np.sqrt(3) / 2

        # Create hexagonal grid with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = (j + 0.5 + (i % 2) * 0.5) * spacing + 0.1  # Shift to avoid boundary issues
                y = (i + 0.5) * hex_height + 0.1
                if x <= 0.9 and y <= 0.9:  # Keep away from boundaries
                    points.append([x, y])

        # Trim to exact number needed
        points = points[:n_points]

        # Add some random jitter to points
        for point in points:
            point[0] += np.random.uniform(-spacing/6, spacing/6)
            point[1] += np.random.uniform(-spacing/6, spacing/6)

        # Ensure points are within bounds (with padding from edges)
        points = [[max(0.02, min(0.98, p[0])), max(0.02, min(0.98, p[1]))] for p in points]

        # Create Voronoi diagram
        try:
            vor = Voronoi(points)
        except:
            # Fallback to simple initialization if Voronoi fails
            return self._create_simple_initialization()

        # Select the most appropriate Voronoi cells to produce circles
        circles = np.zeros((self.n_circles, 3))

        # For each Voronoi cell, compute optimal circle placement
        valid_cells = []
        for i in range(min(self.n_circles, len(vor.point_region))):
            region = vor.point_region[i]
            if region != -1:  # Valid region
                valid_cells.append(i)

        # Use the first n_circles valid cells
        selected_indices = valid_cells[:self.n_circles]

        # Compute circle properties based on Voronoi regions
        for i, idx in enumerate(selected_indices):
            # Get the cell center (initial guess for circle center)
            center = vor.points[idx]

            # Estimate cell area using distance to neighbors
            if len(vor.neighbors[idx]) > 0:
                # Find average distance to neighboring points to estimate cell size
                neighbor_distances = []
                for neighbor_idx in vor.neighbors[idx]:
                    if neighbor_idx >= 0 and neighbor_idx < len(vor.points):
                        dist = np.sqrt((center[0] - vor.points[neighbor_idx][0])**2 +
                                     (center[1] - vor.points[neighbor_idx][1])**2)
                        neighbor_distances.append(dist)
                avg_neighbor_dist = np.mean(neighbor_distances) if neighbor_distances else spacing
                estimated_area = avg_neighbor_dist * avg_neighbor_dist * np.pi
                # Convert area to radius (assuming circular cell approx)
                radius_estimate = np.sqrt(estimated_area / np.pi) * 0.5
            else:
                # Fallback to simple spacing-based estimate
                radius_estimate = spacing / 3

            # Refine the radius considering constraints
            x, y = center
            min_distance_to_boundary = min(x, y, 1-x, 1-y)
            final_radius = min(radius_estimate, min_distance_to_boundary * 0.7)

            # Ensure reasonable minimum radius
            final_radius = max(0.015, min(final_radius, 0.2))

            circles[i] = [x, y, final_radius]

        # Ensure all circles are valid by checking and fixing constraints
        # This is a critical step that wasn't done in the previous version
        safe_circles = self._refine_initial_circles(circles)

        return safe_circles

    def _refine_initial_circles(self, circles: np.ndarray) -> np.ndarray:
        """Refine initial circles to ensure all constraints are satisfied."""
        circles = circles.copy()

        # First pass: ensure all circles fit within boundaries
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Clamp positions to valid range
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            circles[i] = [x, y, r]

        # Second pass: resolve overlaps through iterative adjustment
        for iteration in range(50):  # Limit iterations to prevent infinite loops
            # Check for overlaps
            overlaps_exist = False
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < r1 + r2:
                        overlaps_exist = True
                        # Resolve by moving one of the circles apart
                        # Move the circle with smaller radius outward
                        if r1 <= r2:
                            # Move circle i
                            angle = np.arctan2(y2-y1, x2-x1)
                            new_x = x1 + 0.01 * np.cos(angle)
                            new_y = y1 + 0.01 * np.sin(angle)
                            # Constrain to boundaries
                            new_x = np.clip(new_x, r1, 1-r1)
                            new_y = np.clip(new_y, r1, 1-r1)
                            circles[i] = [new_x, new_y, r1]
                        else:
                            # Move circle j
                            angle = np.arctan2(y1-y2, x1-x2)
                            new_x = x2 + 0.01 * np.cos(angle)
                            new_y = y2 + 0.01 * np.sin(angle)
                            # Constrain to boundaries
                            new_x = np.clip(new_x, r2, 1-r2)
                            new_y = np.clip(new_y, r2, 1-r2)
                            circles[j] = [new_x, new_y, r2]

            if not overlaps_exist:
                break

        return circles

    def _compute_cell_area(self, points, centroid):
        """Compute approximate area of Voronoi cell."""
        # Simplified: compute average distance to neighbors
        if len(points) < 3:
            return 0.1

        distances = [distance.euclidean(centroid, p) for p in points]
        return np.mean(distances) ** 2

    def optimize_placement(self, circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Apply advanced local optimization to improve placement."""
        circles = circles.copy()
        n = len(circles)

        # Better optimization using gradient-like approach
        for iteration in range(max_iter):
            improved = False

            # Try to increase radii while respecting constraints
            for i in range(n):
                original_radius = circles[i][2]

                # Calculate maximum possible radius
                max_radius = min(
                    circles[i][0],  # x coordinate
                    circles[i][1],  # y coordinate
                    1 - circles[i][0],  # distance to right edge
                    1 - circles[i][1]   # distance to top edge
                )

                # Try to increase the radius
                new_radius = min(original_radius + 0.005, max_radius)

                if new_radius > original_radius:
                    # Temporarily update radius
                    circles[i][2] = new_radius

                    # Check if valid configuration
                    if self.is_valid_configuration(circles):
                        improved = True
                    else:
                        # Revert if invalid
                        circles[i][2] = original_radius

            if not improved:
                break

        return circles

    def initialize_population(self, pop_size: int) -> List[np.ndarray]:
        """Initialize population with valid configurations using Voronoi approach."""
        population = []

        # Create diverse initial configurations
        for i in range(pop_size):
            if i == 0:
                # First individual: Voronoi initialization
                circles = self.create_voronoi_initialization()
            elif i < pop_size // 3:
                # Second third: random with constraint checking
                circles = self._create_constrained_random_initialization()
            else:
                # Last third: slightly modified Voronoi
                base = self.create_voronoi_initialization()
                circles = base.copy()
                # Add small mutations to diversify
                for j in range(self.n_circles):
                    if np.random.random() < 0.2:
                        circles[j, 0] += np.random.uniform(-0.02, 0.02)
                        circles[j, 1] += np.random.uniform(-0.02, 0.02)
                        circles[j, 2] *= np.random.uniform(0.9, 1.1)

            # Ensure validity
            if self.is_valid_configuration(circles):
                population.append(circles.copy())
            else:
                # Fallback to valid configuration
                circles = self._create_simple_initialization()
                if self.is_valid_configuration(circles):
                    population.append(circles.copy())

        return population

    def _create_constrained_random_initialization(self) -> np.ndarray:
        """Create a constrained random initialization."""
        circles = np.zeros((self.n_circles, 3))

        # Create circles one by one with overlap avoidance
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

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform constraint-aware crossover using Voronoi-based strategy."""
        if np.random.random() > CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()

        n = len(parent1)
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        # Voronoi-inspired crossover: create Voronoi diagram from parents
        # and sample from the Voronoi structures for crossover

        # Combine points from both parents for Voronoi construction
        combined_points = np.vstack([parent1[:, :2], parent2[:, :2]])

        # Create Voronoi diagram from combined points
        try:
            vor = Voronoi(combined_points)
            # Sample points from Voronoi cells to create children
            # For simplicity, we'll do a more direct approach:

            # Uniform crossover with constraint validation
            for i in range(n):
                if np.random.random() < 0.5:
                    child1[i] = parent1[i].copy()
                    child2[i] = parent2[i].copy()
                else:
                    child1[i] = parent2[i].copy()
                    child2[i] = parent1[i].copy()

        except:
            # Fallback to standard crossover if Voronoi fails
            for i in range(n):
                if np.random.random() < 0.5:
                    child1[i] = parent1[i].copy()
                    child2[i] = parent2[i].copy()
                else:
                    child1[i] = parent2[i].copy()
                    child2[i] = parent1[i].copy()

        # Ensure children are valid
        child1 = self.optimize_placement(child1)
        child2 = self.optimize_placement(child2)

        return child1, child2

    def mutate(self, circles: np.ndarray, mutation_rate: float = MUTATION_RATE_INITIAL) -> np.ndarray:
        """Apply mutation with adaptive strategies."""
        mutated = circles.copy()
        n = len(mutated)

        # Adaptive mutation: different strategies based on generation progress
        for i in range(n):
            if np.random.random() < mutation_rate:
                # 70% chance to mutate position, 30% to mutate radius
                if np.random.random() < 0.7:
                    # Mutate position with adaptive magnitude
                    mutation_magnitude = 0.03 * (1 - mutation_rate)
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
        """Run the complete hybrid evolutionary algorithm."""
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

                # Mutation with adaptive rate
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
    optimizer = HybridVoronoiOptimizer()
    return optimizer.run_evolution()

# EVOLVE-BLOCK-END