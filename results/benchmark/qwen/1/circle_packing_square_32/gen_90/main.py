# EVOLVE-BLOCK-START
import numpy as np
import random
from deap import base, creator, tools, algorithms
import math
from typing import Tuple, List, Optional
import time

# Set seeds for deterministic behavior
random.seed(42)
np.random.seed(42)

class CirclePacker:
    """A modular circle packing optimizer with evolutionary algorithms."""

    def __init__(self, n_circles: int = 32):
        self.n_circles = n_circles
        self.toolbox = None
        self.best_solution = None
        self.best_fitness = float('-inf')

    def initialize_strategic_positions(self) -> List[Tuple[float, float]]:
        """Generate strategic positions for initial circle placement using enhanced Voronoi-inspired approach."""
        # Start with corner and edge positions
        positions = [
            (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),  # corners
            (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),  # edges
            (0.5, 0.5),  # center
        ]

        # Add boundary points that emphasize locations where large circles can fit
        # These are positioned closer to edges to favor larger initial radii
        boundary_points = []
        # Corners with extra emphasis
        for x, y in [(0.05, 0.05), (0.05, 0.95), (0.95, 0.05), (0.95, 0.95)]:
            boundary_points.append((x, y))

        # Edge centers with extra emphasis
        for i in range(1, 10):
            # Top and bottom edges (closer to corners)
            boundary_points.extend([(i*0.1, 0.05), (i*0.1, 0.95)])
            # Left and right edges (closer to corners)
            boundary_points.extend([(0.05, i*0.1), (0.95, i*0.1)])

        # Add more strategic positions in interior using Voronoi-like logic
        # Generate points that would be Voronoi vertices for a uniform grid
        voronoi_positions = []
        for i in range(1, 5):
            for j in range(1, 5):
                # Place points at grid intersections but offset slightly
                # to avoid perfect symmetry and encourage diverse initial placements
                x = i * 0.25 + np.random.uniform(-0.02, 0.02)
                y = j * 0.25 + np.random.uniform(-0.02, 0.02)
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                voronoi_positions.append((x, y))

        # Combine all strategic positions with weighted importance
        # Prioritize boundary positions for larger initial radii
        return positions + boundary_points + voronoi_positions

    def place_circle_greedy(self, circles: np.ndarray) -> np.ndarray:
        """Place circles greedily with maximum radius using strategic positions."""
        new_circles = circles.copy()
        strategic_positions = self.initialize_strategic_positions()

        placed = 0

        # Place initial strategic circles
        for i, (x, y) in enumerate(strategic_positions[:min(len(strategic_positions), self.n_circles)]):
            if placed >= self.n_circles:
                break
            # Try to place with maximum possible radius
            max_radius = min(x, 1-x, y, 1-y)
            if max_radius > 0:
                new_circle = [x, y, max_radius]
                if self.is_valid_position(new_circle, new_circles[:placed]):
                    new_circles[placed] = new_circle
                    placed += 1

        # Fill remaining spots with greedy approach
        while placed < self.n_circles:
            best_circle = None
            best_radius = 0

            # Generate candidates using a mix of random sampling and strategic placement
            candidates = []

            # Strategic sample around corners and edges
            for _ in range(500):
                x = random.uniform(0.01, 0.99)
                y = random.uniform(0.01, 0.99)
                max_radius = min(x, 1-x, y, 1-y)
                candidates.append((x, y, max_radius))

            # Add some strategic samples from our predefined positions
            for x, y in strategic_positions:
                if len(candidates) >= 1000:
                    break
                for dx, dy in [(-0.1, -0.1), (-0.1, 0.1), (0.1, -0.1), (0.1, 0.1)]:
                    nx, ny = x + dx, y + dy
                    if 0.01 <= nx <= 0.99 and 0.01 <= ny <= 0.99:
                        max_radius = min(nx, 1-nx, ny, 1-ny)
                        if max_radius > 0:
                            candidates.append((nx, ny, max_radius))

            # Find the best valid circle among candidates
            for x, y, max_radius in candidates[:1000]:
                if max_radius <= best_radius:
                    continue
                test_circle = [x, y, max_radius]
                if self.is_valid_position(test_circle, new_circles[:placed]):
                    best_circle = test_circle
                    best_radius = max_radius

            if best_circle is None:
                # Fallback to random placement
                x = random.uniform(0.01, 0.99)
                y = random.uniform(0.01, 0.99)
                test_circle = [x, y, 0.01]
                if self.is_valid_position(test_circle, new_circles[:placed]):
                    new_circles[placed] = test_circle
                    placed += 1
                else:
                    break  # Can't place more circles
            else:
                new_circles[placed] = best_circle
                placed += 1

        return new_circles

    def is_valid_position(self, circle: list, circles: np.ndarray) -> bool:
        """Check if a circle position is valid (within bounds and no collisions)."""
        x, y, r = circle

        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

        # Check collision with existing circles (vectorized for efficiency)
        if len(circles) > 0:
            # Extract positions and radii of existing circles
            existing_positions = circles[:, :2]
            existing_radii = circles[:, 2]

            # Vectorized distance calculation using broadcasting
            dx = existing_positions[:, 0] - x
            dy = existing_positions[:, 1] - y
            distances_squared = dx*dx + dy*dy
            radii_sum = existing_radii + r

            # Check if any distance is less than sum of radii
            if np.any(distances_squared < radii_sum * radii_sum):
                return False

        return True

    def calculate_fitness(self, circles: np.ndarray) -> Tuple[float, float]:
        """
        Calculate fitness for circle packing configuration.
        Returns (sum_of_radii, penalty_score)
        """
        n = len(circles)

        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Calculate sum of radii (primary objective)
        sum_radii = np.sum(radii)

        # Penalty for constraint violations
        penalty = 0.0

        # Boundary constraint penalty - vectorized
        boundary_violations = (circles[:, 0] - circles[:, 2] < 0) | \
                              (circles[:, 0] + circles[:, 2] > 1) | \
                              (circles[:, 1] - circles[:, 2] < 0) | \
                              (circles[:, 1] + circles[:, 2] > 1)
        penalty += np.sum(boundary_violations) * 1000

        # Overlap penalty (vectorized for efficiency)
        if n > 1:
            # Compute pairwise squared distances for all pairs
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]

            # Use broadcasting to compute all pairwise differences efficiently
            diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
            diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
            distances_squared = diff_x**2 + diff_y**2

            # Compute sum of radii for all pairs
            radii_i = radii[:, np.newaxis]
            radii_j = radii[np.newaxis, :]
            radii_sums = radii_i + radii_j

            # Mask for upper triangle to avoid double counting
            mask = np.triu(np.ones_like(distances_squared), k=1).astype(bool)

            # Find overlapping pairs
            overlaps = distances_squared < radii_sums**2
            overlaps = overlaps & mask

            # Calculate overlap penalty
            if np.any(overlaps):
                overlap_distances = np.sqrt(np.maximum(0, radii_sums**2 - distances_squared))
                penalty += np.sum(overlap_distances[overlaps]) * 100

        # Return sum of radii with penalty (lower penalty = higher fitness)
        return sum_radii, penalty

    def create_toolbox(self) -> base.Toolbox:
        """Create DEAP toolbox for evolution."""
        # Define fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Create an individual (32 circles, each with 3 values)
        def create_individual():
            # Start with greedy initialization
            individual = np.zeros((self.n_circles, 3))
            individual = self.place_circle_greedy(individual)

            # Apply small random perturbations to create initial diversity
            for i in range(self.n_circles):
                if np.random.random() < 0.3:  # 30% chance to perturb
                    individual[i, 0] += np.random.normal(0, 0.01)
                    individual[i, 1] += np.random.normal(0, 0.01)
                    individual[i, 0] = np.clip(individual[i, 0], 0.01, 0.99)
                    individual[i, 1] = np.clip(individual[i, 1], 0.01, 0.99)

            return creator.Individual(individual)

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register evaluation function
        def evaluate_individual(individual: np.ndarray) -> Tuple[float,]:
            """Evaluate individual and return fitness tuple."""
            sum_radii, penalty = self.calculate_fitness(individual)
            # Fitness is sum of radii minus penalty (since we maximize sum_radii)
            # We penalize invalid configurations heavily
            fitness = sum_radii - penalty
            return (fitness,)

        toolbox.register("evaluate", evaluate_individual)

        # Register genetic operators with improved parameters
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.mutate_individual)
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

    def mutate_individual(self, individual: np.ndarray, indpb: float = 0.1) -> Tuple[np.ndarray,]:
        """Mutate an individual by modifying positions and radii."""
        mutated_individual = individual.copy()

        for i in range(len(mutated_individual)):
            if np.random.random() < indpb:
                # Mutate position with smaller variance for finer control
                mutated_individual[i, 0] += np.random.normal(0, 0.005)
                mutated_individual[i, 1] += np.random.normal(0, 0.005)
                # Clamp to valid range
                mutated_individual[i, 0] = np.clip(mutated_individual[i, 0], 0.01, 0.99)
                mutated_individual[i, 1] = np.clip(mutated_individual[i, 1], 0.01, 0.99)

            if np.random.random() < indpb:
                # Mutate radius with smaller variance for fine tuning
                mutated_individual[i, 2] += np.random.normal(0, 0.002)
                # Ensure positive radius
                mutated_individual[i, 2] = max(0.005, mutated_individual[i, 2])

                # Adjust position if needed due to radius change
                x, y, r = mutated_individual[i]
                max_radius = min(x, 1-x, y, 1-y)
                if r > max_radius:
                    mutated_individual[i, 2] = max_radius * 0.9  # Scale down radius

        return (mutated_individual,)

    def optimize_with_refinement(self, pop_size: int = 50, generations: int = 100) -> np.ndarray:
        """Run optimization with hierarchical approach."""
        # Create toolbox
        self.toolbox = self.create_toolbox()

        # First phase: Coarse optimization with fewer generations
        population = self.toolbox.population(n=pop_size)

        # Statistics tracking
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run first optimization phase
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=0.8,
            mutpb=0.2,
            ngen=min(50, generations),
            stats=stats,
            verbose=False
        )

        # Get best from first phase
        best_individual = tools.selBest(population, 1)[0]
        best_fitness_first = best_individual.fitness.values[0]

        # Second phase: Fine-tune with more generations if needed
        if generations > 50:
            # Continue optimization with the best as starting point
            # Re-initialize population with better starting point
            new_population = self.toolbox.population(n=pop_size//2)
            # Add the best individual to the population
            new_population[0] = best_individual

            # Run second optimization phase
            population, logbook = algorithms.eaSimple(
                new_population, self.toolbox,
                cxpb=0.8,
                mutpb=0.2,
                ngen=generations - 50,
                stats=stats,
                verbose=False
            )

            # Get best from second phase
            best_individual = tools.selBest(population, 1)[0]
            best_fitness_second = best_individual.fitness.values[0]

            # Use the better of both phases
            if best_fitness_second > best_fitness_first:
                return best_individual
            else:
                return best_individual
        else:
            return best_individual

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        packer = CirclePacker(32)

        # Phase 1: Initial greedy placement for baseline
        circles = np.zeros((32, 3))
        circles = packer.place_circle_greedy(circles)

        # Phase 2: Evolutionary optimization
        start_time = time.time()
        optimized_circles = packer.optimize_with_refinement(pop_size=50, generations=100)
        end_time = time.time()

        # Validate final result
        sum_radii, penalty = packer.calculate_fitness(optimized_circles)
        if penalty > 100:  # High penalty indicates serious constraint violations
            # Fall back to greedy solution if optimization failed
            print("Optimization failed, using greedy solution")
            return circles

        # Validate that the optimized solution is actually better
        original_sum = np.sum(circles[:, 2])
        optimized_sum = np.sum(optimized_circles[:, 2])

        if optimized_sum > original_sum:
            return optimized_circles
        else:
            return circles

    except Exception as e:
        # Fallback in case of any error
        print(f"Error in optimization: {e}")
        # Return greedy solution as final fallback
        circles = np.zeros((32, 3))
        packer = CirclePacker(32)
        circles = packer.place_circle_greedy(circles)
        return circles

# EVOLVE-BLOCK-END