# EVOLVE-BLOCK-START
import numpy as np
import random
from copy import deepcopy
from typing import Tuple, List
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist

class SpatialIndexer:
    """Enhanced spatial indexing for efficient overlap detection using KDTree"""

    def __init__(self):
        self.tree = None
        self.circles_data = None

    def build_index(self, circles: np.ndarray):
        """Build KDTree index for efficient neighbor queries"""
        self.circles_data = circles
        if len(circles) > 0:
            self.tree = cKDTree(circles[:, :2])  # Only use (x,y) coordinates for spatial indexing
        else:
            self.tree = None

    def get_potential_overlaps(self, x: float, y: float, radius: float, max_distance: float = None) -> List[int]:
        """Get indices of potentially overlapping circles using KDTree"""
        if self.tree is None:
            return []

        query_point = np.array([[x, y]])
        if max_distance is None:
            # Use a reasonable upper bound for search radius
            max_distance = 2.0

        # Find neighbors within max_distance
        indices = self.tree.query_ball_point(query_point, max_distance)
        return indices

    def check_overlap(self, circles: np.ndarray, i: int, j: int) -> bool:
        """Check if two specific circles overlap"""
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance < r1 + r2

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

    # Global spatial indexer instance for validation
    spatial_indexer = SpatialIndexer()

    def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
        """Initialize population with diverse configurations using Voronoi-inspired method"""
        population = []
        for _ in range(pop_size):
            # Better Voronoi-inspired initialization with systematic grid + jitter
            circles = np.zeros((n_circles, 3))

            # Generate points in a more structured way
            points = []
            num_rows = int(np.ceil(np.sqrt(n_circles)))
            num_cols = int(np.ceil(n_circles / num_rows))

            # Create structured grid points with jitter
            for i in range(num_rows):
                for j in range(num_cols):
                    if len(points) < n_circles:
                        # Add structured positioning with jitter for better distribution
                        x = (j + 0.5 + np.random.normal(0, 0.15)) / num_cols
                        y = (i + 0.5 + np.random.normal(0, 0.15)) / num_rows

                        # Clamp to valid range
                        x = max(0.01, min(0.99, x))
                        y = max(0.01, min(0.99, y))

                        points.append([x, y])

            # Use Voronoi points for better distribution if available
            if len(points) >= n_circles:
                try:
                    # Create Voronoi diagram to get better spaced points
                    vor = Voronoi(points[:n_circles*2])  # Use more points for Voronoi
                    valid_voronoi_points = []

                    # Extract valid Voronoi vertices
                    for vertex in vor.vertices:
                        if 0 <= vertex[0] <= 1 and 0 <= vertex[1] <= 1:
                            valid_voronoi_points.append(vertex)

                    # If we have enough valid Voronoi points
                    if len(valid_voronoi_points) >= n_circles:
                        points = valid_voronoi_points[:n_circles]
                    else:
                        # Fall back to structured grid points
                        points = points[:n_circles]
                except:
                    # If Voronoi fails, use structured grid points
                    points = points[:n_circles]

            # Create circles with distributed positions and appropriate radii
            for i in range(n_circles):
                x, y = points[i]

                # Calculate safe radius based on proximity to other circles
                min_dist = float('inf')
                for other_x, other_y in points[:i]:
                    dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)

                # Set initial radius with boundary constraints and neighbor distance
                # Give first few circles larger radii for better starting configuration
                if i < 5:
                    radius = min(0.1, x, 1-x, y, 1-y, min_dist/2)
                else:
                    radius = min(0.06, x, 1-x, y, 1-y, min_dist/2)

                if radius <= 0:
                    radius = 0.01

                circles[i] = [x, y, radius]

            population.append(circles)
        return population

    def is_valid(circles: np.ndarray) -> bool:
        """Check if all circles are within bounds and non-overlapping"""
        n = len(circles)

        # Build spatial index for overlap checking
        spatial_indexer.build_index(circles)

        # First check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

        # Then check overlap constraints using spatial indexing
        for i in range(n):
            x1, y1, r1 = circles[i]

            # Get potential overlapping candidates with reasonable search radius
            candidate_indices = spatial_indexer.get_potential_overlaps(x1, y1, r1, r1 * 2)

            # Check actual overlaps
            for j in candidate_indices:
                if i != j:  # Skip self-comparison
                    if spatial_indexer.check_overlap(circles, i, j):
                        return False

        return True

    def evaluate_fitness(circles: np.ndarray) -> float:
        """Evaluate fitness of a solution"""
        if not is_valid(circles):
            # Dynamic penalty based on constraint violations
            total_penalty = 0

            # Boundary violations with weighted penalties
            for i in range(len(circles)):
                x, y, r = circles[i]
                boundary_violation = 0
                if x - r < 0:
                    boundary_violation += abs(x - r)
                if x + r > 1:
                    boundary_violation += abs(x + r - 1)
                if y - r < 0:
                    boundary_violation += abs(y - r)
                if y + r > 1:
                    boundary_violation += abs(y + r - 1)
                total_penalty += boundary_violation * 10000

            # Overlap violations with higher penalty
            # Simplified penalty calculation using spatial indexing for efficiency
            spatial_indexer.build_index(circles)
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                candidate_indices = spatial_indexer.get_potential_overlaps(x1, y1, r1, r1 * 2)
                for j in candidate_indices:
                    if i != j:
                        if spatial_indexer.check_overlap(circles, i, j):
                            x2, y2, r2 = circles[j]
                            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            overlap = max(0, r1 + r2 - distance)
                            total_penalty += overlap * 100000

            return -total_penalty

        # Valid configuration: maximize sum of radii
        return np.sum(circles[:, 2])

    def mutate(circles: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
        """Apply mutation to circles with improved adaptive strategy"""
        mutated = deepcopy(circles)

        # More sophisticated adaptive mutation rate with softer decay
        # Start with higher mutation rate but decrease more gradually
        base_mutation_rate = 0.3
        mutation_rate = base_mutation_rate * (1 - generation / max_generations * 0.8)

        # Also adapt mutation intensities based on generation
        pos_mutation_intensity = 0.05 * (1 - generation / max_generations * 0.7)
        rad_mutation_intensity = 0.03 * (1 - generation / max_generations * 0.7)

        # Mutate each circle with some probability
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Mutate position with adaptive intensity
                mutated[i, 0] += np.random.normal(0, pos_mutation_intensity)  # x coordinate
                mutated[i, 1] += np.random.normal(0, pos_mutation_intensity)  # y coordinate

                # Clamp to unit square with safety margin
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0]))
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1]))

                # Mutate radius with adaptive intensity
                mutated[i, 2] += np.random.normal(0, rad_mutation_intensity)
                mutated[i, 2] = max(0.001, mutated[i, 2])

        return mutated

    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Create offspring via crossover of two parents"""
        child = deepcopy(parent1)
        n = len(parent1)

        # Two-point crossover for better mixing
        crossover_point1 = random.randint(1, n//2)
        crossover_point2 = random.randint(crossover_point1, n-1)

        for i in range(crossover_point1, crossover_point2):
            child[i] = parent2[i].copy()

        return child

    def tournament_selection(population: List[np.ndarray], k: int = 4) -> np.ndarray:
        """Select individual using tournament selection"""
        selected = random.sample(population, k)
        return max(selected, key=evaluate_fitness)

    def geometric_refinement(circles: np.ndarray) -> np.ndarray:
        """Apply geometric refinement to improve solution quality"""
        refined = deepcopy(circles)

        # Phase 1: Fix containment issues
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Ensure containment with margin
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]

        # Phase 2: Try to slightly increase radii where possible
        for _ in range(20):  # Limited iterations to avoid infinite loops
            improved = False
            for i in range(len(refined)):
                x, y, r = refined[i]
                # Try to increase radius slightly while maintaining constraints
                new_r = min(r + 0.001, x, 1-x, y, 1-y)

                # Test if we can increase this radius
                valid = True
                temp_r = new_r
                for j in range(len(refined)):
                    if i != j:
                        x2, y2, r2 = refined[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < temp_r + r2:
                            valid = False
                            break

                if valid and new_r > r:
                    refined[i] = [x, y, new_r]
                    improved = True

            if not improved:
                break

        return refined

    def refine_solution(circles: np.ndarray) -> np.ndarray:
        """Apply local refinement to fix minor constraint violations"""
        refined = deepcopy(circles)

        # Fix containment violations first
        for i in range(len(refined)):
            x, y, r = refined[i]
            # Ensure containment with margin
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]

        # Apply geometric refinement for further improvement
        refined = geometric_refinement(refined)

        return refined

    # Initialize population
    population = initialize_population(population_size, n_circles)

    # Evolve
    best_fitness = float('-inf')
    best_solution = None

    for generation in range(max_generations):
        # Evaluate fitness for entire population
        fitness_scores = [evaluate_fitness(individual) for individual in population]

        # Track best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = deepcopy(population[max_fitness_idx])

        # Elitism: keep top 20%
        elite_count = max(1, population_size // 5)
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

            # Local refinement
            refined_child = refine_solution(mutated_child)

            new_population.append(refined_child)

        population = new_population[:population_size]

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to first individual if nothing was found
        return population[0]

# EVOLVE-BLOCK-END