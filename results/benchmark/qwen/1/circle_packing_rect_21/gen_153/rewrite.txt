# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class CirclePackingOptimizer:
    def __init__(self, rect_width: float = 1.5, rect_height: float = 0.5, n_circles: int = 21):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.n_circles = n_circles
        self.population_size = 50
        self.generations = 100
        self.elite_size = 8
        self.tournament_size = 7

    def check_constraints(self, circles: np.ndarray) -> bool:
        """Efficiently check if all circles satisfy the constraints with early termination."""
        # Check boundary constraints first
        for i in range(self.n_circles):
            x, y, r = circles[i]
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                return False

        # Check overlap constraints using spatial indexing for better efficiency
        if self.n_circles > 1:
            # Use grid-based spatial indexing for fast neighbor lookup
            positions = circles[:, :2]
            radii = circles[:, 2]

            # Create spatial grid for efficient overlap checking
            # Grid cell size based on average circle radius
            avg_radius = np.mean(radii)
            if avg_radius > 0:
                cell_size = avg_radius * 2.0
            else:
                cell_size = 0.1

            # Ensure cell size isn't too small
            cell_size = max(cell_size, 0.01)

            # Calculate grid dimensions
            grid_width = int(np.ceil(self.rect_width / cell_size))
            grid_height = int(np.ceil(self.rect_height / cell_size))

            # Initialize grid
            grid = {}

            # Place circles in grid
            for i in range(len(circles)):
                x, y, r = circles[i]
                grid_x = int(x / cell_size)
                grid_y = int(y / cell_size)
                if (grid_x, grid_y) not in grid:
                    grid[(grid_x, grid_y)] = []
                grid[(grid_x, grid_y)].append(i)

            # Check overlaps using spatial indexing
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                # Get grid cell for this circle
                grid_x = int(x1 / cell_size)
                grid_y = int(y1 / cell_size)

                # Check this cell and adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_grid_x = grid_x + dx
                        neighbor_grid_y = grid_y + dy
                        if (neighbor_grid_x, neighbor_grid_y) in grid:
                            for j in grid[(neighbor_grid_x, neighbor_grid_y)]:
                                if i != j:
                                    x2, y2, r2 = circles[j]
                                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                    if dist < (r1 + r2):
                                        return False

        return True

    def evaluate_fitness(self, circles: np.ndarray) -> float:
        """Evaluate fitness as the sum of radii with constraint validation."""
        if not self.check_constraints(circles):
            return -np.inf

        return np.sum(circles[:, 2])

    def get_voronoi_criticality(self, individual):
        """Calculate criticality based on Voronoi diagram - how much room each circle has"""
        circles = individual.copy()
        n = len(circles)

        # Generate Voronoi diagram
        points = circles[:, :2]  # x,y coordinates

        try:
            vor = Voronoi(points)
        except:
            # Fallback: return uniform criticality if Voronoi fails
            return np.ones(n) * 0.1

        criticality_scores = np.zeros(n)

        # For each circle, calculate minimum distance to Voronoi edges
        # This represents how constrained the circle is
        for i in range(n):
            # Get Voronoi regions for this point
            try:
                region_indices = np.where(vor.point_region == i)[0]
                if len(region_indices) > 0:
                    # Calculate minimum distance to region boundaries
                    min_dist_to_edge = float('inf')

                    # Check distance to all Voronoi vertices for this region
                    region_vertices = vor.regions[vor.point_region[i]]
                    if -1 not in region_vertices and len(region_vertices) > 0:
                        for vertex_idx in region_vertices:
                            if vertex_idx != -1 and vertex_idx < len(vor.vertices):
                                vertex = vor.vertices[vertex_idx]
                                dist = np.sqrt((circles[i,0] - vertex[0])**2 + (circles[i,1] - vertex[1])**2)
                                min_dist_to_edge = min(min_dist_to_edge, dist)

                    if min_dist_to_edge < float('inf') and min_dist_to_edge > 0:
                        # Criticality increases with proximity to boundaries
                        # Smaller distances mean less room for growth
                        # Normalize by max possible distance to avoid extreme values
                        criticality_scores[i] = min_dist_to_edge
                    else:
                        criticality_scores[i] = 0.01  # Default when distance is invalid
                else:
                    criticality_scores[i] = 0.01
            except:
                criticality_scores[i] = 0.01

        # Normalize criticality scores to [0,1] range for consistency
        if np.max(criticality_scores) > 0:
            criticality_scores = criticality_scores / np.max(criticality_scores)

        return criticality_scores

    def create_hexagonal_initial_solution(self) -> np.ndarray:
        """Create initial solution using hexagonal lattice pattern for better packing efficiency."""
        circles = np.zeros((self.n_circles, 3))

        # Hexagonal grid parameters
        # Determine grid dimensions
        rows = int(np.ceil(np.sqrt(self.n_circles)))
        cols = int(np.ceil(self.n_circles / rows))

        # Adjust for rectangular container
        if self.rect_width >= self.rect_height:
            # Width is larger, arrange horizontally
            grid_width = self.rect_width * 0.9
            grid_height = self.rect_height * 0.9
        else:
            # Height is larger, arrange vertically
            grid_width = self.rect_width * 0.9
            grid_height = self.rect_height * 0.9

        # Calculate spacing based on rectangle dimensions
        cell_width = grid_width / cols
        cell_height = grid_height / rows
        min_cell_dim = min(cell_width, cell_height)

        # Hexagon radius (circles should fit comfortably)
        hex_radius = min_cell_dim * 0.4

        # Arrange in hexagonal pattern
        placed = 0
        for row in range(rows):
            if placed >= self.n_circles:
                break
            for col in range(cols):
                if placed >= self.n_circles:
                    break

                # Offset every other row for hexagonal pattern
                offset = (row % 2) * (cell_width / 2)
                x = offset + col * cell_width + cell_width / 2
                y = row * cell_height + cell_height / 2

                # Ensure we're within bounds
                x = np.clip(x, hex_radius, self.rect_width - hex_radius)
                y = np.clip(y, hex_radius, self.rect_height - hex_radius)

                # Adjust radius to prevent boundary issues
                max_radius = min(x, y, self.rect_width - x, self.rect_height - y)
                r = min(hex_radius, max_radius * 0.8)

                circles[placed] = [x, y, r]
                placed += 1

        # Fill remaining positions with small random circles
        for i in range(placed, self.n_circles):
            # Place remaining circles randomly but within bounds
            x = np.random.uniform(hex_radius, self.rect_width - hex_radius)
            y = np.random.uniform(hex_radius, self.rect_height - hex_radius)
            r = np.random.uniform(0.005, hex_radius * 0.5)
            circles[i] = [x, y, r]

        return circles

    def create_random_solution(self) -> np.ndarray:
        """Create a random valid solution."""
        circles = np.zeros((self.n_circles, 3))

        # Try to place circles greedily
        placed = 0
        max_attempts = 10000

        for attempt in range(max_attempts):
            if placed >= self.n_circles:
                break

            # Random position and radius
            x = np.random.uniform(0, self.rect_width)
            y = np.random.uniform(0, self.rect_height)
            r = np.random.uniform(0.01, 0.1)

            new_circle = np.array([x, y, r])

            # Check if it overlaps with existing circles
            valid_placement = True
            for i in range(placed):
                existing_x, existing_y, existing_r = circles[i]
                distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if distance < (r + existing_r):
                    valid_placement = False
                    break

            # Check boundary constraints
            if x - r < 0 or x + r > self.rect_width or y - r < 0 or y + r > self.rect_height:
                valid_placement = False

            if valid_placement:
                circles[placed] = new_circle
                placed += 1

        # Fill remaining positions with zeros if needed
        for i in range(placed, self.n_circles):
            circles[i] = [0, 0, 0]

        return circles

    def mutate(self, circles: np.ndarray) -> np.ndarray:
        """Improved mutation operator with adaptive parameters based on constraint density."""
        mutated = circles.copy()

        # First compute constraint density for adaptive mutation
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Compute Voronoi cells to estimate constraint density (using approximate method)
        try:
            vor = Voronoi(positions)
            # Estimate constraint density based on Voronoi cell areas
            cell_areas = np.zeros(len(positions))
            for i in range(len(positions)):
                vertices = vor.points[vor.regions[vor.point_region[i]]]
                if len(vertices) > 2:
                    # Simplified area estimation
                    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
                    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
                    cell_areas[i] = (max_x - min_x) * (max_y - min_y)
                else:
                    cell_areas[i] = 1.0
        except:
            cell_areas = np.ones(len(positions))

        # Adaptive mutation rate based on generation and constraint density
        for i in range(self.n_circles):
            # Base mutation rate
            mutation_rate = 0.3

            # Density-based adjustment: higher density = lower mutation rate
            density_adjustment = 1.0 / (1.0 + cell_areas[i]/np.mean(cell_areas))
            adjusted_mutation_rate = mutation_rate * density_adjustment

            if np.random.random() < adjusted_mutation_rate:
                # Choose mutation type
                mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])

                if mutation_type == 'position':
                    # Mutate position (x, y) with adaptive step size based on density
                    # High density = small step, low density = large step
                    base_step = 0.05 + np.random.random() * 0.05
                    density_factor = 1.0 / (1.0 + cell_areas[i]/np.mean(cell_areas))
                    step_size = base_step * density_factor

                    mutated[i, 0] += np.random.normal(0, step_size)
                    mutated[i, 1] += np.random.normal(0, step_size)
                else:
                    # Mutate radius with log-normal distribution to avoid negative values
                    # Adapted based on density - more conservative in dense regions
                    density_factor = 1.0 / (1.0 + cell_areas[i]/np.mean(cell_areas))
                    scale_factor = np.exp(np.random.normal(0, 0.2 * density_factor))
                    mutated[i, 2] *= scale_factor
                    mutated[i, 2] = max(0.001, mutated[i, 2])

        return mutated

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Improved crossover operator with better recombination."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Get criticality scores for both parents to guide crossover
        crit1 = self.get_voronoi_criticality(parent1)
        crit2 = self.get_voronoi_criticality(parent2)

        # Exchange radii of circles with highest criticality using weighted probabilities
        combined_criticality = np.maximum(crit1, crit2)
        sorted_indices = np.argsort(-combined_criticality)

        # Exchange radii for top 30% of circles with probabilistic selection
        num_exchanges = int(len(parent1) * 0.3)
        for i in range(num_exchanges):
            idx = sorted_indices[i]
            # Apply crossover with probability based on criticality
            # More critical circles have higher chance of exchange
            crossover_prob = 0.7 + 0.3 * (crit1[idx] + crit2[idx]) / 2  # Max prob = 1.0
            if np.random.random() < crossover_prob:
                child1[idx, 2], child2[idx, 2] = child2[idx, 2], child1[idx, 2]

        # For positions, use uniform crossover
        for i in range(self.n_circles):
            if np.random.random() < 0.5:
                child1[i, :2], child2[i, :2] = child2[i, :2], child1[i, :2]

        return child1, child2

    def repair_solution(self, circles: np.ndarray) -> np.ndarray:
        """Enhanced repair mechanism for fixing constraint violations with Voronoi-based adaptive movement."""
        repaired = circles.copy()

        # Ensure positive radii
        repaired[:, 2] = np.maximum(repaired[:, 2], 0.001)

        # Enforce bounds
        for i in range(len(repaired)):
            x, y, r = repaired[i]
            x = np.clip(x, r, self.rect_width - r)
            y = np.clip(y, r, self.rect_height - r)
            repaired[i] = [x, y, r]

        # Resolve overlaps iteratively with better algorithm
        for iteration in range(50):
            # Calculate pairwise distances
            positions = repaired[:, :2]
            radii = repaired[:, 2]
            distances = cdist(positions, positions)

            conflicts = []
            for i in range(len(repaired)):
                for j in range(i+1, len(repaired)):
                    if distances[i, j] < (radii[i] + radii[j]):
                        conflicts.append((i, j))

            if not conflicts:
                break

            # Compute Voronoi diagram to assess local constraint density
            try:
                vor = Voronoi(positions)
                # Calculate approximate Voronoi cell areas for each point
                cell_areas = np.zeros(len(positions))
                for i in range(len(positions)):
                    # Get vertices for the Voronoi cell of point i
                    vertices = vor.points[vor.regions[vor.point_region[i]]]
                    if len(vertices) > 2:
                        # Approximate area using convex hull
                        # For simplicity, we'll use the minimum bounding box area
                        if len(vertices) >= 2:
                            min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
                            min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
                            cell_areas[i] = (max_x - min_x) * (max_y - min_y)
                        else:
                            cell_areas[i] = 1.0
                    else:
                        cell_areas[i] = 1.0
            except:
                # Fallback if Voronoi computation fails
                cell_areas = np.ones(len(positions))

            # Move conflicting pairs apart with adaptive step sizes based on Voronoi density
            for i, j in conflicts:
                x1, y1, r1 = repaired[i]
                x2, y2, r2 = repaired[j]

                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)

                if distance > 0:
                    # Move circles away from each other with density-adaptive movement
                    move_distance = (r1 + r2 - distance) / 2

                    # Adaptive movement based on Voronoi cell area (smaller cells = more constrained)
                    # Inverse relationship: smaller cell area means higher constraint density
                    base_step = 0.5  # Base movement factor
                    density_factor = 1.0 / (1.0 + cell_areas[i] + cell_areas[j])  # Normalize

                    # Apply movement with bounded adjustment and density weighting
                    move1 = move_distance * r2 / (r1 + r2 + 1e-8) * base_step * density_factor
                    move2 = move_distance * r1 / (r1 + r2 + 1e-8) * base_step * density_factor

                    repaired[i, 0] -= dx / distance * move1
                    repaired[i, 1] -= dy / distance * move1
                    repaired[j, 0] += dx / distance * move2
                    repaired[j, 1] += dy / distance * move2

                    # Keep within bounds
                    repaired[i, 0] = np.clip(repaired[i, 0], r1, self.rect_width - r1)
                    repaired[i, 1] = np.clip(repaired[i, 1], r1, self.rect_height - r1)
                    repaired[j, 0] = np.clip(repaired[j, 0], r2, self.rect_width - r2)
                    repaired[j, 1] = np.clip(repaired[j, 1], r2, self.rect_height - r2)

        return repaired

    def evolve(self) -> np.ndarray:
        """Main evolutionary algorithm loop."""
        # Initialize population with better quality solutions
        population = []
        # Start with hexagonal initialization (better packing)
        for _ in range(self.population_size // 2):
            solution = self.create_hexagonal_initial_solution()
            population.append(solution)

        # Fill remaining with random solutions
        for _ in range(self.population_size // 2):
            solution = self.create_random_solution()
            population.append(solution)

        # Track best fitness for convergence detection
        previous_best = -np.inf
        stagnation_count = 0

        # Evolutionary algorithm
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(individual) for individual in population]

            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]

            # Keep elite
            elite = population[:self.elite_size]

            # Generate new population
            new_population = elite[:]

            # Create offspring using tournament selection and crossover
            while len(new_population) < self.population_size:
                # Tournament selection - select two parents
                parent1_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]
                parent2_idx = sorted_indices[np.random.choice(min(self.tournament_size, len(sorted_indices)))]

                parent1 = population[parent1_idx].copy()
                parent2 = population[parent2_idx].copy()

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutate
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # Repair
                child1 = self.repair_solution(child1)
                child2 = self.repair_solution(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            # Convergence detection
            current_best = max(fitness_scores)
            if abs(current_best - previous_best) < 1e-5:
                stagnation_count += 1
            else:
                stagnation_count = 0
            previous_best = current_best

            # Early stopping if stagnated too long
            if stagnation_count > 20:
                print(f"Early stopping at generation {generation} due to convergence")
                break

            # Print progress
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {current_best:.6f}")

        # Return the best solution
        fitness_scores = [self.evaluate_fitness(individual) for individual in population]
        best_idx = np.argmax(fitness_scores)
        best_solution = population[best_idx]

        # Final validation
        final_fitness = self.evaluate_fitness(best_solution)
        if final_fitness == -np.inf:
            print("Warning: Final solution violated constraints. Returning fallback.")
            # Fallback to best valid solution found during evolution
            for i in range(len(population)):
                if self.evaluate_fitness(population[i]) > -np.inf:
                    return population[i]

        return best_solution

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimized rectangle dimensions - using a wider rectangle for better packing
    rect_width = 1.5
    rect_height = 0.5

    # Create optimizer instance
    optimizer = CirclePackingOptimizer(rect_width, rect_height, 21)

    # Run evolution
    circles = optimizer.evolve()

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")