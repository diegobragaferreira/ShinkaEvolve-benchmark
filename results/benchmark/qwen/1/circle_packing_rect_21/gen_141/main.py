# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
from collections import defaultdict
from scipy.spatial import cKDTree

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.0, 1.0

    def is_valid_position(x: float, y: float, r: float) -> bool:
        """Check if circle is within bounds"""
        return (r <= x <= width - r) and (r <= y <= height - r)

    def check_overlap(circle1: Tuple[float, float, float], circle2: Tuple[float, float, float]) -> bool:
        """Check if two circles overlap"""
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance < (r1 + r2)

    def build_spatial_index(circles_list: List[Tuple[float, float, float]]) -> cKDTree:
        """Build spatial index for fast neighbor lookup"""
        if len(circles_list) == 0:
            return None
        points = np.array([[x, y] for x, y, r in circles_list])
        return cKDTree(points)

    def get_potential_collisions(spatial_tree, circle: Tuple[float, float, float],
                               max_distance: float) -> List[int]:
        """Get potential collision candidates using spatial index"""
        if spatial_tree is None:
            return []
        x, y, r = circle
        # Query neighbors within 2*(r_max) distance (where r_max is largest possible radius)
        indices = spatial_tree.query_ball_point([x, y], max_distance)
        return indices

    def calculate_constraint_density(circles_list: List[Tuple[float, float, float]]) -> dict:
        """Calculate constraint density for each circle based on neighbor count and distances"""
        if len(circles_list) < 2:
            return {}

        points = np.array([[x, y] for x, y, r in circles_list])
        try:
            # Build spatial index for efficient neighbor queries
            tree = cKDTree(points)

            # For each circle, count neighbors within a reasonable distance
            # and compute average distance to neighbors
            density_info = {}
            for i, (x, y, r) in enumerate(circles_list):
                # Query neighbors within 3x radius (should be sufficient for most cases)
                # Using a larger radius to capture more neighbors for density estimation
                neighbors = tree.query_ball_point([x, y], r * 3.0)

                # Exclude the circle itself
                neighbors = [n for n in neighbors if n != i]

                # Count neighbors and compute average distance
                neighbor_count = len(neighbors)
                avg_distance = 0.0

                if neighbor_count > 0:
                    # Compute distances to all neighbors
                    neighbor_points = points[neighbors]
                    distances = np.sqrt(np.sum((neighbor_points - [x, y])**2, axis=1))
                    avg_distance = np.mean(distances)

                # Density is inversely related to average distance and directly related to neighbor count
                # Lower average distance = higher density = more constrained
                # More neighbors = higher density = more constrained
                if avg_distance > 0:
                    density = neighbor_count / avg_distance
                else:
                    density = neighbor_count  # If no neighbors, just use count

                # Normalize density to make it comparable across different configurations
                density_info[i] = max(0.1, density)  # Minimum density to avoid division by zero issues

            return density_info
        except:
            # Fallback to simple neighbor counting if Voronoi fails
            density_info = {}
            for i, (x, y, r) in enumerate(circles_list):
                # Simple approach: count neighbors within 3x radius
                neighbor_count = 0
                for j, (ox, oy, oradius) in enumerate(circles_list):
                    if i != j:
                        distance = np.sqrt((x - ox)**2 + (y - oy)**2)
                        if distance < (r + oradius) * 3:  # Within 3x combined radius
                            neighbor_count += 1
                density_info[i] = max(0.1, neighbor_count)
            return density_info

    def evaluate_fitness(circles_array: np.ndarray) -> Tuple[float, float]:
        """Evaluate fitness: sum of radii and penalty for overlaps/bounds"""
        total_radius = np.sum(circles_array[:, 2])

        # Check bounds
        valid = True
        for x, y, r in circles_array:
            if not is_valid_position(x, y, r):
                valid = False
                break

        if not valid:
            return -1e6, total_radius

        # Check overlaps using spatial indexing for efficiency
        circles_list = [(x, y, r) for x, y, r in circles_array]
        spatial_tree = build_spatial_index(circles_list)

        # For each circle, find neighbors and check overlap efficiently
        max_radius = np.max(circles_array[:, 2])
        for i, circle in enumerate(circles_list):
            # Get nearby potential collisions
            potential_neighbors = get_potential_collisions(spatial_tree, circle, 2 * max_radius)

            for j in potential_neighbors:
                if i != j:  # Don't compare with self
                    x1, y1, r1 = circle
                    x2, y2, r2 = circles_list[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2):
                        return -1e6, total_radius  # Overlap penalty

        return total_radius, total_radius

    def initialize_hexagonal_arrangement(width: float, height: float, n_circles: int) -> np.ndarray:
        """Initialize circles in a hexagonal pattern"""
        circles = np.zeros((n_circles, 3))
        base_radius = min(width, height) * 0.05

        # Try to place in hexagonal grid
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))

        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Adjust for hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x / 2
                # Bound checks and adjust radius
                r = min(base_radius,
                       min(x, width-x), min(y, height-y))
                if r > 0.001:
                    circles[idx] = [x, y, r]
                    idx += 1
        return circles

    def adaptive_mutation(circles: np.ndarray, constraint_density: dict, generation: int) -> np.ndarray:
        """Mutate circles with adaptive step sizes based on density"""
        mutated = circles.copy()
        mutation_rate = 0.2 + 0.1 * np.exp(-generation/50)  # Decrease over generations

        for i in range(len(mutated)):
            # Get constraint density for this circle (higher means tighter constraints)
            density = constraint_density.get(i, 1.0)
            # Adaptive step size based on density and generation
            step_size = 0.02 * np.exp(-density) * (0.5 + 0.5 * np.exp(-generation/100))

            if random.random() < mutation_rate:
                # Mutate position
                mutated[i, 0] += np.random.normal(0, step_size)
                mutated[i, 1] += np.random.normal(0, step_size)

                # Mutate radius with different behavior based on density
                if density > 2.0:  # High constraint area
                    radius_step = np.random.normal(0, step_size * 0.3)
                elif density < 0.5:  # Low constraint area
                    radius_step = np.random.normal(0, step_size * 3.0)
                else:  # Medium constraint area
                    radius_step = np.random.normal(0, step_size * 1.5)

                mutated[i, 2] = max(0.001, mutated[i, 2] + radius_step)

                # Fix bounds
                if not is_valid_position(mutated[i, 0], mutated[i, 1], mutated[i, 2]):
                    # Revert to valid position and recompute
                    mutated[i, 0] = max(mutated[i, 2], min(width - mutated[i, 2], mutated[i, 0]))
                    mutated[i, 1] = max(mutated[i, 2], min(height - mutated[i, 2], mutated[i, 1]))

        return mutated

    def local_optimization(circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
        """Perform local optimization focusing on improving radii"""
        current = circles.copy()
        best_fitness = evaluate_fitness(current)[0]

        for iteration in range(max_iterations):
            improved = False
            # Try small adjustments to each circle
            for i in range(len(current)):
                x, y, r = current[i]
                best_r = r
                best_x, best_y = x, y
                best_fitness_local = best_fitness

                # Test nearby positions with different resolution based on radius
                resolution = max(0.002, r * 0.1)  # Finer resolution for smaller circles
                steps = int(0.02 / resolution)
                if steps < 1:
                    steps = 1

                # Test a wider range of positions for better optimization
                dx_range = np.linspace(-0.02, 0.02, steps * 2 + 1)
                dy_range = np.linspace(-0.02, 0.02, steps * 2 + 1)

                for dx in dx_range:
                    for dy in dy_range:
                        new_x, new_y = x + dx, y + dy
                        new_r = r

                        # Check if new position is valid
                        if is_valid_position(new_x, new_y, new_r):
                            # Check collision with others
                            valid = True
                            for j in range(len(current)):
                                if i != j:
                                    ox, oy, oradius = current[j]
                                    distance = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                                    if distance < (new_r + oradius):
                                        valid = False
                                        break

                            if valid:
                                # Test if we can increase radius
                                # Calculate maximum possible radius at new position
                                max_radius_x = min(new_x, width - new_x)
                                max_radius_y = min(new_y, height - new_y)

                                # Consider neighbor interactions when determining max radius
                                # Make sure we don't create overlaps with neighbors
                                max_radius = min(max_radius_x, max_radius_y)

                                # Add safety margin to ensure no overlaps with neighbors
                                for j in range(len(current)):
                                    if i != j:
                                        ox, oy, oradius = current[j]
                                        distance = np.sqrt((new_x - ox)**2 + (new_y - oy)**2)
                                        # Reduce radius to maintain minimum distance from neighbors
                                        max_radius = min(max_radius, distance - oradius - 0.001)

                                test_r = max(0.001, min(new_r, max_radius))

                                test_circles = current.copy()
                                test_circles[i] = [new_x, new_y, test_r]

                                new_fitness, _ = evaluate_fitness(test_circles)

                                if new_fitness > best_fitness_local:
                                    best_fitness_local = new_fitness
                                    best_r = test_r
                                    best_x, best_y = new_x, new_y
                                    improved = True

                current[i] = [best_x, best_y, best_r]

            # Update best fitness
            new_fitness, _ = evaluate_fitness(current)
            if new_fitness > best_fitness:
                best_fitness = new_fitness
            elif not improved:
                break  # No improvement, stop early

        return current

    # Main algorithm

    # Initialize with hexagonal packing
    circles = initialize_hexagonal_arrangement(width, height, 21)

    # Evolution parameters
    generations = 150
    population_size = 50
    elite_size = 5

    # Initialize population
    population = [circles.copy()]

    # Add some variation to initial population
    for _ in range(population_size - 1):
        mutated = circles.copy()
        for i in range(len(mutated)):
            mutated[i, 0] += np.random.uniform(-0.02, 0.02)
            mutated[i, 1] += np.random.uniform(-0.02, 0.02)
            mutated[i, 2] += np.random.uniform(-0.01, 0.01)
            mutated[i, 2] = max(0.001, mutated[i, 2])
        population.append(mutated)

    best_solution = None
    best_fitness = -1e10

    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            score, _ = evaluate_fitness(individual)
            fitness_scores.append(score)

        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]

        # Keep elite
        elite = sorted_population[:elite_size]

        # Update best solution
        if sorted_fitness[0] > best_fitness:
            best_fitness = sorted_fitness[0]
            best_solution = sorted_population[0].copy()

        # Create new population
        new_population = elite.copy()

        # Generate offspring through mutation with adaptation
        while len(new_population) < population_size:
            # Select parent from top half
            parent_idx = random.randint(0, population_size // 2 - 1)
            parent = sorted_population[parent_idx]

            # Add constraint density information
            constraint_density = calculate_constraint_density([(x, y, r) for x, y, r in parent])

            # Mutate parent
            child = adaptive_mutation(parent, constraint_density, gen)

            # Local optimization on child
            child = local_optimization(child, max_iterations=5)

            new_population.append(child)

        population = new_population[:population_size]

    # Final local optimization
    if best_solution is not None:
        final_solution = local_optimization(best_solution, max_iterations=50)
        return final_solution

    # Fallback to hexagonal arrangement if nothing worked
    return initialize_hexagonal_arrangement(width, height, 21)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")