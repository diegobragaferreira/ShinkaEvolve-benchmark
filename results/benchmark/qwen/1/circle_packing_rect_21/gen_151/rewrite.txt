# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
from collections import defaultdict

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
    # Optimized rectangle dimensions based on prior experiments
    width, height = 1.5, 0.5

    def is_valid_position(x: float, y: float, r: float) -> bool:
        """Check if circle is within bounds"""
        return (r <= x <= width - r) and (r <= y <= height - r)

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
        # Query neighbors within 2*(r_max) distance 
        indices = spatial_tree.query_ball_point([x, y], max_distance)
        return indices

    def calculate_combined_constraints(circles_list: List[Tuple[float, float, float]]) -> dict:
        """Calculate combined constraint information using both Voronoi and grid-based approaches"""
        if len(circles_list) < 2:
            return {i: 1.0 for i in range(len(circles_list))}
            
        # Voronoi-based constraint calculation (primary method)
        try:
            points = np.array([[x, y] for x, y, r in circles_list])
            vor = Voronoi(points)
            voronoi_constraints = {}
            for i, (x, y, r) in enumerate(circles_list):
                voronoi_constraints[i] = 1.0  # Default
        except:
            voronoi_constraints = {i: 1.0 for i in range(len(circles_list))}
        
        # Grid-based constraint calculation (secondary backup)
        grid_resolution = 0.2
        grid_width = int(width / grid_resolution) + 1
        grid_height = int(height / grid_resolution) + 1

        grid_counts = defaultdict(int)
        grid_centers = {}

        for x, y, r in circles_list:
            grid_x = int(x / grid_resolution)
            grid_y = int(y / grid_resolution)
            grid_counts[(grid_x, grid_y)] += 1
            if (grid_x, grid_y) not in grid_centers:
                grid_centers[(grid_x, grid_y)] = (
                    (grid_x + 0.5) * grid_resolution,
                    (grid_y + 0.5) * grid_resolution
                )

        grid_constraints = {}
        for i, (x, y, r) in enumerate(circles_list):
            neighborhood_radius = r * 2.0
            neighbors = 0

            grid_x = int(x / grid_resolution)
            grid_y = int(y / grid_resolution)

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = grid_x + dx, grid_y + dy
                    if (nx, ny) in grid_counts:
                        gx, gy = grid_centers[(nx, ny)]
                        dist = np.sqrt((x - gx)**2 + (y - gy)**2)
                        if dist <= neighborhood_radius:
                            neighbors += grid_counts[(nx, ny)]

            effective_neighbors = max(0, neighbors - 1)
            grid_constraints[i] = 1.0 + effective_neighbors * 0.5

        # Combine both constraint measures
        combined_constraints = {}
        for i in range(len(circles_list)):
            voronoi_val = voronoi_constraints.get(i, 1.0)
            grid_val = grid_constraints.get(i, 1.0)
            # Weighted combination favoring Voronoi when available
            combined_constraints[i] = 0.7 * voronoi_val + 0.3 * grid_val
            
        return combined_constraints

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
        """Initialize circles in a hexagonal pattern with better packing efficiency"""
        circles = np.zeros((n_circles, 3))
        
        # Better hexagonal layout using mathematical approximation for optimal packing
        rows = int(np.ceil(np.sqrt(n_circles)))
        cols = int(np.ceil(n_circles / rows))
        
        # Adjust for the specific rectangle dimensions
        grid_width = width * 0.95
        grid_height = height * 0.95
        
        cell_width = grid_width / cols
        cell_height = grid_height / rows
        min_cell_dim = min(cell_width, cell_height)
        
        # Optimize circle radius for better packing
        hex_radius = min_cell_dim * 0.4
        
        placed = 0
        for row in range(rows):
            if placed >= n_circles:
                break
            for col in range(cols):
                if placed >= n_circles:
                    break
                    
                # Offset every other row for hexagonal pattern
                offset = (row % 2) * (cell_width / 2)
                x = offset + col * cell_width + cell_width / 2
                y = row * cell_height + cell_height / 2
                
                # Ensure we're within bounds
                x = np.clip(x, hex_radius, width - hex_radius)
                y = np.clip(y, hex_radius, height - hex_radius)
                
                # Adjust radius to prevent boundary issues
                max_radius = min(x, y, width - x, height - y)
                r = min(hex_radius, max_radius * 0.8)
                
                circles[placed] = [x, y, r]
                placed += 1
                
        # Fill remaining positions with random small circles for diversity
        for i in range(placed, n_circles):
            x = np.random.uniform(hex_radius, width - hex_radius)
            y = np.random.uniform(hex_radius, height - hex_radius)
            r = np.random.uniform(0.005, hex_radius * 0.3)
            circles[i] = [x, y, r]
            
        return circles

    def adaptive_mutation(circles: np.ndarray, constraint_density: dict, generation: int) -> np.ndarray:
        """Mutate circles with adaptive step sizes based on density and improved variance control"""
        mutated = circles.copy()
        mutation_rate = 0.2 + 0.1 * np.exp(-generation/50)  # Decrease over generations

        for i in range(len(mutated)):
            # Get constraint density for this circle (higher means tighter constraints)
            density = constraint_density.get(i, 1.0)
            
            # Adaptive step size based on density and generation
            # Modify the step size formula with better scaling
            base_step = 0.03
            density_factor = 1.0 / (1.0 + density * 0.7)
            generation_factor = 0.5 + 0.5 * np.exp(-generation/100)
            step_size = base_step * density_factor * generation_factor

            if random.random() < mutation_rate:
                # Mutate position with adaptive step
                mutated[i, 0] += np.random.normal(0, step_size)
                mutated[i, 1] += np.random.normal(0, step_size)

                # Mutate radius with density-dependent behavior
                if density > 2.0:  # High constraint area - be very conservative
                    radius_step = np.random.normal(0, step_size * 0.2)
                elif density < 1.0:  # Low constraint area - be more aggressive
                    radius_step = np.random.normal(0, step_size * 2.0)
                else:  # Medium constraint area
                    radius_step = np.random.normal(0, step_size * 1.0)

                mutated[i, 2] = max(0.001, mutated[i, 2] + radius_step)

                # Boundary checking and correction with smarter constraint handling
                mutated[i, 0] = max(mutated[i, 2], min(width - mutated[i, 2], mutated[i, 0]))
                mutated[i, 1] = max(mutated[i, 2], min(height - mutated[i, 2], mutated[i, 1]))

                # Ensure radius remains feasible given position
                max_radius_x = min(mutated[i, 0], width - mutated[i, 0])
                max_radius_y = min(mutated[i, 1], height - mutated[i, 1])
                mutated[i, 2] = min(mutated[i, 2], max_radius_x, max_radius_y)

        return mutated

    def local_optimization(circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
        """Perform enhanced local optimization focusing on improving radii and position"""
        current = circles.copy()
        best_fitness = evaluate_fitness(current)[0]

        # Multi-resolution search for better optimization
        resolutions = [0.02, 0.01, 0.005]  # Different search resolutions
        
        for iteration in range(max_iterations):
            improved = False
            # Try small adjustments to each circle
            for i in range(len(current)):
                x, y, r = current[i]
                best_r = r
                best_x, best_y = x, y
                best_fitness_local = best_fitness

                # Use multi-resolution search based on current radius
                search_resolution = max(0.001, r * 0.1)
                steps = max(3, int(0.02 / search_resolution))
                
                # Test nearby positions with varying resolution
                for dx in np.linspace(-0.02, 0.02, steps):
                    for dy in np.linspace(-0.02, 0.02, steps):
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
                                test_r = min(
                                    new_r,
                                    new_x, width - new_x,
                                    new_y, height - new_y
                                )

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
    generations = 120
    population_size = 60
    elite_size = 8

    # Initialize population with better diversity
    population = [circles.copy()]
    
    # Add variations to initial population
    for _ in range(population_size - 1):
        mutated = circles.copy()
        for i in range(len(mutated)):
            mutated[i, 0] += np.random.uniform(-0.03, 0.03)
            mutated[i, 1] += np.random.uniform(-0.03, 0.03)
            mutated[i, 2] += np.random.uniform(-0.015, 0.015)
            mutated[i, 2] = max(0.001, mutated[i, 2])
        population.append(mutated)

    best_solution = None
    best_fitness = -1e10

    # Evolution loop with convergence tracking
    stagnation_count = 0
    previous_best = -np.inf
    
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
            stagnation_count = 0  # Reset stagnation counter
        else:
            stagnation_count += 1

        # Early stopping if no improvement for many generations
        if stagnation_count > 25:
            print(f"Early stopping at generation {gen} due to lack of improvement")
            break

        # Create new population
        new_population = elite.copy()

        # Generate offspring through mutation with adaptation
        while len(new_population) < population_size:
            # Select parent from top half
            parent_idx = random.randint(0, population_size // 2 - 1)
            parent = sorted_population[parent_idx]

            # Add constraint density information
            constraint_density = calculate_combined_constraints([(x, y, r) for x, y, r in parent])

            # Mutate parent
            child = adaptive_mutation(parent, constraint_density, gen)

            # Local optimization on child
            child = local_optimization(child, max_iterations=8)

            new_population.append(child)

        population = new_population[:population_size]

    # Final local optimization with more iterations
    if best_solution is not None:
        final_solution = local_optimization(best_solution, max_iterations=70)
        return final_solution

    # Fallback to hexagonal arrangement if nothing worked
    return initialize_hexagonal_arrangement(width, height, 21)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")