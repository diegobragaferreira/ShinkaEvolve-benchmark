# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import differential_evolution
import random
from typing import Tuple, List
import time
from collections import defaultdict

# Set seeds for determinism
random.seed(42)
np.random.seed(42)

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_valid_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Check if all circles are within bounds and non-overlapping using efficient spatial hashing."""
    n = len(circles)

    # Check bounds - vectorized
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]

    if np.any(x_coords - radii < 0) or np.any(x_coords + radii > rect_width) or \
       np.any(y_coords - radii < 0) or np.any(y_coords + radii > rect_height):
        return False

    # Use spatial indexing to detect overlaps efficiently
    if len(radii) > 0:
        avg_radius = np.mean(radii)
        # Dynamic cell size based on average radius
        cell_size = max(0.001, avg_radius * 1.2)  # Slightly larger than average radius for efficiency
    else:
        cell_size = 0.01  # Fallback

    # Grid dimensions
    grid_width = int(np.ceil(rect_width / cell_size)) + 1
    grid_height = int(np.ceil(rect_height / cell_size)) + 1

    # Initialize grid with lists for each cell
    grid = {}

    # Place circles into grid cells
    for i in range(n):
        x, y, r = circles[i]
        # Calculate grid bounds for this circle
        min_cell_x = max(0, int((x - r) / cell_size))
        max_cell_x = min(grid_width - 1, int((x + r) / cell_size))
        min_cell_y = max(0, int((y - r) / cell_size))
        max_cell_y = min(grid_height - 1, int((y + r) / cell_size))

        # Add circle to all relevant grid cells
        for gx in range(min_cell_x, max_cell_x + 1):
            for gy in range(min_cell_y, max_cell_y + 1):
                if (gx, gy) not in grid:
                    grid[(gx, gy)] = []
                grid[(gx, gy)].append(i)

    # Check for overlaps within each grid cell and adjacent cells
    for i in range(n):
        x1, y1, r1 = circles[i]
        # Get grid cell coordinates
        cell_x = int(x1 / cell_size)
        cell_y = int(y1 / cell_size)

        # Check nearby cells (3x3 neighborhood)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = cell_x + dx, cell_y + dy
                if (nx, ny) in grid:
                    for j in grid[(nx, ny)]:
                        # Skip self-comparison and ensure we don't double-check
                        if i >= j:
                            continue
                        x2, y2, r2 = circles[j]
                        # Check if circles overlap using squared distances for efficiency
                        dx_squared = (x1 - x2)**2
                        dy_squared = (y1 - y2)**2
                        distance_squared = dx_squared + dy_squared
                        radii_sum = r1 + r2
                        if distance_squared < radii_sum * radii_sum:
                            return False

    return True

def compute_voronoi_density(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Compute Voronoi cell areas for each circle to estimate constraint density."""
    # Add boundary points for proper Voronoi calculation
    points = circles[:, :2].copy()

    # Add boundary points to make Voronoi more meaningful
    boundary_points = [
        [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
        [rect_width/2, 0], [rect_width/2, rect_height],
        [0, rect_height/2], [rect_width, rect_height/2]
    ]
    points = np.vstack([points, boundary_points])

    try:
        vor = Voronoi(points)

        # For each original point, compute Voronoi cell area
        areas = []
        for i in range(len(circles)):
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1

            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    # Compute area of polygon using shoelace formula
                    vertices = np.array([vor.vertices[i] for i in region])
                    if len(vertices) >= 3:
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
            else:
                areas.append(1.0)

        return np.array(areas)
    except:
        # Fallback to uniform distribution if Voronoi fails
        return np.ones(len(circles))

def compute_voronoi_metrics(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Voronoi cell areas and boundary distances for each circle."""
    # Add boundary points for proper Voronoi calculation
    points = circles[:, :2].copy()
    
    # Add boundary points to make Voronoi more meaningful
    boundary_points = [
        [0, 0], [rect_width, 0], [0, rect_height], [rect_width, rect_height],
        [rect_width/2, 0], [rect_width/2, rect_height],
        [0, rect_height/2], [rect_width, rect_height/2]
    ]
    points = np.vstack([points, boundary_points])
    
    try:
        vor = Voronoi(points)
        
        # For each original point, compute Voronoi cell area and boundary distance
        areas = []
        boundary_distances = []
        
        for i in range(len(circles)):
            region_idx = np.where(vor.point_region == i)[0][0] if i in vor.point_region else -1
            
            if region_idx != -1 and region_idx < len(vor.regions):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) >= 3:
                    # Compute area of polygon using shoelace formula
                    vertices = np.array([vor.vertices[i] for i in region])
                    if len(vertices) >= 3:
                        x = vertices[:, 0]
                        y = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        areas.append(area)
                    else:
                        areas.append(1.0)
                else:
                    areas.append(1.0)
            else:
                areas.append(1.0)
            
            # Compute minimum distance to boundary
            x, y = circles[i, 0], circles[i, 1]
            min_dist_to_boundary = min([
                x, rect_width - x, y, rect_height - y
            ])
            boundary_distances.append(min_dist_to_boundary)
                
        return np.array(areas), np.array(boundary_distances)
    except:
        # Fallback to uniform distribution if Voronoi fails
        return np.ones(len(circles)), np.ones(len(circles))

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as sum of radii."""
    return np.sum(circles[:, 2])

def generate_initial_population(n: int, pop_size: int = 50, rect_width: float = 1.0, rect_height: float = 1.0) -> List[np.ndarray]:
    """Generate diverse initial population using multiple packing strategies."""
    population = []
    
    # Strategy 1: Hexagonal packing
    def hexagonal_strategy():
        circles = np.zeros((n, 3))
        rows = int(np.sqrt(n))
        cols = int(n / rows) + 1
        spacing_x = rect_width / (cols + 0.5)
        spacing_y = rect_height / (rows + 0.5)
        initial_radius = min(spacing_x, spacing_y) * 0.3
        
        for i in range(n):
            row = i // cols
            col = i % cols
            x = (col + 0.5) * spacing_x
            y = (row + 0.5) * spacing_y
            if row % 2 == 1:
                x += spacing_x / 2
            circles[i] = [x, y, initial_radius]
        return circles
    
    # Strategy 2: Triangular packing
    def triangular_strategy():
        circles = np.zeros((n, 3))
        sqrt_n = int(np.ceil(np.sqrt(n)))
        spacing_x = rect_width / (sqrt_n + 1.5)
        spacing_y = rect_height / (sqrt_n + 1.5)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                if i % 2 == 1:
                    x += spacing_x / 2
                circles[idx] = [x, y, 0.03]
                idx += 1
            if idx >= n:
                break
        return circles
    
    # Strategy 3: Square packing (denser)
    def square_strategy():
        circles = np.zeros((n, 3))
        sqrt_n = int(np.ceil(np.sqrt(n)))
        spacing_x = rect_width / (sqrt_n + 1.2)
        spacing_y = rect_height / (sqrt_n + 1.2)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                circles[idx] = [x, y, 0.03]
                idx += 1
            if idx >= n:
                break
        return circles
    
    # Strategy 4: Random with clustering
    def random_strategy():
        circles = np.zeros((n, 3))
        # Create clusters of circles
        for i in range(n):
            cluster_center_x = np.random.uniform(0.1, rect_width - 0.1)
            cluster_center_y = np.random.uniform(0.1, rect_height - 0.1)
            # Place circle around cluster center
            x = np.random.normal(cluster_center_x, 0.05)
            y = np.random.normal(cluster_center_y, 0.05)
            # Clip to stay within bounds
            x = np.clip(x, 0.05, rect_width - 0.05)
            y = np.clip(y, 0.05, rect_height - 0.05)
            r = np.random.uniform(0.02, 0.06)
            circles[i] = [x, y, r]
        return circles
    
    strategies = [hexagonal_strategy, triangular_strategy, square_strategy, random_strategy]
    
    # Generate initial population
    for _ in range(pop_size):
        strategy = random.choice(strategies)
        individual = strategy()
        population.append(individual)
    
    # Add some diversity with slight mutations of best patterns
    if population:
        best_candidate = population[0]
        for _ in range(pop_size // 4):
            mutated = best_candidate.copy()
            # Slight random adjustments
            for i in range(len(mutated)):
                if np.random.random() < 0.3:
                    mutated[i, 0] += np.random.uniform(-0.02, 0.02)
                    mutated[i, 1] += np.random.uniform(-0.02, 0.02)
                    mutated[i, 2] += np.random.uniform(-0.01, 0.01)
                    mutated[i, 0] = np.clip(mutated[i, 0], 0.01, rect_width - 0.01)
                    mutated[i, 1] = np.clip(mutated[i, 1], 0.01, rect_height - 0.01)
                    mutated[i, 2] = max(0.001, mutated[i, 2])
            population.append(mutated)
    
    return population

def density_weighted_mutation(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0,
                              mutation_rate: float = 0.3) -> np.ndarray:
    """Mutation operator weighted by Voronoi density for more intelligent adaptation."""
    new_circles = circles.copy()
    
    # Get Voronoi metrics for weighting
    voronoi_areas, boundary_distances = compute_voronoi_metrics(new_circles, rect_width, rect_height)
    
    # Normalize metrics
    normalized_voronoi = voronoi_areas / np.max(voronoi_areas) if np.max(voronoi_areas) > 0 else voronoi_areas
    normalized_boundaries = boundary_distances / np.max(boundary_distances) if np.max(boundary_distances) > 0 else boundary_distances
    
    # Density weight (lower area = higher density/constraint)
    density_weights = 1.0 - normalized_voronoi
    boundary_weights = 1.0 - normalized_boundaries 
    
    # Combined weight - prioritize both high constraint and boundary proximity
    combined_weights = 0.6 * density_weights + 0.4 * boundary_weights
    
    # Apply mutation with weights
    for i in range(len(new_circles)):
        if np.random.random() < mutation_rate:
            # Higher probability for mutation in high-density areas
            prob = combined_weights[i] * 0.3 + 0.1  # Base probability plus density influence
            
            if np.random.random() < prob:
                # Choose mutation type based on weights: position mutations more likely in high constraint areas
                if combined_weights[i] > 0.7:
                    # High constraint: more aggressive radius adjustments
                    if np.random.random() < 0.5:
                        # Radius mutation (larger step)
                        old_r = new_circles[i, 2]
                        delta = np.random.uniform(-0.02, 0.02)
                        new_r = max(0.001, old_r + delta)
                        new_circles[i, 2] = new_r
                    else:
                        # Position mutation (larger step)
                        old_x, old_y = new_circles[i, 0], new_circles[i, 1]
                        delta_x = np.random.uniform(-0.03, 0.03)
                        delta_y = np.random.uniform(-0.03, 0.03)
                        new_x = np.clip(old_x + delta_x, 0.01, rect_width - 0.01)
                        new_y = np.clip(old_y + delta_y, 0.01, rect_height - 0.01)
                        new_circles[i, 0] = new_x
                        new_circles[i, 1] = new_y
                elif combined_weights[i] > 0.4:
                    # Medium constraint: balanced approach
                    if np.random.random() < 0.5:
                        # Radius mutation (medium step)
                        old_r = new_circles[i, 2]
                        delta = np.random.uniform(-0.01, 0.01)
                        new_r = max(0.001, old_r + delta)
                        new_circles[i, 2] = new_r
                    else:
                        # Position mutation (medium step)
                        old_x, old_y = new_circles[i, 0], new_circles[i, 1]
                        delta_x = np.random.uniform(-0.02, 0.02)
                        delta_y = np.random.uniform(-0.02, 0.02)
                        new_x = np.clip(old_x + delta_x, 0.01, rect_width - 0.01)
                        new_y = np.clip(old_y + delta_y, 0.01, rect_height - 0.01)
                        new_circles[i, 0] = new_x
                        new_circles[i, 1] = new_y
                else:
                    # Low constraint: conservative adjustments
                    if np.random.random() < 0.3:
                        # Radius mutation (small step)
                        old_r = new_circles[i, 2]
                        delta = np.random.uniform(-0.005, 0.005)
                        new_r = max(0.001, old_r + delta)
                        new_circles[i, 2] = new_r
                    else:
                        # Position mutation (small step)
                        old_x, old_y = new_circles[i, 0], new_circles[i, 1]
                        delta_x = np.random.uniform(-0.01, 0.01)
                        delta_y = np.random.uniform(-0.01, 0.01)
                        new_x = np.clip(old_x + delta_x, 0.01, rect_width - 0.01)
                        new_y = np.clip(old_y + delta_y, 0.01, rect_height - 0.01)
                        new_circles[i, 0] = new_x
                        new_circles[i, 1] = new_y
    
    return new_circles

def density_weighted_selection(population: List[np.ndarray], rect_width: float = 1.0, rect_height: float = 1.0) -> List[np.ndarray]:
    """Selection operator that favors individuals with fewer constraint conflicts."""
    fitness_scores = []
    for individual in population:
        # Fitness is sum of radii, but penalize for high constraint density
        base_fitness = evaluate_fitness(individual)
        
        # Compute constraint density (lower area = higher density)
        try:
            voronoi_areas, _ = compute_voronoi_metrics(individual, rect_width, rect_height)
            avg_area = np.mean(voronoi_areas)
            # Penalize for low areas (high constraint density)
            constraint_penalty = 1.0 / (avg_area + 0.01)  # Add small value to prevent division by zero
            penalty_factor = constraint_penalty * 0.1  # Scale penalty
            adjusted_fitness = base_fitness - penalty_factor
        except:
            adjusted_fitness = base_fitness
            
        fitness_scores.append(adjusted_fitness)
    
    # Sort by fitness descending
    sorted_indices = np.argsort(fitness_scores)[::-1]
    
    # Select top half and add some diversity
    selected_indices = sorted_indices[:len(sorted_indices)//2]
    
    # Add some random individuals to maintain diversity
    additional_indices = random.sample(range(len(population)), len(population)//4)
    selected_indices.extend(additional_indices)
    
    return [population[i] for i in selected_indices]

def evolve_population(population: List[np.ndarray], rect_width: float = 1.0, rect_height: float = 1.0, 
                     generations: int = 30, mutation_rate: float = 0.3) -> np.ndarray:
    """Evolutionary process with density-aware operators."""
    current_pop = population.copy()
    
    for gen in range(generations):
        # Selection
        selected = density_weighted_selection(current_pop, rect_width, rect_height)
        
        # Create offspring through crossover and mutation
        offspring = []
        
        # Keep best individuals
        best_individuals = selected[:len(selected)//2]
        offspring.extend(best_individuals)
        
        # Generate new individuals through mutation
        while len(offspring) < len(current_pop):
            # Select parent
            parent = random.choice(selected)
            
            # Mutate
            mutated = density_weighted_mutation(parent, rect_width, rect_height, mutation_rate)
            
            # Validate and add to offspring
            if is_valid_solution(mutated, rect_width, rect_height):
                offspring.append(mutated)
            else:
                # Try to repair invalid solution
                repaired = parent.copy()
                # Apply small random fixes to restore validity
                for i in range(len(repaired)):
                    x, y, r = repaired[i]
                    if x - r < 0:
                        repaired[i, 0] = r + 0.01
                    elif x + r > rect_width:
                        repaired[i, 0] = rect_width - r - 0.01
                    if y - r < 0:
                        repaired[i, 1] = r + 0.01
                    elif y + r > rect_height:
                        repaired[i, 1] = rect_height - r - 0.01
                if is_valid_solution(repaired, rect_width, rect_height):
                    offspring.append(repaired)
                else:
                    # Return original if we can't repair
                    offspring.append(parent)
        
        # Trim to correct population size
        current_pop = offspring[:len(current_pop)]
    
    # Return best individual from final population
    fitness_scores = [evaluate_fitness(individual) for individual in current_pop]
    best_index = np.argmax(fitness_scores)
    return current_pop[best_index]

def boundary_aware_refinement(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0, 
                             max_iter: int = 50) -> np.ndarray:
    """Refine solution focusing on boundary constraints."""
    current = circles.copy()
    
    # Identify boundary circles (near edges)
    boundary_indices = []
    for i in range(len(current)):
        x, y, r = current[i]
        if x <= r + 0.02 or x >= rect_width - r - 0.02 or \
           y <= r + 0.02 or y >= rect_height - r - 0.02:
            boundary_indices.append(i)
    
    # Refine boundary circles with stricter constraints
    for _ in range(max_iter):
        improved = False
        
        # Focus on boundary circles first
        for i in boundary_indices:
            # Try various mutations with smaller steps
            mutated_pos = current.copy()
            old_x, old_y = mutated_pos[i, 0], mutated_pos[i, 1]
            
            # Smaller position adjustment
            delta_x = np.random.uniform(-0.005, 0.005)
            delta_y = np.random.uniform(-0.005, 0.005)
            
            new_x = old_x + delta_x
            new_y = old_y + delta_y
            
            # Ensure within bounds
            new_x = np.clip(new_x, 0.01, rect_width - 0.01)
            new_y = np.clip(new_y, 0.01, rect_height - 0.01)
            
            mutated_pos[i, 0] = new_x
            mutated_pos[i, 1] = new_y
            
            # Try radius adjustment
            mutated_rad = current.copy()
            old_r = mutated_rad[i, 2]
            delta = np.random.uniform(-0.002, 0.002)
            new_r = max(0.001, old_r + delta)
            mutated_rad[i, 2] = new_r
            
            # Test both variations
            pos_fitness = evaluate_fitness(mutated_pos)
            rad_fitness = evaluate_fitness(mutated_rad)
            
            # Choose better variation if valid
            if pos_fitness > rad_fitness:
                if is_valid_solution(mutated_pos, rect_width, rect_height):
                    current = mutated_pos
                    improved = True
            else:
                if is_valid_solution(mutated_rad, rect_width, rect_height):
                    current = mutated_rad
                    improved = True
        
        # Then refine non-boundary circles
        non_boundary = [i for i in range(len(current)) if i not in boundary_indices]
        for i in non_boundary:
            mutated_pos = current.copy()
            old_x, old_y = mutated_pos[i, 0], mutated_pos[i, 1]
            
            # Larger adjustments for interior circles
            delta_x = np.random.uniform(-0.01, 0.01)
            delta_y = np.random.uniform(-0.01, 0.01)
            
            new_x = old_x + delta_x
            new_y = old_y + delta_y
            
            # Ensure within bounds
            new_x = np.clip(new_x, 0.01, rect_width - 0.01)
            new_y = np.clip(new_y, 0.01, rect_height - 0.01)
            
            mutated_pos[i, 0] = new_x
            mutated_pos[i, 1] = new_y
            
            # Try radius adjustment
            mutated_rad = current.copy()
            old_r = mutated_rad[i, 2]
            delta = np.random.uniform(-0.005, 0.005)
            new_r = max(0.001, old_r + delta)
            mutated_rad[i, 2] = new_r
            
            # Test both variations
            pos_fitness = evaluate_fitness(mutated_pos)
            rad_fitness = evaluate_fitness(mutated_rad)
            
            # Choose better variation if valid
            if pos_fitness > rad_fitness:
                if is_valid_solution(mutated_pos, rect_width, rect_height):
                    current = mutated_pos
                    improved = True
            else:
                if is_valid_solution(mutated_rad, rect_width, rect_height):
                    current = mutated_rad
                    improved = True
                
        # Early stopping if no improvement
        if not improved:
            break
            
    return current

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    rect_width = 1.0
    rect_height = 1.0
    
    # Phase 1: Evolutionary population optimization
    initial_pop = generate_initial_population(n, pop_size=40, rect_width=rect_width, rect_height=rect_height)
    
    # Evolve population
    evolved_solution = evolve_population(initial_pop, rect_width, rect_height, generations=25, mutation_rate=0.25)
    
    # Phase 2: Boundary-aware refinement
    refined_solution = boundary_aware_refinement(evolved_solution, rect_width, rect_height, max_iter=30)
    
    # Phase 3: Final polishing with local optimization
    for _ in range(20):
        # Try to improve by adjusting each circle
        for i in range(n):
            mutated = refined_solution.copy()
            
            # Try position adjustment with small steps
            old_x, old_y = mutated[i, 0], mutated[i, 1]
            delta_x = np.random.uniform(-0.005, 0.005)
            delta_y = np.random.uniform(-0.005, 0.005)
            new_x = np.clip(old_x + delta_x, 0.01, rect_width - 0.01)
            new_y = np.clip(old_y + delta_y, 0.01, rect_height - 0.01)
            mutated[i, 0] = new_x
            mutated[i, 1] = new_y
            
            # Try radius adjustment
            old_r = mutated[i, 2]
            delta = np.random.uniform(-0.002, 0.002)
            new_r = max(0.001, old_r + delta)
            mutated[i, 2] = new_r
            
            # Accept if valid and better
            if is_valid_solution(mutated, rect_width, rect_height):
                if evaluate_fitness(mutated) > evaluate_fitness(refined_solution):
                    refined_solution = mutated
    
    # Final validation and cleanup
    if not is_valid_solution(refined_solution, rect_width, rect_height):
        # Revert to best known valid solution
        initial_pop = generate_initial_population(n, pop_size=20, rect_width=rect_width, rect_height=rect_height)
        best_individual = max(initial_pop, key=lambda x: evaluate_fitness(x) if is_valid_solution(x, rect_width, rect_height) else -float('inf'))
        refined_solution = best_individual
    
    return refined_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")