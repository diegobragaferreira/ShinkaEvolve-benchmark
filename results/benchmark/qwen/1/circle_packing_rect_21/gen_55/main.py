# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def check_constraints(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> bool:
    """Efficiently check if all circles satisfy the constraints with early termination."""
    n = len(circles)
    
    # Check boundary constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
            return False
    
    # Check overlap constraints efficiently using distance matrix
    if n > 1:
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute distance matrix
        distances = cdist(positions, positions)
        
        # Check for overlaps
        for i in range(n):
            for j in range(i+1, n):
                distance = distances[i, j]
                overlap_distance = radii[i] + radii[j]
                if distance < overlap_distance:
                    return False
    
    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii with constraint validation."""
    if not check_constraints(circles):
        return -np.inf
    
    return np.sum(circles[:, 2])

def compute_voronoi_criticality(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Compute Voronoi cell areas for each circle to determine constraint density."""
    if len(circles) < 2:
        return np.ones(len(circles))
    
    # Extract circle centers
    points = circles[:, :2]
    
    # Add boundary points to ensure proper Voronoi diagram
    boundary_points = [
        [-1, -1], [rect_width + 1, -1], 
        [-1, rect_height + 1], [rect_width + 1, rect_height + 1],
        [rect_width/2, -1], [rect_width/2, rect_height + 1],
        [-1, rect_height/2], [rect_width + 1, rect_height/2]
    ]
    
    all_points = np.vstack([points, boundary_points])
    
    try:
        vor = Voronoi(all_points)
        
        # Calculate area for each Voronoi cell corresponding to original points
        cell_areas = []
        for i in range(len(points)):
            # Find vertices of Voronoi cell for point i
            region_indices = np.where(vor.point_region == i)[0]
            if len(region_indices) > 0:
                region_id = region_indices[0]
                vertices = vor.vertices[vor.regions[region_id]]
                if len(vertices) > 2:
                    # Compute polygon area using shoelace formula
                    vertices = np.array(vertices)
                    # Close the polygon
                    vertices = np.vstack([vertices, vertices[0]])
                    area = 0.5 * abs(sum(vertices[i][0] * vertices[i+1][1] - vertices[i+1][0] * vertices[i][1] 
                                       for i in range(len(vertices)-1)))
                    cell_areas.append(area)
                else:
                    cell_areas.append(1.0)  # Default for degenerate cases
            else:
                cell_areas.append(1.0)
    except:
        # Fallback if Voronoi computation fails
        cell_areas = [1.0] * len(points)
    
    # Normalize to get criticality measure (smaller areas = higher constraint density)
    areas = np.array(cell_areas)
    if np.max(areas) > 0:
        criticality = 1.0 / (areas + 1e-8)  # Higher criticality for smaller cells
        # Normalize to [0.1, 1.0] range
        normalized = 0.1 + 0.9 * (criticality - np.min(criticality)) / (np.max(criticality) - np.min(criticality) + 1e-8)
    else:
        normalized = np.ones(len(points))
    
    return normalized

def create_initial_solution(rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Create an initial solution using a combination of hexagonal packing and strategic placement."""
    circles = np.zeros((21, 3))
    
    # Start with hexagonal arrangement for dense packing
    rows = 5
    cols = 5
    
    # Calculate spacing for hexagonal pattern
    cell_width = rect_width * 0.8 / cols
    cell_height = rect_height * 0.8 / rows
    min_cell_dim = min(cell_width, cell_height)
    
    hex_radius = min_cell_dim * 0.3
    
    # Place circles in hexagonal pattern
    placed = 0
    for row in range(rows):
        if placed >= 21:
            break
        for col in range(cols):
            if placed >= 21:
                break
                
            # Offset every other row for hexagonal pattern
            offset = (row % 2) * (cell_width / 2)
            x = offset + col * cell_width + cell_width / 2
            y = row * cell_height + cell_height / 2
            
            # Adjust to stay within bounds
            x = np.clip(x, hex_radius, rect_width - hex_radius)
            y = np.clip(y, hex_radius, rect_height - hex_radius)
            
            # Calculate max possible radius for this position
            max_radius = min(x, y, rect_width - x, rect_height - y)
            r = min(hex_radius, max_radius * 0.9)
            
            circles[placed] = [x, y, r]
            placed += 1
    
    # Fill remaining positions with carefully placed circles near edges and corners
    edge_positions = [
        (rect_width * 0.2, rect_height * 0.2),
        (rect_width * 0.8, rect_height * 0.2),
        (rect_width * 0.2, rect_height * 0.8),
        (rect_width * 0.8, rect_height * 0.8),
        (rect_width / 2, rect_height * 0.1),
        (rect_width / 2, rect_height * 0.9),
        (rect_width * 0.1, rect_height / 2),
        (rect_width * 0.9, rect_height / 2)
    ]
    
    for i, (x, y) in enumerate(edge_positions):
        if placed >= 21:
            break
        # Place near edge with appropriate radius
        max_radius = min(x, y, rect_width - x, rect_height - y)
        r = min(0.1, max_radius * 0.3)
        circles[placed] = [x, y, r]
        placed += 1
    
    # Fill remaining with random circles
    for i in range(placed, 21):
        x = np.random.uniform(0.05, rect_width - 0.05)
        y = np.random.uniform(0.05, rect_height - 0.05)
        max_radius = min(x, y, rect_width - x, rect_height - y)
        r = min(0.05, max_radius * 0.2)
        circles[i] = [x, y, r]
        
    return circles

def mutate_with_voronoi_guidance(circles: np.ndarray, criticality: np.ndarray, 
                               mutation_rate: float = 0.3) -> np.ndarray:
    """Mutate circles with Voronoi-based guidance - conservative mutations in dense regions."""
    mutated = circles.copy()
    
    for i in range(21):
        if np.random.random() < mutation_rate:
            # Get criticality factor for this circle
            crit_factor = criticality[i]
            
            # Choose mutation type based on criticality
            mutation_type = np.random.choice(['position', 'radius'], 
                                            p=[0.6 + 0.4 * (1 - crit_factor), 0.4 * crit_factor])
            
            if mutation_type == 'position':
                # Mutate position with adaptive step size based on criticality
                base_step = 0.03 + 0.02 * (1 - crit_factor)  # More conservative in dense regions
                step_size = base_step * (0.8 + 0.4 * np.random.random())
                mutated[i, 0] += np.random.normal(0, step_size)
                mutated[i, 1] += np.random.normal(0, step_size)
            else:
                # Mutate radius with criticality-dependent scaling
                # More conservative in dense regions, more aggressive in sparse regions
                scale_factor = 1.0 + 0.5 * (1 - crit_factor) * np.random.normal(0, 0.3)
                mutated[i, 2] *= scale_factor
                mutated[i, 2] = max(0.001, mutated[i, 2])
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Improved crossover with Voronoi-aware recombination."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Select crossover points based on Voronoi criticality
    # More critical regions (smaller cells) get less aggressive recombination
    criticality1 = compute_voronoi_criticality(parent1)
    criticality2 = compute_voronoi_criticality(parent2)
    
    # For each circle, decide whether to keep from parent1 or parent2 based on criticality
    for i in range(21):
        # Use criticality to bias decision - more critical circles get more conservative mixing
        crit_factor = (criticality1[i] + criticality2[i]) / 2
        # Lower criticality (larger cells) gets more mixing
        mix_probability = 0.3 + 0.4 * (1 - crit_factor)
        
        if np.random.random() < mix_probability:
            # Swap genes with some probability
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
    
    return child1, child2

def repair_solution(circles: np.ndarray, rect_width: float = 1.0, rect_height: float = 1.0) -> np.ndarray:
    """Enhanced repair mechanism with Voronoi-based conflict resolution."""
    repaired = circles.copy()
    
    # Ensure positive radii
    repaired[:, 2] = np.maximum(repaired[:, 2], 0.001)
    
    # Enforce bounds
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        x = np.clip(x, r, rect_width - r)
        y = np.clip(y, r, rect_height - r)
        repaired[i] = [x, y, r]
    
    # Resolve overlaps iteratively with Voronoi-guided approach
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
            
        # Move conflicting pairs apart with Voronoi-aware strategy
        for i, j in conflicts:
            x1, y1, r1 = repaired[i]
            x2, y2, r2 = repaired[j]
            
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Move circles away from each other
                move_distance = (r1 + r2 - distance) / 2
                dx_norm = dx / distance
                dy_norm = dy / distance
                
                # Apply movement with bounded adjustment
                move1 = move_distance * r2 / (r1 + r2 + 1e-8)
                move2 = move_distance * r1 / (r1 + r2 + 1e-8)
                
                # Voronoi-aware movement - reduce movement in critical regions
                crit_i = compute_voronoi_criticality(repaired, rect_width, rect_height)[i]
                crit_j = compute_voronoi_criticality(repaired, rect_width, rect_height)[j]
                avg_crit = (crit_i + crit_j) / 2
                
                reduced_move = move_distance * (0.3 + 0.7 * avg_crit)  # Less movement in critical areas
                
                repaired[i, 0] -= dx_norm * move1 * 0.5 * (0.3 + 0.7 * crit_i)
                repaired[i, 1] -= dy_norm * move1 * 0.5 * (0.3 + 0.7 * crit_i)
                repaired[j, 0] += dx_norm * move2 * 0.5 * (0.3 + 0.7 * crit_j)
                repaired[j, 1] += dy_norm * move2 * 0.5 * (0.3 + 0.7 * crit_j)
                
                # Keep within bounds
                repaired[i, 0] = np.clip(repaired[i, 0], r1, rect_width - r1)
                repaired[i, 1] = np.clip(repaired[i, 1], r1, rect_height - r1)
                repaired[j, 0] = np.clip(repaired[j, 0], r2, rect_width - r2)
                repaired[j, 1] = np.clip(repaired[j, 1], r2, rect_height - r2)
    
    return repaired

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimized rectangle dimensions
    rect_width = 1.5
    rect_height = 0.5

    # Parameters for evolutionary algorithm
    population_size = 60
    generations = 120
    elite_size = 10
    tournament_size = 8

    # Initialize population with better quality solutions
    population = []
    for _ in range(population_size):
        solution = create_initial_solution(rect_width, rect_height)
        population.append(solution)

    # Track best fitness for convergence detection
    previous_best = -np.inf
    stagnation_count = 0
    
    # Evolutionary algorithm
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_fitness(individual) for individual in population]
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elite
        elite = population[:elite_size]
        
        # Generate new population
        new_population = elite[:]
        
        # Create offspring using tournament selection and crossover
        while len(new_population) < population_size:
            # Tournament selection - select two parents
            parent1_idx = sorted_indices[np.random.choice(min(tournament_size, len(sorted_indices)))]
            parent2_idx = sorted_indices[np.random.choice(min(tournament_size, len(sorted_indices)))]
            
            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Get criticality information for both parents
            criticality1 = compute_voronoi_criticality(parent1)
            criticality2 = compute_voronoi_criticality(parent2)
            
            # Mutate with Voronoi guidance
            child1 = mutate_with_voronoi_guidance(child1, criticality1)
            child2 = mutate_with_voronoi_guidance(child2, criticality2)
            
            # Repair
            child1 = repair_solution(child1, rect_width, rect_height)
            child2 = repair_solution(child2, rect_width, rect_height)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
        
        # Convergence detection
        current_best = max(fitness_scores)
        if abs(current_best - previous_best) < 1e-6:
            stagnation_count += 1
        else:
            stagnation_count = 0
        previous_best = current_best
        
        # Early stopping if stagnated too long
        if stagnation_count > 25:
            print(f"Early stopping at generation {generation} due to convergence")
            break
            
        # Print progress
        if generation % 20 == 0:
            print(f"Generation {generation}: Best fitness = {current_best:.6f}")
    
    # Return the best solution
    fitness_scores = [evaluate_fitness(individual) for individual in population]
    best_idx = np.argmax(fitness_scores)
    best_solution = population[best_idx]
    
    # Final validation
    final_fitness = evaluate_fitness(best_solution)
    if final_fitness == -np.inf:
        print("Warning: Final solution violated constraints. Returning fallback.")
        # Fallback to best valid solution found during evolution
        for i in range(len(population)):
            if evaluate_fitness(population[i]) > -np.inf:
                return population[i]
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")