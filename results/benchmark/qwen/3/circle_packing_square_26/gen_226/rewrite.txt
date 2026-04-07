# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 120
GENERATIONS = 300
TOURNAMENT_SIZE = 4
MUTATION_RATE_START = 0.25
MUTATION_RATE_END = 0.01
ELITISM_COUNT = 15
BOUNDARY_MARGIN = 0.01
MAX_RADIUS_FACTOR = 0.45
MIN_RADIUS = 0.001

def generate_poisson_disk_points(n_points: int, width: float = 1.0, height: float = 1.0, 
                               min_distance: float = 0.05) -> np.ndarray:
    """Generate points using Poisson disk sampling with better control"""
    if n_points <= 0:
        return np.array([]).reshape(0, 2)
    
    # Use a more robust Poisson disk sampling approach
    points = []
    active_list = []
    
    # Create grid for efficient neighbor searching
    cell_size = min_distance / math.sqrt(2)
    grid_width = int(math.ceil(width / cell_size)) + 1
    grid_height = int(math.ceil(height / cell_size)) + 1
    grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
    
    def get_grid_coords(x, y):
        return int(x / cell_size), int(y / cell_size)
    
    def is_valid_point(x, y):
        gx, gy = get_grid_coords(x, y)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    neighbor = grid[ny][nx]
                    if neighbor is not None:
                        dist = math.sqrt((x - neighbor[0])**2 + (y - neighbor[1])**2)
                        if dist < min_distance:
                            return False
        return True
    
    # Add first point
    first_x = np.random.uniform(BOUNDARY_MARGIN, width - BOUNDARY_MARGIN)
    first_y = np.random.uniform(BOUNDARY_MARGIN, height - BOUNDARY_MARGIN)
    points.append((first_x, first_y))
    active_list.append(0)
    gx, gy = get_grid_coords(first_x, first_y)
    grid[gy][gx] = (first_x, first_y)
    
    attempts = 0
    max_attempts = 10000
    
    while active_list and len(points) < n_points and attempts < max_attempts:
        # Pick random active point
        idx = random.randint(0, len(active_list) - 1)
        point_idx = active_list[idx]
        px, py = points[point_idx]
        
        # Try to generate new point
        found = False
        for _ in range(30):  # Max tries per point
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_distance, min_distance * 2)
            
            new_x = px + radius * math.cos(angle)
            new_y = py + radius * math.sin(angle)
            
            # Check boundaries
            if (new_x < BOUNDARY_MARGIN or new_x > width - BOUNDARY_MARGIN or
                new_y < BOUNDARY_MARGIN or new_y > height - BOUNDARY_MARGIN):
                continue
            
            # Check distance to existing points
            if is_valid_point(new_x, new_y):
                points.append((new_x, new_y))
                active_list.append(len(points) - 1)
                gx, gy = get_grid_coords(new_x, new_y)
                grid[gy][gx] = (new_x, new_y)
                found = True
                break
        
        if not found:
            active_list.pop(idx)
        attempts += 1
    
    # If we couldn't generate enough points, fill with random ones
    while len(points) < n_points:
        x = random.uniform(BOUNDARY_MARGIN, width - BOUNDARY_MARGIN)
        y = random.uniform(BOUNDARY_MARGIN, height - BOUNDARY_MARGIN)
        if is_valid_point(x, y):
            points.append((x, y))
            gx, gy = get_grid_coords(x, y)
            grid[gy][gx] = (x, y)
    
    return np.array(points[:n_points])

def initialize_population(n_circles: int, pop_size: int) -> List[np.ndarray]:
    """Initialize population with enhanced Voronoi + Poisson hybrid approach"""
    population = []
    
    # Generate base points using Poisson disk sampling for good distribution
    base_points = generate_poisson_disk_points(n_circles * 2, 1.0, 1.0, min_distance=0.1)
    
    # If we got insufficient points, fall back to random
    if len(base_points) < n_circles:
        base_points = np.random.rand(n_circles * 2, 2)
        base_points[:, 0] = np.clip(base_points[:, 0], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
        base_points[:, 1] = np.clip(base_points[:, 1], BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
    
    # Use nearest neighbors to determine local density and assign appropriate radii
    if len(base_points) >= 2:
        tree = cKDTree(base_points)
        # Find distances to 5 nearest neighbors (excluding self)
        distances, indices = tree.query(base_points, k=min(6, len(base_points)), p=1)
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Remove first column (distance to self)
        
        # Higher density = lower radius, lower density = higher radius
        # Normalize density estimates
        normalized_densities = 1.0 / (avg_distances + 1e-8)
        # Map to radii: more dense areas get smaller radii
        initial_radii = MAX_RADIUS_FACTOR * (1.0 / (normalized_densities + 1.0)) + 0.005
    else:
        initial_radii = np.full(len(base_points), 0.02)
    
    # Ensure radii are within valid range
    initial_radii = np.maximum(initial_radii, MIN_RADIUS)
    initial_radii = np.minimum(initial_radii, MAX_RADIUS_FACTOR)
    
    for _ in range(pop_size):
        # Create circles with varied initial radii and positions
        circles = np.zeros((n_circles, 3))
        
        # Select subset of base points with appropriate radii
        selected_indices = np.random.choice(len(base_points), n_circles, replace=True)
        selected_points = base_points[selected_indices]
        selected_radii = initial_radii[selected_indices]
        
        # Add noise to make configurations unique
        noise_factor = 0.02
        for i in range(n_circles):
            x, y = selected_points[i]
            r = selected_radii[i]
            
            # Add noise to position
            x += np.random.normal(0, noise_factor * r)
            y += np.random.normal(0, noise_factor * r)
            
            # Ensure within bounds
            x = np.clip(x, BOUNDARY_MARGIN + r, 1 - BOUNDARY_MARGIN - r)
            y = np.clip(y, BOUNDARY_MARGIN + r, 1 - BOUNDARY_MARGIN - r)
            
            # Add noise to radius
            r *= (1 + np.random.normal(0, noise_factor))
            r = np.clip(r, MIN_RADIUS, MAX_RADIUS_FACTOR)
            
            circles[i] = [x, y, r]
        
        # Final adjustment to ensure no overlaps and all constraints met
        circles = _adjust_to_constraints(circles)
        
        population.append(circles)
    
    return population

def _adjust_to_constraints(circles: np.ndarray) -> np.ndarray:
    """Adjust circle configuration to meet all constraints"""
    adjusted = circles.copy()
    
    # First pass: fix boundary violations
    for i in range(len(adjusted)):
        x, y, r = adjusted[i]
        # Ensure circle fits in square
        max_r = min(x, 1-x, y, 1-y)
        adjusted[i, 2] = min(r, max_r * 0.98)
        
        # Re-adjust positions if needed
        if adjusted[i, 2] != r:
            r = adjusted[i, 2]
            adjusted[i, 0] = np.clip(adjusted[i, 0], r, 1-r)
            adjusted[i, 1] = np.clip(adjusted[i, 1], r, 1-r)
    
    # Second pass: resolve overlaps using iterative approach
    max_iterations = 20
    for _ in range(max_iterations):
        # Build spatial tree for efficient neighbor queries
        tree = cKDTree(adjusted[:, :2])
        changed = False
        
        for i in range(len(adjusted)):
            x, y, r = adjusted[i]
            # Find neighbors using the tree
            neighbors = tree.query_ball_point([x, y], 2 * (r + 0.001))
            
            # Check if there are overlaps
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = adjusted[j]
                    dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                    
                    if dist < r + r2:
                        # Resolve by moving circles apart
                        if dist > 1e-8:  # Prevent division by zero
                            dx = (x2 - x) / dist
                            dy = (y2 - y) / dist
                            
                            # Move circles apart proportionally to their radii
                            total_radius = r + r2
                            move_amount = (total_radius - dist) / 2.0
                            
                            adjusted[i, 0] -= dx * move_amount * 0.3
                            adjusted[i, 1] -= dy * move_amount * 0.3
                            adjusted[j, 0] += dx * move_amount * 0.3
                            adjusted[j, 1] += dy * move_amount * 0.3
                            
                            changed = True
        
        # If no changes occurred, stop
        if not changed:
            break
    
    # Final boundary correction
    for i in range(len(adjusted)):
        x, y, r = adjusted[i]
        max_r = min(x, 1-x, y, 1-y)
        adjusted[i, 2] = min(r, max_r * 0.99)
        adjusted[i, 0] = np.clip(adjusted[i, 0], adjusted[i, 2], 1 - adjusted[i, 2])
        adjusted[i, 1] = np.clip(adjusted[i, 1], adjusted[i, 2], 1 - adjusted[i, 2])
    
    return adjusted

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration is valid (no overlaps, fully contained)"""
    if len(circles) == 0:
        return False
    
    # Check boundary containment
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps using efficient distance computation
    if len(circles) < 2:
        return True
        
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Use optimized pairwise distance computation
    distances = cdist(positions, positions)
    
    # Check all pairs for overlap
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            if dist < radii[i] + radii[j]:
                return False
    
    return True

def calculate_fitness(circles: np.ndarray, generation: int = 0, 
                     total_generations: int = 1) -> Tuple[float, float]:
    """
    Calculate fitness with progressive penalty system
    Returns (total_radius, penalty_score)
    """
    total_radius = np.sum(circles[:, 2])
    
    if not is_valid_configuration(circles):
        # Progressive penalty system
        penalty = 0.0
        
        # Boundary violations penalty
        for x, y, r in circles:
            if x - r < 0:
                penalty += abs(x - r) * 100000
            elif x + r > 1:
                penalty += abs(x + r - 1) * 100000
            if y - r < 0:
                penalty += abs(y - r) * 100000
            elif y + r > 1:
                penalty += abs(y + r - 1) * 100000
        
        # Overlap violations penalty
        if len(circles) >= 2:
            positions = circles[:, :2]
            radii = circles[:, 2]
            distances = cdist(positions, positions)
            
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    dist = distances[i, j]
                    overlap = max(0, radii[i] + radii[j] - dist)
                    if overlap > 0:
                        penalty += overlap * 1000000
        
        # Add progressive penalty based on generation
        penalty_factor = (1 - generation / total_generations) * 50000
        penalty += penalty_factor
        
        return total_radius - penalty, penalty
    
    return total_radius, 0.0

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float],
                        tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Select individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = [fitness_scores[i] for i in selected_indices]
    best_idx = selected_indices[np.argmax(selected_fitness)]
    return population[best_idx]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Uniform crossover between two parents with constraint preservation"""
    n_circles = len(parent1)
    child = np.zeros_like(parent1)
    
    # Uniform crossover
    mask = np.random.rand(n_circles) > 0.5
    child[mask] = parent1[mask]
    child[~mask] = parent2[~mask]
    
    # Apply refinement to ensure validity
    child = _adjust_to_constraints(child)
    return child

def mutate(individual: np.ndarray, generation: int, total_generations: int, 
          elite: bool = False) -> np.ndarray:
    """Advanced mutation with dual strategies"""
    mutated = individual.copy()
    n_circles = len(individual)
    
    # Adaptive mutation rate with dual-phase behavior
    # Early generations: more aggressive
    # Later generations: more conservative
    mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * (generation / total_generations)
    
    # Mutation scales - smaller for elite individuals
    pos_scale = 0.02 if not elite else 0.005
    rad_scale = 0.015 if not elite else 0.003
    
    # Mutate each circle
    for i in range(n_circles):
        if np.random.rand() < mutation_rate:
            # Position mutation
            mutated[i, 0] += np.random.normal(0, pos_scale)
            mutated[i, 1] += np.random.normal(0, pos_scale)
            
            # Radius mutation
            mutated[i, 2] += np.random.normal(0, rad_scale)
            
            # Clamp to valid ranges
            mutated[i, 0] = np.clip(mutated[i, 0], BOUNDARY_MARGIN + mutated[i, 2], 1 - BOUNDARY_MARGIN - mutated[i, 2])
            mutated[i, 1] = np.clip(mutated[i, 1], BOUNDARY_MARGIN + mutated[i, 2], 1 - BOUNDARY_MARGIN - mutated[i, 2])
            mutated[i, 2] = np.clip(mutated[i, 2], MIN_RADIUS, MAX_RADIUS_FACTOR)
    
    # Post-mutation refinement to maintain constraints
    mutated = _adjust_to_constraints(mutated)
    return mutated

def evolve_circles(n_circles: int = 26, generations: int = GENERATIONS) -> np.ndarray:
    """Main evolutionary algorithm with enhanced optimization"""
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize population
    population = initialize_population(n_circles, POPULATION_SIZE)
    
    best_fitness_history = []
    
    for gen in range(generations):
        # Evaluate fitness for entire population
        fitness_scores = []
        for ind in population:
            fitness, penalty = calculate_fitness(ind, gen, generations)
            fitness_scores.append(fitness)
        
        # Track best fitness
        best_fitness = max(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        # Print progress every 50 generations
        if gen % 50 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness:.6f}")
        
        # Elitism: keep best individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:ELITISM_COUNT]
        elites = [population[i].copy() for i in elite_indices]
        
        # Create new population
        new_population = []
        
        # Add elites first
        new_population.extend(elites)
        
        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation with elite flag
            elite_flag = len(new_population) < ELITISM_COUNT
            child = mutate(child, gen, generations, elite_flag)
            
            new_population.append(child)
        
        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]
        
        # Early stopping when improvement plateaus
        if len(best_fitness_history) > 20:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-20]
            if recent_improvement < 0.0001:
                print(f"Stopping early at generation {gen} due to minimal improvement")
                break
    
    # Return best individual
    final_fitness_scores = []
    for ind in population:
        fitness, penalty = calculate_fitness(ind, generations, generations)
        final_fitness_scores.append(fitness)
    
    best_idx = np.argmax(final_fitness_scores)
    return population[best_idx]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = evolve_circles(n_circles=26, generations=GENERATIONS)
        return circles
    except Exception as e:
        print(f"Evolution process failed: {e}")
        # Fallback to simple grid arrangement
        circles = np.zeros((26, 3))
        rows = 5
        cols = 5
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.3
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= 26:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1
        
        # Ensure we have exactly 26 circles
        for i in range(idx, 26):
            circles[i] = [0.5, 0.5, 0.01]
        
        return circles

# EVOLVE-BLOCK-END