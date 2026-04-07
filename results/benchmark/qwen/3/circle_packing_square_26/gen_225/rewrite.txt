# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List
import warnings

# Global constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
TOURNAMENT_SIZE = 5
CROSSOVER_RATE = 0.8
MUTATION_RATE_START = 0.3
MUTATION_RATE_END = 0.02
BOUNDARY_PENALTY_WEIGHT = 100000.0
OVERLAP_PENALTY_WEIGHT = 100000.0
GRID_SIZE = 25
VORONOI_REFINEMENT_ITERATIONS = 3

def generate_voronoi_points(n_points: int, method: str = 'hexagonal') -> np.ndarray:
    """Generate well-distributed points using Voronoi-based methods."""
    if method == 'hexagonal':
        # Create hexagonal grid pattern for better distribution
        points = []
        sqrt3 = math.sqrt(3)
        rows = int(math.ceil(math.sqrt(n_points)))
        cols = int(math.ceil(n_points / rows))
        
        # Hexagonal spacing
        spacing = 0.8 / (max(rows, cols) + 1)
        hex_radius = spacing / 2.0
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                # Offset every other row for hexagonal arrangement
                x_offset = j * spacing + (i % 2) * spacing / 2.0
                y_offset = i * spacing * sqrt3 / 2.0
                
                # Add some randomness to create more natural distribution
                x = 0.1 + x_offset + np.random.uniform(-spacing/6, spacing/6)
                y = 0.1 + y_offset + np.random.uniform(-spacing/6, spacing/6)
                
                # Ensure points stay within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                points.append([x, y])
        
        # Fill remaining points randomly
        while len(points) < n_points:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])
        
        return np.array(points[:n_points])
    
    elif method == 'uniform':
        # Simple uniform random distribution
        points = []
        for _ in range(n_points):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])
        return np.array(points)

def create_voronoi_regions(points: np.ndarray) -> Tuple[List, List]:
    """Create Voronoi regions for given points."""
    try:
        vor = Voronoi(points)
        # Get Voronoi vertices and regions
        vertices = vor.vertices
        regions = vor.regions
        point_regions = vor.point_region
        return vor, point_regions
    except Exception:
        # Fallback to simpler approach if Voronoi fails
        return None, None

def estimate_radii_from_voronoi(points: np.ndarray, voronoi_obj=None) -> np.ndarray:
    """Estimate optimal radii based on Voronoi regions."""
    # If Voronoi calculation fails, use a simpler method
    if voronoi_obj is None:
        # Estimate radius as proportion of minimum distance to neighbors
        radii = []
        for i, point in enumerate(points):
            # Calculate distances to all other points
            distances = np.linalg.norm(points - point, axis=1)
            distances[i] = np.inf  # Exclude self-distance
            
            if len(distances) > 1:
                min_distance = np.min(distances)
                # Estimate radius as a fraction of minimum neighbor distance
                # but bounded by containment constraints
                max_allowable = min(point[0], point[1], 1-point[0], 1-point[1])
                estimated_radius = min(min_distance/3.0, max_allowable * 0.5)
                radius = max(0.001, min(estimated_radius, 0.4))
            else:
                # Default radius if insufficient points
                radius = np.random.uniform(0.01, 0.1)
            
            radii.append(radius)
        
        return np.array(radii)
    
    # If Voronoi is available, use it for better estimation
    try:
        # Use the Voronoi approach with area-based estimation
        # First create Voronoi diagram
        vor = Voronoi(points)
        
        # Calculate approximate radius based on Voronoi cell area
        radii = []
        for i in range(len(points)):
            # Get distance to nearest neighbor
            distances = np.linalg.norm(points - points[i], axis=1)
            distances[i] = np.inf  # Remove self-distance
            min_neighbor_dist = np.min(distances) if len(distances) > 1 else 0.1
            
            # Calculate maximum radius that fits in unit square
            max_allowable = min(points[i][0], points[i][1], 1-points[i][0], 1-points[i][1])
            
            # Estimate based on Voronoi cell geometry
            estimated_radius = min(min_neighbor_dist/3.0, max_allowable * 0.6)
            radius = max(0.001, min(estimated_radius, 0.4))
            radii.append(radius)
        
        return np.array(radii)
    except Exception:
        # Fallback to simple method
        return estimate_radii_from_voronoi(points, None)

def create_grid(circles: np.ndarray) -> dict:
    """Create a spatial grid for efficient overlap checking.""" 
    grid = {}
    
    for i, (x, y, r) in enumerate(circles):
        # Determine which grid cells this circle touches
        min_x_cell = max(0, int((x - r) * GRID_SIZE))
        max_x_cell = min(GRID_SIZE - 1, int((x + r) * GRID_SIZE))
        min_y_cell = max(0, int((y - r) * GRID_SIZE))
        max_y_cell = min(GRID_SIZE - 1, int((y + r) * GRID_SIZE))

        # Add circle to all relevant grid cells
        for gx in range(min_x_cell, max_x_cell + 1):
            for gy in range(min_y_cell, max_y_cell + 1):
                grid.setdefault((gx, gy), []).append(i)

    return grid

def check_overlap_with_grid(circles: np.ndarray, grid: dict) -> bool:
    """Check overlaps using spatial grid for improved efficiency."""
    # For each cell in the grid, check if any pairs of circles overlap
    for cell, circle_indices in grid.items():
        # Only check pairs within the same grid cell
        for i in range(len(circle_indices)):
            idx1 = circle_indices[i]
            x1, y1, r1 = circles[idx1]

            for j in range(i + 1, len(circle_indices)):
                idx2 = circle_indices[j]
                x2, y2, r2 = circles[idx2]

                # Calculate distance between circle centers
                dx = x1 - x2
                dy = y1 - y2
                distance_squared = dx*dx + dy*dy

                # Check if circles overlap
                if distance_squared < (r1 + r2)**2:
                    return False

    return True

def is_valid(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping."""
    n = len(circles)

    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False

    # Check overlap constraints using spatial grid
    grid = create_grid(circles)
    return check_overlap_with_grid(circles, grid)

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii."""
    return np.sum(circles[:, 2])

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness of a solution, higher is better."""
    if not is_valid(circles):
        # Apply penalty for constraint violations
        penalty = 0

        # Boundary penalty
        boundary_violations = 0
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0:
                boundary_violations += (r - x)**2
            if x + r > 1:
                boundary_violations += (x + r - 1)**2
            if y - r < 0:
                boundary_violations += (r - y)**2
            if y + r > 1:
                boundary_violations += (y + r - 1)**2

        penalty += BOUNDARY_PENALTY_WEIGHT * boundary_violations

        # Overlap penalty
        overlap_penalty = 0
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    overlap_penalty += (r1 + r2 - distance)**2

        penalty += OVERLAP_PENALTY_WEIGHT * overlap_penalty

        return -penalty

    return calculate_sum_radii(circles)

def initialize_population(size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with Voronoi-based distribution."""
    population = []
    
    # Generate points using Voronoi-based approach
    voronoi_points = generate_voronoi_points(n_circles, 'hexagonal')
    
    # Create multiple populations with variations
    for _ in range(size):
        # Start with Voronoi points
        points = voronoi_points.copy()
        
        # Add slight jitter to points for diversity
        for i in range(len(points)):
            points[i][0] += np.random.normal(0, 0.01)
            points[i][1] += np.random.normal(0, 0.01)
            # Clamp to valid range
            points[i][0] = np.clip(points[i][0], 0.05, 0.95)
            points[i][1] = np.clip(points[i][1], 0.05, 0.95)
        
        # Create circles with radius estimation
        circles = np.zeros((n_circles, 3))
        radii = estimate_radii_from_voronoi(points)
        
        for i in range(n_circles):
            circles[i] = [points[i][0], points[i][1], radii[i]]
        
        # Refine configuration to ensure validity
        for _ in range(VORONOI_REFINEMENT_ITERATIONS):
            grid = create_grid(circles)
            if check_overlap_with_grid(circles, grid):
                break
            # Apply small adjustments to resolve overlaps
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Try to move circle slightly to reduce overlap
                circles[i] = [x + np.random.uniform(-0.001, 0.001),
                             y + np.random.uniform(-0.001, 0.001),
                             r]
                # Clamp to bounds  
                circles[i][0] = np.clip(circles[i][0], r + 0.001, 1 - r - 0.001)
                circles[i][1] = np.clip(circles[i][1], r + 0.001, 1 - r - 0.001)
        
        # Validate final configuration
        if not is_valid(circles):
            # Fallback to a simple configuration
            circles = np.zeros((n_circles, 3))
            # Use circular pattern for fallback
            for i in range(n_circles):
                angle = 2 * math.pi * i / n_circles
                x = 0.5 + 0.3 * math.cos(angle)
                y = 0.5 + 0.3 * math.sin(angle)
                # Smaller radii for more circles
                r = 0.03 + (i % 3) * 0.01
                circles[i] = [x, y, r]
        
        population.append(circles)

    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents using Voronoi-aware method."""
    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Use a more sophisticated crossover that preserves Voronoi structure
    # First, determine crossover points based on proximity to Voronoi centers
    crossover_points = []
    for i in range(n):
        if random.random() < 0.5:
            crossover_points.append(i)
    
    # For chosen points, swap positions and radii
    for point in crossover_points:
        # Swap positions
        child1[point, 0] = parent2[point, 0]
        child1[point, 1] = parent2[point, 1]
        child2[point, 0] = parent1[point, 0]
        child2[point, 1] = parent1[point, 1]
        
        # Swap radii
        child1[point, 2] = parent2[point, 2]
        child2[point, 2] = parent1[point, 2]

    return child1, child2

def adaptive_mutation_rate(generation: int, max_generations: int) -> float:
    """Calculate adaptive mutation rate with sigmoidal decay."""
    # Start high, decay to low with sigmoidal function
    rate = MUTATION_RATE_END + (MUTATION_RATE_START - MUTATION_RATE_END) * \
           (1 / (1 + math.exp(10 * (generation / max_generations - 0.5))))
    return rate

def mutate(circles: np.ndarray, generation: int, max_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive rate and Voronoi-aware strategy."""
    mutation_rate = adaptive_mutation_rate(generation, max_generations)
    mutated = circles.copy()
    n = len(mutated)
    
    for i in range(n):
        if random.random() < mutation_rate:
            # Determine what kind of mutation to apply based on generation stage
            gen_progress = generation / max_generations
            
            if gen_progress < 0.3:
                # Early phase: focus on exploration with position mutations
                choice = random.choices([0, 1, 2], weights=[0.6, 0.6, 0.2])[0]
                strength = 0.05
            elif gen_progress < 0.6:
                # Mid phase: balanced approach
                choice = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.4])[0]
                strength = 0.03
            else:
                # Late phase: emphasize fine-tuning with radius mutations
                choice = random.choices([0, 1, 2], weights=[0.2, 0.2, 0.8])[0]
                strength = 0.015
            
            if choice == 0:  # X coordinate
                mutated[i, 0] = max(0.001, min(0.999, mutated[i, 0] + np.random.normal(0, strength)))
            elif choice == 1:  # Y coordinate
                mutated[i, 1] = max(0.001, min(0.999, mutated[i, 1] + np.random.normal(0, strength)))
            else:  # Radius with log-normal distribution
                log_factor = np.random.normal(0, 0.15)
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] * np.exp(log_factor)))
    
    return mutated

def get_best_individual(population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
    """Get the individual with highest fitness."""
    best_idx = fitnesses.index(max(fitnesses))
    return population[best_idx]

def voronoi_guided_evolution() -> np.ndarray:
    """Main evolutionary algorithm with Voronoi guidance."""
    n = 26
    np.random.seed(42)
    random.seed(42)
    
    # Initialize population with Voronoi-based approach
    population = initialize_population(POPULATION_SIZE, n)
    
    best_fitness_history = []
    
    # Evolutionary loop
    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]
        
        # Track best fitness
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)
        
        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        best_individual = get_best_individual(population, fitnesses)
        new_population.append(best_individual)
        
        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            child1, child2 = crossover(parent1, parent2)
            
            child1 = mutate(child1, generation, NUM_GENERATIONS)
            child2 = mutate(child2, generation, NUM_GENERATIONS)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:POPULATION_SIZE]
    
    # Get final best solution
    final_fitnesses = [evaluate_fitness(individual) for individual in population]
    best_solution = get_best_individual(population, final_fitnesses)
    
    # Final validation and refinement
    if not is_valid(best_solution):
        # Try to improve by making small adjustments
        for i in range(len(best_solution)):
            x, y, r = best_solution[i]
            best_solution[i] = [np.clip(x, r + 0.001, 1 - r - 0.001),
                              np.clip(y, r + 0.001, 1 - r - 0.001),
                              r]
    
    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Run the Voronoi-guided evolution algorithm
    return voronoi_guided_evolution()

# EVOLVE-BLOCK-END