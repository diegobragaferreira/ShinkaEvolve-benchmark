# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Global constants
POPULATION_SIZE = 100
GENERATIONS = 150
TOURNAMENT_SIZE = 5
MUTATION_RATE_START = 0.15
MUTATION_RATE_END = 0.005
CROSSOVER_PROB = 0.8
VALIDITY_THRESHOLD = 1e-6

def generate_voronoi_like_points(n_points: int) -> List[Tuple[float, float]]:
    """Generate points with Voronoi-like distribution using grid with jitter."""
    points = []
    
    # Create a grid-based pattern with some randomness
    rows, cols = 5, 5
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    for i in range(rows):
        for j in range(cols):
            base_x = (j + 1) * spacing_x
            base_y = (i + 1) * spacing_y
            
            # Add jitter to make it more Voronoi-like
            jitter_x = random.uniform(-0.03, 0.03)
            jitter_y = random.uniform(-0.03, 0.03)
            
            x = max(0.01, min(0.99, base_x + jitter_x))
            y = max(0.01, min(0.99, base_y + jitter_y))
            
            points.append((x, y))
    
    # Fill remaining points randomly but ensuring good distribution
    while len(points) < n_points:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        points.append((x, y))
    
    return points[:n_points]

def initialize_population(n: int, population_size: int) -> np.ndarray:
    """Initialize population with Voronoi-based distribution and better initial spacing."""
    population = []
    
    # Generate Voronoi-like points
    voronoi_points = generate_voronoi_like_points(n)
    
    # Create multiple populations with variation
    for _ in range(population_size):
        circles = np.zeros((n, 3))
        
        # Distribute circles using Voronoi points
        for i in range(min(n, len(voronoi_points))):
            x_base, y_base = voronoi_points[i]
            
            # Add more substantial jitter to create better distribution
            x = max(0.01, min(0.99, x_base + random.uniform(-0.05, 0.05)))
            y = max(0.01, min(0.99, y_base + random.uniform(-0.05, 0.05)))
            
            # Initial radius - larger values to promote growth
            circles[i] = [x, y, 0.08]
        
        # Fill remaining circles with carefully placed random positions
        for i in range(len(voronoi_points), n):
            # Try to place near existing circles but with more randomness
            if random.random() < 0.3 and len(circles) > 0:
                # Place near an existing circle
                idx = random.randint(0, len(circles) - 1)
                x_base, y_base, _ = circles[idx]
                x = max(0.01, min(0.99, x_base + random.uniform(-0.1, 0.1)))
                y = max(0.01, min(0.99, y_base + random.uniform(-0.1, 0.1)))
            else:
                # Place randomly
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
            
            circles[i] = [x, y, 0.04]
        
        # Enforce minimal overlap resolution with physics-based approach
        circles = resolve_initial_overlaps_physics(circles)
        population.append(circles)
    
    return np.array(population)

def resolve_initial_overlaps_physics(circles: np.ndarray, max_iterations: int = 20) -> np.ndarray:
    """Resolve overlaps using physics-based force application."""
    resolved = circles.copy()
    
    # Use cKDTree for efficient neighbor searching
    tree = cKDTree(resolved[:, :2])
    
    # Iteratively resolve overlaps
    for iteration in range(max_iterations):
        # Get neighbors within a threshold distance for overlap checking
        distances = tree.query_pairs(0.01, output_type='ndarray')
        
        if len(distances) == 0:
            break
            
        changed = False
        
        # Process pairs that may be overlapping
        for i, j in distances:
            if i >= len(resolved) or j >= len(resolved):
                continue
                
            xi, yi, ri = resolved[i]
            xj, yj, rj = resolved[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            
            if dist < (ri + rj - VALIDITY_THRESHOLD):
                # Move circles apart using force-based approach
                dx = xj - xi
                dy = yj - yi
                distance = max(VALIDITY_THRESHOLD, dist)
                
                # Normalize direction vector
                dx /= distance
                dy /= distance
                
                # Calculate overlap amount and move based on inverse radius ratios
                overlap = (ri + rj - dist)
                move_amount = overlap * 0.5
                
                # Apply movement inversely proportional to radii
                scale_i = rj / (ri + rj + 0.001)
                scale_j = ri / (ri + rj + 0.001)
                
                resolved[i, 0] -= dx * move_amount * scale_i * 0.3
                resolved[i, 1] -= dy * move_amount * scale_i * 0.3
                resolved[j, 0] += dx * move_amount * scale_j * 0.3
                resolved[j, 1] += dy * move_amount * scale_j * 0.3
                changed = True
        
        # Ensure bounds are maintained
        for i in range(len(resolved)):
            x, y, r = resolved[i]
            # Clamp to valid range
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            resolved[i] = [x, y, r]
            
        if not changed:
            break
    
    return resolved

def check_containment(circles: np.ndarray) -> bool:
    """Check if all circles are fully contained in the unit square."""
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    return True

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def check_overlap_efficient(circles: np.ndarray) -> bool:
    """Check if any circles overlap using cKDTree for efficient querying."""
    if len(circles) <= 1:
        return False
    
    # Build KD-tree for efficient neighbor search
    tree = cKDTree(circles[:, :2])
    
    # Query pairs within the sum of radii distance
    query_radius = 0.001  # Very small value to avoid false positives
    pairs = tree.query_pairs(query_radius, output_type='ndarray')
    
    for i, j in pairs:
        if i >= len(circles) or j >= len(circles):
            continue
            
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance = calculate_distance((x1, y1), (x2, y2))
        
        if distance < (r1 + r2 - VALIDITY_THRESHOLD):
            return True
    
    return False

def compute_penalty(circles: np.ndarray, generation: int = 0, total_generations: int = 100) -> float:
    """Compute penalty based on constraint violations with progressive scaling."""
    penalty = 0.0
    
    # Dynamic penalty scaling factor based on progress
    penalty_scale = 1.0 + (generation / total_generations) * 10.0
    
    # Check containment violations with scaled penalties
    for x, y, r in circles:
        # Boundary violations  
        if x - r < 0:
            penalty += (abs(x - r) ** 2) * 5000 * penalty_scale
        elif x + r > 1:
            penalty += (abs(x + r - 1) ** 2) * 5000 * penalty_scale
        if y - r < 0:
            penalty += (abs(y - r) ** 2) * 5000 * penalty_scale
        elif y + r > 1:
            penalty += (abs(y + r - 1) ** 2) * 5000 * penalty_scale
    
    # Check overlap violations with scaled penalties
    if check_overlap_efficient(circles):
        penalty += 5000000.0 * penalty_scale
    
    return penalty

def evaluate_fitness(circles: np.ndarray, generation: int = 0, total_generations: int = 100) -> float:
    """Evaluate fitness of a circle configuration."""
    # If invalid, heavily penalize
    if not check_containment(circles) or check_overlap_efficient(circles):
        penalty = compute_penalty(circles, generation, total_generations)
        return -penalty
    
    # Otherwise, return total radius
    total_radius = np.sum(circles[:, 2])
    return total_radius

def mutate(circles: np.ndarray, generation: int, total_generations: int) -> np.ndarray:
    """Mutate a circle configuration with adaptive rates."""
    mutated = circles.copy()
    
    # Adaptive mutation rate using sigmoid decay
    mutation_rate = MUTATION_RATE_START + (MUTATION_RATE_END - MUTATION_RATE_START) * \
                   (1 / (1 + math.exp(-10 * (generation / total_generations - 0.5))))
    
    n = len(mutated)
    
    # Mutate some circles
    for i in range(n):
        if random.random() < mutation_rate:
            # Randomly choose what to mutate
            choice = random.randint(0, 2)
            
            if choice == 0:  # X coordinate
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + random.gauss(0, 0.03)))
            elif choice == 1:  # Y coordinate
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + random.gauss(0, 0.03)))
            else:  # Radius - allow both increase and decrease
                mutated[i, 2] = max(0.001, min(0.49, mutated[i, 2] + random.gauss(0, 0.025)))
    
    # Ensure valid configuration after mutation
    return enforce_constraints(mutated)

def enforce_constraints(circles: np.ndarray) -> np.ndarray:
    """Enforce constraints on circle positions and radii."""
    result = circles.copy()
    
    # Adjust positions and radii to satisfy bounds
    for i in range(len(result)):
        x, y, r = result[i]
        
        # Ensure circle fits in the unit square
        r = min(r, x, 1-x, y, 1-y)
        r = max(0.001, min(0.49, r))
        
        # Clamp coordinates to valid range
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        result[i] = [x, y, r]
    
    return result

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two parent configurations."""
    if random.random() > CROSSOVER_PROB:
        # Return one of the parents randomly
        return parent1.copy() if random.random() < 0.5 else parent2.copy()
    
    n = len(parent1)
    child = np.zeros_like(parent1)
    
    # Single point crossover - but make it more strategic by swapping radius groups
    crossover_point = random.randint(1, n-1)
    
    # Cross positions and radii separately to preserve meaningful relationships
    child[:crossover_point, :2] = parent1[:crossover_point, :2]
    child[crossover_point:, :2] = parent2[crossover_point:, :2]
    
    child[:crossover_point, 2] = parent1[:crossover_point, 2]
    child[crossover_point:, 2] = parent2[crossover_point:, 2]
    
    # Apply local refinement to ensure validity
    refined_child = refine_configuration(child)
    
    return refined_child

def refine_configuration(circles: np.ndarray) -> np.ndarray:
    """Refine configuration to remove overlaps and correct constraints."""
    refined = circles.copy()
    
    # Multiple refinement passes with decreasing intensity
    for iteration in range(15):
        # Use cKDTree for efficient overlap checking
        tree = cKDTree(refined[:, :2])
        
        # Check for overlaps and resolve them
        resolved = False
        # Get close pairs using cKDTree
        pairs = tree.query_pairs(0.001, output_type='ndarray')  # Small radius for proximity
        
        for i, j in pairs:
            if i >= len(refined) or j >= len(refined):
                continue
                
            xi, yi, ri = refined[i]
            xj, yj, rj = refined[j]
            dist = calculate_distance((xi, yi), (xj, yj))
            
            if dist < (ri + rj - VALIDITY_THRESHOLD):
                # Resolve overlap by moving circles apart with force-based approach
                dx = xj - xi
                dy = yj - yi
                distance = max(VALIDITY_THRESHOLD, dist)
                
                # Normalize direction vector
                dx /= distance
                dy /= distance
                
                # Move circles apart based on their relative sizes and distances
                move_amount = (ri + rj - dist) * 0.5
                
                # Scale by inverse radii to balance movement
                scale_factor = min(1.0, ri / (ri + rj + 0.001))
                refined[i, 0] -= dx * move_amount * scale_factor * 0.2
                refined[i, 1] -= dy * move_amount * scale_factor * 0.2
                refined[j, 0] += dx * move_amount * (1 - scale_factor) * 0.2
                refined[j, 1] += dy * move_amount * (1 - scale_factor) * 0.2
                resolved = True
        
        # Enforce bounds
        for i in range(len(refined)):
            x, y, r = refined[i]
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            refined[i] = [x, y, r]
        
        # Early stopping if no changes made
        if not resolved:
            break
    
    return refined

def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    """Select parent using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 26
    population = initialize_population(n, POPULATION_SIZE)
    
    # Evaluate initial population
    fitnesses = [evaluate_fitness(individual) for individual in population]
    
    # Evolution loop
    for gen in range(GENERATIONS):
        # Selection, crossover, and mutation
        new_population = []
        
        for _ in range(POPULATION_SIZE):
            # Tournament selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, gen, GENERATIONS)
            
            new_population.append(child)
        
        # Evaluate new population
        population = np.array(new_population)
        fitnesses = [evaluate_fitness(individual, gen, GENERATIONS) for individual in population]
        
        # Print progress
        best_fitness = max(fitnesses)
        if gen % 30 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness}")
    
    # Return the best individual
    best_index = np.argmax(fitnesses)
    best_solution = population[best_index]
    
    return best_solution

# EVOLVE-BLOCK-END