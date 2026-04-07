# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional
import time
from collections import defaultdict
import warnings

# Global constants for optimization
POP_SIZE = 150
GENERATIONS = 800
INITIAL_MUTATION_RATE = 0.25
FINAL_MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 7
BOUNDARY_PENALTY_BASE = 500.0
OVERLAP_PENALTY_BASE = 5000.0
ELITISM_COUNT = 3

def generate_voronoi_points(n_points: int, n_circles: int) -> List[Tuple[float, float]]:
    """Generate points using Voronoi diagram approach for better distribution"""
    # Generate more points than needed to ensure good Voronoi coverage
    points = []
    
    # Use a systematic approach to generate initial points
    grid_size = max(6, int(np.ceil(np.sqrt(n_points))))
    
    # Create a grid with some randomness
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_points:
                x = 0.05 + (i / (grid_size - 1)) * 0.90
                y = 0.05 + (j / (grid_size - 1)) * 0.90
                # Add small random perturbation
                x += random.uniform(-0.03, 0.03)
                y += random.uniform(-0.03, 0.03)
                # Clip to valid range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                points.append((x, y))
    
    # Add random points to fill out the space
    while len(points) < n_points:
        points.append((random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)))
    
    # Generate Voronoi diagram and take centroids of cells
    try:
        vor = Voronoi(points)
        centroids = []
        for region in vor.regions:
            if len(region) > 0 and -1 not in region:
                # Calculate centroid of this region
                vertices = [vor.vertices[i] for i in region if i >= 0]
                if vertices:
                    # Only consider centroids inside the unit square
                    centroid_x = np.mean([v[0] for v in vertices])
                    centroid_y = np.mean([v[1] for v in vertices])
                    if 0.05 <= centroid_x <= 0.95 and 0.05 <= centroid_y <= 0.95:
                        centroids.append((centroid_x, centroid_y))
        
        # If we didn't get enough centroids, sample from original points
        if len(centroids) < n_circles:
            centroids.extend(random.sample(points, min(n_circles - len(centroids), len(points))))
        
        return random.sample(centroids, min(n_circles, len(centroids)))
    except Exception:
        # Fallback to regular random points if Voronoi fails
        return [(random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)) for _ in range(n_circles)]

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Initialize population with Voronoi-based distribution"""
    population = []
    
    # Generate points using Voronoi approach
    voronoi_points = generate_voronoi_points(max(36, n_circles * 2), n_circles)
    
    for _ in range(pop_size):
        individual = np.zeros((n_circles, 3))
        
        # Assign positions with slight perturbation
        for i in range(n_circles):
            # Get point from Voronoi distribution or fallback
            if i < len(voronoi_points):
                x, y = voronoi_points[i]
            else:
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
            
            # Perturb slightly
            x += random.uniform(-0.03, 0.03)
            y += random.uniform(-0.03, 0.03)
            
            # Clip to valid range
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            individual[i, 0] = x
            individual[i, 1] = y
            
            # Assign radius based on proximity to edges and other circles
            margin = min(x, y, 1 - x, 1 - y)
            base_radius = min(0.15, margin / 2.0)
            
            # Add randomness to radius
            individual[i, 2] = max(0.005, base_radius * random.uniform(0.5, 1.5))
            
        # Refine the solution to ensure validity
        individual = refine_solution(individual)
        population.append(individual)
    
    return population

def is_valid_position(x: float, y: float, r: float) -> bool:
    """Check if a circle position is valid (within bounds)"""
    return (r <= x <= 1 - r and r <= y <= 1 - r)

def is_valid_circle(circle: np.ndarray, other_circles: np.ndarray) -> bool:
    """Check if a circle is valid (contained and not overlapping)"""
    x, y, r = circle
    
    # Check containment
    if not is_valid_position(x, y, r):
        return False

    # Check overlap with existing circles
    for other in other_circles:
        if other[2] > 0:  # Only check non-zero radius circles
            ox, oy, oradius = other
            distance = np.sqrt((x - ox)**2 + (y - oy)**2)
            if distance < (r + oradius):
                return False
    
    return True

def calculate_penalty(circles: np.ndarray) -> Tuple[float, float, dict]:
    """Calculate penalty based on constraint violations with detailed information"""
    penalty = 0.0
    boundary_violations = 0.0
    overlap_violations = 0.0
    overlap_count = 0
    
    n = len(circles)
    
    # Check containment penalties (more precise calculation)
    for circle in circles:
        x, y, r = circle
        if not is_valid_position(x, y, r):
            # Calculate violation amounts
            left_violation = max(0, r - x)
            right_violation = max(0, r - (1 - x))
            bottom_violation = max(0, r - y)
            top_violation = max(0, r - (1 - y))
            boundary_violations += (left_violation + right_violation + 
                                   bottom_violation + top_violation)
    
    # Use spatial indexing to efficiently check overlaps
    valid_circles = [c for c in circles if c[2] > 0]
    if len(valid_circles) > 1:
        # Precompute distances for efficiency
        positions = np.array([[c[0], c[1]] for c in valid_circles])
        
        # Use distance matrix for overlap detection
        distances = cdist(positions, positions)
        
        # For each pair, check if they're overlapping
        for i in range(len(valid_circles)):
            for j in range(i + 1, len(valid_circles)):
                if distances[i, j] < (valid_circles[i][2] + valid_circles[j][2]):
                    overlap_count += 1
                    # Calculate overlap amount
                    overlap = (valid_circles[i][2] + valid_circles[j][2]) - distances[i, j]
                    overlap_violations += overlap
        
        penalty = BOUNDARY_PENALTY_BASE * boundary_violations + \
                 OVERLAP_PENALTY_BASE * overlap_violations
    
    return penalty, boundary_violations, overlap_violations

def evaluate_fitness(circles: np.ndarray) -> Tuple[float, float, float]:
    """Evaluate the fitness of a solution"""
    # Sum of radii (primary objective)
    total_radius = np.sum(circles[:, 2])

    # Penalty for constraint violations
    penalty, _, _ = calculate_penalty(circles)

    # Fitness is total radius minus penalty
    fitness = total_radius - penalty

    return fitness, total_radius, penalty

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float],
                        tournament_size: int = TOURNAMENT_SIZE) -> np.ndarray:
    """Select an individual using tournament selection"""
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = [fitness_scores[i] for i in selected_indices]

    winner_idx = selected_indices[np.argmax(selected_fitness)]
    return population[winner_idx].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray,
             crossover_rate: float = CROSSOVER_RATE) -> np.ndarray:
    """Perform crossover between two parents"""
    if random.random() > crossover_rate:
        return parent1.copy()

    n = len(parent1)
    child = np.zeros_like(parent1)

    # Multi-point crossover that respects both positions and radii
    crossover_points = sorted(random.sample(range(1, n), min(3, n-1)))

    # Alternate between parents for segments
    last_point = 0
    use_parent1 = True
    for point in crossover_points:
        if use_parent1:
            child[last_point:point, :] = parent1[last_point:point, :]
        else:
            child[last_point:point, :] = parent2[last_point:point, :]
        last_point = point
        use_parent1 = not use_parent1
    
    # Handle final segment
    if use_parent1:
        child[last_point:, :] = parent1[last_point:, :]
    else:
        child[last_point:, :] = parent2[last_point:, :]

    # Local refinement to fix any constraint violations
    child = refine_solution(child)

    return child

def refine_solution(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to fix constraint violations"""
    refined = circles.copy()
    
    # Phase 1: Fix containment violations first
    for i in range(len(refined)):
        x, y, r = refined[i]
        if not is_valid_position(x, y, r):
            # Adjust position to stay within bounds
            if r > x:
                x = r + 0.001
            if r > y:
                y = r + 0.001
            if r > (1 - x):
                x = 1 - r - 0.001
            if r > (1 - y):
                y = 1 - r - 0.001
            refined[i, 0] = x
            refined[i, 1] = y
    
    # Phase 2: Iterative overlap resolution
    max_iter = 100
    for iteration in range(max_iter):
        changed = False
        
        # Build list of valid circles for overlap checking
        valid_indices = [i for i in range(len(refined)) if refined[i, 2] > 0]
        
        for i in valid_indices:
            x, y, r = refined[i]
            
            # Check overlap with all others
            for j in valid_indices:
                if i != j:
                    ox, oy, oradius = refined[j]
                    distance = np.sqrt((x - ox)**2 + (y - oy)**2)
                    
                    if distance < (r + oradius):
                        # Try to resolve overlap by adjusting current circle
                        if distance > 0.0001:
                            # Normalize direction vector
                            dx = (x - ox) / distance
                            dy = (y - oy) / distance
                            
                            # Reduce radius to prevent further overlap
                            new_r = max(0.001, (r + oradius) * 0.99 - distance)
                            if new_r < r and new_r > 0.001:
                                refined[i, 2] = new_r
                                changed = True
                                
                            # Adjust position to separate circles
                            separation = 0.001
                            refined[i, 0] = x + dx * separation
                            refined[i, 1] = y + dy * separation
                            
                            # Ensure containment after adjustment
                            refined[i, 0] = np.clip(refined[i, 0], refined[i, 2], 1 - refined[i, 2])
                            refined[i, 1] = np.clip(refined[i, 1], refined[i, 2], 1 - refined[i, 2])
                        else:
                            # If circles are at same position, move one slightly
                            refined[i, 0] += random.uniform(-0.001, 0.001)
                            refined[i, 1] += random.uniform(-0.001, 0.001)
                            changed = True
        
        if not changed:
            break
    
    # Final containment check and correction
    for i in range(len(refined)):
        x, y, r = refined[i]
        r = max(0.001, r)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        refined[i, 0] = x
        refined[i, 1] = y
        refined[i, 2] = r
    
    return refined

def adaptive_mutation_rate(generation: int, max_generations: int) -> float:
    """Adaptive mutation rate that decreases over time"""
    # Start high, decrease gradually
    return INITIAL_MUTATION_RATE - (INITIAL_MUTATION_RATE - FINAL_MUTATION_RATE) * (generation / max_generations)

def mutate(individual: np.ndarray, mutation_rate: float = 0.2) -> np.ndarray:
    """Mutate an individual with different strategies"""
    mutated = individual.copy()
    n = len(mutated)
    
    # Apply mutations based on probabilities
    for i in range(n):
        if random.random() < mutation_rate:
            # Choose mutation type with preference for position changes
            mutation_type = random.choices(
                [0, 1, 2, 3], 
                weights=[0.5, 0.5, 0.2, 0.3]  # Position more likely
            )[0]
            
            if mutation_type == 0:  # Mutate x position (larger change)
                mutated[i, 0] = np.clip(mutated[i, 0] + random.gauss(0, 0.05), 0.05, 0.95)
            elif mutation_type == 1:  # Mutate y position (larger change)
                mutated[i, 1] = np.clip(mutated[i, 1] + random.gauss(0, 0.05), 0.05, 0.95)
            elif mutation_type == 2:  # Mutate radius (smaller change)
                mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, 0.01), 0.001, 0.2)
            else:  # Mutate both position and radius
                mutated[i, 0] = np.clip(mutated[i, 0] + random.gauss(0, 0.02), 0.05, 0.95)
                mutated[i, 1] = np.clip(mutated[i, 1] + random.gauss(0, 0.02), 0.05, 0.95)
                mutated[i, 2] = np.clip(mutated[i, 2] + random.gauss(0, 0.005), 0.001, 0.2)
    
    # Local refinement after mutation
    mutated = refine_solution(mutated)
    return mutated

def evolve_population(population: List[np.ndarray]) -> Tuple[List[np.ndarray], float, float, float]:
    """Evolve the population for one generation"""
    # Evaluate fitness
    fitness_scores = []
    total_radii = []
    penalties = []

    for individual in population:
        fitness, total_radius, penalty = evaluate_fitness(individual)
        fitness_scores.append(fitness)
        total_radii.append(total_radius)
        penalties.append(penalty)

    # Track best individual
    best_idx = np.argmax(fitness_scores)
    best_fitness = fitness_scores[best_idx]
    best_total_radius = total_radii[best_idx]
    best_penalty = penalties[best_idx]

    # Create new population
    new_population = []

    # Elitism: keep the best individuals
    elite_indices = np.argsort(fitness_scores)[-ELITISM_COUNT:]
    for idx in elite_indices:
        new_population.append(population[idx].copy())

    # Generate rest of population
    while len(new_population) < len(population):
        # Selection
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)

        # Crossover
        child = crossover(parent1, parent2)

        # Mutation with adaptive rate
        mut_rate = adaptive_mutation_rate(len(new_population), POP_SIZE)
        child = mutate(child, mut_rate)

        new_population.append(child)

    return new_population, best_fitness, best_total_radius, best_penalty

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 26
    population = initialize_population(POP_SIZE, n)

    best_total_radius = 0.0
    best_individual = None
    best_penalty = float('inf')

    # Evolution loop
    start_time = time.time()
    for generation in range(GENERATIONS):
        population, gen_fitness, gen_radius, gen_penalty = evolve_population(population)

        if gen_radius > best_total_radius:
            best_total_radius = gen_radius
            best_individual = population[0]  # Keep track of best individual
            best_penalty = gen_penalty

        # Print progress every 100 generations
        if generation % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation}: Best radius sum = {gen_radius:.6f} (penalty={gen_penalty:.2f}) Time: {elapsed:.2f}s")

    elapsed = time.time() - start_time
    print(f"Final result: Best radius sum = {best_total_radius:.6f} (penalty={best_penalty:.2f}) Time: {elapsed:.2f}s")
    print(f"Benchmark ratio: {best_total_radius / 2.6358627564136983:.6f}")

    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to returning first individual if something went wrong
        return population[0]

# EVOLVE-BLOCK-END