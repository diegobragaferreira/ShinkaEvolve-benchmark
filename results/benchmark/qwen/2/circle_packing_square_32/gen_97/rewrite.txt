# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random
import math

def is_valid_placement(circles, x, y, r):
    """Check if placing a circle at (x,y) with radius r is valid."""
    # Check boundary constraints
    if r > x or r > y or r > (1-x) or r > (1-y):
        return False

    # Check overlap with existing circles
    for i in range(len(circles)):
        cx, cy, cr = circles[i]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        if distance < (r + cr):
            return False

    return True

def compute_local_density(circles, point, k=5):
    """Compute local density around a point using k-nearest neighbors."""
    if len(circles) < 2:
        return 0.0

    # Convert to numpy array for efficient processing
    pts = np.array(circles)[:, :2]
    tree = cKDTree(pts)

    # Query k nearest neighbors (excluding the point itself if it exists)
    distances, indices = tree.query(point, k=min(k+1, len(pts)), p=2)

    # Average distance to neighbors (excluding self if present)
    if len(distances) > 1:
        avg_distance = np.mean(distances[1:])  # Skip the first (distance to itself)
    else:
        avg_distance = distances[0]

    # Density is inversely proportional to average distance
    if avg_distance > 0:
        return 1.0 / avg_distance
    else:
        return float('inf')

def evaluate_fitness(circles):
    """Calculate the fitness (sum of radii) of the current configuration."""
    return np.sum(circles[:, 2])

def calculate_constraints(circles):
    """Calculate constraint violations for all circles."""
    violations = []
    n = len(circles)
    
    # Boundary violations
    for i in range(n):
        x, y, r = circles[i]
        boundary_violation = 0
        
        # Check all boundaries
        if r > x:
            boundary_violation += (r - x)
        if r > y:
            boundary_violation += (r - y)
        if r > (1 - x):
            boundary_violation += (r - (1 - x))
        if r > (1 - y):
            boundary_violation += (r - (1 - y))
            
        violations.append(boundary_violation)
    
    # Overlap violations
    tree = cKDTree(circles[:, :2])
    for i in range(n):
        x, y, r = circles[i]
        
        # Find nearby circles using cKDTree for efficiency
        neighbors = tree.query_ball_point([x, y], 2 * r)
        for j in neighbors:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                overlap_violation = max(0, (r + r2) - distance)
                violations[i] += overlap_violation
    
    return violations

def initialize_population(pop_size, n=32):
    """Initialize population using evolutionary approach."""
    population = []
    
    for _ in range(pop_size):
        circles = []
        attempts = 0
        
        # Try to place circles with density-aware approach
        while len(circles) < n and attempts < 10000:
            attempts += 1
            
            # Random sampling with preference for less dense areas
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Estimate max radius at this location
            r_max = min(x, y, 1-x, 1-y)
            if r_max <= 0:
                continue

            # Compute local density at this point for adaptive sizing
            density = compute_local_density(circles, [x, y], k=5)
            radius_adjustment = 1.0 / (1.0 + 0.3 * density)
            r_adjusted = min(r_max * 0.4 * radius_adjustment, r_max * 0.4)
            
            # Try different radii
            test_radii = np.linspace(0.01, r_adjusted, 10)
            for r in test_radii:
                if is_valid_placement(circles, x, y, r):
                    circles.append([x, y, r])
                    break
        
        if len(circles) == n:
            population.append(np.array(circles))
    
    return population

def crossover(parent1, parent2):
    """Perform crossover between two parent individuals."""
    n = len(parent1)
    child = np.zeros_like(parent1)
    
    # Use uniform crossover
    for i in range(n):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    
    return child

def mutate(individual, mutation_rate=0.1, max_attempts=100):
    """Mutate an individual by adjusting circle positions and radii."""
    mutated = individual.copy()
    
    # Apply mutations to some circles
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate radius
            old_radius = mutated[i][2]
            radius_change = np.random.normal(0, 0.01)
            new_radius = max(0.001, old_radius + radius_change)
            
            # Try to find a new valid position for this circle
            attempts = 0
            while attempts < max_attempts:
                # Slightly perturb the position
                dx = np.random.normal(0, 0.01)
                dy = np.random.normal(0, 0.01)
                
                new_x = max(0.001, min(0.999, mutated[i][0] + dx))
                new_y = max(0.001, min(0.999, mutated[i][1] + dy))
                
                # Check if new position is valid with adjusted radius
                if is_valid_placement(mutated, new_x, new_y, new_radius):
                    mutated[i] = [new_x, new_y, new_radius]
                    break
                    
                attempts += 1
    
    return mutated

def tournament_selection(population, fitnesses, tournament_size=3):
    """Select an individual using tournament selection."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def evolutionary_optimization(n=32, pop_size=50, generations=100):
    """Evolutionary optimization approach for circle packing."""
    # Initialize population
    population = initialize_population(pop_size, n)
    
    # If we couldn't generate enough valid individuals, fill with heuristics
    while len(population) < pop_size:
        # Use heuristic initialization for missing individuals
        circles = []
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                
                r_min = min(x, y, 1-x, 1-y)
                r = min(r_min * 0.3, 0.15)
                
                if is_valid_placement(circles, x, y, r):
                    circles.append([x, y, r])
        
        # Fill remaining spots
        while len(circles) < n:
            best_r = 0
            best_x, best_y = 0, 0
            
            for _ in range(1000):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                r_max = min(x, y, 1-x, 1-y)
                if r_max <= 0:
                    continue
                
                density = compute_local_density(circles, [x, y], k=5)
                radius_adjustment = 1.0 / (1.0 + 0.3 * density)
                r_adjusted = min(r_max * 0.4 * radius_adjustment, r_max * 0.4)
                
                test_radii = np.linspace(0.01, r_adjusted, 10)
                for r in test_radii:
                    if is_valid_placement(circles, x, y, r):
                        if r > best_r:
                            best_r = r
                            best_x, best_y = x, y
                            break
            
            if best_r > 0:
                circles.append([best_x, best_y, best_r])
        
        population.append(np.array(circles[:n]))
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(individual) for individual in population]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individual
        best_idx = np.argmax(fitnesses)
        new_population.append(population[best_idx].copy())
        
        # Generate offspring through selection, crossover, and mutation
        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best individual
    final_fitnesses = [evaluate_fitness(individual) for individual in population]
    best_idx = np.argmax(final_fitnesses)
    return population[best_idx]

def local_refinement(circles, max_iterations=50):
    """Apply local refinement to improve the solution."""
    n = len(circles)
    
    for iteration in range(max_iterations):
        old_circles = circles.copy()
        
        # Optimize each circle individually with better constraint handling
        for i in range(n):
            def objective(r):
                temp_circles = circles.copy()
                temp_circles[i, 2] = r[0]
                
                # Calculate constraints with proper penalties
                violations = calculate_constraints(temp_circles)
                penalty = sum(violations) * 1000  # Large penalty for violations
                
                # Maximize sum of radii (negative because minimize)
                return -(evaluate_fitness(temp_circles) - penalty)
            
            # Get current values
            current_radius = circles[i, 2]
            
            # Optimization bounds
            bounds = [(0.001, 0.5)]
            
            # Optimize just this circle
            try:
                result = minimize(objective, [current_radius], method='L-BFGS-B', bounds=bounds)
                if result.success:
                    circles[i, 2] = result.x[0]
            except:
                # If optimization fails, keep current radius
                pass
        
        # Check for convergence
        if np.allclose(old_circles, circles, atol=1e-6):
            break
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Use evolutionary optimization to find good initial configuration
    circles = evolutionary_optimization(32, pop_size=50, generations=100)
    
    # Apply local refinement to improve the solution
    circles = local_refinement(circles, max_iterations=30)
    
    # Final local optimization pass
    circles = local_refinement(circles, max_iterations=20)
    
    return circles

# EVOLVE-BLOCK-END