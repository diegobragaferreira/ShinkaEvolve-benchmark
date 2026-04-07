# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from typing import Tuple, List

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circle_placement(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and don't overlap using KDTree for efficiency."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]
    tree = cKDTree(points)
    
    for i in range(n):
        x, y, r = circles[i]
        # Find nearby circles (within 2*r distance)
        indices = tree.query_ball_point([x, y], 2*r)
        for j in indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < r + r2:
                    return False

    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate sum of all radii."""
    return np.sum(circles[:, 2])

def local_optimize(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Apply local optimization to improve a solution by making small adjustments."""
    optimized = circles.copy()
    
    # Try to increase radii where possible
    for _ in range(max_iterations):
        improved = False
        
        # For each circle, try to increase its radius
        for i in range(len(optimized)):
            current_r = optimized[i, 2]
            
            # Try increasing radius slightly
            new_r = min(current_r * 1.05, 0.5)  # Cap at reasonable size
            
            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(len(optimized)):
                if i != j:
                    dist = np.sqrt((optimized[i, 0] - optimized[j, 0])**2 +
                                 (optimized[i, 1] - optimized[j, 1])**2)
                    if dist < new_r + optimized[j, 2]:
                        valid = False
                        break
            
            # Update if improvement is possible and within bounds
            if valid and new_r <= 1 - max(optimized[i, 0], 1 - optimized[i, 0]) and \
               new_r <= 1 - max(optimized[i, 1], 1 - optimized[i, 1]):
                optimized[i, 2] = new_r
                improved = True
        
        # If no improvement made, stop early
        if not improved:
            break
    
    return optimized

def create_initial_population(pop_size: int, n_circles: int) -> list:
    """Create initial population of valid circle arrangements with enhanced initialization."""
    population = []
    
    # First, try to create valid individuals with structured grid initialization
    for _ in range(pop_size):
        # Create circles in a grid-like pattern with better spacing and strategic perturbations
        circles = np.zeros((n_circles, 3))
        
        # Determine grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Initial grid placement with more conservative radius
        radius = min(spacing_x, spacing_y) * 0.4
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                
                # Apply strategic perturbation to positions
                # Use different perturbation sizes for different positions
                if i % 2 == 0 and j % 2 == 0:
                    # Corner positions get larger perturbations
                    x += np.random.uniform(-spacing_x/4, spacing_x/4)
                    y += np.random.uniform(-spacing_y/4, spacing_y/4)
                elif i % 2 == 1 and j % 2 == 1:
                    # Center positions get moderate perturbations
                    x += np.random.uniform(-spacing_x/8, spacing_x/8)
                    y += np.random.uniform(-spacing_y/8, spacing_y/8)
                else:
                    # Edge positions get smaller perturbations
                    x += np.random.uniform(-spacing_x/12, spacing_x/12)
                    y += np.random.uniform(-spacing_y/12, spacing_y/12)
                
                # Ensure it stays within bounds
                x = np.clip(x, radius, 1 - radius)
                y = np.clip(y, radius, 1 - radius)
                
                circles[count] = [x, y, radius]
                count += 1
            if count >= n_circles:
                break
        
        # Fine-tune radius values to allow for better packing
        # Try to increase radii in a controlled way that respects constraints
        for i in range(n_circles):
            # Start with a better approximation of maximum possible radius
            max_radius = radius
            
            # Check neighbor constraints to determine actual max radius
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
                    # Maximum radius that doesn't cause overlap
                    max_radius = min(max_radius, dist - circles[j, 2] - 0.001)  # Small safety margin
            
            # Ensure we don't exceed bounds
            max_radius = min(max_radius, circles[i, 0] - 0.001, 1 - circles[i, 0] - 0.001,
                            circles[i, 1] - 0.001, 1 - circles[i, 1] - 0.001)
            
            # Set a reasonable radius that's close to maximum possible
            if max_radius > 0.001:
                new_radius = max_radius * 0.9  # Use 90% of max possible to ensure safety
                circles[i, 2] = new_radius
        
        # Ensure all circles are valid and fix any issues
        for i in range(n_circles):
            # If still invalid due to rounding issues, fix manually
            if not validate_circle_placement(circles):
                # Try to adjust position to make it valid
                x, y, r = circles[i]
                # Find a valid location near current position
                for _ in range(100):
                    test_x = np.clip(x + np.random.uniform(-r/2, r/2), r, 1-r)
                    test_y = np.clip(y + np.random.uniform(-r/2, r/2), r, 1-r)
                    circles[i] = [test_x, test_y, r]
                    if validate_circle_placement(circles):
                        break
        
        # Final cleanup to ensure all positions are valid
        for i in range(n_circles):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        # Apply local optimization to improve the initial solution
        circles = local_optimize(circles)
        
        population.append(circles.copy())
    
    return population

def mutate_individual(circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Apply mutation to an individual with constraint awareness and adaptive strengths."""
    mutated = circles.copy()
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Calculate constraint violation measure for this circle
            x, y, r = mutated[i]
            
            # Measure how close we are to boundary constraints
            boundary_distance = min(x, y, 1 - x, 1 - y)
            
            # Adjust mutation strength based on constraint proximity
            # Closer to boundary = smaller mutation to avoid going out of bounds
            if boundary_distance < 0.1:
                pos_mutation_strength = 0.005
                radius_mutation_strength = 0.005
            else:
                pos_mutation_strength = 0.02
                radius_mutation_strength = 0.01
            
            # Mutate either position or radius
            if np.random.random() < 0.5:
                # Mutate position with adaptive bounds
                mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, pos_mutation_strength), 0.01, 0.99)
                mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, pos_mutation_strength), 0.01, 0.99)
            else:
                # Mutate radius with adaptive bounds
                mutated[i, 2] = np.clip(mutated[i, 2] + np.random.normal(0, radius_mutation_strength), 0.001, 0.2)
    
    # Fix any invalid placements
    for i in range(len(mutated)):
        if not validate_circle_placement(mutated):
            # Try to fix by adjusting position and radius
            attempts = 0
            while not validate_circle_placement(mutated) and attempts < 100:
                mutated[i, 0] = np.random.uniform(0.01, 0.99)
                mutated[i, 1] = np.random.uniform(0.01, 0.99)
                mutated[i, 2] = np.random.uniform(0.001, 0.1)
                attempts += 1
    
    return mutated

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parents with overlap risk awareness."""
    # Uniform crossover with better mixing strategy
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Calculate overlap risk for each pair of circles between parents
    # and use this information to guide crossover decisions
    for i in range(len(child1)):
        # Calculate distance between circles from both parents
        x1, y1, r1 = parent1[i]
        x2, y2, r2 = parent2[i]
        
        # Estimate overlap risk based on proximity
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        overlap_risk = max(0, (r1 + r2) - distance) / (r1 + r2 + 1e-8)
        
        # Higher overlap risk means lower chance of copying from that parent
        crossover_prob = 0.8 if overlap_risk < 0.5 else 0.3
        
        if np.random.random() < crossover_prob:
            child1[i] = parent2[i].copy()
            child2[i] = parent1[i].copy()
    
    return child1, child2

def tournament_selection(population: List[np.ndarray], fitnesses: List[float], 
                        tournament_size: int = 3) -> np.ndarray:
    """Select parents using tournament selection with better probability calculation."""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_idx].copy()

def optimize_circles() -> np.ndarray:
    """Main optimization function using evolutionary algorithm."""
    n_circles = 26
    pop_size = 50
    generations = 100
    
    # Create initial population with better starting points
    population = create_initial_population(pop_size, n_circles)
    
    best_fitness = 0
    best_individual = None
    
    for generation in range(generations):
        # Adaptive mutation rate with exponential decay
        mutation_rate = 0.1 * np.exp(-generation / 75.0) + 0.01
        
        # Evaluate fitness for all individuals
        fitnesses = []
        for individual in population:
            if validate_circle_placement(individual):
                fitnesses.append(calculate_sum_radii(individual))
            else:
                fitnesses.append(-1000000)  # Invalid solutions get very low fitness
        
        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Select parents
        parents = [tournament_selection(population, fitnesses) for _ in range(pop_size)]
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Elitism: keep best individual
        if best_individual is not None:
            new_population.append(best_individual)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate_individual(child1, mutation_rate)
            child2 = mutate_individual(child2, mutation_rate)
            
            # Apply local optimization to refined solutions
            child1 = local_optimize(child1)
            child2 = local_optimize(child2)
            
            # Ensure children meet constraints
            if validate_circle_placement(child1):  # Check if valid
                new_population.append(child1)
            if len(new_population) < pop_size and validate_circle_placement(child2):
                new_population.append(child2)
        
        # Trim to population size
        population = new_population[:pop_size]
    
    # Final local optimization on the best solution
    if best_individual is not None:
        best_individual = local_optimize(best_individual, max_iterations=100)
    
    return best_individual if best_individual is not None else population[0]

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    try:
        circles = optimize_circles()
        return circles
    except Exception as e:
        print(f"Error during optimization: {e}")
        # Fallback to improved heuristic
        circles = np.zeros((26, 3))
        
        # Try to create a more organized pattern
        grid_size = int(np.ceil(np.sqrt(26)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        radius = spacing_x / 3.0
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= 26:
                    break
                x = spacing_x * (i + 1)
                y = spacing_y * (j + 1)
                # Slightly randomize to avoid perfect grid issues
                x += np.random.uniform(-spacing_x/10, spacing_x/10)
                y += np.random.uniform(-spacing_y/10, spacing_y/10)
                circles[count] = [x, y, radius]
                count += 1
            if count >= 26:
                break
        
        # Ensure constraints are satisfied
        for i in range(count):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        return circles

# EVOLVE-BLOCK-END