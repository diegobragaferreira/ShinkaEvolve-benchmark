# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List
import random
from collections import defaultdict

# Global constants for the optimization
INITIAL_POPULATION_SIZE = 100
MAX_GENERATIONS = 400
TOURNAMENT_SIZE = 5
INITIAL_MUTATION_RATE = 0.2
ELITISM_COUNT = 10
MIN_MUTATION_RATE = 0.05
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 150
SPECIES_THRESHOLD = 0.08

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if a configuration of circles is valid (no overlaps, fully contained)."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r + BOUNDARY_MARGIN or x > 1-r - BOUNDARY_MARGIN or y < r + BOUNDARY_MARGIN or y > 1-r - BOUNDARY_MARGIN:
            return False

    # Check overlap constraints efficiently using KDTree for large configurations
    if n > 50:
        try:
            positions = circles[:, :2]
            tree = cKDTree(positions)
            pairs = tree.query_pairs(0.001, p=2)  # Very small distance to catch overlaps
            for i, j in pairs:
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance_squared = (x1-x2)**2 + (y1-y2)**2
                    min_distance_squared = (r1 + r2)**2
                    if distance_squared < min_distance_squared:
                        return False
        except:
            # Fallback to brute force if KDTree fails
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance_squared = (x1-x2)**2 + (y1-y2)**2
                    min_distance_squared = (r1 + r2)**2
                    if distance_squared < min_distance_squared:
                        return False
    else:
        # Brute force for small configurations
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_squared = (x1-x2)**2 + (y1-y2)**2
                min_distance_squared = (r1 + r2)**2
                if distance_squared < min_distance_squared:
                    return False

    return True

def evaluate_fitness(circles: np.ndarray) -> float:
    """Evaluate fitness as the sum of radii."""
    return np.sum(circles[:, 2])

def create_speciated_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration with speciation-aware point generation."""
    # Generate diverse candidate points with better distribution
    candidates = []
    
    # Add strategic points for better coverage
    strategic_points = [
        [0, 0], [0, 1], [1, 0], [1, 1],  # Corners
        [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5],  # Midpoints
        [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75],  # Diagonals
        [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],  # Additional corners
        [0.33, 0.33], [0.33, 0.67], [0.67, 0.33], [0.67, 0.67],  # Central points
        [0.5, 0.5],  # Center
    ]
    
    candidates.extend(strategic_points)
    
    # Add random points
    for _ in range(n_circles * 2):
        candidates.append([np.random.rand(), np.random.rand()])
    
    # Convert to numpy array and clip to bounds
    candidates = np.array(candidates)
    candidates = np.clip(candidates, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
    
    # Remove duplicates
    unique_candidates = np.unique(candidates, axis=0)
    
    # Ensure we have enough points
    if len(unique_candidates) < n_circles:
        additional_points = n_circles - len(unique_candidates)
        for _ in range(additional_points):
            unique_candidates = np.vstack([unique_candidates, [np.random.rand(), np.random.rand()]])
    
    # Compute Voronoi diagram (only use valid points)
    try:
        vor = Voronoi(unique_candidates)
        # Use Voronoi cell centers as initial circle positions
        centroids = vor.points[vor.point_region[:-1]]  # Exclude infinite region

        # Limit to number of circles needed
        selected_centroids = centroids[:n_circles]

        # Create circles with initial radii based on Voronoi cell properties
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x, y = selected_centroids[i]
            
            # Calculate maximum possible radius that accounts for geometry
            max_radius = min(x, 1-x, y, 1-y)

            # Adjust based on position: center gets larger radius, edges smaller
            center_distance = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            if center_distance < 0.25:  # Center region
                max_radius = min(max_radius, 0.25)
            elif center_distance < 0.4:  # Middle region
                max_radius = min(max_radius, 0.2)
            else:  # Edge regions
                max_radius = min(max_radius, 0.15)

            # Ensure minimum reasonable value
            max_radius = max(0.005, max_radius)

            circles[i] = [x, y, max_radius]

        return circles
    except:
        # Fallback to grid-based initialization if Voronoi fails
        return generate_grid_initialization(n_circles)

def generate_grid_initialization(n_circles: int) -> np.ndarray:
    """Generate grid-based initial configuration."""
    circles = np.zeros((n_circles, 3))
    grid_size = int(np.ceil(np.sqrt(n_circles)))
    spacing = 1.0 / grid_size
    r = spacing * 0.3
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count < n_circles:
                x = (j + 0.5) * spacing
                y = (i + 0.5) * spacing
                # Adjust for boundary constraints
                x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                circles[count] = [x, y, r]
                count += 1
    
    return circles

def get_species_id(circle, species_centers):
    """Get species ID for a circle based on proximity to species centers."""
    if len(species_centers) == 0:
        return 0
    distances = [np.sqrt((circle[0] - center[0])**2 + (circle[1] - center[1])**2) for center in species_centers]
    min_dist = min(distances)
    if min_dist < SPECIES_THRESHOLD:
        return distances.index(min_dist)
    else:
        return len(species_centers)

def speciate_population(population):
    """Group similar individuals into species to maintain diversity."""
    if len(population) <= 1:
        return [list(range(len(population)))]

    # Use a simplified approach: group by cluster of center positions
    positions = np.array([ind[:, :2].mean(axis=0) for ind in population])
    
    # Simple clustering based on distance
    species = []
    assigned = [False] * len(population)
    
    for i in range(len(population)):
        if not assigned[i]:
            species_group = [i]
            assigned[i] = True
            
            # Group nearby individuals
            for j in range(i+1, len(population)):
                if not assigned[j]:
                    dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                    if dist < SPECIES_THRESHOLD:
                        species_group.append(j)
                        assigned[j] = True
                        
            species.append(species_group)
    
    return species

def constraint_aware_local_search(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Apply a constrained local search optimization to improve solution quality.
    Focuses on efficient radius expansion and intelligent position adjustments.
    """
    # Clone input to avoid modifying original
    current_solution = circles.copy()
    
    # Keep track of best solution found during local search
    best_solution = current_solution.copy()
    best_fitness = evaluate_fitness(current_solution)

    for iteration in range(max_iterations):
        improved = False
        
        # Strategy 1: Try to expand each circle's radius while maintaining constraints
        for i in range(len(current_solution)):
            original_r = current_solution[i, 2]
            x, y, r = current_solution[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                           y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
            
            # Try to expand radius
            if max_radius > r:
                # Binary search for maximum safe expansion
                low = 0.0
                high = max_radius - r
                best_expansion = 0.0
                
                # Early exit for very small improvements
                if high < 0.0001:
                    continue
                    
                for _ in range(20):  # More iterations for better precision
                    test_expansion = (low + high) / 2
                    test_radius = r + test_expansion
                    
                    # Check if expansion violates any constraints
                    valid = True
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                            if dist < (test_radius + r_j):
                                valid = False
                                break
                                
                    if valid:
                        best_expansion = test_expansion
                        low = test_expansion
                    else:
                        high = test_expansion
                
                if best_expansion > 0.0001:  # Only if significant improvement
                    current_solution[i, 2] = r + best_expansion
                    improved = True
        
        # Strategy 2: Try smarter position adjustments to resolve overlaps
        for i in range(len(current_solution)):
            x, y, r = current_solution[i]
            
            # Collect overlapping neighbors more efficiently
            neighbors_to_adjust = []
            for j in range(len(current_solution)):
                if i != j:
                    pos_j = current_solution[j, :2]
                    r_j = current_solution[j, 2]
                    dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                    if dist < (r + r_j):
                        neighbors_to_adjust.append((j, dist))
            
            # If there are overlaps, try adjusting position intelligently
            if neighbors_to_adjust:
                # Try several moves
                best_move = [0.0, 0.0]
                best_score = -1000
                
                # Search in a smaller neighborhood around current position
                search_moves = []
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        search_moves.append((dx, dy))
                        
                # Prioritize moves that directly address overlaps
                for dx, dy in search_moves:
                    test_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + dx))
                    test_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + dy))
                    
                    # Check if move helps
                    score = 0
                    valid = True
                    
                    # Check overlap with others
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                            if dist < (r + r_j):
                                valid = False
                                break
                            # Score based on how much we improve separation
                            if dist < (r + r_j + 0.005):
                                score -= 10  # Penalty for remaining overlap
                            else:
                                score += (dist - (r + r_j))  # Reward for better separation
                    
                    if valid and score > best_score:
                        best_score = score
                        best_move = [dx, dy]
                
                # Apply best move if beneficial
                if best_score > -1000 and best_score > 0.001:
                    current_solution[i, 0] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + best_move[0]))
                    current_solution[i, 1] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + best_move[1]))
                    improved = True
        
        # Strategy 3: Global improvement by adjusting all positions carefully
        if not improved and iteration % 4 == 0:  # Only do occasionally
            # Try small systematic adjustments for all circles
            for i in range(len(current_solution)):
                x, y, r = current_solution[i]
                # Only adjust if it's not too close to boundary
                if (r + BOUNDARY_MARGIN < x < 1 - r - BOUNDARY_MARGIN and 
                    r + BOUNDARY_MARGIN < y < 1 - r - BOUNDARY_MARGIN):
                    dx = np.random.normal(0, 0.005)  # Smaller perturbation
                    dy = np.random.normal(0, 0.005)
                    
                    test_x = np.clip(x + dx, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    test_y = np.clip(y + dy, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    
                    # Check overlap - be more conservative here
                    valid = True
                    for j in range(len(current_solution)):
                        if i != j:
                            pos_j = current_solution[j, :2]
                            r_j = current_solution[j, 2]
                            dist = np.sqrt((test_x - pos_j[0])**2 + (test_y - pos_j[1])**2)
                            if dist < (r + r_j):
                                valid = False
                                break
                    
                    if valid:
                        current_solution[i, 0] = test_x
                        current_solution[i, 1] = test_y
                        improved = True
        
        # Update best solution if current is better
        current_fitness = evaluate_fitness(current_solution)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution.copy()
        
        if not improved:
            break
    
    return best_solution

def initialize_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create an initial population of valid circle configurations."""
    population = []
    
    # Use enhanced Voronoi-based initialization for first few individuals
    for i in range(min(30, pop_size)):
        circles = create_speciated_initialization(n_circles)
        if is_valid_configuration(circles):
            population.append(circles)
    
    # Fill up with grid-based initializations
    while len(population) < pop_size:
        circles = generate_grid_initialization(n_circles)
        if is_valid_configuration(circles):
            population.append(circles)
    
    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float], tournament_size: int) -> np.ndarray:
    """Select an individual using tournament selection."""
    tournament_indices = np.random.choice(len(population), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform crossover between two parent configurations."""
    if np.random.random() > 0.8:  # Lower crossover rate for better preservation
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    crossover_point = np.random.randint(1, n)

    child1 = np.vstack([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.vstack([parent2[:crossover_point], parent1[crossover_point:]])

    # Ensure children are valid
    child1 = enforce_boundaries(child1)
    child2 = enforce_boundaries(child2)
    
    return child1, child2

def enforce_boundaries(circles: np.ndarray) -> np.ndarray:
    """Ensure circles respect boundary constraints."""
    result = circles.copy()
    for i in range(len(result)):
        x, y, r = result[i]
        # Clip position to stay within boundaries
        x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
        y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
        result[i] = [x, y, r]
    return result

def mutate(circles: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
    """Mutate a circle configuration with improved strategy."""
    mutated = circles.copy()

    # Mutate each circle with some probability
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Decide what to mutate with bias towards position
            mutation_type = np.random.choice(['position', 'radius'], p=[0.7, 0.3])

            if mutation_type == 'position':
                # Slightly perturb position with bounded adjustment
                dx = np.random.normal(0, 0.01)
                dy = np.random.normal(0, 0.01)
                
                mutated[i][0] = np.clip(mutated[i][0] + dx,
                                       mutated[i][2] + BOUNDARY_MARGIN, 1 - mutated[i][2] - BOUNDARY_MARGIN)
                mutated[i][1] = np.clip(mutated[i][1] + dy,
                                       mutated[i][2] + BOUNDARY_MARGIN, 1 - mutated[i][2] - BOUNDARY_MARGIN)
            else:
                # Mutate radius with careful bounds
                dr = np.random.normal(0, 0.01)
                new_radius = mutated[i][2] + dr
                # Ensure radius stays positive and reasonable
                mutated[i][2] = np.clip(new_radius, 0.001, min(0.3, 1 - mutated[i][0], mutated[i][0], 
                                                              1 - mutated[i][1], mutated[i][1]))

    return mutated

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 26
    best_solution = None
    best_fitness = -np.inf

    # Initialize population with evolutionary approach
    population = initialize_population(INITIAL_POPULATION_SIZE, n)

    # Remove invalid solutions
    valid_population = [ind for ind in population if is_valid_configuration(ind)]
    if not valid_population:
        # Fallback to simple initialization
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < n:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    circles[count] = [x, y, r]
                    count += 1
        return circles

    population = valid_population

    start_time = time.time()
    
    for generation in range(MAX_GENERATIONS):
        # Calculate adaptive mutation rate
        # Decrease over time to reduce exploration and increase exploitation
        adaptive_mutation_rate = INITIAL_MUTATION_RATE * (1 - generation / MAX_GENERATIONS)
        adaptive_mutation_rate = max(adaptive_mutation_rate, MIN_MUTATION_RATE)

        # Evaluate fitness for all individuals
        fitnesses = [evaluate_fitness(ind) for ind in population]

        # Track best individual
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        elite_indices = np.argsort(fitnesses)[-ELITISM_COUNT:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # Generate offspring
        while len(new_population) < INITIAL_POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, adaptive_mutation_rate)
            child2 = mutate(child2, adaptive_mutation_rate)

            # Ensure validity of children
            if is_valid_configuration(child1):
                new_population.append(child1)
            if len(new_population) < INITIAL_POPULATION_SIZE and is_valid_configuration(child2):
                new_population.append(child2)

        population = new_population[:INITIAL_POPULATION_SIZE]

        # Early stopping check
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break

    # If evolutionary approach didn't find a good solution, fall back to local search
    if best_solution is None:
        # Generate initial candidates with enhanced Voronoi approach
        initial_candidates = []
        for _ in range(10):  # Generate several candidates
            circles = create_speciated_initialization(n)
            if is_valid_configuration(circles):
                initial_candidates.append(circles)
        
        # If no valid candidates, fall back to grid
        if not initial_candidates:
            return generate_grid_initialization(n)
        
        # Select the best initial candidate
        initial_fitnesses = [evaluate_fitness(c) for c in initial_candidates]
        best_initial_idx = np.argmax(initial_fitnesses)
        current_solution = initial_candidates[best_initial_idx].copy()
        
        # Apply local search to the best initial configuration
        refined_solution = constraint_aware_local_search(current_solution)
        current_fitness = evaluate_fitness(refined_solution)
        
        # Track best solution so far
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = refined_solution.copy()
    
    # Do additional local search refinements
    if best_solution is not None:
        # Apply local search to final solution
        refined_solution = constraint_aware_local_search(best_solution)
        refined_fitness = evaluate_fitness(refined_solution)
        
        if refined_fitness > best_fitness:
            best_solution = refined_solution.copy()
            
        # Multiple rounds of refinement
        for _ in range(2):  # Additional refinement rounds
            # Apply local search again
            local_refined = constraint_aware_local_search(refined_solution, 100)
            local_fitness = evaluate_fitness(local_refined)
            
            if local_fitness > best_fitness:
                best_fitness = local_fitness
                best_solution = local_refined.copy()
                refined_solution = local_refined.copy()
            else:
                # Try another approach if not improving
                # Perturb slightly and try again
                perturbed = refined_solution.copy()
                for i in range(len(perturbed)):
                    # Small random perturbation
                    perturbed[i, 0] += np.random.normal(0, 0.005)
                    perturbed[i, 1] += np.random.normal(0, 0.005)
                    # Keep within bounds
                    perturbed[i, 0] = np.clip(perturbed[i, 0], 
                                            perturbed[i, 2] + BOUNDARY_MARGIN, 
                                            1 - perturbed[i, 2] - BOUNDARY_MARGIN)
                    perturbed[i, 1] = np.clip(perturbed[i, 1], 
                                            perturbed[i, 2] + BOUNDARY_MARGIN, 
                                            1 - perturbed[i, 2] - BOUNDARY_MARGIN)
                
                # Validate and update
                if is_valid_configuration(perturbed):
                    perturbed_fitness = evaluate_fitness(perturbed)
                    if perturbed_fitness > best_fitness:
                        best_fitness = perturbed_fitness
                        best_solution = perturbed.copy()
                        refined_solution = perturbed.copy()

    # Return the best solution found
    if best_solution is not None:
        return best_solution
    else:
        # Fallback to grid initialization if no good solution was found
        return generate_grid_initialization(n)


# EVOLVE-BLOCK-END