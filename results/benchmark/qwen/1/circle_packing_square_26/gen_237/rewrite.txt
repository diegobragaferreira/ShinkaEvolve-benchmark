# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List

# Global constants for the optimization
INITIAL_POPULATION_SIZE = 150
MAX_GENERATIONS = 600
TOURNAMENT_SIZE = 5
INITIAL_MUTATION_RATE = 0.15
ELITISM_COUNT = 8
MIN_MUTATION_RATE = 0.02
BOUNDARY_MARGIN = 0.01
MAX_LOCAL_SEARCH_ITERATIONS = 200

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if a configuration of circles is valid (no overlaps, fully contained)."""
    n = len(circles)

    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < r + BOUNDARY_MARGIN or x > 1-r - BOUNDARY_MARGIN or y < r + BOUNDARY_MARGIN or y > 1-r - BOUNDARY_MARGIN:
            return False

    # Check overlap constraints with early termination
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

def create_weighted_voronoi_initialization(n_circles: int) -> np.ndarray:
    """Create initial configuration using enhanced Voronoi-based spreading with weighted placement."""
    # Create initial points with strategic distribution
    points = []

    # 1. Grid-based points with randomness
    grid_size = int(np.ceil(np.sqrt(n_circles)) * 1.2)
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n_circles * 3:
                x = (j + 0.5) / grid_size
                y = (i + 0.5) / grid_size
                points.append([x, y])

    # 2. Random points
    random_points = np.random.rand(n_circles * 2, 2)
    points.extend(random_points.tolist())

    # 3. Boundary points for edge coverage
    for _ in range(n_circles * 2):
        side = np.random.randint(0, 4)
        if side == 0:  # Top
            points.append([np.random.rand(), 1.0 - BOUNDARY_MARGIN])
        elif side == 1:  # Bottom
            points.append([np.random.rand(), BOUNDARY_MARGIN])
        elif side == 2:  # Left
            points.append([BOUNDARY_MARGIN, np.random.rand()])
        else:  # Right
            points.append([1.0 - BOUNDARY_MARGIN, np.random.rand()])

    # 4. Corner points
    corners = [[BOUNDARY_MARGIN, BOUNDARY_MARGIN],
               [BOUNDARY_MARGIN, 1-BOUNDARY_MARGIN],
               [1-BOUNDARY_MARGIN, BOUNDARY_MARGIN],
               [1-BOUNDARY_MARGIN, 1-BOUNDARY_MARGIN]]
    points.extend(corners)

    points = np.array(points)[:n_circles * 6]  # Take enough points
    points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)

    # Compute Voronoi diagram
    try:
        vor = Voronoi(points)
        # Use Voronoi cell centers as initial circle positions
        centroids = vor.points[vor.point_region[:-1]]  # Exclude infinite region

        # Limit to number of circles needed
        selected_centroids = centroids[:n_circles]

        # Create circles with initial radii using weighted importance
        circles = np.zeros((n_circles, 3))

        # Compute importance weights based on distance from center and boundary
        weights = []
        for x, y in selected_centroids:
            # Weight based on distance to center (closer to center = higher weight)
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            # Invert so that center points get higher weights
            center_weight = 1.0 - center_dist / np.sqrt(0.5**2 + 0.5**2)
            
            # Also consider boundary proximity (lower weight near edges)
            boundary_min_dist = min(x, 1-x, y, 1-y)
            boundary_weight = boundary_min_dist / 0.5
            
            # Combined weight
            combined_weight = 0.6 * center_weight + 0.4 * boundary_weight
            weights.append(max(0, combined_weight))

        # Sort by weights (higher first) to prioritize important locations
        sorted_indices = np.argsort(weights)[::-1]
        
        for idx, actual_idx in enumerate(sorted_indices):
            x, y = selected_centroids[actual_idx]
            
            # Find nearest neighbor to estimate appropriate radius
            distances = np.sqrt(np.sum((selected_centroids - [x, y])**2, axis=1))
            distances = distances[distances > 0]  # Exclude self-distance
            
            if len(distances) > 0:
                avg_distance = np.min(distances) * 0.35
                radius = min(avg_distance, 0.3)  # Adjusted max radius
            else:
                radius = 0.1

            # Adjust radius based on weight (larger for important locations)
            weight = weights[actual_idx]
            adjusted_radius = radius * (0.5 + 0.5 * weight)  # Blend with base radius
            
            # Ensure it's within bounds
            adjusted_radius = min(adjusted_radius, 
                                x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                                y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)

            circles[idx] = [x, y, max(adjusted_radius, 0.001)]

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

def constraint_aware_local_search(circles: np.ndarray, max_iterations: int = MAX_LOCAL_SEARCH_ITERATIONS) -> np.ndarray:
    """
    Apply a constrained local search optimization to improve solution quality.
    Enhanced with force-based overlap resolution and improved position adjustments.
    """
    # Clone input to avoid modifying original
    current_solution = circles.copy()
    
    # Keep track of best solution during local search
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
                
                for _ in range(12):  # More iterations for better precision
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
                
                if best_expansion > 0:
                    current_solution[i, 2] = r + best_expansion
                    improved = True
        
        # Strategy 2: Force-based position adjustment to resolve overlaps
        for i in range(len(current_solution)):
            x, y, r = current_solution[i]
            
            # Collect overlapping neighbors
            overlapping_pairs = []
            for j in range(len(current_solution)):
                if i != j:
                    pos_j = current_solution[j, :2]
                    r_j = current_solution[j, 2]
                    dist = np.sqrt((x - pos_j[0])**2 + (y - pos_j[1])**2)
                    if dist < (r + r_j):
                        overlapping_pairs.append((j, dist, pos_j, r_j))
            
            # If there are overlaps, use force-based resolution
            if overlapping_pairs:
                # Calculate repulsive forces from overlapping neighbors
                total_force_x = 0.0
                total_force_y = 0.0
                
                for other_i, dist, other_pos, other_r in overlapping_pairs:
                    # Compute direction vector from other circle to this one
                    dx = x - other_pos[0]
                    dy = y - other_pos[1]
                    
                    # Normalize
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx /= length
                        dy /= length
                        
                        # Repulsion force (stronger when closer)
                        force_magnitude = 0.02 * (1 - dist/(r + other_r))
                        total_force_x += dx * force_magnitude
                        total_force_y += dy * force_magnitude
                
                # Apply force to position (with bounds checking)
                if abs(total_force_x) > 0 or abs(total_force_y) > 0:
                    new_x = x + total_force_x
                    new_y = y + total_force_y
                    
                    # Ensure new position is valid
                    new_x = max(r + BOUNDARY_MARGIN, min(1 - r - BOUNDARY_MARGIN, new_x))
                    new_y = max(r + BOUNDARY_MARGIN, min(1 - r - BOUNDARY_MARGIN, new_y))
                    
                    # Check if the move actually improves overlap situation
                    valid_move = True
                    for other_i, _, other_pos, other_r in overlapping_pairs:
                        dist_after = np.sqrt((new_x - other_pos[0])**2 + (new_y - other_pos[1])**2)
                        if dist_after < (r + other_r):
                            valid_move = False
                            break
                    
                    if valid_move:
                        current_solution[i, 0] = new_x
                        current_solution[i, 1] = new_y
                        improved = True
        
        # Strategy 3: Systematic nearby position searches
        if not improved:
            for i in range(len(current_solution)):
                x, y, r = current_solution[i]
                
                # Try various small moves systematically
                best_move = [0.0, 0.0]
                best_score = -1000
                
                # Use a more extensive grid around the current position
                moves = []
                step = 0.01
                for dx in [-step*2, -step, 0, step, step*2]:
                    for dy in [-step*2, -step, 0, step, step*2]:
                        moves.append((dx, dy))
                
                for dx, dy in moves:
                    test_x = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + dx))
                    test_y = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + dy))
                    
                    # Score based on improvement in overlap resolution and separation
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
                            # Reward good separation
                            score += max(0, dist - (r + r_j))  # Positive reward for distance above minimum
                        
                    if valid and score > best_score:
                        best_score = score
                        best_move = [dx, dy]
                
                # Apply best move if beneficial
                if best_score > 0:
                    current_solution[i, 0] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, x + best_move[0]))
                    current_solution[i, 1] = max(BOUNDARY_MARGIN + r, min(1 - BOUNDARY_MARGIN - r, y + best_move[1]))
                    improved = True
        
        # Strategy 4: Random perturbations to escape local minima
        if not improved and iteration % 7 == 0:  # Occasionally try random moves
            for i in range(len(current_solution)):
                if np.random.random() < 0.3:  # 30% chance per circle
                    x, y, r = current_solution[i]
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    
                    test_x = np.clip(x + dx, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    test_y = np.clip(y + dy, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    
                    # Check overlap with others
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
    
    # Use weighted Voronoi-based initialization for majority of individuals
    for i in range(int(pop_size * 0.7)):  # 70% weighted Voronoi
        circles = create_weighted_voronoi_initialization(n_circles)
        if is_valid_configuration(circles):
            population.append(circles)
    
    # Add some random configurations for diversity
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
    """Perform crossover between two parent configurations with improved strategy."""
    if np.random.random() > 0.75:  # Even lower crossover rate to preserve good solutions
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    # Use a more robust uniform crossover approach
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Cross half the circles from each parent, but with more attention to geometric compatibility
    crossover_point = n // 2
    
    # Swap segments
    child1[:crossover_point] = parent1[:crossover_point]
    child2[:crossover_point] = parent2[:crossover_point]
    
    # Apply repair mechanism to fix constraint violations
    child1 = repair_offspring(child1)
    child2 = repair_offspring(child2)
    
    return child1, child2

def repair_offspring(circles: np.ndarray) -> np.ndarray:
    """Repair offspring solutions to ensure they satisfy all constraints."""
    repaired = circles.copy()

    # Fix boundary violations first
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        # Ensure position respects boundaries
        repaired[i] = [
            np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN),
            np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN),
            r
        ]

    # Resolve overlap violations through iterative adjustment
    max_iterations = 30
    for iteration in range(max_iterations):
        violated = False
        # Check all pairs for overlap
        for i in range(len(repaired)):
            for j in range(i + 1, len(repaired)):
                x1, y1, r1 = repaired[i]
                x2, y2, r2 = repaired[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                if distance < (r1 + r2):
                    violated = True
                    # Move circles apart with more precise calculation
                    if distance > 1e-8:  # Avoid division by zero
                        dx = (x2 - x1) / distance * (r1 + r2 - distance) * 0.3
                        dy = (y2 - y1) / distance * (r1 + r2 - distance) * 0.3
                    else:
                        # If circles are at same position, move them apart randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * (r1 + r2) * 0.3
                        dy = np.sin(angle) * (r1 + r2) * 0.3

                    # Apply small adjustments
                    repaired[i][0] = np.clip(repaired[i][0] - dx,
                                           repaired[i][2] + BOUNDARY_MARGIN,
                                           1 - repaired[i][2] - BOUNDARY_MARGIN)
                    repaired[i][1] = np.clip(repaired[i][1] - dy,
                                           repaired[i][2] + BOUNDARY_MARGIN,
                                           1 - repaired[i][2] - BOUNDARY_MARGIN)
                    repaired[j][0] = np.clip(repaired[j][0] + dx,
                                           repaired[j][2] + BOUNDARY_MARGIN,
                                           1 - repaired[j][2] - BOUNDARY_MARGIN)
                    repaired[j][1] = np.clip(repaired[j][1] + dy,
                                           repaired[j][2] + BOUNDARY_MARGIN,
                                           1 - repaired[j][2] - BOUNDARY_MARGIN)

        if not violated:
            break

    return repaired

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
            # Decide what to mutate with bias towards position (more impactful)
            mutation_type = np.random.choice(['position', 'radius'], p=[0.75, 0.25])

            if mutation_type == 'position':
                # Larger position perturbation for better exploration
                dx = np.random.normal(0, 0.02)
                dy = np.random.normal(0, 0.02)
                
                mutated[i][0] = np.clip(mutated[i][0] + dx,
                                       mutated[i][2] + BOUNDARY_MARGIN, 1 - mutated[i][2] - BOUNDARY_MARGIN)
                mutated[i][1] = np.clip(mutated[i][1] + dy,
                                       mutated[i][2] + BOUNDARY_MARGIN, 1 - mutated[i][2] - BOUNDARY_MARGIN)
            else:
                # Mutate radius with careful bounds
                dr = np.random.normal(0, 0.012)
                new_radius = mutated[i][2] + dr
                # Ensure radius stays positive and reasonable
                mutated[i][2] = np.clip(new_radius, 0.001, min(0.45, 1 - mutated[i][0], mutated[i][0], 
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
        adaptive_mutation_rate = MAX_GENERATIONS * INITIAL_MUTATION_RATE / (MAX_GENERATIONS + generation * 2)
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
        # Generate initial candidates with weighted Voronoi approach
        initial_candidates = []
        for _ in range(10):  # Generate several candidates
            circles = create_weighted_voronoi_initialization(n)
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
        for _ in range(3):  # Additional refinement rounds
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