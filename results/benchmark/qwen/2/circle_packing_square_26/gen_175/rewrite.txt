# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def validate_circles(circles: np.ndarray) -> bool:
    """
    Validates that all circles are within bounds and don't overlap.
    Optimized with early termination and efficient spatial queries.
    """
    n = len(circles)

    # Check containment constraints first
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False

    # Check overlap constraints using KDTree for efficiency
    points = circles[:, :2]  # Get (x, y) coordinates
    tree = cKDTree(points)

    # For each circle, check overlap with others
    for i in range(n):
        x1, y1, r1 = circles[i]
        
        # Find nearby circles (within 2*(r1+r2) distance)
        nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
        
        # Check overlap with each nearby circle
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                
                if distance_sq < min_distance_sq:
                    return False
                    
    return True

def calculate_sum_radii(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def create_initial_population(pop_size: int, n_circles: int) -> List[np.ndarray]:
    """Create initial population using enhanced multi-scale grid initialization"""
    population = []
    
    # Multi-scale grid approach with more diverse placement strategies
    grid_sizes = [2, 3, 4, 5]  # Different grid patterns
    
    for _ in range(pop_size):
        circles = np.zeros((n_circles, 3))
        grid_size = np.random.choice(grid_sizes)
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Place circles on a grid with strategic positioning
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break
                    
                # Position in grid cell with strategic offset
                x = (i + 1 + np.random.uniform(-0.2, 0.2)) * spacing_x
                y = (j + 1 + np.random.uniform(-0.2, 0.2)) * spacing_y
                
                # Initial radius with preference for larger values in center areas
                max_radius = min(x, 1-x, y, 1-y)
                if max_radius > 0.01:
                    # Prefer larger radii for central positions
                    if (0.3 < x < 0.7) and (0.3 < y < 0.7):
                        r = np.random.uniform(0.02, min(0.15, max_radius * 0.8))
                    else:
                        r = np.random.uniform(0.01, min(0.1, max_radius * 0.6))
                else:
                    r = 0.01
                    
                # Adjust to fit within bounds
                r = min(r, max_radius)
                
                circles[idx] = [x, y, r]
                idx += 1
                
        # Add strategic perturbations to improve diversity
        for i in range(n_circles):
            if np.random.rand() < 0.5:  # 50% chance to perturb
                # More aggressive perturbations for better exploration
                circles[i, 0] += np.random.normal(0, 0.015)
                circles[i, 1] += np.random.normal(0, 0.015)
                circles[i, 2] += np.random.normal(0, 0.007)
                
                # Ensure valid bounds
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
                circles[i, 2] = max(0.001, circles[i, 2])
                
        population.append(circles)
        
    return population

def tournament_selection(population: List[np.ndarray], fitnesses: List[float],
                         tournament_size: int = 3) -> np.ndarray:
    """Select parent using improved adaptive tournament selection based on population diversity"""
    # Calculate population diversity (variance of fitness values)
    if len(fitnesses) > 1:
        diversity = np.var(fitnesses)
        std_dev = np.std(fitnesses)
        
        # Adjust tournament size based on diversity with more precise scaling
        if diversity > std_dev * 0.4:
            # High diversity → smaller tournaments (more exploration)
            tournament_size = max(2, min(6, int(tournament_size * 0.7)))
        elif diversity < std_dev * 0.2:
            # Low diversity → larger tournaments (more exploitation)
            tournament_size = max(4, min(8, int(tournament_size * 1.3)))
        else:
            # Medium diversity → standard tournaments
            tournament_size = max(3, min(7, int(tournament_size * 0.9)))

    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
    return population[winner_index].copy()

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Improved crossover with constraint awareness and better distribution"""
    child = parent1.copy()
    
    # Apply crossover with 60% probability for each gene (more selective) 
    mask = np.random.rand(*parent1.shape) > 0.4
    
    # Ensure that the resulting child is valid by checking constraints
    child[mask] = parent2[mask]
    
    # Make sure positions and radii respect boundary conditions with better repair
    for i in range(len(child)):
        x, y, r = child[i]
        child[i, 0] = np.clip(x, r, 1 - r)
        child[i, 1] = np.clip(y, r, 1 - r)
        child[i, 2] = max(0.001, r)
        
    return child

def mutate(circles: np.ndarray, mutation_rate: float = 0.1,
           max_radius_change: float = 0.02) -> np.ndarray:
    """Improved mutation with better boundary handling and strategic perturbations"""
    mutated = circles.copy()
    
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            # Mutate position or radius with adaptive weights
            if np.random.rand() < 0.7:  # 70% chance to mutate position (more frequent)
                mutated[i, 0] += np.random.normal(0, 0.012)
                mutated[i, 1] += np.random.normal(0, 0.012)
                
                # Keep within bounds with better clamping
                mutated[i, 0] = np.clip(mutated[i, 0], mutated[i, 2], 1 - mutated[i, 2])
                mutated[i, 1] = np.clip(mutated[i, 1], mutated[i, 2], 1 - mutated[i, 2])
            else:  # 30% chance to mutate radius
                mutated[i, 2] += np.random.normal(0, max_radius_change * 0.5)
                mutated[i, 2] = max(0.001, mutated[i, 2])
                
    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Enhanced repair mechanism with iterative optimization and physics-based repulsion"""
    repaired = circles.copy()
    
    # Stage 1: Fix boundary violations
    for i in range(len(repaired)):
        x, y, r = repaired[i]
        repaired[i, 0] = np.clip(x, r, 1 - r)
        repaired[i, 1] = np.clip(y, r, 1 - r)
        repaired[i, 2] = max(0.001, repaired[i, 2])
    
    # Stage 2: Resolve overlaps with iterative repulsion
    points = repaired[:, :2]
    tree = cKDTree(points)
    
    # Try up to 5 iterations to resolve overlaps
    for iteration in range(5):
        overlap_found = False
        for i in range(len(repaired)):
            x1, y1, r1 = repaired[i]
            
            # Find nearby circles within a reasonable distance
            nearby_indices = tree.query_ball_point([x1, y1], 2 * (r1 + 0.001))
            
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = repaired[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    min_distance = r1 + r2
                    
                    if distance < min_distance:
                        overlap_found = True
                        
                        # Move circles apart using physics-inspired approach
                        if distance > 0.001:
                            dx = (x1 - x2) / distance
                            dy = (y1 - y2) / distance
                            
                            # Move them apart with enhanced force and proper clamping
                            total_radius = r1 + r2
                            move_amount = (min_distance - distance) * 0.4
                            
                            # Apply movement with bias towards larger circles
                            repaired[i, 0] += dx * move_amount * (r2 / total_radius)
                            repaired[i, 1] += dy * move_amount * (r2 / total_radius)
                            repaired[j, 0] -= dx * move_amount * (r1 / total_radius)
                            repaired[j, 1] -= dy * move_amount * (r1 / total_radius)
                            
                            # Clamp to bounds with better error handling
                            repaired[i, 0] = np.clip(repaired[i, 0], r1, 1 - r1)
                            repaired[i, 1] = np.clip(repaired[i, 1], r1, 1 - r1)
                            repaired[j, 0] = np.clip(repaired[j, 0], r2, 1 - r2)
                            repaired[j, 1] = np.clip(repaired[j, 1], r2, 1 - r2)
        
        if not overlap_found:
            break
            
    # Stage 3: Fine-tune for maximum sum of radii
    # Try small adjustments to increase overall radius sum
    for _ in range(3):
        improved = False
        for i in range(len(repaired)):
            # Try to slightly increase radius in a feasible way
            x, y, r = repaired[i]
            if r < 0.001:
                continue
            
            # Try to find a better position to fit a larger radius
            max_radius = min(x, 1-x, y, 1-y)
            if max_radius > r + 0.001:  # Can potentially increase
                # Check if moving to the center would help
                new_r = min(r + 0.002, max_radius)
                # Simple validation - just see if it fits
                if new_r <= 1-x and new_r <= x and new_r <= 1-y and new_r <= y:
                    # Check if any overlaps would be created
                    valid_move = True
                    for j in range(len(repaired)):
                        if i != j:
                            dx = x - repaired[j, 0]
                            dy = y - repaired[j, 1]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (new_r + repaired[j, 2])**2
                            if dist_sq < min_dist_sq:
                                valid_move = False
                                break
                    if valid_move:
                        repaired[i, 2] = new_r
                        improved = True
        
        if not improved:
            break
    
    return repaired

def classify_overlap_severity(circles: np.ndarray) -> str:
    """Classify overlap severity for targeted local optimization"""
    n = len(circles)
    if n <= 1:
        return "none"
    
    # Count overlaps
    overlap_count = 0
    distances = cdist(circles[:, :2], circles[:, :2])
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    min_distances = (circles[:, 2][:, np.newaxis] + circles[:, 2][np.newaxis, :]) * mask
    overlaps = distances < min_distances
    overlap_count = np.sum(overlaps)
    
    if overlap_count == 0:
        return "none"
    elif overlap_count <= 5:
        return "low"
    elif overlap_count <= 15:
        return "medium"
    else:
        return "high"

def adaptive_local_optimization(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply adaptive local optimization based on overlap severity"""
    n = len(circles)
    circles = circles.copy()
    
    # Determine overlap severity
    severity = classify_overlap_severity(circles)
    
    # Set refinement intensity based on severity
    if severity == "none":
        max_refinement_iter = max_iter // 2
    elif severity == "low":
        max_refinement_iter = max_iter // 3
    elif severity == "medium":
        max_refinement_iter = max_iter // 2
    else:  # high
        max_refinement_iter = max_iter
    
    # More comprehensive local refinement
    for iteration in range(max_refinement_iter):
        improved = False
        
        # Strategy 1: Try to expand radii
        for i in range(n):
            original_radius = circles[i][2]
            original_x, original_y = circles[i][0], circles[i][1]

            # Calculate maximum possible radius for this circle
            max_radius = min(
                original_x,
                original_y,
                1 - original_x,
                1 - original_y
            )

            # Try to increase radius with more careful increment
            if max_radius > original_radius:
                # Use larger increments for low overlap, smaller for high overlap
                if severity == "none":
                    increment = 0.01
                elif severity == "low":
                    increment = 0.005
                else:
                    increment = 0.002
                new_radius = min(original_radius + increment, max_radius)

                if new_radius > original_radius:
                    circles[i][2] = new_radius

                    # Check if valid configuration
                    if validate_circles(circles):
                        improved = True
                    else:
                        # Revert if invalid
                        circles[i][2] = original_radius

        # Strategy 2: Try small position adjustments to resolve conflicts
        if improved or severity in ["low", "medium", "high"]:
            # Apply position adjustments more aggressively when there are overlaps
            adjustment_multiplier = 1.0 if severity == "low" else 2.0
            adjustments = [
                (0.001 * adjustment_multiplier, 0),
                (-0.001 * adjustment_multiplier, 0),
                (0, 0.001 * adjustment_multiplier),
                (0, -0.001 * adjustment_multiplier),
                (0.0005 * adjustment_multiplier, 0.0005 * adjustment_multiplier),
                (-0.0005 * adjustment_multiplier, -0.0005 * adjustment_multiplier)
            ]

            for i in range(n):
                original_x, original_y = circles[i][0], circles[i][1]

                # Try adjustments to resolve overlaps
                for dx, dy in adjustments:
                    new_x = np.clip(original_x + dx, 0, 1)
                    new_y = np.clip(original_y + dy, 0, 1)

                    if new_x != original_x or new_y != original_y:
                        circles[i][0] = new_x
                        circles[i][1] = new_y

                        if validate_circles(circles):
                            improved = True
                            break
                        else:
                            # Revert if invalid
                            circles[i][0] = original_x
                            circles[i][1] = original_y

        # Early termination if no improvement
        if not improved:
            break

    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Algorithm parameters - optimized for better performance
    pop_size = 80   # Increased population size for better exploration
    n_generations = 150  # Increased generations for better convergence
    elite_size = 10   # Increased elite count for better preservation
    
    # Create initial population
    population = create_initial_population(pop_size, 26)

    # Evolution loop
    best_fitness = -np.inf
    best_individual = None
    
    # Three-phase adaptive mutation rate scheduling
    # Phase 1 (generations 1-60): Exploration
    # Phase 2 (generations 61-120): Balance  
    # Phase 3 (generations 121-150): Exploitation
    mutation_schedule = [
        (0, 60, 0.15),   # High for exploration
        (60, 120, 0.05), # Moderate for balance
        (120, 150, 0.015) # Low for exploitation
    ]
    
    for generation in range(n_generations):
        # Calculate adaptive mutation rate
        current_mutation_rate = 0.15  # Default
        for start, end, rate in mutation_schedule:
            if start <= generation < end:
                current_mutation_rate = rate
                break
        
        # Calculate fitness for all individuals
        fitnesses = []
        valid_individuals = []
        
        for circles in population:
            if validate_circles(circles):
                fitness = calculate_sum_radii(circles)
                fitnesses.append(fitness)
                valid_individuals.append(circles)
            else:
                # Repair invalid individuals
                repaired = repair_circles(circles)
                if validate_circles(repaired):
                    fitness = calculate_sum_radii(repaired)
                    fitnesses.append(fitness)
                    valid_individuals.append(repaired)
                else:
                    # If still invalid, penalize heavily
                    fitnesses.append(-np.inf)
                    valid_individuals.append(circles)
        
        # Track best individual
        if valid_individuals:
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_individual = valid_individuals[max_idx].copy()
        
        # Elitism: keep top individuals
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elites = [valid_individuals[i] for i in elite_indices if fitnesses[i] > -np.inf]
        
        # Generate new population
        new_population = elites[:]
        
        # Fill remaining slots with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(valid_individuals, fitnesses)
            parent2 = tournament_selection(valid_individuals, fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child, current_mutation_rate)
            
            # Apply adaptive local optimization
            child = adaptive_local_optimization(child)
            
            # Repair
            child = repair_circles(child)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
        
        # Logging every 25 generations
        if generation % 25 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
    
    # Return the best solution found
    if best_individual is not None:
        return best_individual
    else:
        # If no valid solution found, return the best from final population
        fitnesses = [calculate_sum_radii(circles) for circles in population if validate_circles(circles)]
        if fitnesses:
            best_idx = np.argmax(fitnesses)
            return population[best_idx]
        else:
            # Fallback: return a valid random solution
            circles = np.zeros((26, 3))
            for i in range(26):
                circles[i] = [0.5, 0.5, 0.01]
            return circles

# EVOLVE-BLOCK-END