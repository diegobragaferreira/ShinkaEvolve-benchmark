# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools
import time
from itertools import combinations

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    N_CIRCLES = 26
    
    def initialize_voronoi_population(pop_size):
        """Initialize population using enhanced Voronoi-based approach with better geometric properties"""
        population = []
        for _ in range(pop_size):
            # Generate initial points using hexagonal-like grid for better spread
            points = []
            grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            # Create points in a grid with alternating offset for hexagonal packing
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < N_CIRCLES:
                        # Hexagonal offset pattern with structured randomness
                        offset = (j % 2) * spacing_x / 2
                        jitter_x = np.random.uniform(-spacing_x/10, spacing_x/10)
                        jitter_y = np.random.uniform(-spacing_y/10, spacing_y/10)
                        x = (i + 1) * spacing_x + offset + jitter_x
                        y = (j + 1) * spacing_y + jitter_y
                        points.append([x, y])
            
            # Fill remaining positions with strategic placement
            while len(points) < N_CIRCLES:
                # Try to place point far from existing ones with better rejection sampling
                max_attempts = 100
                placed = False
                for attempt in range(max_attempts):
                    x = np.random.uniform(0.02, 0.98)
                    y = np.random.uniform(0.02, 0.98)

                    # Check distance to existing points - early termination
                    min_dist = float('inf')
                    for existing_point in points:
                        dist = np.sqrt((x - existing_point[0])**2 + (y - existing_point[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                        # Early termination if too close to any existing point
                        if min_dist < 0.03:  # Early termination threshold
                            break

                    # Place if sufficiently far from others
                    if min_dist > 0.03:
                        points.append([x, y])
                        placed = True
                        break

                # If couldn't place far enough, just place randomly
                if not placed:
                    x = np.random.uniform(0.02, 0.98)
                    y = np.random.uniform(0.02, 0.98)
                    points.append([x, y])
            
            # Create Voronoi diagram and get cell centroids as initial circle positions
            points_array = np.array(points)
            
            try:
                # Compute Voronoi diagram
                vor = Voronoi(points_array)
                centroids = []
                
                # Use centroids of finite Voronoi cells as circle centers
                for i, (x, y) in enumerate(vor.points):
                    # Skip infinite cells
                    if i < len(vor.point_region) and vor.point_region[i] >= 0:
                        region = vor.regions[vor.point_region[i]]
                        if len(region) > 0 and all(r >= 0 for r in region):  # Only consider finite regions
                            # Compute centroid of the Voronoi cell
                            vertices = np.array([vor.vertices[r] for r in region])
                            if len(vertices) > 0:
                                centroid = np.mean(vertices, axis=0)
                                # Ensure centroid is within bounds
                                centroid[0] = np.clip(centroid[0], 0.02, 0.98)
                                centroid[1] = np.clip(centroid[1], 0.02, 0.98)
                                centroids.append(centroid)
                
                # If we don't have enough centroids, use original points
                if len(centroids) < N_CIRCLES:
                    centroids = points_array[:N_CIRCLES].tolist()
                else:
                    centroids = centroids[:N_CIRCLES]
                
                # Compute radii based on Voronoi cell sizes with better estimation
                circles = np.zeros((N_CIRCLES, 3))
                for i, (cx, cy) in enumerate(centroids):
                    # Calculate minimum distance to other points
                    min_dist = float('inf')
                    for j, (px, py) in enumerate(centroids):
                        if i != j:
                            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                            min_dist = min(min_dist, dist)
                    
                    # Set radius with improved heuristic based on Voronoi cell geometry
                    if min_dist > 0:
                        # Conservative estimate based on Voronoi structure
                        r = min(0.12, min_dist/4.0)
                    else:
                        r = np.random.uniform(0.01, 0.05)
                    
                    # Enforce hard bounds
                    r = max(0.005, min(0.2, r))
                    circles[i] = [cx, cy, r]
                    
            except Exception:
                # Fall back to simple initialization if Voronoi fails
                circles = np.zeros((N_CIRCLES, 3))
                for i in range(N_CIRCLES):
                    x = np.random.uniform(0.02, 0.98)
                    y = np.random.uniform(0.02, 0.98)
                    r = np.random.uniform(0.01, 0.08)
                    circles[i] = [x, y, r]
            
            population.append(circles)
        return population
    
    def is_valid_solution(circles):
        """Check if solution satisfies all constraints efficiently"""
        # Check containment first for early exit
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check non-overlap using optimized KDTree approach with early termination
        try:
            points = circles[:, :2]
            radii = circles[:, 2]
            
            # Create KDTree for efficient neighbor queries
            tree = KDTree(points)
            # Use a tighter ball query for better performance
            pairs = tree.query_pairs(0.5, return_distance=False)
            
            # Check only pairs that could potentially overlap (faster than full comparison)
            for i, j in pairs:
                if i < j:  # Avoid checking same pair twice
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
                        
        except Exception:
            # Fallback to brute force if KDTree fails
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        
        return True
    
    def evaluate_fitness(individual):
        """Evaluate fitness as negative sum of radii (since we want to maximize)"""
        # Only consider valid solutions
        if not is_valid_solution(individual):
            # Apply large penalty for invalid solutions
            return (-np.sum(individual[:, 2]) - 1000,)
        
        # Return negative sum of radii for maximization problem
        return (-np.sum(individual[:, 2]),)
    
    def mut_gaussian(individual, mu=0, sigma=0.01, indpb=0.2):
        """Custom mutator that keeps circles within bounds and maintains validity"""
        for i in range(len(individual)):
            if random.random() < indpb:
                # Mutate position with careful boundary handling
                individual[i][0] += np.random.normal(mu, sigma)
                individual[i][1] += np.random.normal(mu, sigma)
                
                # Mutate radius with smaller changes
                individual[i][2] += np.random.normal(mu, sigma/3)
                
                # Ensure boundaries
                individual[i][0] = np.clip(individual[i][0], individual[i][2], 1 - individual[i][2])
                individual[i][1] = np.clip(individual[i][1], individual[i][2], 1 - individual[i][2])
                individual[i][2] = np.clip(individual[i][2], 0.001, 0.5)
        
        # Attempt to repair collisions if they occur
        repair_individual(individual)
        return individual,
    
    def repair_individual(individual):
        """Repair an individual to ensure no overlaps and containment using gradient-based approach"""
        # First fix containment issues
        for i in range(len(individual)):
            x, y, r = individual[i]
            
            # Ensure containment
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            individual[i][0] = x
            individual[i][1] = y
        
        # Resolve overlaps with gradient-based separation
        max_iterations = 5
        for iteration in range(max_iterations):
            overlapped = False
            # Build adjacency matrix for efficient overlap checking
            points = individual[:, :2]
            radii = individual[:, 2]
            
            # Use KDTree to find nearby circles
            tree = KDTree(points)
            pairs = tree.query_pairs(0.5, return_distance=False)
            
            for i, j in pairs:
                if i < j:
                    x1, y1, r1 = individual[i]
                    x2, y2, r2 = individual[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if dist < r1 + r2:
                        overlapped = True
                        # Gradient-based separation using attractive forces
                        dx = x2 - x1
                        dy = y2 - y1
                        
                        # Handle special case of identical positions
                        if dx == 0 and dy == 0:
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                        
                        # Normalize direction vector
                        length = np.sqrt(dx*dx + dy*dy)
                        if length > 0:
                            dx /= length
                            dy /= length
                        
                        # Calculate required separation
                        separation = (r1 + r2) - dist
                        # Move both circles outward from each other
                        individual[i][0] -= dx * separation * 0.3
                        individual[i][1] -= dy * separation * 0.3
                        individual[j][0] += dx * separation * 0.3
                        individual[j][1] += dy * separation * 0.3
                        
                        # Clip to bounds
                        individual[i][0] = np.clip(individual[i][0], r1, 1 - r1)
                        individual[i][1] = np.clip(individual[i][1], r1, 1 - r1)
                        individual[j][0] = np.clip(individual[j][0], r2, 1 - r2)
                        individual[j][1] = np.clip(individual[j][1], r2, 1 - r2)
            
            if not overlapped:
                break
    
    def cx_uniform(individual1, individual2):
        """Uniform crossover for circle positions and radii"""
        size = len(individual1)
        for i in range(size):
            if random.random() < 0.5:
                individual1[i], individual2[i] = individual2[i], individual1[i]
        
        # Repair offspring
        repair_individual(individual1)
        repair_individual(individual2)
        return individual1, individual2
    
    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: [np.random.uniform(0.02, 0.98), 
                              np.random.uniform(0.02, 0.98), 
                              np.random.uniform(0.01, 0.08)], 
                     n=N_CIRCLES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", cx_uniform)
    toolbox.register("mutate", mut_gaussian)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population with Voronoi-based starting points
    pop = initialize_voronoi_population(75)
    
    # Convert to DEAP individuals
    deap_pop = []
    for p in pop:
        ind = creator.Individual(p.tolist())
        ind.fitness.values = evaluate_fitness(p)
        deap_pop.append(ind)
    
    # Run evolution with refined parameters
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evolution parameters - optimized for better convergence
    n_generations = 120  # Reduced generations for faster execution
    mutation_rate = 0.12
    crossover_rate = 0.85
    
    # Run evolution with dynamic parameter adjustment
    last_improvement = 0
    best_fitness = float('-inf')
    stagnation_counter = 0
    
    for gen in range(n_generations):
        # Adaptive mutation rate with exponential decay
        current_mutation_rate = max(0.005, mutation_rate * (0.95 ** gen))
        
        # Select parents
        offspring = toolbox.select(deap_pop, len(deap_pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_rate:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation with adaptive rate
        for mutant in offspring:
            if random.random() < current_mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not hasattr(ind.fitness, 'values') or len(ind.fitness.values) == 0]
        for ind in invalid_ind:
            ind.fitness.values = evaluate_fitness(np.array(ind))
        
        # Update population
        deap_pop[:] = offspring
        
        # Update hall of fame
        hof.update(deap_pop)
        
        # Track best fitness for early stopping
        current_best_fitness = max([ind.fitness.values[0] for ind in deap_pop if hasattr(ind.fitness, 'values')])
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            last_improvement = gen
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            if stagnation_counter > 20:  # Early stopping if no improvement for 20 generations
                break
        
        # Print progress
        if gen % 15 == 0:
            try:
                best_fit = max([ind.fitness.values[0] for ind in deap_pop])
                print(f"Generation {gen}: Best fitness = {-best_fit}")
            except:
                pass
    
    # Get best solution
    best_individual = hof[0] if hof else deap_pop[0]
    best_solution = np.array(best_individual)
    
    # Final validation and repair
    if not is_valid_solution(best_solution):
        repair_individual(best_solution)
    
    # Enhanced refinement using better local search
    try:
        final_circles = best_solution.copy()
        
        # Multi-phase local refinement with improved optimization
        for phase in range(3):
            improved = False
            phase_improvements = 0
            
            # Phase 1: Radius maximization
            if phase == 0:
                for iteration in range(15):
                    for i in range(N_CIRCLES):
                        # Store original values
                        orig_x, orig_y, orig_r = final_circles[i]
                        
                        # Try to increase radius while maintaining constraints
                        # Find the maximum possible radius
                        min_dist_to_others = float('inf')
                        for j in range(N_CIRCLES):
                            if i != j:
                                x2, y2, r2 = final_circles[j]
                                dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                                min_dist_to_others = min(min_dist_to_others, dist)
                        
                        # New radius can be up to min_dist_to_others - margin
                        max_new_radius = min_dist_to_others - 0.001 if min_dist_to_others > 0.001 else orig_r
                        
                        if max_new_radius > orig_r:
                            # Try to increase radius
                            target_r = min(max_new_radius, orig_r + 0.015)
                            target_r = max(0.001, min(target_r, 0.5))  # Bound radius
                            
                            # Check if new configuration is valid
                            temp_circles = final_circles.copy()
                            temp_circles[i][2] = target_r
                            
                            if is_valid_solution(temp_circles):
                                final_circles[i][2] = target_r
                                improved = True
                                phase_improvements += 1
                
                if phase_improvements == 0:
                    break
                
            # Phase 2: Position refinement
            elif phase == 1:
                for iteration in range(10):
                    for i in range(N_CIRCLES):
                        # Store original values
                        orig_x, orig_y, orig_r = final_circles[i]
                        
                        # Try small position adjustments in multiple directions
                        best_position = [orig_x, orig_y]
                        best_radius = orig_r
                        best_valid = False
                        
                        # Sample several potential moves
                        for _ in range(10):
                            new_x = orig_x + np.random.uniform(-0.005, 0.005)
                            new_y = orig_y + np.random.uniform(-0.005, 0.005)
                            
                            # Clip to bounds
                            new_x = np.clip(new_x, orig_r, 1 - orig_r)
                            new_y = np.clip(new_y, orig_r, 1 - orig_r)
                            
                            # Check validity
                            temp_circles = final_circles.copy()
                            temp_circles[i][0] = new_x
                            temp_circles[i][1] = new_y
                            
                            if is_valid_solution(temp_circles):
                                # Update to this better position
                                best_position = [new_x, new_y]
                                best_valid = True
                                
                        if best_valid:
                            final_circles[i][0] = best_position[0]
                            final_circles[i][1] = best_position[1]
                            improved = True
            
            # Phase 3: Combined refinement
            elif phase == 2:
                for iteration in range(8):
                    # Try to simultaneously improve multiple circles
                    improvement_count = 0
                    for i in range(N_CIRCLES):
                        # Store original values
                        orig_x, orig_y, orig_r = final_circles[i]
                        
                        # Find the maximum possible radius
                        min_dist_to_others = float('inf')
                        for j in range(N_CIRCLES):
                            if i != j:
                                x2, y2, r2 = final_circles[j]
                                dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                                min_dist_to_others = min(min_dist_to_others, dist)
                        
                        # New radius can be up to min_dist_to_others - margin
                        max_new_radius = min_dist_to_others - 0.001 if min_dist_to_others > 0.001 else orig_r
                        
                        if max_new_radius > orig_r:
                            # Try to increase radius
                            target_r = min(max_new_radius, orig_r + 0.01)
                            target_r = max(0.001, min(target_r, 0.5))  # Bound radius
                            
                            # Check if new configuration is valid
                            temp_circles = final_circles.copy()
                            temp_circles[i][2] = target_r
                            
                            if is_valid_solution(temp_circles):
                                final_circles[i][2] = target_r
                                improvement_count += 1
                                improved = True
                    
                    if improvement_count == 0:
                        break
            
            if not improved:
                break
                
        # Repair if needed
        if not is_valid_solution(final_circles):
            repair_individual(final_circles)
        
        best_solution = final_circles
        
    except Exception as e:
        # If optimization fails, just return the best evolved solution
        pass
    
    return best_solution

# EVOLVE-BLOCK-END