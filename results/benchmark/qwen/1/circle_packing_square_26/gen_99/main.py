# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools
import time
from itertools import combinations
from collections import defaultdict

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
    
    def initialize_multi_scale_population(pop_size):
        """Initialize population using multi-scale approach with different initialization strategies"""
        population = []
        for _ in range(pop_size):
            circles = np.zeros((N_CIRCLES, 3))
            
            # Strategy 1: Hexagonal grid with variation
            grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < N_CIRCLES:
                        offset = (j % 2) * spacing_x / 2
                        jitter_x = np.random.uniform(-spacing_x/10, spacing_x/10)
                        jitter_y = np.random.uniform(-spacing_y/10, spacing_y/10)
                        x = (i + 1) * spacing_x + offset + jitter_x
                        y = (j + 1) * spacing_y + jitter_y
                        points.append([x, y])
            
            # Strategy 2: Fill remaining positions with scattered points
            while len(points) < N_CIRCLES:
                # Prefer placing points near edges for better space utilization
                if random.random() < 0.7:
                    # Random interior placement
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                else:
                    # Edge placement
                    side = random.randint(0, 3)
                    if side == 0:  # Top
                        x = np.random.uniform(0.05, 0.95)
                        y = 0.99
                    elif side == 1:  # Right
                        x = 0.99
                        y = np.random.uniform(0.05, 0.95)
                    elif side == 2:  # Bottom
                        x = np.random.uniform(0.05, 0.95)
                        y = 0.01
                    else:  # Left
                        x = 0.01
                        y = np.random.uniform(0.05, 0.95)
                points.append([x, y])
            
            points = points[:N_CIRCLES]
            
            # Initialize circles with proper radii
            for i in range(N_CIRCLES):
                x, y = points[i]
                # Ensure boundary constraints
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                
                # Calculate minimum distance to neighbors
                min_dist = float('inf')
                for j in range(N_CIRCLES):
                    if i != j:
                        dist = np.sqrt((x - points[j][0])**2 + (y - points[j][1])**2)
                        min_dist = min(min_dist, dist)
                
                # Set radius based on neighbor distances with some randomness
                if min_dist > 0:
                    r = min(0.15, min_dist/3.0 * (0.8 + random.random() * 0.4))
                else:
                    r = np.random.uniform(0.01, 0.05)
                
                # Clamp radius to reasonable bounds
                r = max(0.005, min(0.2, r))
                circles[i] = [x, y, r]
            
            population.append(circles)
        return population
    
    def is_valid_solution(circles):
        """Efficient constraint checking with early termination"""
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Fast overlap check using spatial data structures
        try:
            points = circles[:, :2]
            radii = circles[:, 2]
            
            # Create a hierarchical structure for faster neighbor querying
            # Use a simple grid-based approach for initial coarse filtering
            grid_size = 20
            grid = defaultdict(list)
            
            # Distribute points into grid cells
            for i, (x, y) in enumerate(points):
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                grid[(grid_x, grid_y)].append(i)
            
            # Check for potential overlaps in adjacent cells
            for (gx, gy), indices in grid.items():
                # Check against same cell and adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if (gx + dx, gy + dy) in grid:
                            for i in indices:
                                for j in grid[(gx + dx, gy + dy)]:
                                    if i < j:  # Avoid duplicate checks
                                        x1, y1, r1 = circles[i]
                                        x2, y2, r2 = circles[j]
                                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                        if dist < r1 + r2:
                                            return False
                                            
        except Exception:
            # Fallback to brute force if grid fails
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        
        return True
    
    def evaluate_fitness(individual):
        """Improved fitness evaluation with soft constraints"""
        # Check constraints
        if not is_valid_solution(individual):
            # Soft penalty based on how much constraints are violated
            penalty = 0
            for i in range(len(individual)):
                x, y, r = individual[i]
                # Boundary violations
                penalty += max(0, r - x)  # Left boundary
                penalty += max(0, r - (1 - x))  # Right boundary
                penalty += max(0, r - y)  # Bottom boundary
                penalty += max(0, r - (1 - y))  # Top boundary
                
                # Overlap violations
                for j in range(len(individual)):
                    if i != j:
                        x2, y2, r2 = individual[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        overlap = max(0, (r + r2) - dist)
                        penalty += overlap * 100
                        
            return (-np.sum(individual[:, 2]) - penalty,)
        
        # Valid solution - maximize sum of radii
        return (-np.sum(individual[:, 2]),)
    
    def mut_smart(individual, mu=0, sigma=0.01, indpb=0.2):
        """Smart mutation that preserves geometric relationships"""
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if random.random() < indpb:
                # Get current position and radius
                x, y, r = mutated_individual[i]
                
                # Mutate position with correlation to neighbor influence
                # If neighbors are close, allow tighter moves; otherwise, more freedom
                neighbor_distances = []
                for j in range(len(mutated_individual)):
                    if i != j:
                        x2, y2, r2 = mutated_individual[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        neighbor_distances.append(dist)
                
                # Base mutation strength on neighbor distances
                if neighbor_distances:
                    avg_dist = np.mean(neighbor_distances)
                    mutation_strength = sigma * (0.5 + 0.5 * min(1.0, avg_dist))
                else:
                    mutation_strength = sigma
                
                # Mutate position
                mutated_individual[i][0] += np.random.normal(mu, mutation_strength)
                mutated_individual[i][1] += np.random.normal(mu, mutation_strength)
                
                # Mutate radius with smaller changes
                mutated_individual[i][2] += np.random.normal(mu, sigma/3)
                
                # Apply boundary constraints
                mutated_individual[i][0] = np.clip(mutated_individual[i][0], 
                                                  mutated_individual[i][2], 
                                                  1 - mutated_individual[i][2])
                mutated_individual[i][1] = np.clip(mutated_individual[i][1], 
                                                  mutated_individual[i][2], 
                                                  1 - mutated_individual[i][2])
                mutated_individual[i][2] = np.clip(mutated_individual[i][2], 0.001, 0.5)
        
        # Repair to resolve any overlaps
        repair_individual(mutated_individual)
        return mutated_individual,
    
    def repair_individual(individual):
        """Advanced repairing algorithm with multiple refinement strategies"""
        # Phase 1: Fix containment issues
        for i in range(len(individual)):
            x, y, r = individual[i]
            # Ensure containment
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            individual[i][0] = x
            individual[i][1] = y
        
        # Phase 2: Resolve overlaps using multiple strategies
        max_iterations = 10
        for iteration in range(max_iterations):
            overlapped = False
            
            # Build neighbor lookup for faster processing
            points = [(ind[0], ind[1]) for ind in individual]
            tree = KDTree(points)
            pairs = tree.query_pairs(0.001, return_distance=False)
            
            # Process overlaps systematically
            processed_pairs = set()
            for i, j in pairs:
                if i > j:  # Ensure consistent ordering
                    i, j = j, i
                
                if (i, j) in processed_pairs:
                    continue
                    
                processed_pairs.add((i, j))
                
                x1, y1, r1 = individual[i]
                x2, y2, r2 = individual[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if dist < r1 + r2:
                    overlapped = True
                    # Resolve by moving both circles apart
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        # If same position, move in random direction
                        angle = np.random.uniform(0, 2*np.pi)
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                    
                    # Normalize direction vector
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx /= length
                        dy /= length
                    
                    # Separate with proportional movement based on radii
                    separation = (r1 + r2) - dist
                    move_ratio = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
                    move1 = separation * move_ratio * 0.5
                    move2 = separation * (1 - move_ratio) * 0.5
                    
                    individual[i][0] -= dx * move1
                    individual[i][1] -= dy * move1
                    individual[j][0] += dx * move2
                    individual[j][1] += dy * move2
                    
                    # Reapply bounds
                    individual[i][0] = np.clip(individual[i][0], individual[i][2], 1 - individual[i][2])
                    individual[i][1] = np.clip(individual[i][1], individual[i][2], 1 - individual[i][2])
                    individual[j][0] = np.clip(individual[j][0], individual[j][2], 1 - individual[j][2])
                    individual[j][1] = np.clip(individual[j][1], individual[j][2], 1 - individual[j][2])
            
            if not overlapped:
                break
    
    def cx_smart(individual1, individual2):
        """Smart crossover that maintains geometric properties"""
        # Use uniform crossover with careful attention to bounds
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
                     lambda: [np.random.uniform(0.01, 0.99), 
                              np.random.uniform(0.01, 0.99), 
                              np.random.uniform(0.01, 0.1)], 
                     n=N_CIRCLES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", cx_smart)
    toolbox.register("mutate", mut_smart)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population with multi-scale starting points
    pop = initialize_multi_scale_population(80)
    
    # Convert to DEAP individuals
    deap_pop = []
    for p in pop:
        ind = creator.Individual(p.tolist())
        ind.fitness.values = evaluate_fitness(p)
        deap_pop.append(ind)
    
    # Run evolution with optimized parameters
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evolution parameters - optimized for efficiency
    n_generations = 150
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Run evolution with adaptive parameters
    for gen in range(n_generations):
        # Adaptive mutation rate
        current_mutation_rate = max(0.01, mutation_rate * (1 - gen/n_generations)**1.2)
        
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
        
        # Print progress
        if gen % 20 == 0:
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
    
    # Enhanced local optimization using gradient-like approach
    try:
        final_circles = best_solution.copy()
        
        # Gradient-based refinement
        improved = True
        iterations = 0
        while improved and iterations < 30:
            improved = False
            iterations += 1
            
            # Try to increase each circle's radius independently
            for i in range(N_CIRCLES):
                orig_x, orig_y, orig_r = final_circles[i]
                
                # Calculate how much we can increase the radius
                min_dist_to_others = float('inf')
                for j in range(N_CIRCLES):
                    if i != j:
                        x2, y2, r2 = final_circles[j]
                        dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                        min_dist_to_others = min(min_dist_to_others, dist)
                
                # Maximum possible radius
                max_new_radius = min_dist_to_others - 0.001 if min_dist_to_others > 0.001 else orig_r
                
                if max_new_radius > orig_r + 0.001:  # Only if there's room to grow
                    # Simple greedy improvement
                    new_r = min(max_new_radius, orig_r + 0.005)
                    new_r = max(0.001, min(new_r, 0.5))
                    
                    # Test validity
                    temp_circles = final_circles.copy()
                    temp_circles[i][2] = new_r
                    
                    if is_valid_solution(temp_circles):
                        final_circles[i][2] = new_r
                        improved = True
            
            # Try small position tweaks
            for i in range(N_CIRCLES):
                orig_x, orig_y, orig_r = final_circles[i]
                
                # Try small adjustment to position
                best_x, best_y = orig_x, orig_y
                best_r = orig_r
                best_valid = True
                
                # Try several small moves
                for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                        test_x = max(orig_r, min(1 - orig_r, orig_x + dx))
                        test_y = max(orig_r, min(1 - orig_r, orig_y + dy))
                        
                        temp_circles = final_circles.copy()
                        temp_circles[i][0] = test_x
                        temp_circles[i][1] = test_y
                        
                        if is_valid_solution(temp_circles):
                            # Prefer position that allows larger radius
                            best_x, best_y = test_x, test_y
                            best_valid = True
                
                if best_valid:
                    final_circles[i][0] = best_x
                    final_circles[i][1] = best_y
                    improved = True
            
            # Repair if needed
            if not is_valid_solution(final_circles):
                repair_individual(final_circles)
        
        best_solution = final_circles
        
    except Exception as e:
        # If optimization fails, just return the best evolved solution
        pass
    
    return best_solution

# EVOLVE-BLOCK-END