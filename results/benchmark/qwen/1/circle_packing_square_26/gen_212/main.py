# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools
import time
import math

# Set fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 26
    
    def generate_voronoi_initialization(n_circles):
        """Generate initial configuration using enhanced Voronoi-based approach."""
        # Create hexagonal grid pattern
        sqrt_n = int(math.ceil(math.sqrt(n_circles)))
        rows = int(math.ceil(n_circles / sqrt_n))
        cols = int(math.ceil(n_circles / rows))
        
        points = []
        spacing_x = 0.95 / (cols + 1)
        spacing_y = 0.95 / (rows + 1)
        
        # Generate points in hexagonal pattern with random jittering
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_circles:
                    break
                offset = (j % 2) * spacing_x / 2
                x = (j + 1) * spacing_x + offset + 0.025 + random.uniform(-spacing_x/8, spacing_x/8)
                y = (i + 1) * spacing_y + 0.025 + random.uniform(-spacing_y/8, spacing_y/8)
                points.append([x, y])
        
        # Fill remaining spots with random points that avoid close proximity
        while len(points) < n_circles:
            x = random.uniform(0.025, 0.975)
            y = random.uniform(0.025, 0.975)
            # Ensure minimum distance to existing points
            min_dist = float('inf')
            for px, py in points:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                min_dist = min(min_dist, dist)
            if min_dist > 0.05:
                points.append([x, y])
        
        points = points[:n_circles]
        
        # Create Voronoi-inspired radii based on neighbor distances
        circles = np.zeros((n_circles, 3))
        for i, (x, y) in enumerate(points):
            # Calculate minimum distance to neighbors
            min_dist = float('inf')
            for j, (other_x, other_y) in enumerate(points):
                if i != j:
                    dist = math.sqrt((x - other_x)**2 + (y - other_y)**2)
                    min_dist = min(min_dist, dist)
            
            # Set radius with boundary constraint consideration
            if min_dist < float('inf') and min_dist > 0:
                base_radius = min(0.15, min_dist * 0.3)
            else:
                base_radius = 0.05
            
            # Boundary constraint
            boundary_radius = min(x, 1-x, y, 1-y)
            initial_r = min(base_radius, boundary_radius * 0.8)
            
            # Ensure reasonable minimum radius
            initial_r = max(0.005, min(0.15, initial_r))
            
            circles[i] = [x, y, initial_r]
        
        return circles
    
    def is_valid_solution(circles):
        """Check if solution satisfies all constraints."""
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check non-overlap using KDTree for efficiency
        try:
            points = circles[:, :2]
            tree = KDTree(points)
            pairs = tree.query_pairs(0, return_distance=False)
            for i, j in pairs:
                if i < j:  # Avoid checking same pair twice
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        except Exception:
            # Fallback to brute force if KDTree fails
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        
        return True
    
    def evaluate_fitness(circles):
        """Evaluate fitness as sum of radii (maximization problem)."""
        if not is_valid_solution(circles):
            return -float('inf')
        return float(np.sum(circles[:, 2]))
    
    def local_search_refinement(circles):
        """Apply local search refinement to improve the configuration."""
        refined_circles = circles.copy()
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Phase 1: Maximize individual radii
            for i in range(N_CIRCLES):
                x, y, r = refined_circles[i]
                
                # Find minimum distance to other circles
                min_dist = float('inf')
                for j in range(N_CIRCLES):
                    if i != j:
                        x2, y2, r2 = refined_circles[j]
                        dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                        min_dist = min(min_dist, dist)
                
                # Calculate maximum possible radius
                max_radius = min(x, 1-x, y, 1-y)
                if min_dist < float('inf') and min_dist > 0:
                    max_radius = min(max_radius, min_dist - 0.001)
                
                if max_radius > r:
                    # Binary search for optimal radius
                    low, high = r, max_radius
                    best_radius = r
                    
                    for _ in range(15):  # More precise binary search
                        test_r = (low + high) / 2
                        # Check validity with this radius
                        temp_circles = refined_circles.copy()
                        temp_circles[i, 2] = test_r
                        
                        if is_valid_solution(temp_circles):
                            best_radius = test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_radius > r:
                        refined_circles[i, 2] = best_radius
                        improved = True
            
            # Phase 2: Position refinement
            for i in range(N_CIRCLES):
                x, y, r = refined_circles[i]
                
                # Spiral search pattern for position improvements
                best_x, best_y = x, y
                best_radius = r
                best_score = r
                
                # Use spiral pattern for sampling positions
                for k in range(1, 20):  # Spiral steps
                    angle = k * 0.2 * math.pi
                    radius = k * 0.005
                    dx = radius * math.cos(angle)
                    dy = radius * math.sin(angle)
                    
                    test_x = max(r, min(1-r, x + dx))
                    test_y = max(r, min(1-r, y + dy))
                    
                    # Check validity
                    temp_circles = refined_circles.copy()
                    temp_circles[i, 0] = test_x
                    temp_circles[i, 1] = test_y
                    
                    if is_valid_solution(temp_circles):
                        # Evaluate improvement (focus on radius preservation)
                        score = temp_circles[i, 2]  # We prioritize keeping radius high
                        if score > best_score:
                            best_score = score
                            best_x, best_y = test_x, test_y
                
                if best_x != x or best_y != y:
                    refined_circles[i, 0] = best_x
                    refined_circles[i, 1] = best_y
                    improved = True
        
        return refined_circles
    
    def constraint_aware_mutation(individual, generation=0):
        """Mutation operator that respects constraints."""
        mutated = individual.copy()
        # Apply mutations to circle parameters
        for i in range(len(mutated)):
            if random.random() < 0.15:  # Mutation probability
                if i % 3 == 0:  # x-coordinate
                    mutated[i] = max(0.005, min(0.995, mutated[i] + random.gauss(0, 0.01)))
                elif i % 3 == 1:  # y-coordinate
                    mutated[i] = max(0.005, min(0.995, mutated[i] + random.gauss(0, 0.01)))
                else:  # radius
                    mutated[i] = max(0.005, min(0.45, mutated[i] * random.uniform(0.9, 1.1)))
        
        # Repair if necessary
        circles = np.array(mutated).reshape(-1, 3)
        if not is_valid_solution(circles):
            # Simple repair: reset to valid nearby values
            for j in range(len(circles)):
                x, y, r = circles[j]
                circles[j, 0] = max(r, min(1-r, x))
                circles[j, 1] = max(r, min(1-r, y))
        
        return circles.flatten().tolist(),
    
    def constraint_aware_crossover(ind1, ind2):
        """Crossover that maintains feasibility."""
        size = len(ind1)
        # Uniform crossover
        for i in range(size):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        
        # Repair offspring
        circles1 = np.array(ind1).reshape(-1, 3)
        circles2 = np.array(ind2).reshape(-1, 3)
        
        if not is_valid_solution(circles1):
            # Simple repair
            for j in range(len(circles1)):
                x, y, r = circles1[j]
                circles1[j, 0] = max(r, min(1-r, x))
                circles1[j, 1] = max(r, min(1-r, y))
        
        if not is_valid_solution(circles2):
            # Simple repair
            for j in range(len(circles2)):
                x, y, r = circles2[j]
                circles2[j, 0] = max(r, min(1-r, x))
                circles2[j, 1] = max(r, min(1-r, y))
        
        ind1[:] = circles1.flatten()
        ind2[:] = circles2.flatten()
        
        return ind1, ind2

    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: [random.uniform(0.05, 0.95),
                              random.uniform(0.05, 0.95),
                              random.uniform(0.01, 0.1)],
                     n=N_CIRCLES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", lambda ind: (evaluate_fitness(np.array(ind).reshape(-1, 3)),))
    toolbox.register("mate", constraint_aware_crossover)
    toolbox.register("mutate", constraint_aware_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Generate initial population with Voronoi seeds
    pop = []
    for _ in range(100):  # Larger population for better exploration
        circles = generate_voronoi_initialization(N_CIRCLES)
        individual = circles.flatten().tolist()
        # Add some random perturbation
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += random.uniform(-0.02, 0.02)
                individual[i] = max(0.005, min(0.995, individual[i]))
            else:  # radius
                individual[i] *= random.uniform(0.9, 1.1)
                individual[i] = max(0.005, min(0.45, individual[i]))
        pop.append(creator.Individual(individual))
    
    # Run evolution with adaptive parameters
    n_generations = 200
    
    for gen in range(n_generations):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.15:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate fitness
        invalid_ind = [ind for ind in offspring if not hasattr(ind.fitness, 'values') or len(ind.fitness.values) == 0]
        for ind in invalid_ind:
            ind.fitness.values = (evaluate_fitness(np.array(ind).reshape(-1, 3)),)
        
        # Replace population
        pop[:] = offspring
    
    # Find best solution
    best_individual = tools.selBest(pop, 1)[0]
    best_solution = np.array(best_individual).reshape(-1, 3)
    
    # Apply final local search refinement
    best_solution = local_search_refinement(best_solution)
    
    # Final validation
    if not is_valid_solution(best_solution):
        # If invalid, try another approach
        fallback_solution = generate_voronoi_initialization(N_CIRCLES)
        fallback_solution = local_search_refinement(fallback_solution)
        if is_valid_solution(fallback_solution):
            return fallback_solution
    
    return best_solution

# EVOLVE-BLOCK-END