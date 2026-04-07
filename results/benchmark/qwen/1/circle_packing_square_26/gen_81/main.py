# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

# Fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid gradient-guided evolutionary approach for improved optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    N_CIRCLES = 26
    POP_SIZE = 100
    GEN_COUNT = 50
    MUTPB = 0.3
    CXPB = 0.5
    
    class CirclePackingProblem:
        def __init__(self, n_circles=N_CIRCLES):
            self.N_CIRCLES = n_circles
            
        def evaluate_fitness(self, individual):
            """Evaluate fitness with penalty for constraint violations."""
            circles = np.array(individual).reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            total_radius = np.sum(radii)
            penalty = 0
            
            # Check containment constraints
            for i, (pos, r) in enumerate(zip(positions, radii)):
                x, y = pos
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    penalty += 10000
                    
            # Check overlap constraints efficiently using cKDTree
            try:
                tree = cKDTree(positions)
                pairs = tree.query_pairs(radii.sum() + 0.001, p=2)
                for i, j in pairs:
                    r_i = radii[i]
                    r_j = radii[j]
                    pos_i = positions[i]
                    pos_j = positions[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        penalty += 1000 * (r_i + r_j - dist)
            except:
                # Fallback to brute force
                for i in range(len(circles)):
                    for j in range(i+1, len(circles)):
                        pos_i = circles[i, :2]
                        pos_j = circles[j, :2]
                        r_i = circles[i, 2]
                        r_j = circles[j, 2]
                        dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                        if dist < (r_i + r_j):
                            penalty += 1000 * (r_i + r_j - dist)
            
            return (total_radius - penalty,)
            
        def initialize_population(self, pop_size):
            """Initialize population using adaptive Voronoi-like approach."""
            population = []
            for _ in range(pop_size):
                individual = self._generate_initial_individual()
                population.append(individual)
            return population
            
        def _generate_initial_individual(self):
            """Generate a single initial individual with adaptive Voronoi-based seeding."""
            circles = np.zeros((self.N_CIRCLES, 3))
            
            # Create hexagonal grid with adaptive distribution
            sqrt_n = int(math.ceil(math.sqrt(self.N_CIRCLES)))
            rows = int(math.ceil(self.N_CIRCLES / sqrt_n))
            cols = int(math.ceil(self.N_CIRCLES / rows))
            
            spacing_x = 0.9 / (cols + 1)
            spacing_y = 0.9 / (rows + 1)
            
            points = []
            for i in range(rows):
                for j in range(cols):
                    if len(points) >= self.N_CIRCLES:
                        break
                    x_offset = 0 if i % 2 == 0 else spacing_x / 2
                    x = (j + 1) * spacing_x + x_offset
                    y = (i + 1) * spacing_y
                    points.append([x, y])
                    
            # Fill remaining positions randomly but ensure good distribution
            while len(points) < self.N_CIRCLES:
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                points.append([x, y])
                
            points = points[:self.N_CIRCLES]
            
            # Assign initial radii based on neighbor distances and boundary constraints
            for i, (x, y) in enumerate(points):
                # Calculate minimum distance to neighbors
                min_dist = float('inf')
                for j, (other_x, other_y) in enumerate(points):
                    if i != j:
                        dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                        min_dist = min(min_dist, dist)
                
                # Calculate initial radius based on neighbor density and boundaries
                if min_dist < float('inf') and min_dist > 0:
                    base_radius = min(0.15, min_dist * 0.15)
                else:
                    base_radius = 0.05
                    
                # Respect boundary constraints
                boundary_radius = min(x, 1-x, y, 1-y)
                initial_r = min(base_radius, boundary_radius * 0.8)
                initial_r = max(0.001, min(0.15, initial_r))
                
                circles[i] = [x, y, initial_r]
                
            return circles.flatten().tolist()
            
        def mutate_individual(self, individual):
            """Mutate individual with adaptive parameters."""
            individual_array = np.array(individual).reshape(-1, 3)
            radii = individual_array[:, 2]
            
            # Calculate diversity measure
            diversity = np.std(radii) / (np.mean(radii) + 1e-8) if np.mean(radii) > 1e-8 else 0
            
            # Adaptive mutation rate
            adaptive_mutation_rate = MUTPB * (1 - min(0.8, diversity))
            
            for i in range(len(individual)):
                if random.random() < adaptive_mutation_rate:
                    idx = i % 3
                    if idx == 2:  # radius
                        old_r = individual[i]
                        mutation_strength = 0.015 * (1 + diversity)
                        new_r = old_r + random.gauss(0, mutation_strength)
                        individual[i] = max(0.001, min(0.5, new_r))
                    else:  # position
                        old_val = individual[i]
                        mutation_strength = 0.02 * (1 + diversity)
                        new_val = old_val + random.gauss(0, mutation_strength)
                        individual[i] = max(0, min(1, new_val))
            return individual
            
        def crossover_individuals(self, ind1, ind2):
            """Perform crossover on two individuals."""
            # Uniform crossover with constraint awareness
            child1 = ind1.copy()
            child2 = ind2.copy()
            
            for i in range(len(child1)):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
                    
            # Repair if necessary
            child1 = self._repair_individual(child1)
            child2 = self._repair_individual(child2)
            
            return child1, child2
            
        def _repair_individual(self, individual):
            """Repair individual to satisfy constraints."""
            circles = np.array(individual).reshape(-1, 3)
            
            # Fix containment first
            for i in range(len(circles)):
                x, y, r = circles[i]
                if x - r < 0:
                    x = r
                elif x + r > 1:
                    x = 1 - r
                if y - r < 0:
                    y = r
                elif y + r > 1:
                    y = 1 - r
                circles[i] = [x, y, r]
                
            # Fix overlaps with simple geometric adjustment
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    pos_i = circles[i, :2]
                    pos_j = circles[j, :2]
                    r_i = circles[i, 2]
                    r_j = circles[j, 2]
                    dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                    if dist < (r_i + r_j):
                        # Move circles apart along the connecting line
                        if dist > 1e-8:
                            dx, dy = pos_i - pos_j
                            dx /= dist
                            dy /= dist
                            # Calculate shift amount
                            shift = (r_i + r_j - dist) * 0.5
                            circles[i, 0] += dx * shift
                            circles[i, 1] += dy * shift
                            circles[j, 0] -= dx * shift
                            circles[j, 1] -= dy * shift
                            
            # Ensure containment after adjustment
            for i in range(len(circles)):
                x, y, r = circles[i]
                x = np.clip(x, r, 1-r)
                y = np.clip(y, r, 1-r)
                circles[i] = [x, y, r]
                
            return circles.flatten().tolist()
            
        def gradient_refinement(self, individual, max_iterations=50):
            """
            Apply gradient-based refinement to improve individual.
            Uses a penalty method approach to handle constraints.
            """
            circles = np.array(individual).reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Define the objective and constraints for scipy optimization
            def objective(params):
                # Reshape parameters back to circles format
                new_circles = params.reshape(-1, 3)
                new_positions = new_circles[:, :2]
                new_radii = new_circles[:, 2]
                
                # Calculate radius sum (negative because we're minimizing)
                obj_value = -np.sum(new_radii)
                
                # Calculate penalty for constraint violations
                penalty = 0
                
                # Containment penalties
                for i, (pos, r) in enumerate(zip(new_positions, new_radii)):
                    x, y = pos
                    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                        penalty += 10000
                
                # Overlap penalties using cKDTree
                try:
                    tree = cKDTree(new_positions)
                    pairs = tree.query_pairs(new_radii.sum() + 0.001, p=2)
                    for i, j in pairs:
                        r_i = new_radii[i]
                        r_j = new_radii[j]
                        pos_i = new_positions[i]
                        pos_j = new_positions[j]
                        dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                        if dist < (r_i + r_j):
                            penalty += 1000 * (r_i + r_j - dist)
                except:
                    for i in range(len(new_circles)):
                        for j in range(i+1, len(new_circles)):
                            pos_i = new_circles[i, :2]
                            pos_j = new_circles[j, :2]
                            r_i = new_circles[i, 2]
                            r_j = new_circles[j, 2]
                            dist = np.sqrt(np.sum((pos_i - pos_j)**2))
                            if dist < (r_i + r_j):
                                penalty += 1000 * (r_i + r_j - dist)
                
                return obj_value + penalty
            
            # Initial values
            x0 = circles.flatten()
            
            # Optimize using scipy minimize with L-BFGS-B method
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                options={'maxiter': max_iterations, 'ftol': 1e-6})
                if result.success:
                    refined_circles = result.x.reshape(-1, 3)
                    # Clip positions to ensure they stay within bounds
                    for i in range(len(refined_circles)):
                        refined_circles[i, 0] = np.clip(refined_circles[i, 0], 
                                                      refined_circles[i, 2], 
                                                      1 - refined_circles[i, 2])
                        refined_circles[i, 1] = np.clip(refined_circles[i, 1], 
                                                      refined_circles[i, 2], 
                                                      1 - refined_circles[i, 2])
                    return refined_circles.flatten().tolist()
            except:
                pass
            
            return individual
            
    # Initialize problem and population
    problem = CirclePackingProblem()
    population = problem.initialize_population(POP_SIZE)
    
    # Main evolutionary loop
    best_fitness = float('-inf')
    best_individual = None
    
    for generation in range(GEN_COUNT):
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness = problem.evaluate_fitness(individual)[0]
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()
                
        # Select parents using tournament selection
        selected_parents = []
        tournament_size = 3
        
        for _ in range(POP_SIZE // 2):
            # Tournament selection
            tournament_indices = random.sample(range(POP_SIZE), tournament_size)
            tournament_scores = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            selected_parents.append(population[winner_idx])
            
        # Create new population through crossover and mutation
        new_population = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[min(i+1, len(selected_parents)-1)]
            
            # Crossover
            if random.random() < CXPB:
                child1, child2 = problem.crossover_individuals(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
                
            # Mutation
            child1 = problem.mutate_individual(child1)
            child2 = problem.mutate_individual(child2)
            
            # Add to new population
            new_population.extend([child1, child2])
            
        # Trim to exact population size
        population = new_population[:POP_SIZE]
        
        # Periodically refine with gradient-based optimization
        if generation % 5 == 0 and generation > 0:
            for i in range(len(population)):
                population[i] = problem.gradient_refinement(population[i])
    
    # Final refinement of best individual
    if best_individual is not None:
        final_best = problem.gradient_refinement(best_individual)
        result = np.array(final_best).reshape(-1, 3)
    else:
        # Fallback to heuristic if something went wrong
        result = problem.initialize_population(1)[0]
        result = np.array(result).reshape(-1, 3)
        
    return result

# EVOLVE-BLOCK-END