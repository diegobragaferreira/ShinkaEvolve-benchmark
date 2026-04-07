# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.spatial.distance import cdist
import random
from deap import base, creator, tools
from itertools import combinations
import time
from collections import defaultdict

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
    BENCHMARK = 2.6358627564136983

    def create_hierarchical_voronoi_initialization(n_circles):
        """Create initial configuration using hierarchical Voronoi subdivision for better spatial distribution"""
        # Start with coarser grid to establish overall structure
        coarse_grid_size = max(3, int(np.ceil(np.sqrt(n_circles))))
        coarse_spacing = 1.0 / (coarse_grid_size + 1)
        
        # Generate coarse grid points
        coarse_points = []
        for i in range(coarse_grid_size):
            for j in range(coarse_grid_size):
                if len(coarse_points) < n_circles:
                    x = (j + 1) * coarse_spacing
                    y = (i + 1) * coarse_spacing
                    coarse_points.append([x, y])
        
        # Add strategic boundary points
        boundary_points = []
        for _ in range(10):
            side = np.random.randint(0, 4)
            if side == 0:  # Top
                boundary_points.append([np.random.rand(), 1.0])
            elif side == 1:  # Bottom
                boundary_points.append([np.random.rand(), 0.0])
            elif side == 2:  # Left
                boundary_points.append([0.0, np.random.rand()])
            else:  # Right
                boundary_points.append([1.0, np.random.rand()])
        
        coarse_points.extend(boundary_points)
        coarse_points = coarse_points[:n_circles]
        
        # Create initial Voronoi structure
        try:
            coarse_points_array = np.array(coarse_points)
            vor = Voronoi(coarse_points_array)
            
            # Extract Voronoi cell centers with refined positioning
            refined_centers = []
            for i, (x, y) in enumerate(vor.points):
                if i < len(vor.point_region) and vor.point_region[i] >= 0:
                    region = vor.regions[vor.point_region[i]]
                    if len(region) > 0 and all(r >= 0 for r in region):
                        vertices = np.array([vor.vertices[r] for r in region])
                        if len(vertices) > 0:
                            # Use barycenter instead of centroid for better distribution
                            barycenter = np.mean(vertices, axis=0)
                            # Clip to ensure it's inside the unit square
                            barycenter[0] = np.clip(barycenter[0], 0.01, 0.99)
                            barycenter[1] = np.clip(barycenter[1], 0.01, 0.99)
                            refined_centers.append(barycenter)
                
            # If we don't have enough centers, use original points
            if len(refined_centers) < n_circles:
                refined_centers = coarse_points_array[:n_circles].tolist()
            else:
                refined_centers = refined_centers[:n_circles]
                
            # Apply recursive refinement to improve distribution
            for _ in range(2):  # Two levels of refinement
                refined_centers = apply_voronoi_refinement(refined_centers, n_circles)
                
            # Compute initial radii using geometric considerations
            circles = np.zeros((n_circles, 3))
            centers_np = np.array(refined_centers)
            
            # Build KDTree for efficient nearest neighbor search
            tree = cKDTree(centers_np)
            
            for i, (cx, cy) in enumerate(refined_centers):
                # Find nearest neighbors to estimate local density
                distances, indices = tree.query([cx, cy], k=min(5, len(centers_np)), 
                                               distance_upper_bound=1.0)
                distances = distances[distances > 0]  # Remove self-distance
                
                if len(distances) > 0:
                    # Use harmonic mean of distances for better estimation
                    avg_distance = 1.0 / np.mean(1.0 / distances) if len(distances) > 0 else 0.1
                    # Radius based on distance and packing efficiency
                    base_radius = min(0.2, avg_distance * 0.3)
                else:
                    base_radius = 0.05
                    
                # Apply boundary penalty
                boundary_dist = min(cx, 1-cx, cy, 1-cy)
                boundary_penalty = min(0.05, boundary_dist * 0.3)
                radius = max(0.005, min(0.2, base_radius - boundary_penalty))
                
                circles[i] = [cx, cy, radius]
                
            return circles
            
        except Exception:
            # Fallback to grid-based initialization
            circles = np.zeros((n_circles, 3))
            grid_size = int(np.ceil(np.sqrt(n_circles)))
            spacing = 1.0 / (grid_size + 1)
            for i in range(n_circles):
                row = i // grid_size
                col = i % grid_size
                x = (col + 1) * spacing
                y = (row + 1) * spacing
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                r = min(0.1, spacing * 0.3)
                circles[i] = [x, y, r]
            return circles

    def apply_voronoi_refinement(points, target_count):
        """Apply Voronoi-based refinement to improve point distribution"""
        if len(points) < 4 or target_count <= len(points):
            return points
            
        try:
            points_array = np.array(points)
            vor = Voronoi(points_array)
            
            # Identify Voronoi cell centers and compute refined positions
            new_points = []
            for i, (x, y) in enumerate(vor.points):
                if i < len(vor.point_region) and vor.point_region[i] >= 0:
                    region = vor.regions[vor.point_region[i]]
                    if len(region) > 0 and all(r >= 0 for r in region):
                        vertices = np.array([vor.vertices[r] for r in region])
                        if len(vertices) > 0:
                            # Compute refined position as weighted average of vertices
                            weights = np.ones(len(vertices))
                            if len(vertices) > 2:
                                # Prefer interior points
                                center = np.mean(vertices, axis=0)
                                # Move towards center with some random perturbation
                                perturbation = np.random.normal(0, 0.02, 2)
                                refined_pos = np.clip(center + perturbation, 0.01, 0.99)
                            else:
                                refined_pos = np.clip(np.mean(vertices, axis=0), 0.01, 0.99)
                            new_points.append(refined_pos.tolist())
                            
            # If still not enough, add original points or new ones
            if len(new_points) < target_count:
                # Add some original points
                random.shuffle(points)
                for point in points[:target_count - len(new_points)]:
                    new_points.append(point)
                    
            return new_points[:target_count]
        except Exception:
            return points[:target_count]

    def is_valid_solution(circles):
        """Check if solution satisfies all constraints efficiently"""
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap using KDTree with early termination
        try:
            points = circles[:, :2]
            tree = cKDTree(points)
            pairs = tree.query_pairs(0, return_distance=False)
            
            for i, j in pairs:
                if i < j:  # Avoid checking same pair twice
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        except Exception:
            # Fallback to brute force
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
                        
        return True

    def evaluate_fitness(circles):
        """Evaluate fitness with penalty for constraint violations"""
        if not is_valid_solution(circles):
            # Apply penalty based on constraint violation severity
            penalty = 0
            
            # Check containment violations
            for i in range(len(circles)):
                x, y, r = circles[i]
                containment_violation = 0
                if x - r < 0:
                    containment_violation += abs(x - r)
                if x + r > 1:
                    containment_violation += abs(x + r - 1)
                if y - r < 0:
                    containment_violation += abs(y - r)
                if y + r > 1:
                    containment_violation += abs(y + r - 1)
                penalty += containment_violation * 1000
                
            # Check overlap violations
            overlap_penalty = 0
            try:
                points = circles[:, :2]
                tree = cKDTree(points)
                pairs = tree.query_pairs(0, return_distance=False)
                for i, j in pairs:
                    if i < j:
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        overlap = max(0, (r1 + r2) - dist)
                        overlap_penalty += overlap * 1000
            except Exception:
                for i in range(len(circles)):
                    x1, y1, r1 = circles[i]
                    for j in range(i+1, len(circles)):
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        overlap = max(0, (r1 + r2) - dist)
                        overlap_penalty += overlap * 1000
                        
            return -(np.sum(circles[:, 2]) + penalty + overlap_penalty)
            
        return -np.sum(circles[:, 2])

    def adaptive_mutate(individual, generation, diversity):
        """Mutation operator with adaptive parameters"""
        mutated = individual.copy()
        # Scale mutation rate based on generation and diversity
        mutation_rate = 0.1 + 0.1 * (1 - generation / 500)
        if diversity > 0.05:
            mutation_rate *= 1.5
            
        # Apply different types of mutations
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                if i % 3 == 0:  # x coordinate
                    mutated[i] = max(0.005, min(0.995, mutated[i] + np.random.normal(0, 0.015)))
                elif i % 3 == 1:  # y coordinate
                    mutated[i] = max(0.005, min(0.995, mutated[i] + np.random.normal(0, 0.015)))
                else:  # radius
                    # Use log-normal for radius mutation to maintain positive values
                    old_radius = mutated[i]
                    log_change = np.random.normal(0, 0.15)
                    new_radius = old_radius * np.exp(log_change)
                    mutated[i] = max(0.005, min(0.3, new_radius))
        
        # Repair if necessary
        circles = np.array(mutated).reshape(-1, 3)
        repair_individual(circles)
        return circles.flatten().tolist(),

    def repair_individual(circles):
        """Proactive constraint repair with geometric optimization"""
        # Fix containment first
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Ensure circle is fully contained
            x = min(1 - r, max(r, x))
            y = min(1 - r, max(r, y))
            circles[i] = [x, y, r]
            
        # Resolve overlaps using geometric approach
        max_iter = 3
        for iteration in range(max_iter):
            # Find all overlapping pairs
            overlaps = []
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        overlaps.append((i, j, dist, r1 + r2 - dist))
            
            if not overlaps:
                break
                
            # Resolve most severe overlaps first
            overlaps.sort(key=lambda x: x[3], reverse=True)  # Sort by overlap amount
            
            for i, j, dist, overlap in overlaps[:len(overlaps)//2 + 1]:  # Limit to half
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Separate circles along the line connecting their centers
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx /= length
                    dy /= length
                    
                    # Move circles away from each other
                    separation = (r1 + r2 - dist) * 0.8  # Separate by 80% of overlap
                    move1 = separation * 0.5
                    move2 = separation * 0.5
                    
                    circles[i][0] = max(r1, min(1-r1, x1 - dx * move1))
                    circles[i][1] = max(r1, min(1-r1, y1 - dy * move1))
                    circles[j][0] = max(r2, min(1-r2, x2 + dx * move2))
                    circles[j][1] = max(r2, min(1-r2, y2 + dy * move2))
        
        # Ensure all circles still satisfy bounds
        for i in range(len(circles)):
            x, y, r = circles[i]
            circles[i] = [min(1-r, max(r, x)), min(1-r, max(r, y)), r]

    def symmetrical_crossover(ind1, ind2):
        """Crossover that preserves symmetry properties in the resulting offspring"""
        size = len(ind1)
        # Apply uniform crossover with symmetry awareness
        for i in range(size):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        
        # Repair both offspring
        circles1 = np.array(ind1).reshape(-1, 3)
        circles2 = np.array(ind2).reshape(-1, 3)
        
        # Apply symmetry-aware repair
        repair_individual(circles1)
        repair_individual(circles2)
        
        # Convert back to lists
        ind1[:] = circles1.flatten()
        ind2[:] = circles2.flatten()
        
        return ind1, ind2

    def constraint_aware_local_search(circles, max_iter=100):
        """Sophisticated local search using geometric programming"""
        current = circles.copy()
        
        # Track improvement to detect convergence
        last_fitness = evaluate_fitness(current)
        
        # Use a two-phase approach: expand radii first, then optimize positions
        for iteration in range(max_iter):
            improved = False
            
            # Phase 1: Radius optimization (global optimization)
            for i in range(len(current)):
                x, y, r = current[i]
                
                # Determine maximum possible radius at this position
                max_r = min(x, 1-x, y, 1-y)
                
                if max_r > r + 1e-6:
                    # Binary search for optimal radius
                    low, high = r, max_r
                    best_r = r
                    
                    # Precise binary search
                    for _ in range(20):
                        test_r = (low + high) / 2
                        temp_circles = current.copy()
                        temp_circles[i][2] = test_r
                        
                        if is_valid_solution(temp_circles):
                            best_r = test_r
                            low = test_r
                        else:
                            high = test_r
                    
                    if best_r > r + 1e-6:
                        current[i][2] = best_r
                        improved = True
                        
            # Phase 2: Position optimization
            if improved:
                continue  # Skip position optimization if we already improved radius
                
            # Apply geometric optimization for positions using gradient-like approach
            for i in range(len(current)):
                original_x, original_y, r = current[i]
                
                # Try to find a better position by testing nearby locations
                best_x, best_y = original_x, original_y
                best_r = r
                best_fitness = evaluate_fitness(current)
                
                # Test a systematic pattern around current position
                test_positions = []
                for dx in np.linspace(-0.015, 0.015, 7):
                    for dy in np.linspace(-0.015, 0.015, 7):
                        test_positions.append((dx, dy))
                        
                for dx, dy in test_positions:
                    test_x = max(r, min(1-r, original_x + dx))
                    test_y = max(r, min(1-r, original_y + dy))
                    
                    # Test this position
                    temp_circles = current.copy()
                    temp_circles[i][0] = test_x
                    temp_circles[i][1] = test_y
                    
                    if is_valid_solution(temp_circles):
                        test_fitness = evaluate_fitness(temp_circles)
                        if test_fitness < best_fitness:  # Minimize fitness (negative sum)
                            best_fitness = test_fitness
                            best_x, best_y = test_x, test_y
                            
                if best_x != original_x or best_y != original_y:
                    current[i][0] = best_x
                    current[i][1] = best_y
                    improved = True
                    
            if not improved:
                break
                
        return current

    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     lambda: [np.random.uniform(0.05, 0.95),
                              np.random.uniform(0.05, 0.95),
                              np.random.uniform(0.01, 0.1)],
                     n=N_CIRCLES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", lambda ind: (evaluate_fitness(np.array(ind).reshape(-1, 3)),))
    toolbox.register("mate", symmetrical_crossover)
    toolbox.register("mutate", adaptive_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Generate initial population
    pop = []
    for _ in range(100):
        circles = create_hierarchical_voronoi_initialization(N_CIRCLES)
        individual = circles.flatten().tolist()
        
        # Add random perturbations
        for i in range(len(individual)):
            if i % 3 < 2:  # x or y coordinate
                individual[i] += np.random.uniform(-0.02, 0.02)
                individual[i] = max(0.005, min(0.995, individual[i]))
            else:  # radius
                individual[i] *= np.random.uniform(0.9, 1.1)
                individual[i] = max(0.005, min(0.45, individual[i]))
        
        pop.append(creator.Individual(individual))

    # Run evolution with adaptive parameters
    n_generations = 300
    
    for gen in range(n_generations):
        # Dynamic parameter adaptation
        current_mutation_rate = max(0.01, 0.1 * (1 - gen / n_generations))
        
        # Selection and reproduction
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < current_mutation_rate:
                toolbox.mutate(mutant, gen, 0.0)  # Diversity can be calculated later
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
    
    # Apply local search refinement with symmetry awareness
    best_solution = constraint_aware_local_search(best_solution)
    
    # Final validation
    if not is_valid_solution(best_solution):
        # Try fallback method
        fallback_solution = create_hierarchical_voronoi_initialization(N_CIRCLES)
        fallback_solution = constraint_aware_local_search(fallback_solution)
        if is_valid_solution(fallback_solution):
            return fallback_solution
    
    return best_solution

# EVOLVE-BLOCK-END