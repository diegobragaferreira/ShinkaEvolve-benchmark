# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
import time
import random
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist

    def compute_voronoi_uniformity(points):
        """Compute uniformity based on Voronoi cell areas."""
        try:
            vor = Voronoi(points)
            areas = []
            for region in vor.regions:
                if not any(v == -1 for v in region):  # Skip infinite regions
                    polygon = [vor.vertices[i] for i in region]
                    if len(polygon) >= 3:
                        area = 0.5 * abs(sum(polygon[i][0] * polygon[(i+1)%len(polygon)][1] -
                                           polygon[(i+1)%len(polygon)][0] * polygon[i][1]
                                           for i in range(len(polygon))))
                        areas.append(area)
            if not areas:
                return 0.0
            avg_area = np.mean(areas)
            # Uniformity is how close average area is to ideal area (1/16)
            ideal_area = 1.0 / len(points)
            uniformity = 1.0 - abs(avg_area - ideal_area) / ideal_area if ideal_area > 0 else 0.0
            return uniformity
        except:
            return 0.0

    def compute_combined_fitness(points):
        """Compute combined fitness using both distance ratio and Voronoi uniformity."""
        ratio = compute_min_max_ratio(points)
        uniformity = compute_voronoi_uniformity(points)
        return ratio * (1.0 + 0.5 * uniformity)  # Weighted combination

    def generate_initial_voronoi_config():
        """Generate initial configuration based on Voronoi uniformity principles."""
        # Start with a hexagonal grid pattern
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                # Add deterministic perturbation to break symmetry
                perturbation = np.sin(i * 1.7 + j * 0.3) * 0.005
                x += perturbation * np.sin(i * 0.5)
                y += perturbation * np.cos(j * 0.5)
                points.append([x, y])
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points

    def generate_spiral_config():
        """Generate points using Fibonacci spiral pattern."""
        points = np.zeros((16, 2))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(16):
            angle = i * golden_ratio * np.pi * 2
            radius = 0.4 * np.sqrt(i / 15.0)  # Scale appropriately
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points[i] = [x, y]
        points = np.clip(points, 0, 1)
        return points

    def generate_random_config():
        """Generate random configuration."""
        return np.random.rand(16, 2)

    def mutate_individual(points, mutation_rate=0.2, strength=0.01):
        """Apply mutations to points."""
        mutated = points.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] += np.random.normal(0, strength, 2)
        return np.clip(mutated, 0, 1)

    def crossover_parents(parent1, parent2, crossover_rate=0.8):
        """Create offspring from two parents."""
        if random.random() < crossover_rate:
            # Uniform crossover
            mask = np.random.rand(*parent1.shape) > 0.5
            child = np.where(mask, parent1, parent2).copy()
        else:
            # Clone one parent
            child = parent1.copy()
        return child

    def evolutionary_search(max_generations=50, population_size=30):
        """Perform evolutionary search."""
        # Initialize population
        population = []
        population.append(generate_initial_voronoi_config())
        population.append(generate_spiral_config())
        population.append(generate_random_config())
        
        # Generate rest randomly
        for _ in range(population_size - 3):
            population.append(generate_random_config())
        
        best_individual = None
        best_fitness = -np.inf
        
        for generation in range(max_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = compute_combined_fitness(individual)
                fitness_scores.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            
            # Create new population
            new_population = population[:5]  # Keep top 5 (elitism)
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 5
                tournament_indices = np.random.choice(len(population), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                tournament_indices2 = np.random.choice(len(population), tournament_size)
                tournament_fitness2 = [fitness_scores[i] for i in tournament_indices2]
                parent2_idx = tournament_indices2[np.argmax(tournament_fitness2)]
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                child = crossover_parents(parent1, parent2, crossover_rate=0.8)
                child = mutate_individual(child, mutation_rate=0.2, strength=0.01)
                new_population.append(child)
            
            population = new_population[:population_size]
        
        return best_individual if best_individual is not None else generate_initial_voronoi_config()

    def neighborhood_move(points, neighbor_size=3):
        """Move a neighbor sub-set of points together."""
        new_points = points.copy()
        # Select random subset of points to move
        indices = np.random.choice(len(points), min(neighbor_size, len(points)), replace=False)
        
        # Calculate centroid of selected points
        centroid = np.mean(new_points[indices], axis=0)
        
        # Apply movement to centroid and propagate
        move_vector = np.random.normal(0, 0.01, 2)
        new_centroid = np.clip(centroid + move_vector, 0, 1)
        delta = new_centroid - centroid
        
        for idx in indices:
            new_points[idx] += delta
            
        return np.clip(new_points, 0, 1)

    def simulated_annealing_refinement(initial_points, max_iter=1000):
        """Refine solution using simulated annealing with neighborhood moves."""
        current_points = initial_points.copy()
        current_fitness = compute_combined_fitness(current_points)
        best_points = current_points.copy()
        best_fitness = current_fitness
        
        # Initial temperature and cooling schedule
        temp = 0.1
        cooling_rate = 0.999
        min_temp = 1e-6
        
        for iteration in range(max_iter):
            # Generate neighbor
            new_points = neighborhood_move(current_points, neighbor_size=2)
            new_fitness = compute_combined_fitness(new_points)
            
            # Accept or reject
            if new_fitness > current_fitness or random.random() < np.exp((new_fitness - current_fitness) / temp):
                current_points = new_points
                current_fitness = new_fitness
                
                if new_fitness > best_fitness:
                    best_points = new_points
                    best_fitness = new_fitness
            
            # Cool down
            temp *= cooling_rate
            if temp < min_temp:
                break
                
        return best_points

    def smart_gradient_refinement(points, max_iter=200):
        """Apply gradient-based refinement."""
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            return -compute_min_max_ratio(points)  # Minimize negative ratio
        
        def constraint_function(x_flat):
            points = x_flat.reshape(-1, 2)
            violations = np.concatenate([
                np.minimum(points[:, 0], 0),
                np.minimum(points[:, 1], 0),
                np.maximum(points[:, 0] - 1, 0),
                np.maximum(points[:, 1] - 1, 0)
            ])
            return violations
        
        # Flatten for optimization
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(len(x0))]
        constraints = {'type': 'ineq', 'fun': constraint_function}
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                return np.clip(refined_points, 0, 1)
        except:
            pass
            
        return points

    # Main optimization process
    np.random.seed(42)
    
    # Multi-start approach with different initialization strategies
    start_configs = [
        generate_initial_voronoi_config(),
        generate_spiral_config(),
        generate_random_config()
    ]
    
    best_solution = None
    best_ratio = -np.inf
    
    # Try different starting points
    for i, initial_config in enumerate(start_configs):
        # Stage 1: Evolutionary search
        evolved = evolutionary_search(max_generations=30, population_size=20)
        
        # Stage 2: Simulated Annealing refinement
        sa_refined = simulated_annealing_refinement(evolved, max_iter=500)
        
        # Stage 3: Gradient-based refinement
        final_points = smart_gradient_refinement(sa_refined, max_iter=100)
        
        # Evaluate final result
        final_ratio = compute_min_max_ratio(final_points)
        
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_solution = final_points.copy()
    
    # Fallback if nothing worked
    if best_solution is None:
        fallback_points = generate_initial_voronoi_config()
        best_solution = smart_gradient_refinement(fallback_points)
    
    return best_solution

# EVOLVE-BLOCK-END
