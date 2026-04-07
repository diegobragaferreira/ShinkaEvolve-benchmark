# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import random
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel spherical evolution optimizer approach combining geometric sampling, evolutionary algorithms,
    and adaptive local refinement.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Constants
    N_POINTS = 14
    MAX_ITERATIONS = 1000
    POPULATION_SIZE = 20
    ELITE_SIZE = 4
    MUTATION_RATE = 0.3
    CROSSOVER_RATE = 0.7
    
    # Helper functions
    def normalize_to_unit_sphere(points):
        """Normalize points to lie exactly on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / np.maximum(norms, 1e-12)
    
    def calculate_ratio(points):
        """Calculate min/max distance ratio with improved numerical stability"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        # Filter out very small distances to avoid numerical issues
        distances = distances[distances > 1e-12]
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max < 1e-12:
            return 0.0
        return d_min / d_max
    
    def fitness_function(points):
        """Enhanced fitness function that balances ratio with distribution uniformity"""
        ratio = calculate_ratio(points)
        if ratio <= 0:
            return -np.inf
            
        # Add penalty for high variance in distances (encourages uniform distribution)
        distances = pdist(points)
        distances = distances[distances > 1e-12]
        if len(distances) > 0:
            dist_variance = np.var(distances)
            dist_mean = np.mean(distances)
            if dist_mean > 1e-12:
                # Penalty based on relative variance
                uniformity_penalty = 0.1 * (dist_variance / dist_mean)
                # Combine with ratio (negative because we minimize in optimization)
                return -(ratio - uniformity_penalty)
        return -ratio
    
    def fibonacci_spiral(n):
        """Generate points on sphere using Fibonacci spiral"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(max(0, 1 - y * y))  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def icosahedron_points(n=14):
        """Generate points using icosahedron-based construction"""
        # Vertices of a regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # If we need more than 12 points, distribute additional points
        if n <= 12:
            return vertices[:n]
        else:
            # For 14 points, we'll start with icosahedron vertices and add two more
            points = vertices.copy()
            
            # Add two more points that are well-distributed
            # Add points along major axes
            points = np.vstack([points, [[0, 0, 1], [0, 0, -1]]])
            
            # Apply slight random perturbation to ensure good distribution
            np.random.seed(42)
            points += np.random.normal(0, 0.05, (points.shape[0], 3))
            
            # Normalize again to maintain unit sphere
            norms = np.linalg.norm(points, axis=1)
            points = points / np.maximum(norms[:, np.newaxis], 1e-12)
            
            return points[:n]
    
    def generate_population():
        """Generate diverse initial population using multiple sampling strategies"""
        population = []
        
        # Strategy 1: Fibonacci spiral with perturbation
        fib_points = fibonacci_spiral(N_POINTS)
        for _ in range(3):
            perturbed = fib_points + np.random.normal(0, 0.02, fib_points.shape)
            population.append(normalize_to_unit_sphere(perturbed))
        
        # Strategy 2: Icosahedron-based points
        ico_points = icosahedron_points(N_POINTS)
        for _ in range(3):
            perturbed = ico_points + np.random.normal(0, 0.02, ico_points.shape)
            population.append(normalize_to_unit_sphere(perturbed))
        
        # Strategy 3: Random points with spherical constraint
        for _ in range(14):
            points = np.random.randn(N_POINTS, 3)
            population.append(normalize_to_unit_sphere(points))
        
        return population
    
    def mutate_individual(individual):
        """Apply geometric mutation to individual while maintaining spherical constraint"""
        mutated = individual.copy()
        
        # Select points to mutate
        num_mutate = max(1, int(MUTATION_RATE * N_POINTS))
        indices = random.sample(range(N_POINTS), num_mutate)
        
        for idx in indices:
            # Add small random displacement on tangent plane
            # Generate random vector
            random_vector = np.random.randn(3)
            # Project onto tangent plane at current point
            tangent_vector = random_vector - np.dot(random_vector, individual[idx]) * individual[idx]
            # Apply small perturbation
            perturbation_scale = 0.01 + 0.02 * random.random()
            mutated[idx] += tangent_vector * perturbation_scale
            # Project back to sphere
            mutated[idx] = normalize_to_unit_sphere(mutated[idx].reshape(1, 3))[0]
        
        return mutated
    
    def crossover_parents(parent1, parent2):
        """Perform spherical crossover between two parent individuals"""
        # Blend points from parents
        offspring = np.zeros_like(parent1)
        crossover_point = random.randint(1, N_POINTS-1)
        
        # Take first part from parent1, second part from parent2
        offspring[:crossover_point] = parent1[:crossover_point]
        offspring[crossover_point:] = parent2[crossover_point:]
        
        # Ensure offspring stays on unit sphere
        return normalize_to_unit_sphere(offspring)
    
    def adaptive_local_search(points, max_iter=200):
        """Apply adaptive local search focused on improving ratio while maintaining spherical constraints"""
        
        def objective_func(x_flat):
            points_reshaped = x_flat.reshape(-1, 3)
            # Ensure points are on unit sphere
            norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
            normalized_points = points_reshaped / np.maximum(norms, 1e-12)
            
            ratio = calculate_ratio(normalized_points)
            # Negative because we want to maximize ratio (minimize negative)
            return -ratio
            
        def constraint_sphere(x_flat):
            points_reshaped = x_flat.reshape(-1, 3)
            norms = np.linalg.norm(points_reshaped, axis=1)
            return norms - 1.0
            
        # Apply local optimization with multiple methods
        try:
            # Multi-start local optimization
            best_points = points.copy()
            best_ratio = calculate_ratio(points)
            
            # Try multiple local optimization methods with different parameters
            methods_params = [
                ('L-BFGS-B', {'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'ftol': 1e-12, 'gtol': 1e-12}),
                ('trust-constr', {'xtol': 1e-12, 'gtol': 1e-12})
            ]
            
            for method, options in methods_params:
                try:
                    result = minimize(
                        objective_func,
                        points.flatten(),
                        method=method,
                        bounds=[(-2, 2)] * (N_POINTS * 3),
                        constraints={'type': 'eq', 'fun': constraint_sphere},
                        options={**options, 'maxiter': max_iter//3}
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                        optimized_points = optimized_points / np.maximum(norms, 1e-12)
                        
                        ratio = calculate_ratio(optimized_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except:
                    continue
                    
            return best_points
            
        except Exception:
            return points
    
    # Main evolutionary optimization loop
    np.random.seed(42)
    random.seed(42)
    
    # Phase 1: Generate initial population
    population = generate_population()
    
    best_overall_ratio = -np.inf
    best_overall_points = None
    
    # Phase 2: Evolutionary optimization
    for generation in range(MAX_ITERATIONS):
        # Evaluate fitness of entire population
        fitness_scores = []
        for individual in population:
            score = fitness_function(individual)
            fitness_scores.append(score)
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Track best solution so far
        if sorted_fitness[0] > best_overall_ratio:
            best_overall_ratio = sorted_fitness[0]
            best_overall_points = sorted_population[0].copy()
        
        # Early stopping condition
        if best_overall_ratio > 0.49:  # Close to target
            break
            
        # Create new population through selection, crossover, and mutation
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(ELITE_SIZE):
            new_population.append(sorted_population[i].copy())
        
        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            # Tournament selection
            tournament_size = 3
            selected_indices = random.sample(range(POPULATION_SIZE), tournament_size)
            selected_fitness = [sorted_fitness[i] for i in selected_indices]
            winner_index = selected_indices[np.argmax(selected_fitness)]
            
            parent1 = sorted_population[winner_index]
            
            # Another parent
            if random.random() < CROSSOVER_RATE and len(new_population) < POPULATION_SIZE - 1:
                # Crossover
                parent2 = sorted_population[random.randint(0, POPULATION_SIZE-1)]
                offspring = crossover_parents(parent1, parent2)
                new_population.append(offspring)
            else:
                # Mutation only
                mutated = mutate_individual(parent1)
                new_population.append(mutated)
        
        population = new_population
    
    # Phase 3: Final adaptive local refinement
    if best_overall_points is not None:
        refined_points = adaptive_local_search(best_overall_points, max_iter=300)
        refined_ratio = calculate_ratio(refined_points)
        
        if refined_ratio > best_overall_ratio:
            best_overall_points = refined_points
            best_overall_ratio = refined_ratio
    
    # Final fallback to best initialization if nothing worked well
    if best_overall_points is None:
        # Generate several candidates and pick the best
        candidates = []
        for _ in range(5):
            # Try different initialization methods
            fib_points = fibonacci_spiral(N_POINTS)
            ico_points = icosahedron_points(N_POINTS)
            
            # Random points
            rand_points = np.random.randn(N_POINTS, 3)
            rand_points = normalize_to_unit_sphere(rand_points)
            
            candidates.append(rand_points)
            
        # Evaluate all candidates
        best_candidate_ratio = -np.inf
        best_candidate = None
        for candidate in candidates:
            ratio = calculate_ratio(candidate)
            if ratio > best_candidate_ratio:
                best_candidate_ratio = ratio
                best_candidate = candidate.copy()
                
        best_overall_points = best_candidate if best_candidate is not None else np.random.rand(N_POINTS, 3)
        best_overall_points = normalize_to_unit_sphere(best_overall_points)
    
    return best_overall_points

# EVOLVE-BLOCK-END