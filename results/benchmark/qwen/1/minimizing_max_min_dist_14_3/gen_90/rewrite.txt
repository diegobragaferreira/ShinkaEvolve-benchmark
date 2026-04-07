# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import random
from copy import deepcopy

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    
    def calculate_min_max_ratio(points):
        """Helper function to compute the min/max distance ratio"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def fitness_function(points):
        """Custom fitness function that rewards good distribution"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        
        # Reward based on ratio, but also penalize extreme disparities
        ratio = min_dist / max_dist
        # Penalize if there's too much variance in distances
        mean_dist = np.mean(distances[distances != np.inf])
        std_dist = np.std(distances[distances != np.inf])
        # Encourage more uniform distribution
        uniformity_penalty = std_dist / mean_dist if mean_dist > 0 else 0
        
        # Combined fitness: ratio minus penalty for non-uniformity
        return ratio - 0.1 * uniformity_penalty
    
    def constraint_func(x):
        # Ensure points are on unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    def generate_voronoi_initial():
        """Generate initial points using spherical Voronoi construction"""
        # Start with an icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Create a more refined distribution
        points = []
        # Add vertices of icosahedron
        for v in vertices:
            points.append(v)
        
        # Add additional points by subdividing faces
        # For simplicity, add some random points near existing ones
        for i in range(n - len(vertices)):
            # Pick a random vertex and add a nearby point
            base_idx = random.randint(0, len(vertices) - 1)
            base_point = vertices[base_idx]
            # Add small random displacement
            displacement = np.random.randn(3) * 0.1
            new_point = base_point + displacement
            # Project back to sphere
            new_point = new_point / np.linalg.norm(new_point)
            points.append(new_point)
        
        return np.array(points[:n])
    
    def population_based_optimization():
        """Use a population-based approach for better exploration"""
        population_size = 20
        generations = 30
        mutation_rate = 0.1
        
        # Initialize population
        population = []
        for _ in range(population_size):
            initial_points = generate_voronoi_initial()
            population.append(initial_points)
        
        best_solution = None
        best_fitness = -np.inf
        
        for generation in range(generations):
            # Evaluate fitness for all individuals
            fitness_values = []
            for individual in population:
                fitness_val = fitness_function(individual)
                fitness_values.append(fitness_val)
                if fitness_val > best_fitness:
                    best_fitness = fitness_val
                    best_solution = deepcopy(individual)
            
            # Selection: keep top half
            sorted_indices = np.argsort(fitness_values)[::-1]
            selected_population = [population[i] for i in sorted_indices[:population_size//2]]
            
            # Create new population through crossover and mutation
            new_population = selected_population[:]
            
            while len(new_population) < population_size:
                # Tournament selection
                parent1_idx = random.randint(0, len(selected_population) - 1)
                parent2_idx = random.randint(0, len(selected_population) - 1)
                parent1 = selected_population[parent1_idx]
                parent2 = selected_population[parent2_idx]
                
                # Crossover: blend points
                alpha = random.random()
                child = parent1 * alpha + parent2 * (1 - alpha)
                
                # Normalize children back to sphere
                for i in range(len(child)):
                    child[i] = child[i] / np.linalg.norm(child[i])
                
                # Mutation
                if random.random() < mutation_rate:
                    # Mutate one point
                    point_idx = random.randint(0, n - 1)
                    mutation_strength = 0.05
                    mutation = np.random.randn(3) * mutation_strength
                    child[point_idx] += mutation
                    child[point_idx] = child[point_idx] / np.linalg.norm(child[point_idx])
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        return best_solution
    
    # Phase 1: Population-based evolutionary search
    np.random.seed(42)
    try:
        evolved_points = population_based_optimization()
    except Exception:
        # Fallback to simpler initialization if evolution fails
        evolved_points = generate_voronoi_initial()
    
    # Phase 2: Local refinement with constrained optimization
    def objective(x):
        points = x.reshape(-1, 3)
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    # Refine using L-BFGS-B
    x0 = evolved_points.flatten()
    cons = {'type': 'eq', 'fun': constraint_func}
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', constraints=cons, 
                         options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 500})
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            # Ensure they're still on the sphere
            for i in range(len(optimized_points)):
                optimized_points[i] = optimized_points[i] / np.linalg.norm(optimized_points[i])
        else:
            optimized_points = evolved_points
            
    except Exception:
        optimized_points = evolved_points
    
    # Phase 3: Final optimization with SLSQP for better constrained handling
    try:
        x0_final = optimized_points.flatten()
        result = minimize(objective, x0_final, method='SLSQP', constraints=cons,
                         options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 300})
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            # Ensure they're still on the sphere
            for i in range(len(final_points)):
                final_points[i] = final_points[i] / np.linalg.norm(final_points[i])
        else:
            final_points = optimized_points
            
    except Exception:
        final_points = optimized_points
    
    return final_points

# EVOLVE-BLOCK-END