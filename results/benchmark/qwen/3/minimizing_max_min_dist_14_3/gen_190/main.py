# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    class SpherePackingOptimizer:
        def __init__(self, n_points=14, dimensions=3, seed=42):
            self.n_points = n_points
            self.dimensions = dimensions
            self.seed = seed
            np.random.seed(seed)
            
        def _generate_initial_configurations(self):
            """Generate diverse initial configurations using multiple strategies"""
            configs = []
            
            # Strategy 1: Fibonacci spiral on sphere
            fib_points = self._fibonacci_sphere(14)
            fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
            configs.append(("fibonacci", fib_points.copy()))
            
            # Strategy 2: Spherical Voronoi
            try:
                sv_points = self._spherical_voronoi_points(14)
                sv_points = (sv_points + 1) / 2
                configs.append(("spherical_voronoi", sv_points.copy()))
            except:
                # Fallback to fibonacci
                configs.append(("spherical_voronoi", fib_points.copy()))
            
            # Strategy 3: Cube grid with jitter
            cube_points = self._cube_grid_with_jitter(14, 0.05)
            configs.append(("cube_grid", cube_points.copy()))
            
            # Strategy 4: Random with clustering
            random_points = np.random.rand(14, 3)
            configs.append(("random", random_points.copy()))
            
            # Strategy 5: Perturbed Fibonacci
            perturbed = fib_points + np.random.normal(0, 0.02, (14, 3))
            perturbed = np.clip(perturbed, 0, 1)
            configs.append(("perturbed", perturbed.copy()))
            
            return configs
            
        def _fibonacci_sphere(self, n):
            """Generate points on sphere using Fibonacci spiral"""
            points = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            
            for i in range(n):
                y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                
                theta = golden_angle * i  # golden angle increment
                
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                
                points.append([x, y, z])
                
            return np.array(points)
            
        def _spherical_voronoi_points(self, n):
            """Generate points using spherical Voronoi diagram"""
            points = np.random.randn(n, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
            sv = SphericalVoronoi(points)
            voronoi_centers = sv.vertices
            voronoi_centers = voronoi_centers / np.linalg.norm(voronoi_centers, axis=1, keepdims=True)
            
            if len(voronoi_centers) >= n:
                return voronoi_centers[:n]
            else:
                return np.vstack([voronoi_centers, points[:n-len(voronoi_centers)]])
                
        def _cube_grid_with_jitter(self, n, jitter_magnitude=0.05):
            """Generate points in cube grid with jitter"""
            grid_size = int(np.ceil(n**(1/3)))
            coords = np.linspace(0, 1, grid_size)
            grid_points = []

            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        if len(grid_points) < n:
                            grid_points.append([coords[i], coords[j], coords[k]])

            points = np.array(grid_points[:n])
            # Add jitter
            jitter = np.random.normal(0, jitter_magnitude, points.shape)
            points = points + jitter
            points = np.clip(points, 0, 1)
            return points
            
        def _evaluate_fitness(self, points):
            """Evaluate fitness as minimum/maximum distance ratio"""
            if len(points) < 2:
                return 0.0
                
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
                
            # Remove infinities and filter non-finite values
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) == 0:
                return 0.0
                
            d_min = np.min(finite_distances)
            d_max = np.max(finite_distances)
            
            if d_max <= 1e-12:
                return 0.0
                
            return d_min / d_max
            
        def _sphere_pack_constraint(self, points, min_dist):
            """Penalize violations of sphere packing constraints"""
            if len(points) < 2:
                return 0.0
                
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
                
            # Remove infinities and filter non-finite values
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) == 0:
                return 0.0
                
            # Count violations (distance less than min_dist)
            violations = np.sum(finite_distances < min_dist)
            penalty = violations * 10000.0
            return penalty
            
        def _evolutionary_population_search(self, initial_points, max_generations=200):
            """Evolutionary search with adaptive operators"""
            population_size = 30
            population = []
            
            # Initialize population with variations of initial points
            for i in range(population_size):
                individual = initial_points.copy()
                # Add small random variation
                noise = np.random.normal(0, 0.01, individual.shape)
                individual = individual + noise
                individual = np.clip(individual, 0, 1)
                population.append(individual)
            
            best_individual = initial_points.copy()
            best_fitness = self._evaluate_fitness(best_individual)
            
            # Evolution loop
            for generation in range(max_generations):
                # Evaluate fitness of entire population
                fitness_scores = []
                for individual in population:
                    fitness = self._evaluate_fitness(individual)
                    fitness_scores.append(fitness)
                
                # Find best individual in current population
                max_fitness_idx = np.argmax(fitness_scores)
                if fitness_scores[max_fitness_idx] > best_fitness:
                    best_fitness = fitness_scores[max_fitness_idx]
                    best_individual = population[max_fitness_idx].copy()
                
                # Selection (tournament selection)
                selected_population = []
                tournament_size = 3
                for _ in range(population_size):
                    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                    selected_population.append(population[winner_idx].copy())
                
                # Crossover and mutation
                new_population = []
                for i in range(0, population_size, 2):
                    parent1 = selected_population[i]
                    parent2 = selected_population[min(i+1, population_size-1)]
                    
                    # Uniform crossover
                    mask = np.random.rand(*parent1.shape) > 0.5
                    child1 = np.where(mask, parent1, parent2)
                    child2 = np.where(mask, parent2, parent1)
                    
                    # Mutation - add gaussian noise
                    mutation_strength = max(0.001, 0.1 * (1 - generation/max_generations))
                    noise1 = np.random.normal(0, mutation_strength, child1.shape)
                    noise2 = np.random.normal(0, mutation_strength, child2.shape)
                    child1 = child1 + noise1
                    child2 = child2 + noise2
                    child1 = np.clip(child1, 0, 1)
                    child2 = np.clip(child2, 0, 1)
                    
                    new_population.extend([child1, child2])
                
                # Trim population to exact size
                population = new_population[:population_size]
            
            return best_individual
            
        def _hybrid_optimization(self, initial_points):
            """Combine evolutionary search with local optimization"""
            # Step 1: Evolutionary search for global exploration
            evolved_points = self._evolutionary_population_search(initial_points, max_generations=100)
            
            # Step 2: Local refinement using L-BFGS
            refined_points = self._local_refinement(evolved_points)
            
            # Step 3: Multi-stage refinement with varying tolerances
            final_points = self._multi_stage_refinement(refined_points)
            
            return final_points
            
        def _local_refinement(self, points):
            """Refine using gradient-based optimization"""
            def objective(x):
                points_local = x.reshape(-1, 3)
                distances = pdist(points_local)
                
                if len(distances) == 0:
                    return -np.inf
                    
                d_min = np.min(distances)
                d_max = np.max(distances)
                
                if d_max > 1e-12:
                    return -(d_min / d_max)
                else:
                    return -np.inf
            
            try:
                x0 = points.flatten()
                bounds = [(0, 1)] * self.n_points * 3
                
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-10, 'gtol': 1e-10},
                    tol=1e-10
                )
                
                if result.success:
                    refined = result.x.reshape(-1, 3)
                    refined = np.clip(refined, 0, 1)
                    return refined
            except:
                pass
                
            return points
            
        def _multi_stage_refinement(self, points):
            """Apply multi-stage local optimization with adaptive tolerances"""
            refined_points = points.copy()
            
            # Stage 1: Coarse refinement
            try:
                def obj_coarse(x):
                    points_local = x.reshape(-1, 3)
                    distances = pdist(points_local)
                    
                    if len(distances) == 0:
                        return -np.inf
                        
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 1e-12:
                        return -(d_min / d_max)
                    else:
                        return -np.inf
                
                x0 = refined_points.flatten()
                bounds = [(0, 1)] * self.n_points * 3
                
                result = minimize(
                    obj_coarse,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-8, 'gtol': 1e-8},
                    tol=1e-8
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 3)
                    refined_points = np.clip(refined_points, 0, 1)
            except:
                pass
            
            # Stage 2: Fine refinement
            try:
                def obj_fine(x):
                    points_local = x.reshape(-1, 3)
                    distances = pdist(points_local)
                    
                    if len(distances) == 0:
                        return -np.inf
                        
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    
                    if d_max > 1e-12:
                        return -(d_min / d_max)
                    else:
                        return -np.inf
                
                x0 = refined_points.flatten()
                bounds = [(0, 1)] * self.n_points * 3
                
                result = minimize(
                    obj_fine,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 3)
                    refined_points = np.clip(refined_points, 0, 1)
            except:
                pass
                
            return refined_points
    
    # Initialize optimizer
    optimizer = SpherePackingOptimizer(n_points=14, dimensions=3, seed=42)
    
    # Generate initial configurations
    configs = optimizer._generate_initial_configurations()
    
    # Evaluate all initial configurations and select best
    best_config = None
    best_fitness = -np.inf
    
    for name, points in configs:
        fitness = optimizer._evaluate_fitness(points)
        if fitness > best_fitness:
            best_fitness = fitness
            best_config = points.copy()
    
    # Apply hybrid optimization to best configuration
    final_points = optimizer._hybrid_optimization(best_config)
    
    # Final validation and cleanup
    final_points = np.clip(final_points, 0, 1)
    
    return final_points

# EVOLVE-BLOCK-END