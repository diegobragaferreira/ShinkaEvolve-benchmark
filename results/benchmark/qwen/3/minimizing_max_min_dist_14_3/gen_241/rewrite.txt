# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective_function(x):
        # Reshape flat array back to 14x3 points
        points = x.reshape((14, 3))

        # Compute pairwise distances efficiently
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -min_dist / max_dist

    def spherical_voronoi_init(n_points=14, max_iter=100):
        """Initialize points using spherical Voronoi relaxation approach"""
        # Start with Fibonacci sphere sampling for good initial distribution
        points = fibonacci_sphere_sampling(n_points)
        
        # Apply iterative relaxation using spherical geometry
        for _ in range(max_iter):
            # Convert to spherical coordinates
            spherical_coords = cartesian_to_spherical(points)
            
            # Apply relaxation update (simplified electrostatic repulsion)
            new_points = np.zeros_like(points)
            
            for i in range(n_points):
                force = np.zeros(3)
                for j in range(n_points):
                    if i != j:
                        # Calculate repulsion force with proper spherical distance
                        diff = points[i] - points[j]
                        dist_sq = np.sum(diff**2)
                        
                        if dist_sq > 1e-12:
                            # Force inversely proportional to distance squared
                            force_magnitude = 1.0 / dist_sq
                            force += force_magnitude * diff
                
                # Update position with damping
                new_points[i] = points[i] + 0.01 * force
                
                # Project back to unit sphere
                norm = np.linalg.norm(new_points[i])
                if norm > 1e-12:
                    new_points[i] = new_points[i] / norm
                    
            points = new_points
            
        return points

    def fibonacci_sphere_sampling(n):
        """Generate points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)

    def cartesian_to_spherical(cartesian_points):
        """Convert Cartesian coordinates to spherical (radius, theta, phi)"""
        r = np.linalg.norm(cartesian_points, axis=1, keepdims=True)
        theta = np.arctan2(cartesian_points[:, 1], cartesian_points[:, 0])  # azimuthal angle
        phi = np.arccos(cartesian_points[:, 2] / np.maximum(r[:, 0], 1e-12))  # polar angle
        
        return np.column_stack([r[:, 0], theta, phi])

    def spherical_to_cartesian(spherical_coords):
        """Convert spherical coordinates to Cartesian"""
        r = spherical_coords[:, 0]
        theta = spherical_coords[:, 1]
        phi = spherical_coords[:, 2]
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        return np.column_stack([x, y, z])

    def adaptive_evolutionary_optimization(initial_points, max_generations=100):
        """Specialized evolutionary algorithm operating in spherical coordinates"""
        n_points = len(initial_points)
        
        # Convert to spherical representation
        spherical_init = cartesian_to_spherical(initial_points)
        base_radius = np.mean(spherical_init[:, 0])  # Average radius
        
        # Evolution parameters
        population_size = 20
        mutation_rate = 0.1
        elite_size = 2
        
        # Initialize population in spherical coordinates
        population = []
        for _ in range(population_size):
            # Add small random variations to initial points
            individual = spherical_init.copy()
            noise = np.random.normal(0, 0.1, individual.shape)
            individual += noise
            # Keep radius approximately constant
            individual[:, 0] = base_radius
            # Keep angles within valid ranges
            individual[:, 1] = np.clip(individual[:, 1], -np.pi, np.pi)
            individual[:, 2] = np.clip(individual[:, 2], 0, np.pi)
            population.append(individual.copy())
        
        best_individual = None
        best_fitness = -np.inf
        
        for generation in range(max_generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                # Convert back to cartesian
                cartesian_points = spherical_to_cartesian(individual)
                # Ensure points are normalized to unit sphere
                norms = np.linalg.norm(cartesian_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                cartesian_points = cartesian_points / norms
                
                # Evaluate objective
                distances = pdist(cartesian_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    fitness = min_dist / max_dist
                else:
                    fitness = -np.inf
                    
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [population[i] for i in sorted_indices]
            sorted_fitness = [fitness_scores[i] for i in sorted_indices]
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            for i in range(elite_size):
                new_population.append(sorted_population[i].copy())
            
            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = tournament_selection(sorted_population, sorted_fitness, 3)
                parent2 = tournament_selection(sorted_population, sorted_fitness, 3)
                
                # Crossover
                child = crossover(parent1, parent2)
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = mutate(child)
                
                # Ensure spherical constraint
                child[:, 0] = base_radius
                child[:, 1] = np.clip(child[:, 1], -np.pi, np.pi)
                child[:, 2] = np.clip(child[:, 2], 0, np.pi)
                
                new_population.append(child)
            
            population = new_population[:population_size]
            
        # Return best solution
        if best_individual is not None:
            final_points = spherical_to_cartesian(best_individual)
            # Normalize to ensure they're on unit sphere
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            final_points = final_points / norms
            return final_points
        else:
            return initial_points

    def tournament_selection(population, fitness_scores, tournament_size):
        """Select an individual using tournament selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
        return population[best_idx].copy()

    def crossover(parent1, parent2):
        """Single-point crossover for spherical coordinates"""
        crossover_point = np.random.randint(1, len(parent1))
        child = np.vstack([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        return child

    def mutate(individual):
        """Mutate individual with spherical coordinate adaptation"""
        mutation_strength = 0.2
        for i in range(len(individual)):
            if np.random.random() < 0.3:  # 30% chance to mutate each gene
                # Mutate radius (small change)
                individual[i, 0] += np.random.normal(0, mutation_strength * 0.1)
                # Mutate theta (azimuthal angle)
                individual[i, 1] += np.random.normal(0, mutation_strength)
                # Mutate phi (polar angle) 
                individual[i, 2] += np.random.normal(0, mutation_strength)
        return individual

    def project_to_cube(points):
        """Project points from unit sphere to [0,1]^3 cube"""
        # Map from [-1,1]^3 to [0,1]^3
        scaled_points = (points + 1) / 2
        return np.clip(scaled_points, 0, 1)

    def validate_and_correct_bounds(points):
        """Ensure all points are within [0,1]^3 bounds"""
        corrected_points = np.clip(points, 0, 1)
        return corrected_points

    # Initialize using spherical Voronoi approach
    np.random.seed(42)
    initial_points_spherical = spherical_voronoi_init(14)
    
    # Convert to cube space
    initial_points_cube = project_to_cube(initial_points_spherical)
    
    # Apply evolutionary optimization
    evolved_points = adaptive_evolutionary_optimization(initial_points_cube, max_generations=50)
    
    # Final refinement with local optimization
    try:
        # Flatten points for scipy optimization
        flattened_points = evolved_points.flatten()
        
        def refined_objective(x):
            points = x.reshape((14, 3))
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return -np.inf
            return -min_dist / max_dist
        
        # Local optimization with bounds
        result = minimize(
            refined_objective,
            flattened_points,
            method='L-BFGS-B',
            bounds=[(0, 1)] * 14 * 3,
            options={'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        # Reshape result
        final_points = result.x.reshape((14, 3))
        
    except Exception:
        final_points = evolved_points
    
    # Ensure bounds are properly maintained
    final_points = validate_and_correct_bounds(final_points)
    
    return final_points

# EVOLVE-BLOCK-END