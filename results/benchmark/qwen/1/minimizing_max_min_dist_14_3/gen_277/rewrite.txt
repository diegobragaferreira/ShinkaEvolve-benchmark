# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import math
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def generate_icosahedron_points():
        """Generate points using icosahedron vertices"""
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices
    
    def fibonacci_like_distribution(n_points):
        """Generate Fibonacci-like distribution on sphere"""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            # Use golden ratio multiple for better spread
            theta = math.acos(y)  # polar angle
            phi_angle = (i * 2.414213562) % (2 * math.pi)  # golden ratio multiple
            
            x = radius * math.cos(phi_angle)
            z = radius * math.sin(phi_angle)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def initialize_points():
        """Create initial configuration with geometrically informed points"""
        # Start with icosahedron vertices
        ico_points = generate_icosahedron_points()
        
        # Add more points using Fibonacci distribution
        additional_points = fibonacci_like_distribution(14 - len(ico_points))
        
        # Combine and ensure we have exactly 14 points
        if len(ico_points) >= 14:
            points = ico_points[:14]
        else:
            points = np.vstack([ico_points, additional_points[:14-len(ico_points)]])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / np.maximum(norms, 1e-12)
        
        # Add small jitter to break symmetry
        np.random.seed(42)
        jitter = np.random.normal(0, 0.01, normalized_points.shape)
        jittered_points = normalized_points + jitter
        
        # Renormalize after jitter
        norms = np.linalg.norm(jittered_points, axis=1, keepdims=True)
        final_points = jittered_points / np.maximum(norms, 1e-12)
        
        return final_points
    
    def calculate_voronoi_uniformity(points):
        """Calculate uniformity of Voronoi cells"""
        try:
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            # Return coefficient of variation (lower is better for uniformity)
            mean_area = np.mean(areas)
            if mean_area > 0:
                cv = np.std(areas) / mean_area
                return cv
            return 1.0
        except:
            return 1.0
    
    def distance_ratio(points):
        """Calculate the ratio of minimum to maximum distance"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def fitness_function(points):
        """Combined fitness based on distance ratio and Voronoi uniformity"""
        # Calculate main objective: distance ratio
        ratio = distance_ratio(points)
        
        # Calculate Voronoi uniformity penalty (lower is better)
        uniformity = calculate_voronoi_uniformity(points)
        
        # We want to maximize ratio while minimizing uniformity penalty
        # Tradeoff parameter: weight towards ratio vs uniformity
        alpha = 0.7
        beta = 0.3
        
        # Combined fitness: maximize ratio with uniformity penalty
        fitness = alpha * ratio - beta * uniformity
        
        return fitness
    
    def voronoi_mutation(parent_points, mutation_rate=0.05):
        """Mutation operator based on Voronoi structure"""
        # Create Voronoi diagram
        sv = SphericalVoronoi(parent_points)
        areas = sv.voronoi_cell_areas()
        
        # Select points with larger Voronoi cells for mutation (they're likely under-represented)
        # This creates pressure to increase density in sparse regions
        mutation_indices = np.where(areas > np.median(areas))[0]
        
        if len(mutation_indices) == 0:
            mutation_indices = np.random.choice(len(parent_points), size=max(1, int(len(parent_points)*mutation_rate)), replace=False)
        
        mutated_points = parent_points.copy()
        
        # Apply mutation to selected points
        for idx in mutation_indices:
            # Small random movement on sphere using spherical coordinates
            np.random.seed(idx)  # For reproducibility
            displacement = np.random.normal(0, mutation_rate, 3)
            
            # Convert to spherical coordinates
            r = np.linalg.norm(mutated_points[idx])
            theta = np.arccos(mutated_points[idx][2] / (r + 1e-12))
            phi = np.arctan2(mutated_points[idx][1], mutated_points[idx][0])
            
            # Apply displacement in spherical space
            new_theta = theta + displacement[0] * 0.5
            new_phi = phi + displacement[1] * 0.5
            
            # Clamp theta to valid range
            new_theta = np.clip(new_theta, 1e-6, np.pi - 1e-6)
            
            # Convert back to Cartesian
            new_r = r + displacement[2] * 0.1  # Small radial change
            new_r = np.clip(new_r, 0.1, 2.0)  # Prevent degenerate points
            
            new_x = new_r * np.sin(new_theta) * np.cos(new_phi)
            new_y = new_r * np.sin(new_theta) * np.sin(new_phi)
            new_z = new_r * np.cos(new_theta)
            
            mutated_points[idx] = [new_x, new_y, new_z]
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(mutated_points, axis=1, keepdims=True)
        normalized_points = mutated_points / np.maximum(norms, 1e-12)
        
        return normalized_points
    
    def spherical_crossover(parent1, parent2):
        """Crossover operator that maintains spherical constraint"""
        # Blend points with random weights
        weights = np.random.rand(len(parent1))
        weights = weights / np.sum(weights)  # Normalize
        
        # Linear blend
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            child[i] = weights[i] * parent1[i] + (1 - weights[i]) * parent2[i]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(child, axis=1, keepdims=True)
        normalized_child = child / np.maximum(norms, 1e-12)
        
        return normalized_child
    
    def evolutive_optimization():
        """Main evolutionary optimization loop"""
        # Initialize population
        population_size = 20
        population = []
        
        # Create diverse initial population
        for i in range(population_size):
            np.random.seed(42 + i)
            # Add different noise to each individual
            points = initialize_points()
            # Apply slight variations
            noise = np.random.normal(0, 0.02, points.shape)
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / np.maximum(norms, 1e-12)
            population.append(points)
        
        best_individual = None
        best_fitness = -np.inf
        
        # Evolutionary loop
        for generation in range(50):
            # Evaluate fitness for entire population
            fitness_scores = []
            for individual in population:
                score = fitness_function(individual)
                fitness_scores.append(score)
            
            # Track best individual
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_individual = population[max_idx].copy()
            
            # Selection: tournament selection
            selected_parents = []
            tournament_size = 3
            for _ in range(population_size // 2):
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_parents.append(population[winner_idx])
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-4:]  # Top 4
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Crossover
                if len(selected_parents) >= 2:
                    parent1 = selected_parents[np.random.randint(len(selected_parents))]
                    parent2 = selected_parents[np.random.randint(len(selected_parents))]
                    child = spherical_crossover(parent1, parent2)
                else:
                    # If no parents available, mutate random individual
                    parent = population[np.random.randint(len(population))]
                    child = voronoi_mutation(parent, 0.1)
                
                # Mutation
                if np.random.random() < 0.7:  # 70% mutation rate
                    child = voronoi_mutation(child, 0.08)
                
                new_population.append(child)
            
            # Trim to exact population size
            population = new_population[:population_size]
            
            # Periodic refinement with local optimization
            if generation % 5 == 0:
                for i in range(len(population)):
                    # Local refinement with constrained optimization
                    def objective(x_flat):
                        points = x_flat.reshape(-1, 3)
                        return -fitness_function(points)
                    
                    def constraint_sphere(x):
                        points = x.reshape(-1, 3)
                        norms = np.linalg.norm(points, axis=1)
                        return norms - 1.0
                    
                    constraints = {'type': 'eq', 'fun': constraint_sphere}
                    bounds = [(-1.5, 1.5)] * len(population[i].flatten())
                    
                    try:
                        result = minimize(
                            objective,
                            population[i].flatten(),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
                        )
                        if result.success:
                            refined_points = result.x.reshape(-1, 3)
                            norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
                            population[i] = refined_points / np.maximum(norms, 1e-12)
                    except:
                        pass
        
        return best_individual if best_individual is not None else population[0]
    
    # Execute the evolutionary optimization
    try:
        final_points = evolutive_optimization()
        return final_points
    except:
        # Fallback to initialization
        return initialize_points()

# EVOLVE-BLOCK-END