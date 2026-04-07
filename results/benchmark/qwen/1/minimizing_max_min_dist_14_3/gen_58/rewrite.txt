# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel spherical Voronoi evolution approach for improved performance.
    """
    
    def spherical_code_14():
        """Generate initial configuration using known spherical code for 14 points"""
        # Generate points using a modified Fibonacci-like approach on sphere
        # This provides a good starting configuration that's already relatively uniform
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(14):
            y = 1 - (i / float(13)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def normalize_to_unit_sphere(points):
        """Normalize points to lie exactly on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def voronoi_uniformity_score(points):
        """Calculate how uniform the Voronoi cells are on the sphere"""
        try:
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            # Return variance of areas - lower is better (more uniform)
            return np.var(areas)
        except:
            # Fallback to inverse of min distance spread
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            return max_dist / (min_dist + 1e-12) if min_dist > 1e-12 else 1000
    
    def geometric_objective(points):
        """Objective function that combines distance ratios with geometric uniformity"""
        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max < 1e-10:
            return -1e10
        
        # Ratio of min to max distance (what we want to maximize)
        ratio = d_min / d_max
        
        # Add penalty for non-uniform Voronoi cells (higher variance means less uniform)
        uniformity_penalty = 1.0 / (1.0 + voronoi_uniformity_score(points))
        
        # Combined objective (maximize ratio AND uniformity)
        return ratio * (1.0 + 0.1 * uniformity_penalty)
    
    def mutate_points(points, strength=0.05):
        """Apply mutation to points with adaptive scaling"""
        # Create different mutation types
        mutated = points.copy()
        
        # Random perturbation with adaptive strength
        noise = np.random.normal(0, strength, points.shape)
        mutated += noise
        
        # Normalize back to sphere
        mutated = normalize_to_unit_sphere(mutated)
        
        return mutated
    
    def adaptive_cooling_schedule(iteration, max_iterations):
        """Adaptive cooling schedule for mutation strength"""
        # Start with higher mutation strength, cool down gradually
        base_strength = 0.1
        min_strength = 0.001
        return max(min_strength, base_strength * (1 - iteration / max_iterations))
    
    def evolutionary_refinement(initial_points, max_iterations=500):
        """Main evolutionary refinement process"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_score = geometric_objective(current_points)
        
        # Evolutionary parameters
        population_size = 10
        elite_count = 2
        
        for iteration in range(max_iterations):
            # Adaptive cooling
            mutation_strength = adaptive_cooling_schedule(iteration, max_iterations)
            
            # Generate new population via mutations
            population = [current_points]
            
            # Add mutated versions
            for _ in range(population_size - 1):
                mutated = mutate_points(current_points, mutation_strength)
                population.append(mutated)
            
            # Evaluate population
            scores = [geometric_objective(p) for p in population]
            
            # Find best in population
            best_in_pop_idx = np.argmax(scores)
            best_in_pop = population[best_in_pop_idx]
            best_in_pop_score = scores[best_in_pop_idx]
            
            # Update global best
            if best_in_pop_score > best_score:
                best_score = best_in_pop_score
                best_points = best_in_pop.copy()
            
            # Selection: keep top performers
            sorted_indices = np.argsort(scores)[::-1]
            selected = [population[i] for i in sorted_indices[:elite_count]]
            
            # Continue evolution with selected
            current_points = selected[0]  # Keep the best
            
            # Occasionally add some diversity through random restarts
            if iteration % 50 == 0 and iteration > 0:
                current_points = spherical_code_14()
                current_points = normalize_to_unit_sphere(current_points)
        
        return best_points
    
    def geometric_local_refinement(points, iterations=100):
        """Local refinement using gradient-based approach on sphere"""
        # Convert to flattened representation for optimization
        flattened = points.flatten()
        
        def objective_flat(flat_points):
            points_matrix = flat_points.reshape(-1, 3)
            return -geometric_objective(points_matrix)
        
        # Use differential evolution with spherical constraints
        bounds = [(-1.0, 1.0)] * len(flattened)
        
        try:
            result = differential_evolution(
                objective_flat,
                bounds,
                seed=42,
                maxiter=iterations,
                popsize=8,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            refined_points = result.x.reshape(-1, 3)
            # Ensure points remain on sphere
            refined_points = normalize_to_unit_sphere(refined_points)
            return refined_points
            
        except:
            # Fallback to simple iterative refinement
            return points
    
    # Main algorithm flow
    np.random.seed(42)
    
    # Step 1: Generate initial spherical configuration
    initial_config = spherical_code_14()
    initial_config = normalize_to_unit_sphere(initial_config)
    
    # Step 2: Apply evolutionary refinement
    evolved_points = evolutionary_refinement(initial_config, max_iterations=300)
    
    # Step 3: Local refinement
    refined_points = geometric_local_refinement(evolved_points, iterations=100)
    
    # Step 4: Final validation and cleanup
    final_points = normalize_to_unit_sphere(refined_points)
    
    # Ensure all points are within [0,1]^3 by centering and scaling
    # This ensures compatibility with the original requirement
    center = np.mean(final_points, axis=0)
    final_points = final_points - center
    
    # Scale to fit in [0,1]^3
    max_extent = np.max(np.abs(final_points))
    if max_extent > 0:
        final_points = final_points / max_extent * 0.5 + 0.5
    
    # Final validation check
    distances = cdist(final_points, final_points)
    np.fill_diagonal(distances, np.inf)
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist < 1e-10 or min_dist < 1e-10:
        # If invalid, return the spherical code
        spherical_points = spherical_code_14()
        spherical_points = normalize_to_unit_sphere(spherical_points)
        # Scale to [0,1]^3
        center = np.mean(spherical_points, axis=0)
        spherical_points = spherical_points - center
        max_extent = np.max(np.abs(spherical_points))
        if max_extent > 0:
            spherical_points = spherical_points / max_extent * 0.5 + 0.5
        return spherical_points
    
    return final_points

# EVOLVE-BLOCK-END