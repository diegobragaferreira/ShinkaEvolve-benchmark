# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import random
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_distance_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        try:
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0 or np.isinf(min_dist):
                return 0.0
                
            return min_dist / max_dist
        except Exception:
            return 0.0

    def initialize_geometric_points():
        """Initialize points with geometric awareness for good starting configuration."""
        np.random.seed(42)
        
        # Create a 4x4 grid with deliberate spacing
        grid_size = 4
        points = []
        
        # Generate points in a structured way that avoids clustering
        for i in range(grid_size):
            for j in range(grid_size):
                # Apply offset to create non-uniform spacing
                x = 0.1 + 0.8 * j / (grid_size - 1) if grid_size > 1 else 0.5
                y = 0.1 + 0.8 * i / (grid_size - 1) if grid_size > 1 else 0.5
                
                # Add structured perturbations to break symmetry
                if i % 2 == 0 and j % 2 == 0:
                    x += 0.02
                    y += 0.02
                elif i % 2 == 1 and j % 2 == 1:
                    x -= 0.02
                    y -= 0.02
                    
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Add small random noise to prevent perfect symmetries
        noise = np.random.normal(0, 0.005, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        
        return points

    def evaluate_fitness(points):
        """Custom fitness function that rewards good distribution."""
        if len(points) < 2:
            return 0.0
            
        try:
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 0.0
                
            ratio = min_dist / max_dist
            
            # Additional penalty for extreme clustering or dispersion
            avg_dist = np.mean(distances[distances != np.inf])
            if avg_dist < 0.05:  # Too clustered
                ratio *= 0.8
            elif avg_dist > 0.4:  # Too dispersed
                ratio *= 0.9
                
            return ratio
        except Exception:
            return 0.0

    def genetic_crossover(parent1, parent2):
        """Specialized crossover that maintains geometric properties."""
        # Blend points with preference for parent1
        alpha = 0.7
        child = alpha * parent1 + (1 - alpha) * parent2
        return np.clip(child, 0, 1)

    def genetic_mutation(individual, mutation_rate=0.1, mutation_strength=0.02):
        """Specialized mutation that preserves point distribution properties."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # Apply localized perturbations
                noise = np.random.normal(0, mutation_strength, 2)
                mutated[i] += noise
                mutated[i] = np.clip(mutated[i], 0, 1)
        return mutated

    def distance_aware_local_refinement(points, max_iter=50):
        """Refinement that focuses on improving minimum distances."""
        def objective(x):
            pts = x.reshape(-1, 2)
            distances = squareform(pdist(pts))
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            # Maximize min distance, so minimize -min_dist
            return -min_dist
            
        bounds = [(0, 1)] * (len(points) * 2)
        
        try:
            result = minimize(objective, points.flatten(), 
                            method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': max_iter, 'ftol': 1e-12})
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
        return points

    def progressive_optimization(initial_points, max_time=170):
        """Progressive optimization starting with coarse search."""
        start_time = time.time()
        
        # Start with geometric initialization
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_distance_ratio(current_points)
        
        # Phase 1: Coarse global search
        if time.time() - start_time > max_time * 0.9:
            return best_points
            
        # Simple evolutionary approach with smaller population
        population_size = 20
        population = [current_points.copy()]
        
        # Generate diverse initial population
        for _ in range(population_size - 1):
            mutated = genetic_mutation(current_points.copy(), 0.15, 0.03)
            population.append(mutated)
        
        # Evolutionary cycles
        for epoch in range(100):
            if time.time() - start_time > max_time * 0.9:
                break
                
            # Evaluate fitness
            fitness_scores = [evaluate_fitness(ind) for ind in population]
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Update best solution
            current_ratio = fitness_scores[0]
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = population[0].copy()
            
            # Keep top performers
            elites = population[:5]
            
            # Generate new population
            new_population = elites.copy()
            
            # Crossover and mutation
            while len(new_population) < population_size:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                
                child = genetic_crossover(parent1, parent2)
                child = genetic_mutation(child, 0.1, 0.015)
                new_population.append(child)
                
            population = new_population[:population_size]
        
        # Phase 2: Local refinement
        if time.time() - start_time > max_time * 0.95:
            return best_points
            
        # Refine with distance-aware optimization
        refined_points = distance_aware_local_refinement(best_points, 100)
        refined_ratio = compute_distance_ratio(refined_points)
        
        if refined_ratio > best_ratio:
            best_points = refined_points
        
        return best_points

    # Main optimization process
    try:
        # Initialize with geometric approach
        initial_points = initialize_geometric_points()
        
        # Run progressive optimization
        final_points = progressive_optimization(initial_points, max_time=170)
        
        # Final verification and refinement
        final_ratio = compute_distance_ratio(final_points)
        
        # Try one more local refinement if needed
        if final_ratio < 0.25:  # If still low, do more aggressive refinement
            try:
                bounds = [(0, 1)] * 32
                def objective(x):
                    pts = x.reshape(-1, 2)
                    distances = squareform(pdist(pts))
                    np.fill_diagonal(distances, np.inf)
                    min_dist = np.min(distances)
                    return -min_dist  # Minimize negative of min distance
                
                result = minimize(objective, final_points.flatten(), 
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 200, 'ftol': 1e-15})
                if result.success:
                    better_points = result.x.reshape(-1, 2)
                    better_ratio = compute_distance_ratio(better_points)
                    if better_ratio > final_ratio:
                        final_points = better_points
            except:
                pass
        
        return final_points
        
    except Exception:
        # Fallback to simple initialization if anything fails
        return initialize_geometric_points()

# EVOLVE-BLOCK-END