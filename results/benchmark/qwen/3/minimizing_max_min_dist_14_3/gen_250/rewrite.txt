# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.spatial.distance import pdist
import warnings
import time
from scipy.spatial.transform import Rotation as R


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    def fibonacci_sphere(samples=14):
        """Generate points on sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def spherical_to_cartesian(theta, phi):
        """Convert spherical coordinates to cartesian"""
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.array([x, y, z])
    
    def cartesian_to_spherical(x, y, z):
        """Convert cartesian to spherical coordinates"""
        r = np.sqrt(x*x + y*y + z*z)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r) if r != 0 else 0
        return r, theta, phi
    
    def distance_guided_mutation(current_points, best_points, ratio_stats, mutation_rate=0.1):
        """Mutation strategy guided by current distance statistics"""
        # Calculate current distance ratio statistics
        distances = pdist(current_points)
        if len(distances) == 0:
            return current_points.copy()
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        d_mean = np.mean(distances)
        
        # Adjust mutation rate based on distance ratio
        if d_max > 0:
            ratio = d_min / d_max
            # If ratio is low (bad), increase mutation rate
            if ratio < 0.3:
                mutation_rate *= 1.5
            elif ratio > 0.45:
                mutation_rate *= 0.8
        
        # Apply mutation with adaptive rates
        mutated = current_points.copy()
        for i in range(len(mutated)):
            # Apply small random perturbations
            if np.random.random() < mutation_rate:
                # Perturb with different magnitudes depending on context
                magnitude = 0.01 if ratio < 0.3 else 0.005
                mutated[i] += np.random.normal(0, magnitude, 3)
                
        # Keep within bounds
        mutated = np.clip(mutated, 0, 1)
        return mutated
    
    def adaptive_evolution_step(population, fitness_scores, generation, time_limit):
        """Perform one adaptive evolution step"""
        start_time = time.time()
        
        # Sort population by fitness (ascending since we minimize negative ratio)
        sorted_indices = np.argsort(fitness_scores)
        top_30_percent = sorted_indices[:len(population)//3]
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        for idx in top_30_percent:
            new_population.append(population[idx].copy())
        
        # Crossover and mutation
        while len(new_population) < len(population):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Select parents
            parent1 = population[np.random.choice(top_30_percent)]
            parent2 = population[np.random.choice(top_30_percent)]
            
            # Crossover (uniform)
            child = np.zeros_like(parent1)
            mask = np.random.random(14) < 0.5
            child[mask] = parent1[mask]
            child[~mask] = parent2[~mask]
            
            # Mutation with distance-guided strategy
            child = distance_guided_mutation(child, population[top_30_percent[0]], 
                                           (np.min(pdist(parent1)), np.max(pdist(parent1))), 
                                           mutation_rate=0.1)
            
            new_population.append(child)
        
        return new_population[:len(population)]
    
    def evaluate_population(population):
        """Evaluate fitness of entire population"""
        fitness_scores = []
        for points in population:
            # Calculate pairwise distances
            distances = pdist(points)
            
            if len(distances) == 0:
                fitness_scores.append(float('inf'))
                continue
            
            # Remove any NaN or infinite values
            distances = distances[np.isfinite(distances)]
            
            if len(distances) == 0:
                fitness_scores.append(float('inf'))
                continue
            
            # Calculate min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # Return negative ratio to maximize (since we're minimizing)
            if d_max <= 0:
                fitness_scores.append(float('inf'))
            else:
                fitness_scores.append(-d_min / d_max)
        
        return np.array(fitness_scores)
    
    def generate_initial_population(size=20):
        """Generate diverse initial population"""
        population = []
        
        # Base spherical distribution
        base_points = fibonacci_sphere(14)
        base_points = base_points - np.mean(base_points, axis=0)
        max_coord = np.max(np.abs(base_points))
        if max_coord > 0:
            base_points = base_points / (2 * max_coord) + 0.5
        
        # Create diverse variants
        for i in range(size):
            np.random.seed(i * 42)
            points = base_points.copy()
            
            # Add different types of perturbations
            if i % 3 == 0:
                # Small random jitter
                points += np.random.normal(0, 0.01, points.shape)
            elif i % 3 == 1:
                # More aggressive perturbation
                points += np.random.normal(0, 0.03, points.shape)
            else:
                # Rotate slightly
                rotation = R.from_euler('xyz', np.random.uniform(0, 2*np.pi, 3)).as_matrix()
                points = points @ rotation.T
            
            # Keep within bounds
            points = np.clip(points, 0, 1)
            population.append(points)
        
        return population
    
    def optimize_with_evolution(time_limit):
        """Use evolutionary algorithm to optimize"""
        start_time = time.time()
        
        # Generate initial population
        population = generate_initial_population(25)
        
        # Evolution parameters
        max_generations = 50
        generation = 0
        
        # Store best solution so far
        best_fitness = float('inf')
        best_individual = None
        
        while generation < max_generations and (time.time() - start_time) < time_limit - 15:
            # Evaluate population
            fitness_scores = evaluate_population(population)
            
            # Update best solution
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()
            
            # Check if we've converged
            if generation > 5:
                recent_fitnesses = fitness_scores[-10:] if len(fitness_scores) >= 10 else fitness_scores
                if len(recent_fitnesses) > 2:
                    std_dev = np.std(recent_fitnesses)
                    if std_dev < 1e-8:
                        # Converged, stop evolution
                        break
            
            # Next generation
            population = adaptive_evolution_step(population, fitness_scores, generation, time_limit - (time.time() - start_time))
            generation += 1
        
        return best_individual if best_individual is not None else population[0]
    
    def calculate_ratio(points):
        """Calculate min/max distance ratio with robust error handling"""
        if len(points) < 2:
            return 0.0
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
            # Filter out invalid distances
            finite_distances = distances[np.isfinite(distances)]
            if len(finite_distances) == 0:
                return 0.0
            d_min = np.min(finite_distances)
            d_max = np.max(finite_distances)
            if d_max <= 0:
                return 0.0
            return d_min / d_max
        except:
            return 0.0

    # Main optimization process
    start_time = time.time()
    time_limit = 345  # seconds (leave some buffer for final steps)
    
    # Initial spherical configuration
    initial_points = fibonacci_sphere(14)
    initial_points = initial_points - np.mean(initial_points, axis=0)
    max_coord = np.max(np.abs(initial_points))
    if max_coord > 0:
        initial_points = initial_points / (2 * max_coord) + 0.5
    
    # Phase 1: Evolutionary optimization
    evolved_points = optimize_with_evolution(time_limit)
    
    # Phase 2: Local refinement with Basin-hopping
    try:
        # Prepare objective function for basin-hopping
        def bh_objective(x):
            points = x.reshape(-1, 3)
            distances = pdist(points)
            if len(distances) == 0:
                return float('inf')
            distances = distances[np.isfinite(distances)]
            if len(distances) == 0:
                return float('inf')
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max <= 0:
                return float('inf')
            return -d_min / d_max  # Negative because we want to maximize

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(0, 1) for _ in range(42)]}
        result_bh = basinhopping(
            bh_objective,
            evolved_points.flatten(),
            niter=20,
            T=1.0,
            stepsize=0.05,
            minimizer_kwargs=minimizer_kwargs,
            seed=42,
            callback=lambda x, f, accepted: time.time() - start_time > time_limit - 10
        )
        
        if result_bh.success:
            refined_points = result_bh.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            
            # Check if refinement improved the solution
            orig_ratio = calculate_ratio(evolved_points)
            refined_ratio = calculate_ratio(refined_points)
            
            if refined_ratio > orig_ratio:
                optimized_points = refined_points
            else:
                optimized_points = evolved_points
        else:
            optimized_points = evolved_points

    except Exception as e:
        warnings.warn(f"Basin-hopping failed: {e}")
        optimized_points = evolved_points
    
    # Final validation
    final_ratio = calculate_ratio(optimized_points)
    if final_ratio <= 0:
        # Fallback to initial points if something went wrong
        return initial_points
    
    return optimized_points

# EVOLVE-BLOCK-END