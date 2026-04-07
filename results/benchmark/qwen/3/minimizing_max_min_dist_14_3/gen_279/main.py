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
        """Generate points on sphere using Fibonacci spiral method"""
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
    
    def cartesian_to_spherical(x, y, z):
        """Convert cartesian to spherical coordinates (r, theta, phi)"""
        r = np.sqrt(x*x + y*y + z*z)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r) if r != 0 else 0
        return r, theta, phi
    
    def spherical_to_cartesian(r, theta, phi):
        """Convert spherical to cartesian coordinates (x, y, z)"""
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z
    
    def evaluate_distance_ratio(points):
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
    
    def distance_guided_mutation(spherical_points, current_ratio, mutation_rate=0.05):
        """Mutation strategy guided by current distance statistics"""
        # Calculate current distance ratio statistics
        mutated = spherical_points.copy()
        
        # Adjust mutation rate based on current ratio
        if current_ratio < 0.3:
            # If ratio is low (bad), increase mutation rate to explore more
            mutation_rate *= 2.0
        elif current_ratio > 0.45:
            # If ratio is high (good), decrease mutation rate to refine
            mutation_rate *= 0.7
        
        # Apply mutation to each point
        for i in range(len(mutated)):
            # Only mutate if random condition is met
            if np.random.random() < mutation_rate:
                # Mutate each spherical coordinate
                # r (radius): small perturbation, but keep within reasonable bounds
                r_delta = np.random.normal(0, 0.02) 
                mutated[i][0] = np.clip(mutated[i][0] + r_delta, 0.1, 1.5)
                
                # theta (azimuthal angle): random change
                theta_delta = np.random.normal(0, 0.2)
                mutated[i][1] = (mutated[i][1] + theta_delta) % (2 * np.pi)
                
                # phi (polar angle): random change, bounded away from poles
                phi_delta = np.random.normal(0, 0.2)
                mutated[i][2] = np.clip(mutated[i][2] + phi_delta, 0.01, np.pi - 0.01)
        
        return mutated
    
    def adaptive_evolution_step(population, fitness_scores, generation, time_limit):
        """Perform one adaptive evolution step using spherical coordinates"""
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
                
            # Select parents (from top performers)
            parent1 = population[np.random.choice(top_30_percent)]
            parent2 = population[np.random.choice(top_30_percent)]
            
            # Crossover: blend spherical coordinates
            child = np.zeros_like(parent1)
            mask = np.random.random(14) < 0.5
            child[mask] = parent1[mask]
            child[~mask] = parent2[~mask]
            
            # Mutation with distance-guided strategy
            # We estimate the current ratio to guide mutation
            parent_points_cart = np.array([spherical_to_cartesian(*point) for point in parent1])
            estimated_ratio = evaluate_distance_ratio(parent_points_cart)
            child = distance_guided_mutation(child, estimated_ratio)
            
            new_population.append(child)
        
        return new_population[:len(population)]
    
    def evaluate_spherical_population(population):
        """Evaluate fitness of entire population in spherical coordinates"""
        fitness_scores = []
        for spherical_points in population:
            # Convert spherical to cartesian
            points = []
            for r, theta, phi in spherical_points:
                x, y, z = spherical_to_cartesian(r, theta, phi)
                points.append([x, y, z])
            points = np.array(points)
            
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
    
    def generate_initial_spherical_population(size=20):
        """Generate diverse initial population in spherical coordinates"""
        population = []
        
        # Base spherical distribution
        base_points = fibonacci_sphere(14)
        # Normalize to unit sphere
        norms = np.linalg.norm(base_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        base_points = base_points / norms
        
        # Convert to spherical coordinates
        base_spherical = []
        for x, y, z in base_points:
            r, theta, phi = cartesian_to_spherical(x, y, z)
            base_spherical.append([r, theta, phi])
        base_spherical = np.array(base_spherical)
        
        # Create diverse variants in spherical space
        for i in range(size):
            np.random.seed(i * 42)
            spherical_points = base_spherical.copy()
            
            # Add different types of perturbations in spherical space
            if i % 3 == 0:
                # Small random jitter to radius
                spherical_points[:, 0] += np.random.normal(0, 0.01, 14)
            elif i % 3 == 1:
                # More aggressive perturbation to angles
                spherical_points[:, 1] += np.random.normal(0, 0.1, 14)
                spherical_points[:, 2] += np.random.normal(0, 0.1, 14)
            else:
                # Rotate slightly in spherical space
                rotation = R.from_euler('xyz', np.random.uniform(0, 2*np.pi, 3)).as_matrix()
                for j in range(14):
                    x, y, z = cartesian_to_spherical(*base_points[j])
                    # Apply rotation to direction vector
                    dir_vec = np.array([x, y, z])
                    rotated_vec = rotation @ dir_vec
                    # Convert back to spherical
                    r_new, theta_new, phi_new = cartesian_to_spherical(rotated_vec[0], rotated_vec[1], rotated_vec[2])
                    spherical_points[j] = [r_new, theta_new, phi_new]
            
            # Keep radii positive and angles within bounds
            spherical_points[:, 0] = np.clip(spherical_points[:, 0], 0.1, 1.5)
            spherical_points[:, 1] = spherical_points[:, 1] % (2 * np.pi)
            spherical_points[:, 2] = np.clip(spherical_points[:, 2], 0.01, np.pi - 0.01)
            
            population.append(spherical_points)
        
        return population
    
    def optimize_with_evolution(time_limit):
        """Use evolutionary algorithm to optimize in spherical coordinates"""
        start_time = time.time()
        
        # Generate initial population in spherical coordinates
        population = generate_initial_spherical_population(25)
        
        # Evolution parameters
        max_generations = 50
        generation = 0
        
        # Store best solution so far
        best_fitness = float('inf')
        best_individual = None
        
        while generation < max_generations and (time.time() - start_time) < time_limit - 15:
            # Evaluate population
            fitness_scores = evaluate_spherical_population(population)
            
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
    
    def convert_spherical_to_cartesian(spherical_points):
        """Convert spherical coordinates to cartesian points"""
        points = []
        for r, theta, phi in spherical_points:
            x, y, z = spherical_to_cartesian(r, theta, phi)
            points.append([x, y, z])
        return np.array(points)
    
    def refine_in_cartesian(spherical_points, time_limit):
        """Refine solution in cartesian space using basin-hopping and L-BFGS-B"""
        start_time = time.time()
        
        # Convert to cartesian for refinement
        points = convert_spherical_to_cartesian(spherical_points)
        
        # Prepare objective function for refinement
        def objective(x):
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
        
        # Basin-hopping refinement
        try:
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": [(0, 1) for _ in range(42)]}
            result_bh = basinhopping(
                objective,
                points.flatten(),
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
                return refined_points
        except Exception as e:
            warnings.warn(f"Basin-hopping failed: {e}")
        
        # Fall back to L-BFGS-B with tight tolerances
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(42)],
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
                callback=lambda x: time.time() - start_time > time_limit - 10
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)
                return refined_points
        except Exception as e:
            warnings.warn(f"L-BFGS-B refinement failed: {e}")
        
        return points
    
    # Main optimization process
    start_time = time.time()
    time_limit = 345  # seconds (leave some buffer for final steps)
    
    # Phase 1: Evolutionary optimization in spherical coordinates
    try:
        evolved_spherical = optimize_with_evolution(time_limit)
    except Exception as e:
        warnings.warn(f"Evolutionary optimization failed: {e}")
        # Fallback to basic initialization
        initial_points = fibonacci_sphere(14)
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / (2 * max_coord) + 0.5
        return initial_points
    
    # Phase 2: Refinement in cartesian space
    try:
        refined_points = refine_in_cartesian(evolved_spherical, time_limit)
    except Exception as e:
        warnings.warn(f"Refinement failed: {e}")
        # Fallback to evolved solution
        refined_points = convert_spherical_to_cartesian(evolved_spherical)
    
    # Final validation
    final_ratio = evaluate_distance_ratio(refined_points)
    if final_ratio <= 0:
        # Fallback to initial points if something went wrong
        initial_points = fibonacci_sphere(14)
        initial_points = initial_points - np.mean(initial_points, axis=0)
        max_coord = np.max(np.abs(initial_points))
        if max_coord > 0:
            initial_points = initial_points / (2 * max_coord) + 0.5
        return initial_points
    
    return refined_points

# EVOLVE-BLOCK-END