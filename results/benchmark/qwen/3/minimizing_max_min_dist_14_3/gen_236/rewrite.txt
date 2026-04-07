# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

class HybridEvolutionaryOptimizer:
    """Hybrid optimizer combining evolutionary and deterministic approaches for 14-point 3D distribution."""
    
    def __init__(self, num_points=14):
        self.num_points = num_points
        self.best_ratio = 0
        self.best_points = None
        self.improvement_streak = 0
        self.max_improvement_streak = 8
        
    def fibonacci_sphere(self, n):
        """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def latin_hypercube_sampling(self, n, d, seed=42):
        """Generate n points using Latin Hypercube Sampling in d dimensions."""
        np.random.seed(seed)
        samples = np.zeros((n, d))

        for i in range(d):
            # Generate random permutation for each dimension
            perm = np.random.permutation(n)
            samples[:, i] = perm

        # Normalize to [0, 1]
        samples = samples / (n - 1)

        return samples
    
    def normalize_to_sphere(self, points):
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def project_to_cube(self, points):
        """Project points from sphere to unit cube [0,1]^3."""
        # Normalize to unit sphere first
        sphere_points = self.normalize_to_sphere(points)
        # Map to cube [0,1]^3
        return (sphere_points + 1) / 2
    
    def spherical_voronoi_quality(self, sphere_points):
        """Calculate quality based on Voronoi cell areas on sphere."""
        if len(sphere_points) < 2:
            return 0
        try:
            sv = SphericalVoronoi(sphere_points)
            cell_areas = sv.calculate_areas()
            if len(cell_areas) > 0:
                mean_area = np.mean(cell_areas)
                if mean_area > 0:
                    variance = np.var(cell_areas)
                    # Return inverse variance (higher is better) - more uniform distribution
                    return 1.0 / (1.0 + variance / mean_area**2)
        except Exception:
            pass
        return 0
    
    def min_max_ratio(self, points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0
    
    def spherical_evolution_operator(self, parents, mutation_rate=0.1, generation=0):
        """Specialized evolution operator working in spherical space with adaptive mutation."""
        children = []
        
        # Adapt mutation rate based on generation (less mutation in later generations)
        adaptive_mutation = mutation_rate * (1.0 - generation * 0.05)
        adaptive_mutation = max(0.01, adaptive_mutation)
        
        for parent in parents:
            # Create offspring through spherical mutation
            child = parent.copy()
            
            # Apply spherical mutation - perturb along tangent plane then reproject
            for i in range(len(child)):
                if np.random.random() < adaptive_mutation:
                    # Generate random displacement in tangent plane
                    tangent_displacement = np.random.normal(0, 0.03, 3)
                    
                    # Ensure we maintain sphere constraint
                    point = child[i]
                    normal = point / np.linalg.norm(point)
                    # Remove component parallel to normal (tangent plane)
                    tangent_displacement = tangent_displacement - np.dot(tangent_displacement, normal) * normal
                    
                    # Apply displacement
                    child[i] = point + tangent_displacement
                    
            # Re-project to unit sphere and handle numerical errors
            child = self.normalize_to_sphere(child)
            children.append(child)
        return children
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """Tournament selection for evolutionary algorithm with adaptive strength."""
        selected = []
        for _ in range(len(population)):
            # Tournament selection with variable size
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_index].copy())
        return selected
    
    def evaluate_individual(self, individual, use_quality=True):
        """Evaluate individual using combined fitness function."""
        # Project to cube for distance calculations
        cube_points = self.project_to_cube(individual)
        ratio = self.min_max_ratio(cube_points)
        
        if use_quality:
            voronoi_quality = self.spherical_voronoi_quality(individual)
            # Combined fitness: prioritize min/max ratio but reward good distribution
            # Weight quality contribution with generation factor
            return ratio + 0.15 * voronoi_quality
        else:
            return ratio

    def generate_initial_population(self, pop_size=25):
        """Generate diverse initial population using multiple strategies."""
        population = []
        
        # Strategy 1: Fibonacci sphere
        points_fib = self.fibonacci_sphere(14)
        population.append(points_fib)
        
        # Strategy 2: Random points on sphere
        np.random.seed(42)
        points_random = np.random.randn(14, 3)
        points_random = self.normalize_to_sphere(points_random)
        population.append(points_random)
        
        # Strategy 3: Perturbed Fibonacci
        points_perturbed = points_fib + np.random.normal(0, 0.05, (14, 3))
        points_perturbed = self.normalize_to_sphere(points_perturbed)
        population.append(points_perturbed)
        
        # Strategy 4: Two-layered approach
        layer1 = np.random.randn(7, 3)
        layer2 = np.random.randn(7, 3) * 0.5  # Smaller spread for second layer
        layer1 = self.normalize_to_sphere(layer1)
        layer2 = self.normalize_to_sphere(layer2)
        points_two_layer = np.vstack([layer1, layer2])
        population.append(points_two_layer)
        
        # Strategy 5: Enhanced Fibonacci (improved uniformity)
        enhanced_fib = self.create_enhanced_fibonacci_placement()
        population.append(enhanced_fib)
        
        # Fill remaining slots with random spherical points
        while len(population) < pop_size:
            np.random.seed(len(population) + 42)
            points = np.random.randn(14, 3)
            points = self.normalize_to_sphere(points)
            population.append(points)
            
        return population[:pop_size]

    def create_enhanced_fibonacci_placement(self):
        """Create enhanced Fibonacci sphere placement with better uniformity."""
        # Generate points using Fibonacci-like distribution
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(self.num_points):
            y = 1 - (i / float(self.num_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        initial_points = np.array(points)

        # Improve distribution by applying multiple small perturbations with better control
        np.random.seed(42)
        for _ in range(20):  # More perturbations for better uniformity
            # Add small random perturbations but with decreasing magnitude
            perturbation = np.random.normal(0, 0.01, (self.num_points, 3)) * (1.0 - _ * 0.01)
            initial_points += perturbation

            # Project back to sphere surface
            initial_points = self.normalize_to_sphere(initial_points)

        # Normalize to unit sphere and scale to unit cube [0,1]^3
        initial_points = (initial_points + 1) / 2

        return initial_points

    def run_evolutionary_search(self, max_generations=60):
        """Run evolutionary search with adaptive parameters and multiple restarts."""
        # Initialize population
        population = self.generate_initial_population()
        best_individual = None
        best_fitness = -np.inf
        improvement_history = []
        stagnation_count = 0
        max_stagnation = 10
        
        for generation in range(max_generations):
            # Evaluate population
            fitnesses = [self.evaluate_individual(ind, use_quality=True) for ind in population]
            
            # Track best
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                improvement_history.append(best_fitness)
                stagnation_count = 0
            else:
                stagnation_count += 1
                improvement_history.append(best_fitness)
                
            # Early stopping if no improvement
            if stagnation_count > max_stagnation:
                break
            
            # Selection
            selected = self.tournament_selection(population, fitnesses)
            
            # Create new population through crossover and mutation
            new_population = []
            
            # Elitism: keep best individual
            if best_individual is not None:
                new_population.append(best_individual)
                
            # Generate offspring
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
                
                # Crossover (uniform)
                child1 = parent1.copy()
                child2 = parent2.copy()
                mask = np.random.random(14) < 0.5
                child1[mask] = parent2[mask]
                child2[mask] = parent1[mask]
                
                # Mutation with adaptive rate
                child1 = self.spherical_evolution_operator([child1], mutation_rate=0.1, generation=generation)[0]
                child2 = self.spherical_evolution_operator([child2], mutation_rate=0.1, generation=generation)[0]
                
                new_population.extend([child1, child2])
            
            # Trim to original population size
            population = new_population[:len(population)]
            
        return best_individual

    def local_refinement(self, points, max_iter=150):
        """Local refinement using L-BFGS-B with adaptive tolerances."""
        try:
            # Convert to flat array for optimization
            x0 = points.flatten()
            
            def obj(x):
                points_refined = x.reshape((14, 3))
                # Ensure points are on sphere
                points_refined = self.normalize_to_sphere(points_refined)
                cube_points = self.project_to_cube(points_refined)
                distances = pdist(cube_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    # Return negative for minimization (we want to maximize ratio)
                    return -min_dist / max_dist
                else:
                    return 0
            
            # Use adaptive tolerances based on optimization progress
            current_tolerance = 1e-12
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(None, None)] * 42,  # No bounds for internal optimization
                options={'ftol': current_tolerance, 'gtol': current_tolerance, 'maxiter': max_iter},
                tol=current_tolerance
            )
            
            refined_points = result.x.reshape((14, 3))
            refined_points = self.normalize_to_sphere(refined_points)
            return refined_points
            
        except Exception:
            return points

    def adaptive_global_optimization(self, initial_points, max_restarts=3):
        """Perform adaptive differential evolution optimization with multiple restarts."""
        best_points = initial_points.copy()
        best_ratio = self.min_max_ratio(self.project_to_cube(best_points))
        
        for restart in range(max_restarts):
            try:
                # Flatten initial points for optimization
                x0 = initial_points.flatten()
                
                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(self.num_points * 3)]
                
                # Dynamic parameters based on restart count
                popsize = 20 + restart * 5  # Increase population size
                maxiter = 100 + restart * 50  # More iterations
                
                result = differential_evolution(
                    lambda x: -self.min_max_ratio(self.project_to_cube(x.reshape((self.num_points, 3)))),  # Negative for maximization
                    bounds,
                    seed=42 + restart,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-9,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False
                )

                # Extract optimized points
                optimized_points = result.x.reshape((self.num_points, 3))
                optimized_points = np.clip(optimized_points, 0, 1)  # Ensure bounds
                
                # Calculate final ratio
                final_ratio = self.min_max_ratio(optimized_points)

                # Update best if improved
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()

            except Exception:
                continue  # Continue if optimization fails
                
        return best_points

    def optimize(self):
        """Main optimization routine."""
        # Phase 1: Try evolutionary approach first with increased generations
        try:
            evolved_points = self.run_evolutionary_search(max_generations=60)
            evolved_ratio = self.min_max_ratio(self.project_to_cube(evolved_points))
            
            if evolved_ratio > self.best_ratio:
                self.best_ratio = evolved_ratio
                self.best_points = evolved_points.copy()
        except Exception as e:
            pass
        
        # Phase 2: If evolutionary wasn't successful or gave poor results, 
        # fallback to the more robust deterministic approach  
        if self.best_points is None or self.best_ratio < 0.1:
            # Multiple initialization strategies (from original optimizer)
            strategies = []
            
            # Strategy 1: Enhanced Fibonacci sphere scaled to unit cube
            enhanced_fib = self.create_enhanced_fibonacci_placement()
            strategies.append(("enhanced_fibonacci", enhanced_fib))
            
            # Strategy 2: Standard Fibonacci sphere scaled to unit cube
            standard_fib = (self.fibonacci_sphere(self.num_points) + 1) / 2
            strategies.append(("standard_fibonacci", standard_fib))

            # Strategy 3: Latin Hypercube Sampling
            lhs_points = self.latin_hypercube_sampling(self.num_points, 3, seed=42)
            strategies.append(("latin_hypercube", lhs_points))

            # Strategy 4: Random initialization
            random_points = np.random.rand(self.num_points, 3)
            strategies.append(("random", random_points))

            # Strategy 5: Perturbed random points for diversity
            perturbed_random = random_points + np.random.normal(0, 0.02, (self.num_points, 3))
            perturbed_random = np.clip(perturbed_random, 0, 1)
            strategies.append(("perturbed_random", perturbed_random))

            # Strategy 6: Two-layered approach - spread out initial points
            layer1 = np.random.rand(self.num_points//2, 3)
            layer2 = np.random.rand(self.num_points//2, 3) + 0.5  # Offset to second half
            two_layer = np.vstack([layer1, layer2])
            strategies.append(("two_layer", two_layer))
            
            # Select best initialization
            best_ratio = -np.inf
            best_points = None
            
            for name, points in strategies:
                ratio = self.min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
            
            if best_points is not None:
                self.best_ratio = best_ratio
                self.best_points = best_points.copy()
        
        # Phase 3: Global optimization with differential evolution (as in original optimizer)
        if self.best_points is not None:
            initial_points = self.best_points.copy()
            
            # Multiple restart rounds for better exploration
            for restart in range(3):  # Increase to 3 restart rounds for better exploration
                # Global optimization
                optimized_points, final_ratio = self.global_optimization_phase(initial_points, restart)

                # Store best result
                if final_ratio > self.best_ratio:
                    self.best_ratio = final_ratio
                    self.best_points = optimized_points.copy()
                    self.improvement_streak = 0  # Reset streak
                else:
                    self.improvement_streak += 1

                # Early stopping if no improvement for too many attempts
                if self.improvement_streak >= self.max_improvement_streak:
                    break

                # Use current result as starting point for next restart
                initial_points = optimized_points

        # Phase 4: Final local refinement
        if self.best_points is not None:
            refined_points = self.local_refinement(self.best_points)
            refined_ratio = self.min_max_ratio(self.project_to_cube(refined_points))

            # Update if improved
            if refined_ratio > self.best_ratio:
                self.best_points = refined_points
                
        # Ensure valid bounds and return
        if self.best_points is not None:
            final_points = self.project_to_cube(self.best_points)
            final_points = np.clip(final_points, 0, 1)
            return final_points
        else:
            # Fallback to enhanced Fibonacci if all methods fail
            fallback_points = self.create_enhanced_fibonacci_placement()
            return (fallback_points + 1) / 2

    def global_optimization_phase(self, initial_points, restart_round):
        """Perform global differential evolution optimization."""
        # Flatten initial points for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(self.num_points * 3)]

        # Use adaptive population size based on restart round and convergence behavior
        base_popsize = 15 + restart_round * 5  # Increase population size with restarts
        maxiter = 100 + restart_round * 50  # More iterations with restarts

        # Dynamic population sizing based on problem difficulty
        popsize = base_popsize
        if restart_round > 0:
            # If we're on a second restart and didn't improve much, increase population
            if hasattr(self, 'prev_ratio') and self.prev_ratio > 0:
                improvement = (self.best_ratio - self.prev_ratio) / self.prev_ratio if self.prev_ratio > 0 else 0
                if improvement < 0.01:  # Less than 1% improvement
                    popsize = min(40, base_popsize + 10)  # Increase population size

        # Adjust mutation and recombination rates based on restart round for better exploration
        mutation_rates = [(0.5, 1.0), (0.7, 1.0), (0.8, 1.0)]
        mutation_strategy = mutation_rates[min(restart_round, len(mutation_rates)-1)]

        recombination_rates = [0.7, 0.8, 0.9]
        recombination_rate = recombination_rates[min(restart_round, len(recombination_rates)-1)]

        try:
            result = differential_evolution(
                lambda x: self.adaptive_penalty_objective(x, iteration=restart_round),
                bounds,
                seed=42 + restart_round * 10,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                mutation=mutation_strategy,
                recombination=recombination_rate,
                disp=False
            )

            # Extract optimized points
            optimized_points = result.x.reshape((self.num_points, 3))

            # Calculate final ratio
            final_ratio = self.min_max_ratio(optimized_points)

            # Store previous ratio for next iteration
            self.prev_ratio = final_ratio

            return optimized_points, final_ratio

        except Exception as e:
            return initial_points, self.min_max_ratio(initial_points)

    def adaptive_penalty_objective(self, x_flat, penalty_weight=1e6, iteration=0):
        """Objective function with adaptive penalty for out-of-bounds points."""
        # Reshape flat array back to points
        points = x_flat.reshape((self.num_points, 3))

        # Apply penalty for constraint violations
        penalty = 0
        for i in range(self.num_points):
            for j in range(3):  # x, y, z coordinates
                if points[i, j] < 0:
                    penalty += penalty_weight * (0 - points[i, j])**2 * (1 + iteration * 0.1)
                elif points[i, j] > 1:
                    penalty += penalty_weight * (points[i, j] - 1)**2 * (1 + iteration * 0.1)

        # Calculate min/max ratio
        ratio = self.min_max_ratio(points)

        # Return value to minimize (negative ratio + penalty)
        return -ratio + penalty

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    optimizer = HybridEvolutionaryOptimizer(num_points=14)
    return optimizer.optimize()

# EVOLVE-BLOCK-END