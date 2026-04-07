# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

def adaptive_evolutionary_optimizer() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses an adaptive evolutionary approach with multiple initialization strategies and progressive refinement.
    """

    def min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0

    def spherical_voronoi_quality(sphere_points):
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

    def normalize_to_sphere(points):
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def project_to_cube(points):
        """Project points from sphere to unit cube [0,1]^3."""
        # Normalize to unit sphere first
        sphere_points = normalize_to_sphere(points)
        # Map to cube [0,1]^3
        return (sphere_points + 1) / 2

    def spherical_evolution_operator(parents, mutation_rate=0.1, generation=0):
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
            child = normalize_to_sphere(child)
            children.append(child)
        return children

    def evaluate_individual(individual, use_quality=True):
        """Evaluate individual using combined fitness function."""
        # Project to cube for distance calculations
        cube_points = project_to_cube(individual)
        ratio = min_max_ratio(cube_points)

        if use_quality:
            voronoi_quality = spherical_voronoi_quality(individual)
            # Combined fitness: prioritize min/max ratio but reward good distribution
            # Weight quality contribution with generation factor
            return ratio + 0.15 * voronoi_quality
        else:
            return ratio

    def generate_initial_population(pop_size=30):
        """Generate diverse initial population using multiple strategies."""
        population = []
        
        # Strategy 1: Fibonacci sphere
        points_fib = fibonacci_sphere(14)
        population.append(points_fib)
        
        # Strategy 2: Random points on sphere
        np.random.seed(42)
        points_random = np.random.randn(14, 3)
        points_random = normalize_to_sphere(points_random)
        population.append(points_random)
        
        # Strategy 3: Perturbed Fibonacci
        points_perturbed = points_fib + np.random.normal(0, 0.05, (14, 3))
        points_perturbed = normalize_to_sphere(points_perturbed)
        population.append(points_perturbed)
        
        # Strategy 4: Two-layered approach
        layer1 = np.random.randn(7, 3)
        layer2 = np.random.randn(7, 3) * 0.5  # Smaller spread for second layer
        layer1 = normalize_to_sphere(layer1)
        layer2 = normalize_to_sphere(layer2)
        points_two_layer = np.vstack([layer1, layer2])
        population.append(points_two_layer)
        
        # Strategy 5: Enhanced Fibonacci (improved uniformity)
        enhanced_fib = create_enhanced_fibonacci_placement()
        population.append(enhanced_fib)
        
        # Strategy 6: Random points on sphere (different seed)
        np.random.seed(123)
        points_random2 = np.random.randn(14, 3)
        points_random2 = normalize_to_sphere(points_random2)
        population.append(points_random2)
        
        # Strategy 7: Clustered initialization
        clustered = create_clustered_initialization()
        population.append(clustered)
        
        # Strategy 8: Spiral-based distribution
        spiral_points = create_spiral_initialization()
        population.append(spiral_points)
        
        # Fill remaining slots with random spherical points
        while len(population) < pop_size:
            np.random.seed(len(population) + 42)
            points = np.random.randn(14, 3)
            points = normalize_to_sphere(points)
            population.append(points)
            
        return population[:pop_size]

    def fibonacci_sphere(n):
        """Generate n points evenly distributed on a unit sphere."""
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

    def create_enhanced_fibonacci_placement():
        """Create enhanced Fibonacci sphere placement with better uniformity."""
        # Generate points using Fibonacci-like distribution
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(14):
            y = 1 - (i / float(14 - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        initial_points = np.array(points)

        # Improve distribution by applying multiple small perturbations with better control
        np.random.seed(42)
        for _ in range(25):  # More perturbations for better uniformity
            # Add small random perturbations but with decreasing magnitude
            perturbation = np.random.normal(0, 0.01, (14, 3)) * (1.0 - _ * 0.01)
            initial_points += perturbation

            # Project back to sphere surface
            initial_points = normalize_to_sphere(initial_points)

        # Normalize to unit sphere and scale to unit cube [0,1]^3
        initial_points = (initial_points + 1) / 2

        return initial_points

    def create_clustered_initialization():
        """Create initialization with clustered points for diversity."""
        points = []
        # Create clusters in different regions of the sphere
        for cluster in range(4):
            np.random.seed(cluster * 10 + 42)
            center = np.random.randn(3)
            center = normalize_to_sphere(center.reshape(1, 3)).flatten()
            
            # Generate points around this center
            for i in range(4):
                # Small random displacement from center
                displacement = np.random.normal(0, 0.1, 3)
                point = center + displacement
                point = normalize_to_sphere(point.reshape(1, 3)).flatten()
                points.append(point)
        
        # Fill remaining spots
        while len(points) < 14:
            point = np.random.randn(3)
            point = normalize_to_sphere(point.reshape(1, 3)).flatten()
            points.append(point)
            
        return np.array(points[:14])

    def create_spiral_initialization():
        """Create points using spiral arrangement for better coverage."""
        points = []
        # Create spirals along different axes
        for i in range(14):
            # Spiral along z-axis
            t = i * 0.5
            r = 0.5 + 0.5 * np.sin(t * 0.5)
            theta = t * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 1 - 2 * (i / 13.0) if i < 14 else 0
            points.append([x, y, z])
            
        # Normalize to unit sphere
        points = np.array(points)
        points = normalize_to_sphere(points)
        return points

    def tournament_selection(population, fitnesses, tournament_size=4):
        """Tournament selection for evolutionary algorithm with adaptive strength."""
        selected = []
        for _ in range(len(population)):
            # Tournament selection with variable size
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_index].copy())
        return selected

    def run_evolutionary_search(max_generations=80):
        """Run evolutionary search with adaptive parameters and multiple restarts."""
        # Initialize population
        population = generate_initial_population(pop_size=30)
        best_individual = None
        best_fitness = -np.inf
        improvement_history = []
        stagnation_count = 0
        max_stagnation = 10

        for generation in range(max_generations):
            # Evaluate population
            fitnesses = [evaluate_individual(ind, use_quality=True) for ind in population]

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
            selected = tournament_selection(population, fitnesses, tournament_size=4)

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
                child1 = spherical_evolution_operator([child1], mutation_rate=0.1, generation=generation)[0]
                child2 = spherical_evolution_operator([child2], mutation_rate=0.1, generation=generation)[0]

                new_population.extend([child1, child2])

            # Trim to original population size
            population = new_population[:len(population)]

        return best_individual

    def local_refinement(points, max_iter=200):
        """Local refinement using L-BFGS-B with adaptive tolerances."""
        try:
            # Convert to flat array for optimization
            x0 = points.flatten()

            def obj(x):
                points_refined = x.reshape((14, 3))
                # Ensure points are on sphere
                points_refined = normalize_to_sphere(points_refined)
                cube_points = project_to_cube(points_refined)
                distances = pdist(cube_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    # Return negative for minimization (we want to maximize ratio)
                    return -min_dist / max_dist
                else:
                    return 0

            # Use adaptive tolerances based on optimization progress
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(None, None)] * 42,  # No bounds for internal optimization
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': max_iter},
                tol=1e-12
            )

            refined_points = result.x.reshape((14, 3))
            refined_points = normalize_to_sphere(refined_points)
            return refined_points

        except Exception:
            return points

    def adaptive_global_optimization(initial_points, max_restarts=5):
        """Perform adaptive differential evolution optimization with multiple restarts."""
        best_points = initial_points.copy()
        best_ratio = min_max_ratio(project_to_cube(best_points))
        
        # Track convergence history for adaptive population sizing
        convergence_history = []
        stagnation_threshold = 5
        max_stagnation = 10
        improvement_counter = 0

        for restart in range(max_restarts):
            try:
                # Flatten initial points for optimization
                x0 = initial_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(14 * 3)]

                # Dynamic parameters based on restart count and convergence behavior
                base_popsize = 15 + restart * 5  # Increase population size with restarts

                # Adaptive population sizing based on recent convergence
                if len(convergence_history) >= stagnation_threshold:
                    recent_changes = convergence_history[-stagnation_threshold:]
                    # Check if there's consistent improvement
                    if len(recent_changes) >= 2:
                        improvements = [recent_changes[i] - recent_changes[i-1] 
                                       for i in range(1, len(recent_changes))]
                        avg_improvement = np.mean(improvements) if improvements else 0
                        
                        # If improvement is minimal, increase population size to enhance exploration
                        if avg_improvement < 1e-6:
                            popsize = min(40, base_popsize + 15)
                        elif avg_improvement > 1e-4:
                            # If making good progress, reduce population slightly  
                            popsize = max(10, base_popsize - 5)
                        else:
                            popsize = base_popsize
                    else:
                        popsize = base_popsize
                else:
                    popsize = base_popsize

                # Progressive increase in DE parameters
                maxiter = 150 + restart * 50  # More iterations with restarts
                mutation_rate = min(0.9, 0.5 + restart * 0.1)  # Increase mutation rate
                recomb_rate = min(0.9, 0.7 + restart * 0.05)   # Increase recombination rate

                result = differential_evolution(
                    lambda x: -min_max_ratio(project_to_cube(x.reshape((14, 3)))),  # Negative for maximization
                    bounds,
                    seed=42 + restart,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-9,
                    mutation=(mutation_rate, 1.0),
                    recombination=recomb_rate,
                    disp=False
                )

                # Extract optimized points
                optimized_points = result.x.reshape((14, 3))
                optimized_points = np.clip(optimized_points, 0, 1)  # Ensure bounds

                # Calculate final ratio
                final_ratio = min_max_ratio(optimized_points)

                # Update best if improved
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()
                    improvement_counter = 0  # Reset counter on improvement
                else:
                    improvement_counter += 1

                # Track convergence for adaptive sizing
                convergence_history.append(final_ratio)
                if len(convergence_history) > max_stagnation:
                    convergence_history.pop(0)

                # Early stopping if no improvement for consecutive restarts
                if improvement_counter >= 3:
                    break

            except Exception:
                continue  # Continue if optimization fails

        return best_points

    # Main evolutionary process
    # Phase 1: Evolutionary search on spherical space with more generations
    best_spherical_points = run_evolutionary_search(max_generations=80)

    # Phase 2: Local refinement with increased iterations
    refined_points = local_refinement(best_spherical_points, max_iter=200)

    # Phase 3: Final adaptive global optimization with more restarts
    final_points = adaptive_global_optimization(refined_points, max_restarts=5)

    # Convert to cube coordinates and ensure valid bounds
    final_points = project_to_cube(final_points)
    final_points = np.clip(final_points, 0, 1)

    return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    return adaptive_evolutionary_optimizer()

# EVOLVE-BLOCK-END