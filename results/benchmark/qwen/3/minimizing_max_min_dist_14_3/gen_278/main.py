# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

def fibonacci_sphere(n):
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

def min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    d_min = np.min(distances)
    d_max = np.max(distances)
    return d_min / d_max if d_max > 0 else 0

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

def adaptive_penalty_objective(x_flat, penalty_weight=1e6, iteration=0):
    """Objective function with adaptive penalty for out-of-bounds points."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Apply penalty for constraint violations
    penalty = 0
    for i in range(n_points):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2 * (1 + iteration * 0.1)
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2 * (1 + iteration * 0.1)

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Return value to minimize (negative ratio + penalty)
    return -ratio + penalty

def adaptive_differential_evolution(x0, bounds, seed, maxiter, popsize, tol, mutation, recombination):
    """
    Adaptive differential evolution that increases population size when convergence stalls
    """
    # Track convergence
    prev_best = np.inf
    convergence_stall_count = 0
    max_stall_count = 10

    # Initial run
    result = differential_evolution(
        adaptive_penalty_objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        disp=False
    )

    # Monitor for convergence
    current_best = -result.fun
    if abs(prev_best - current_best) < 1e-6:
        convergence_stall_count += 1
    else:
        convergence_stall_count = 0
    prev_best = current_best

    # If convergence stalled, try with larger population
    if convergence_stall_count >= max_stall_count:
        larger_popsize = min(50, popsize + 10)  # Increase population size
        try:
            result = differential_evolution(
                adaptive_penalty_objective,
                bounds,
                seed=seed,
                maxiter=maxiter,
                popsize=larger_popsize,
                tol=tol,
                mutation=mutation,
                recombination=recombination,
                disp=False
            )
        except:
            pass  # Fall back to previous result

    return result

def enhanced_fibonacci_with_voronoi(n):
    """Enhanced Fibonacci sphere with Voronoi quality consideration"""
    points = fibonacci_sphere(n)
    
    # Add perturbations to break symmetries and improve distribution
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, points.shape)
    points += noise
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Scale to make better use of volume
    points *= 0.9
    
    return points

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

def generate_initial_population(pop_size=25):
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
    
    # Fill remaining slots with random spherical points
    while len(population) < pop_size:
        np.random.seed(len(population) + 42)
        points = np.random.randn(14, 3)
        points = normalize_to_sphere(points)
        population.append(points)
        
    return population[:pop_size]

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
    for _ in range(20):  # More perturbations for better uniformity
        # Add small random perturbations but with decreasing magnitude
        perturbation = np.random.normal(0, 0.01, (14, 3)) * (1.0 - _ * 0.01)
        initial_points += perturbation

        # Project back to sphere surface
        initial_points = normalize_to_sphere(initial_points)

    # Normalize to unit sphere and scale to unit cube [0,1]^3
    initial_points = (initial_points + 1) / 2

    return initial_points

def tournament_selection(population, fitnesses, tournament_size=3):
    """Tournament selection for evolutionary algorithm with adaptive strength."""
    selected = []
    for _ in range(len(population)):
        # Tournament selection with variable size
        tournament_indices = np.random.choice(len(population), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_index].copy())
    return selected

def run_evolutionary_search(max_generations=60):
    """Run evolutionary search with adaptive parameters and multiple restarts."""
    # Initialize population
    population = generate_initial_population()
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
        selected = tournament_selection(population, fitnesses)
        
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

def local_refinement(points, max_iter=150):
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
        refined_points = normalize_to_sphere(refined_points)
        return refined_points
        
    except Exception:
        return points

def adaptive_global_optimization(initial_points, max_restarts=3):
    """Perform adaptive differential evolution optimization with multiple restarts."""
    best_points = initial_points.copy()
    best_ratio = min_max_ratio(project_to_cube(best_points))
    
    for restart in range(max_restarts):
        try:
            # Flatten initial points for optimization
            x0 = initial_points.flatten()
            
            # Define bounds for each coordinate (0 to 1)
            bounds = [(0, 1) for _ in range(14 * 3)]
            
            # Dynamic parameters based on restart count
            popsize = 20 + restart * 5  # Increase population size
            maxiter = 100 + restart * 50  # More iterations
            
            result = differential_evolution(
                lambda x: -min_max_ratio(project_to_cube(x.reshape((14, 3)))),  # Negative for maximization
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
            optimized_points = result.x.reshape((14, 3))
            optimized_points = np.clip(optimized_points, 0, 1)  # Ensure bounds
            
            # Calculate final ratio
            final_ratio = min_max_ratio(optimized_points)

            # Update best if improved
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = optimized_points.copy()

        except Exception:
            continue  # Continue if optimization fails
            
    return best_points

def improved_local_refinement(points, method='L-BFGS-B'):
    """Apply improved local optimization refinement using multiple techniques"""
    try:
        # Convert to flat array for optimization
        x0 = points.flatten()
        
        def obj_func(x):
            points_refined = x.reshape((14, 3))
            ratio = min_max_ratio(points_refined)
            return -ratio  # Return negative for minimization
            
        if method == 'L-BFGS-B':
            result = minimize(
                obj_func,
                x0,
                method=method,
                bounds=[(0, 1)] * 14 * 3,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000},
                tol=1e-12
            )
            
            refined_points = result.x.reshape((14, 3))
            return np.clip(refined_points, 0, 1)
            
    except Exception:
        return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    best_ratio = 0
    best_points = None

    # Hybrid approach: Start with evolutionary search for global optimization
    # Phase 1: Evolutionary search on spherical space
    best_spherical_points = run_evolutionary_search(max_generations=60)
    
    # Phase 2: Local refinement 
    refined_points = local_refinement(best_spherical_points)
    
    # Phase 3: Final adaptive global optimization
    final_points = adaptive_global_optimization(refined_points)
    
    # Convert to cube coordinates and ensure valid bounds
    final_points_cube = project_to_cube(final_points)
    final_points_cube = np.clip(final_points_cube, 0, 1)
    
    # Also try direct optimization from multiple initializations
    initialization_strategies = [
        # Strategy 1: Enhanced Fibonacci sphere scaled to unit cube
        lambda: (fibonacci_sphere(n) + 1) / 2,

        # Strategy 2: Latin Hypercube Sampling (simulated with random sampling)
        lambda: np.random.rand(n, 3),

        # Strategy 3: Random initialization
        lambda: np.random.rand(n, 3),

        # Strategy 4: Two-layered approach
        lambda: np.vstack([
            np.random.rand(n//2, 3),
            np.random.rand(n//2, 3) + 0.5
        ]),

        # Strategy 5: Enhanced Fibonacci with Voronoi quality
        lambda: enhanced_fibonacci_with_voronoi(n),
        
        # Strategy 6: Perturbed Fibonacci
        lambda: np.clip((fibonacci_sphere(n) + 1) / 2 + np.random.normal(0, 0.03, (n, 3)), 0, 1)
    ]

    # Try different initialization strategies with multiple restarts
    for restart in range(3):  # Reduced restarts for efficiency
        for i, init_func in enumerate(initialization_strategies):
            try:
                # Generate initial points
                initial_points = init_func()

                # Flatten initial points for optimization
                x0 = initial_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(n * 3)]

                # Phase 1: Global optimization with differential evolution
                # Use adaptive parameters based on restart round
                base_popsize = 20 + restart * 5  # Increase population size with restarts
                maxiter = 100 + restart * 50  # More iterations with restarts

                # Adaptive population sizing based on convergence behavior
                result = adaptive_differential_evolution(
                    x0, bounds, seed=42 + restart * 10 + i,
                    maxiter=maxiter, popsize=base_popsize,
                    tol=1e-6, mutation=(0.5, 1.0), recombination=0.7
                )

                # Extract optimized points
                optimized_points = result.x.reshape((n, 3))
                optimized_points = np.clip(optimized_points, 0, 1)

                # Calculate final ratio
                final_ratio = min_max_ratio(optimized_points)

                # Store best result
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()

            except Exception as e:
                continue  # Skip this strategy if optimization fails

    # Phase 2: Local refinement with L-BFGS-B if we found a good candidate
    if best_points is not None and best_ratio > 0:
        try:
            # Second refinement stage with L-BFGS-B
            refined_points = improved_local_refinement(best_points, 'L-BFGS-B')
            final_ratio = min_max_ratio(refined_points)

            # Update if improved
            if final_ratio > best_ratio:
                best_points = refined_points

        except Exception as e:
            pass  # Keep original best points if refinement fails

    # Phase 3: Additional refinement using alternate method
    if best_points is not None:
        try:
            # Try another refinement with slightly different tolerance
            refined_points = improved_local_refinement(best_points, 'L-BFGS-B')
            final_ratio = min_max_ratio(refined_points)
            
            if final_ratio > best_ratio:
                best_points = refined_points

        except Exception as e:
            pass  # Keep original best points if refinement fails

    # Final selection: compare evolutionary approach with direct optimization
    evol_ratio = min_max_ratio(final_points_cube)
    if evol_ratio > best_ratio:
        return final_points_cube
    else:
        return best_points if best_points is not None else (fibonacci_sphere(n) + 1) / 2

# EVOLVE-BLOCK-END