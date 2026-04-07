# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from scipy.stats import qmc
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute pairwise distances
        distances = cdist(points, points)
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Minimize negative of minimum distance (maximize minimum distance)
        return -np.min(distances)

    def objective_with_uniformity(x):
        """Enhanced objective that considers both minimum distance and distribution uniformity"""
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # If max_dist is too small, return a large penalty
        if max_dist < 1e-12:
            return 1e10
            
        ratio = min_dist / max_dist
        
        # Add uniformity penalty (lower is better)
        try:
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            if len(areas) > 0:
                mean_area = np.mean(areas)
                if mean_area > 0:
                    uniformity_penalty = np.std(areas) / mean_area
                    # Weight the penalty (0.1 seems reasonable)
                    return -ratio + 0.1 * uniformity_penalty
            return -ratio
        except:
            return -ratio

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_max_distance(x, max_dist_bound=2.0):
        # Ensure maximum distance doesn't exceed some reasonable bound
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, 0)
        max_dist = np.max(distances)
        return max_dist_bound - max_dist  # Should be >= 0
        
    def normalize_points(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def generate_sobol_points(n):
        """Generate points using Sobol sequence for better space-filling properties"""
        # Create Sobol sampler
        sampler = qmc.Sobol(d=3, seed=42)
        # Generate points in [0,1]^3
        sample = sampler.random(n)
        # Map to unit sphere using inverse transform
        points = np.zeros((n, 3))
        for i in range(n):
            u1, u2, u3 = sample[i]
            theta = np.arccos(1 - 2*u1)  # Polar angle
            phi = 2 * np.pi * u2         # Azimuthal angle
            r = u3 ** (1/3)              # Radius (cube root for uniform volume)
            points[i] = [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ]
        return points
    
    def generate_fibonacci_points(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def generate_random_points(n):
        """Generate random points on unit sphere"""
        points = np.random.randn(n, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return points

    def generate_perturbed_fibonacci_points(n, perturbation_strength=0.05):
        """Generate fibonacci points with small random perturbations"""
        base_points = generate_fibonacci_points(n)
        perturbations = np.random.normal(0, perturbation_strength, (n, 3))
        perturbed_points = base_points + perturbations
        # Normalize back to unit sphere
        perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        return perturbed_points

    def generate_voronoi_based_config(n):
        """Generate points using spherical Voronoi construction"""
        # Start with random points and iteratively improve
        points = generate_random_points(n)
        # Simple iterative improvement
        for _ in range(10):
            try:
                sv = SphericalVoronoi(points)
                centroids = sv._voronoi_cell_centroids()
                # Move points towards centroids
                points = centroids
                points = points / np.linalg.norm(points, axis=1, keepdims=True)
            except:
                break
        return points

    # Multiple starting configurations
    configs = []
    
    # Configuration 1: Sobol sequence points
    configs.append(generate_sobol_points(14))
    
    # Configuration 2: Fibonacci spiral on sphere
    configs.append(generate_fibonacci_points(14))
    
    # Configuration 3: Random points on sphere
    np.random.seed(42)
    random_points = np.random.randn(14, 3)
    random_points = normalize_points(random_points)
    configs.append(random_points)
    
    # Configuration 4: Perturbed Fibonacci
    fib_points = generate_fibonacci_points(14)
    np.random.seed(100)
    perturbed_fib = fib_points + np.random.normal(0, 0.02, fib_points.shape)
    perturbed_fib = normalize_points(perturbed_fib)
    configs.append(perturbed_fib)
    
    # Configuration 5: Another random seed
    np.random.seed(200)
    random_points2 = np.random.randn(14, 3)
    random_points2 = normalize_points(random_points2)
    configs.append(random_points2)
    
    # Configuration 6: Fibonacci points with larger perturbation
    np.random.seed(300)
    perturbed_fib_large = generate_fibonacci_points(14) + np.random.normal(0, 0.05, (14, 3))
    perturbed_fib_large = normalize_points(perturbed_fib_large)
    configs.append(perturbed_fib_large)
    
    # Configuration 7: Voronoi-based configuration
    configs.append(generate_voronoi_based_config(14))
    
    # Configuration 8: Another Sobol configuration with different seed
    np.random.seed(500)
    configs.append(generate_sobol_points(14))

    # Main optimization loop with multiple restarts
    best_final_points = None
    best_ratio = 0
    
    # First phase: Use differential evolution with adaptive bounds
    for i, initial_config in enumerate(configs):
        try:
            # Use differential evolution for global search first
            n_points = 14
            n_vars = n_points * 3  # 14 points * 3 coordinates each
            
            # Bounds for each coordinate: [-1, 1] to allow for sphere constraint
            bounds = [(-1, 1) for _ in range(n_vars)]
            
            # Run differential evolution with increased population size
            result = differential_evolution(
                objective_with_uniformity,  # Use enhanced objective
                bounds,
                seed=42 + i,
                maxiter=1000,  # Increased iterations
                popsize=25,    # Larger population
                tol=1e-8,
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False
            )
            
            # Extract optimized points and normalize
            optimized_points = result.x.reshape(-1, 3)
            optimized_points = normalize_points(optimized_points)
            
            # Evaluate this solution
            distances = cdist(optimized_points, optimized_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = optimized_points.copy()
                    
        except Exception as e:
            continue
    
    # Second phase: Two-stage refinement using local optimization
    if best_final_points is not None:
        # Stage 1: SLSQP optimization with tighter constraints
        try:
            x0 = best_final_points.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere},
                {'type': 'ineq', 'fun': lambda x: constraint_max_distance(x, max_dist_bound=2.0)}
            ]

            # More aggressive optimization parameters
            result = minimize(
                objective_with_uniformity, 
                x0, 
                method='SLSQP', 
                constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 1000, 'eps': 1e-6},
                tol=1e-12
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_points = normalize_points(refined_points)
                
                # Re-evaluate
                distances = cdist(refined_points, refined_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_final_points = refined_points.copy()
        except:
            pass

        # Stage 2: L-BFGS-B refinement for local improvement
        try:
            x0 = best_final_points.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere},
                {'type': 'ineq', 'fun': lambda x: constraint_max_distance(x, max_dist_bound=2.0)}
            ]

            result = minimize(
                objective_with_uniformity, 
                x0, 
                method='L-BFGS-B', 
                constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 500},
                tol=1e-12
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_points = normalize_points(refined_points)
                
                # Re-evaluate
                distances = cdist(refined_points, refined_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_final_points = refined_points.copy()
        except:
            pass

    # If still no solution, try alternate approach with different objectives
    if best_final_points is None:
        # Try with the simpler objective first
        for i, initial_config in enumerate(configs[:4]):  # Try only first 4 configs
            try:
                # Local optimization around initial point
                x0 = initial_config.flatten()
                cons = [
                    {'type': 'ineq', 'fun': constraint_sphere},
                    {'type': 'ineq', 'fun': lambda x: constraint_max_distance(x, max_dist_bound=2.0)}
                ]
                
                result = minimize(
                    objective, 
                    x0, 
                    method='SLSQP', 
                    constraints=cons,
                    options={'ftol': 1e-10, 'maxiter': 1000}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 3)
                    optimized_points = normalize_points(optimized_points)
                    
                    # Evaluate this solution
                    distances = cdist(optimized_points, optimized_points)
                    np.fill_diagonal(distances, np.inf)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_final_points = optimized_points.copy()
                            
            except Exception as e:
                continue
    
    # If still no solution, return the best initial configuration
    if best_final_points is None:
        # Find best among initial configs based on simple min/max ratio
        best_initial_ratio = 0
        best_initial_points = None
        
        for i, initial_config in enumerate(configs):
            try:
                distances = cdist(initial_config, initial_config)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_initial_ratio:
                        best_initial_ratio = ratio
                        best_initial_points = initial_config.copy()
            except:
                continue
        
        if best_initial_points is not None:
            return best_initial_points
        else:
            # Fallback to Fibonacci points
            return generate_fibonacci_points(14)
        
    return best_final_points

# EVOLVE-BLOCK-END