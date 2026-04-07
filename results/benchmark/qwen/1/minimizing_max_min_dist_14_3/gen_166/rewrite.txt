# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective_ratio(x):
        """Objective function to maximize min/max distance ratio"""
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def objective_negative_ratio(x):
        """Negative of ratio for minimization"""
        return -objective_ratio(x)

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_bounds(x):
        # Ensure all points are within [0,1]^3 bounds
        points = x.reshape(-1, 3)
        return np.concatenate([
            points.flatten() - 0.0,      # lower bound
            1.0 - points.flatten()       # upper bound
        ])

    def generate_sobol_like_points(n):
        """Generate points using a Sobol-like sequence for better space filling"""
        # Generate points on sphere using modified Fibonacci approach
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Use a more robust Fibonacci construction
        for i in range(n):
            # Distribute points more evenly
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            
            # Add some perturbation to avoid perfect symmetry
            if i > 0:
                phi += np.sin(i * 0.5) * 0.02
                
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        
        return np.array(points)

    def generate_initial_points(n):
        """Generate good initial point configuration using Sobol-like approach"""
        # Start with Sobol-like points
        initial_points = generate_sobol_like_points(n)
        
        # Add controlled randomness for diversity
        np.random.seed(42)
        noise = np.random.normal(0, 0.03, (n, 3))
        initial_points += noise
        
        # Normalize to unit sphere
        norms = np.linalg.norm(initial_points, axis=1)
        initial_points = initial_points / norms[:, np.newaxis]
        
        return initial_points

    def compute_voronoi_uniformity(points):
        """Compute a measure of how uniform the spherical Voronoi cells are"""
        try:
            # Only compute if we have enough points
            if len(points) >= 4:
                sv = SphericalVoronoi(points)
                areas = sv.voronoi_cell_areas()
                # Return coefficient of variation of cell areas (lower is more uniform)
                if len(areas) > 0:
                    mean_area = np.mean(areas)
                    if mean_area > 0:
                        cv = np.std(areas) / mean_area
                        return cv
            return 1.0
        except:
            return 1.0

    def adaptive_local_optimization(initial_points, max_iter=200):
        """Apply adaptive local optimization with multiple stages"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = 0.0
        
        # Stage 1: Coarse optimization with relaxed tolerances
        try:
            x0 = current_points.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere},
                {'type': 'ineq', 'fun': constraint_bounds}
            ]
            
            result = minimize(objective_negative_ratio, x0, method='SLSQP', constraints=cons,
                            options={'ftol': 1e-6, 'maxiter': max_iter//2})
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                optimized_points = optimized_points / norms
                
                # Evaluate ratio
                ratio = objective_ratio(result.x)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except:
            pass
            
        # Stage 2: Fine-tune with tighter tolerances
        try:
            x0 = best_points.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere},
                {'type': 'ineq', 'fun': constraint_bounds}
            ]
            
            result = minimize(objective_negative_ratio, x0, method='L-BFGS-B', constraints=cons,
                            options={'ftol': 1e-10, 'maxiter': max_iter//2})
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                optimized_points = optimized_points / norms
                
                # Evaluate ratio
                ratio = objective_ratio(result.x)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except:
            pass
            
        return best_points

    # Multiple starting configurations for robustness
    configs = []
    
    # Configuration 1: Sobol-like points
    configs.append(generate_initial_points(14))
    
    # Configuration 2: Random points on sphere
    np.random.seed(100)
    random_points = np.random.randn(14, 3)
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    configs.append(random_points / norms)
    
    # Configuration 3: Another Sobol-like variant
    np.random.seed(200)
    configs.append(generate_initial_points(14))
    
    # Configuration 4: Perturbed Fibonacci
    np.random.seed(300)
    fib_points = generate_sobol_like_points(14)
    perturbations = np.random.normal(0, 0.02, fib_points.shape)
    perturbed_fib = fib_points + perturbations
    norms = np.linalg.norm(perturbed_fib, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    configs.append(perturbed_fib / norms)
    
    # Configuration 5: Another deterministic seed
    np.random.seed(400)
    configs.append(generate_initial_points(14))

    # Main optimization loop with multiple restarts
    best_final_points = None
    best_ratio = 0
    
    # First try global optimization with DE on each configuration
    for i, initial_config in enumerate(configs):
        try:
            # Flatten initial configuration
            x0 = initial_config.flatten()
            
            # Use differential evolution for global search
            bounds = [(-1.5, 1.5)] * 14 * 3
            result = differential_evolution(
                objective_negative_ratio,
                bounds,
                seed=42 + i,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )
            
            # Extract optimized points and normalize
            optimized_points = result.x.reshape(-1, 3)
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            optimized_points = optimized_points / norms
            
            # Evaluate this solution using the ratio objective
            ratio = objective_ratio(result.x)
            if ratio > best_ratio:
                best_ratio = ratio
                best_final_points = optimized_points.copy()
                    
        except Exception as e:
            continue
    
    # If no good solution from DE, try adaptive local optimization
    if best_final_points is None:
        # Try optimizing from the best initial configurations using adaptive approach
        for i, initial_config in enumerate(configs[:3]):  # Try first 3 configs
            try:
                # Adaptive local optimization
                refined_points = adaptive_local_optimization(initial_config, max_iter=100)
                
                # Evaluate this solution using the ratio objective
                ratio = objective_ratio(refined_points.flatten())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = refined_points.copy()
                        
            except Exception as e:
                continue
    
    # Final refinement step with L-BFGS-B from the best found configuration
    if best_final_points is not None:
        try:
            x0 = best_final_points.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere},
                {'type': 'ineq', 'fun': constraint_bounds}
            ]
            
            # Use L-BFGS-B for final polishing
            refined_result = minimize(objective_negative_ratio, x0, method='L-BFGS-B', constraints=cons,
                                    options={'ftol': 1e-12, 'maxiter': 200})
            
            refined_points = refined_result.x.reshape(-1, 3)
            norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            refined_points = refined_points / norms
            
            # Re-evaluate final solution
            refined_ratio = objective_ratio(refined_result.x)
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_final_points = refined_points.copy()
        except Exception as e:
            pass
    
    # If still no solution, return the last attempt or fallback to Sobol-like points
    if best_final_points is None:
        return generate_initial_points(14)
        
    return best_final_points

# EVOLVE-BLOCK-END