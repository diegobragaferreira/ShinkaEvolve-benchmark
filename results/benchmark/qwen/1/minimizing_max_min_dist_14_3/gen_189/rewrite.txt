# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.stats import qmc
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

    def objective_with_regularization(x):
        """Objective function with distance variance regularization to promote uniformity"""
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        var_dist = np.var(distances)
        
        if max_dist == 0:
            return 0.0
            
        # Regularized objective: maximize ratio + penalize distance variance
        ratio = min_dist / max_dist
        regularization = 0.15 * (var_dist / (mean_dist**2 + 1e-12))  # Weighted penalty
        return -(ratio - regularization)  # Negative because we minimize

    def constraint_bounds(x):
        # Ensure all points are within [0,1]^3 bounds
        points = x.reshape(-1, 3)
        return np.concatenate([
            points.flatten() - 0.0,      # lower bound
            1.0 - points.flatten()       # upper bound
        ])

    def fibonacci_spiral_points(n):
        """Generate points on sphere using Fibonacci spiral"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = np.arctan2(np.sin(i * 2 * np.pi / golden_ratio), np.cos(i * 2 * np.pi / golden_ratio))
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        
        return np.array(points)

    def sobol_sequence_points(n):
        """Generate points using 3D Sobol sequence for better space-filling"""
        sampler = qmc.Sobol(d=3, scramble=True, seed=42)
        points = sampler.random(n)
        # Scale to unit sphere
        points = points * 2 - 1  # map to [-1,1]^3
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms
        return points

    def generate_diverse_initial_points(n):
        """Generate a diverse set of initial configurations"""
        configs = []
        
        # Configuration 1: Pure Fibonacci spiral
        configs.append(fibonacci_spiral_points(n))
        
        # Configuration 2: Perturbed Fibonacci with small noise
        np.random.seed(100)
        fib_points = fibonacci_spiral_points(n)
        noise = np.random.normal(0, 0.02, fib_points.shape)
        configs.append(fib_points + noise)
        
        # Configuration 3: Perturbed Fibonacci with medium noise  
        np.random.seed(200)
        fib_points = fibonacci_spiral_points(n)
        noise = np.random.normal(0, 0.05, fib_points.shape)
        configs.append(fib_points + noise)
        
        # Configuration 4: Sobol sequence points
        configs.append(sobol_sequence_points(n))
        
        # Configuration 5: Random points on sphere
        np.random.seed(300)
        random_points = np.random.randn(n, 3)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        configs.append(random_points / norms)
        
        # Configuration 6: Another Fibonacci variant with offset
        np.random.seed(400)
        fib_points = fibonacci_spiral_points(n)
        noise = np.random.normal(0, 0.03, fib_points.shape)
        configs.append(fib_points + noise)
        
        # Configuration 7: Perturbed Fibonacci with larger noise
        np.random.seed(500)
        fib_points = fibonacci_spiral_points(n)
        noise = np.random.normal(0, 0.1, fib_points.shape)
        configs.append(fib_points + noise)
        
        # Configuration 8: Another random seed
        np.random.seed(600)
        random_points = np.random.randn(n, 3)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        configs.append(random_points / norms)
        
        # Normalize all configurations to unit sphere
        normalized_configs = []
        for config in configs:
            norms = np.linalg.norm(config, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized_configs.append(config / norms)
            
        return normalized_configs

    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_points(14)

    # Main optimization loop with multiple restarts
    best_final_points = None
    best_ratio = 0
    
    # First try differential evolution on each configuration
    for i, initial_config in enumerate(initial_configs):
        try:
            # Flatten initial configuration
            x0 = initial_config.flatten()
            
            # Use differential evolution for global search with enhanced parameters
            bounds = [(0, 1) for _ in range(14 * 3)]
            
            result = differential_evolution(
                objective_with_regularization,
                bounds,
                seed=42 + i,
                maxiter=1500,   # Increased iterations
                popsize=30,     # Larger population size for better exploration
                tol=1e-8,       # Tighter tolerance
                mutation=(0.5, 1),
                recombination=0.7,
                disp=False
            )
            
            # Extract optimized points 
            optimized_points = result.x.reshape(-1, 3)
            optimized_points = np.clip(optimized_points, 0, 1)
            
            # Evaluate this solution
            ratio = objective_ratio(result.x)
            if ratio > best_ratio:
                best_ratio = ratio
                best_final_points = optimized_points.copy()
                    
        except Exception as e:
            continue
    
    # If no good solution from DE, try local optimization from best configurations
    if best_final_points is None:
        # Try optimizing from the best initial configurations using local method
        for i, initial_config in enumerate(initial_configs[:4]):  # Try first 4 configs
            try:
                # Local optimization around initial point
                x0 = initial_config.flatten()
                cons = [{'type': 'ineq', 'fun': constraint_bounds}]
                
                # Use SLSQP for local optimization
                result = minimize(objective_with_regularization, x0, method='SLSQP', constraints=cons,
                                options={'ftol': 1e-10, 'maxiter': 1000, 'disp': False})
                
                optimized_points = result.x.reshape(-1, 3)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                # Evaluate this solution
                ratio = objective_ratio(result.x)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = optimized_points.copy()
                        
            except Exception as e:
                continue

    # Final refinement step with hybrid approach
    if best_final_points is not None:
        try:
            x0 = best_final_points.flatten()
            cons = [{'type': 'ineq', 'fun': constraint_bounds}]
            
            # Stage 1: SLSQP for constraint handling
            refined_result = minimize(objective_with_regularization, x0, method='SLSQP', constraints=cons,
                                    options={'ftol': 1e-12, 'maxiter': 500, 'disp': False})
            
            refined_points = refined_result.x.reshape(-1, 3)
            refined_points = np.clip(refined_points, 0, 1)
            
            # Stage 2: L-BFGS-B for fine-tuning
            refined_result2 = minimize(objective_with_regularization, refined_points.flatten(), method='L-BFGS-B', 
                                     constraints=cons, options={'ftol': 1e-14, 'maxiter': 300, 'disp': False})
            
            final_points = refined_result2.x.reshape(-1, 3)
            final_points = np.clip(final_points, 0, 1)
            
            # Final validation
            final_ratio = objective_ratio(final_points.flatten())
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_final_points = final_points.copy()
                
        except Exception as e:
            pass
    
    # If still no solution, return the best configuration from initial set
    if best_final_points is None:
        return initial_configs[0]
        
    return best_final_points

# EVOLVE-BLOCK-END