# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a quadratic programming reformulation approach for better convergence.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective_quadratic(x):
        """
        Quadratic objective that approximates the ratio maximization by focusing on 
        minimizing squared distances while maintaining reasonable spread.
        """
        points = x.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return np.inf
        
        # We want to maximize min_dist/max_dist, which is equivalent to minimizing max_dist/min_dist
        # Using a quadratic approximation that penalizes both small min_dist and large max_dist
        return -(min_dist / max_dist)
    
    def create_balanced_initial():
        """
        Create an initial configuration that's balanced in terms of distance distribution
        by using a combination of structured approach and geometric considerations.
        """
        np.random.seed(42)
        
        # Start with a regular grid pattern
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + j * 0.225
                y = 0.1 + i * 0.225
                grid_points.append([x, y])
        
        points = np.array(grid_points)
        
        # Add controlled perturbations to break symmetry and improve distribution
        # Use a smaller perturbation scale than previous attempts
        perturbation_scale = 0.01
        noise = np.random.normal(0, perturbation_scale, points.shape)
        points = points + noise
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        
        return points
    
    def compute_distance_ratios(points):
        """
        Helper function to compute the ratio metrics for evaluation.
        """
        distances = pdist(points)
        if len(distances) == 0:
            return 0, 0, 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0, 0, 0
        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist
    
    def solve_reformulated_problem():
        """
        Solve a reformulation of the problem using sequential quadratic programming
        that focuses on maintaining balanced distance distribution.
        """
        # Start with a good initial configuration
        points = create_balanced_initial()
        x0 = points.flatten()
        
        # Bounds for all coordinates
        bounds = [(0, 1) for _ in range(32)]
        
        # Use a hybrid approach: first try L-BFGS-B then fallback to SLSQP
        try:
            # Main optimization with high precision
            result = minimize(
                objective_quadratic,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                return final_points
        except:
            pass
        
        # Fallback to SLSQP if L-BFGS-B fails
        try:
            result = minimize(
                objective_quadratic,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                return final_points
        except:
            pass
        
        # Return initial configuration if all optimization fails
        return points
    
    def improve_with_local_search(initial_points):
        """
        Apply a local search improvement heuristic to further refine the solution.
        """
        points = initial_points.copy()
        best_ratio, _, _ = compute_distance_ratios(points)
        
        # Try several local perturbations
        for iter_num in range(10):
            # Create perturbed version
            np.random.seed(42 + iter_num)
            noise = np.random.normal(0, 0.001, points.shape)
            perturbed = points + noise
            perturbed = np.clip(perturbed, 0, 1)
            
            # Check if this improves the ratio
            ratio, _, _ = compute_distance_ratios(perturbed)
            if ratio > best_ratio:
                best_ratio = ratio
                points = perturbed.copy()
                
        return points
    
    # Main execution flow
    try:
        # Step 1: Solve the reformulated problem
        optimized_points = solve_reformulated_problem()
        
        # Step 2: Apply local search improvement
        improved_points = improve_with_local_search(optimized_points)
        
        # Final evaluation
        final_ratio, _, _ = compute_distance_ratios(improved_points)
        
        # If we still have a poor solution, fall back to a better structured approach
        if final_ratio < 0.25:  # If ratio is still quite low
            # Try a different initialization strategy
            np.random.seed(42)
            points = np.random.rand(16, 2)
            points = np.clip(points, 0, 1)
            x0 = points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            try:
                result = minimize(
                    objective_quadratic,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success:
                    points = result.x.reshape(-1, 2)
                    points = np.clip(points, 0, 1)
                    final_ratio, _, _ = compute_distance_ratios(points)
                    
                    if final_ratio > 0.25:
                        return points
            except:
                pass
                
        return improved_points
        
    except Exception as e:
        # Fallback to simplest approach if everything else fails
        points = create_balanced_initial()
        return points

# EVOLVE-BLOCK-END