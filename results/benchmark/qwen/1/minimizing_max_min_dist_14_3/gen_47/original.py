# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def distance_ratio(points_flat):
        """Calculate the ratio of minimum to maximum distance"""
        points = points_flat.reshape(-1, 3)
        distances = squareform(pdist(points))
        # Set diagonal to large value so it doesn't affect min/max
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def objective_function(points_flat):
        """Minimize negative of distance ratio (since we want to maximize)"""
        return -distance_ratio(points_flat)
    
    def fibonacci_spiral_points(n):
        """Generate points on sphere using Fibonacci spiral"""
        points = []
        phi = math.acos(-1 + 2 * (0 / (n - 1)))  # theta
        theta = math.sqrt(n * math.pi) * phi
        
        for i in range(1, n + 1):
            phi = math.acos(-1 + 2 * (i - 1) / (n - 1))
            theta = math.sqrt(n * math.pi) * phi
            
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def get_initial_points():
        """Get good initial point configuration"""
        # Start with Fibonacci spiral points
        initial_points = fibonacci_spiral_points(14)
        
        # Add some randomness to avoid local minima
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (14, 3))
        initial_points += noise
        
        # Normalize to unit sphere
        norms = np.linalg.norm(initial_points, axis=1)
        initial_points = initial_points / norms[:, np.newaxis]
        
        return initial_points.flatten()
    
    def optimize_with_constraints(x0, maxiter=500):
        """Optimize with constraints using L-BFGS-B"""
        # Define constraints for normalization (points should be on unit sphere)
        constraints = []
        
        def constraint_func(x):
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0  # Should equal 0 for unit sphere
        
        # Add constraint for each point to lie on unit sphere
        for i in range(14):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]})
        
        # Initial bounds (allow slight deviation from sphere)
        bounds = [(-1.5, 1.5)] * len(x0)
        
        # Try multiple optimization methods
        options = {'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-8}
        
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-8
        )
        
        return result.x
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with different initializations
    for restart in range(5):
        np.random.seed(42 + restart)
        
        # Get initial points
        x0 = get_initial_points()
        
        # Optimize
        try:
            optimized_points = optimize_with_constraints(x0, maxiter=500)
            
            # Calculate final ratio
            ratio = distance_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    if best_points is None:
        # Fallback to Fibonacci if optimization fails
        best_points = get_initial_points()
    
    # Convert back to 14x3 array
    final_points = best_points.reshape(14, 3)
    
    return final_points

# EVOLVE-BLOCK-END
