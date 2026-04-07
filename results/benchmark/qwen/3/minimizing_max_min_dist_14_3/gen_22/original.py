# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances efficiently
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0
            
        return d_min / d_max
    
    def objective_function(points_flat):
        """Objective function to minimize (negative ratio of min/max distances)"""
        points = points_flat.reshape(-1, 3)
        # Compute pairwise distances
        distances = cdist(points, points)
        # Set diagonal to large value to avoid zero distances
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return float('inf')
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        return -min_dist / max_dist
    
    def optimize_with_de_and_local_refinement(initial_points):
        """Optimize using differential evolution followed by local refinement"""
        # Use differential evolution for global optimization
        bounds = [(-1, 1)] * (14 * 3)
        
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                disp=False
            )
            
            # Extract final points from DE result
            final_points = result.x.reshape(-1, 3)
        except:
            # Fallback to simple local optimization if DE fails
            final_points = initial_points.copy()
        
        # Local refinement with hill climbing
        for _ in range(50):
            current_ratio = -objective_function(final_points.flatten())
            best_points = final_points.copy()
            best_ratio = current_ratio
            
            # Try small perturbations
            step_size = 0.01
            for i in range(14):
                for dim in range(3):
                    # Try moving in positive direction
                    test_points = final_points.copy()
                    test_points[i, dim] += step_size
                    test_points = np.clip(test_points, -1, 1)
                    test_ratio = -objective_function(test_points.flatten())
                    
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()
                    
                    # Try moving in negative direction
                    test_points = final_points.copy()
                    test_points[i, dim] -= step_size
                    test_points = np.clip(test_points, -1, 1)
                    test_ratio = -objective_function(test_points.flatten())
                    
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()
            
            # If no improvement, stop
            if best_ratio <= current_ratio:
                break
                
            final_points = best_points
        
        # Final cleanup to ensure points are within bounds
        final_points = np.clip(final_points, -1, 1)
        
        return final_points
    
    def generate_initial_points():
        """Generate good initial points using multiple strategies"""
        # Strategy 1: Start with spherical distribution
        np.random.seed(42)
        points = np.random.randn(14, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True) * 0.5  # Scale to unit sphere
        
        # Add some randomness to break symmetry
        points += np.random.normal(0, 0.05, points.shape)
        
        # Ensure points stay within reasonable bounds
        points = np.clip(points, -1, 1)
        
        return points
    
    # Multiple restart strategy with better initialization
    best_solution = None
    best_ratio = 0.0
    
    # Try multiple random initializations with different strategies
    for restart in range(10):
        # Generate initial points
        points = generate_initial_points()
        
        # Optimize this initialization
        optimized_points = optimize_with_de_and_local_refinement(points)
        ratio = compute_min_max_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()
    
    # Final optimization with the best found solution
    if best_solution is not None:
        final_points = optimize_with_de_and_local_refinement(best_solution)
        return final_points
    
    # Fallback to the best random solution if no optimization worked
    return generate_initial_points()

# EVOLVE-BLOCK-END
