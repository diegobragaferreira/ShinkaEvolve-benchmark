# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0
            
        return d_min / d_max
    
    def optimize_points(initial_points, max_iter=1000, temp_start=1.0, cooling_rate=0.995):
        """Optimize point placement using simulated annealing."""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Temperature schedule
        temp = temp_start
        
        for iteration in range(max_iter):
            # Perturb one point at random
            idx = np.random.randint(len(current_points))
            new_points = current_points.copy()
            
            # Add small random perturbation
            perturbation = np.random.normal(0, 0.01, 3)
            new_points[idx] = current_points[idx] + perturbation
            
            # Keep within bounds [0,1]
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
            # Compute new ratio
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) / temp):
                current_points = new_points
                if new_ratio > best_ratio:
                    best_points = new_points
                    best_ratio = new_ratio
            
            # Cool down temperature
            temp *= cooling_rate
            
            # Early stopping if barely improving
            if iteration > 100 and abs(new_ratio - best_ratio) < 1e-8:
                break
                
        return best_points, best_ratio
    
    # Multiple restart strategy
    best_solution = None
    best_ratio = 0.0
    
    # Try multiple random initializations
    for restart in range(10):
        # Create initial points with different strategies
        if restart == 0:
            # Random initialization
            points = np.random.rand(14, 3)
        else:
            # Slightly perturbed version of previous best
            points = np.random.rand(14, 3) * 0.8 + 0.1
            # Add some structure to avoid too random
            if restart % 3 == 0:
                # Add some regular pattern
                points[0:3] = np.random.rand(3, 3) * 0.5
                points[3:6] = np.random.rand(3, 3) * 0.5 + 0.5
                points[6:9] = np.random.rand(3, 3) * 0.3 + 0.35
                points[9:11] = np.random.rand(2, 3) * 0.2 + 0.4
                points[11:] = np.random.rand(3, 3) * 0.2 + 0.8
                
        # Optimize this initialization
        optimized_points, ratio = optimize_points(points, max_iter=1000)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()
    
    # Final optimization with the best found solution
    if best_solution is not None:
        final_points, _ = optimize_points(best_solution, max_iter=500)
        return final_points
    
    # Fallback to the best random solution if no optimization worked
    return best_solution if best_solution is not None else np.random.rand(14, 3)

# EVOLVE-BLOCK-END
