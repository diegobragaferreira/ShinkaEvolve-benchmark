# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def initialize_spherical_points(n_points):
        """Initialize points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)
        
        # Ensure points are within bounds [0,1]^3
        points = np.clip(points, 0, 1)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Handle edge case of identical points
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero or near-zero cases
        if d_max <= 1e-12:
            return -np.inf
            
        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)
    
    def optimize_with_restart():
        """Run differential evolution with multiple restarts and early stopping"""
        best_result = None
        best_ratio = -np.inf
        max_restarts = 5
        max_iter_per_restart = 1000
        
        # Try multiple initializations
        for restart in range(max_restarts):
            # Initialize with spherical arrangement
            initial_points = initialize_spherical_points(14)
            
            # Add some randomness to break symmetry
            np.random.seed(restart + 42)
            noise = np.random.normal(0, 0.05, (14, 3))
            initial_points += noise
            initial_points = np.clip(initial_points, 0, 1)
            
            # Flatten for optimization
            x0 = initial_points.flatten()
            
            # Define bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
            bounds = [(0, 1)] * 14 * 3
            
            try:
                # Run optimization with reasonable settings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = differential_evolution(
                        objective,
                        bounds,
                        x0=x0,
                        seed=42 + restart,
                        maxiter=max_iter_per_restart,
                        popsize=15,
                        tol=1e-6,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        disp=False,
                        callback=None
                    )
                
                # Check if this result is better
                if result.success:
                    points = result.x.reshape(-1, 3)
                    distances = pdist(points)
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 1e-12:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_result = result
                                
            except Exception:
                continue
                
        return best_result
    
    # Perform optimization with restarts
    result = optimize_with_restart()
    
    # If optimization failed, fall back to spherical initialization
    if result is None or not result.success:
        points = initialize_spherical_points(14)
        points = np.clip(points, 0, 1)
    else:
        points = result.x.reshape(-1, 3)
        points = np.clip(points, 0, 1)
    
    return points

# EVOLVE-BLOCK-END