# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        # Reshape x back to points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio to maximize ratio (minimize negative)
        if max_dist == 0:
            return float('inf')
        return -min_dist / max_dist
    
    def constraint_func(x):
        # Ensure points stay within unit sphere (for better conditioning)
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return positive values where constraint is satisfied
        return 1.0 - norms  # Positive when norm <= 1
    
    # Initialize with a good starting configuration
    # Using a known good configuration from literature
    np.random.seed(42)
    
    # Start with a spherical code-like arrangement
    # Generate points on a sphere, then slightly perturb
    points = []
    
    # Create points using Fibonacci spiral on sphere (approximation)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(14):
        theta = np.arccos(1 - 2 * (i / 13))
        phi = np.mod(i * golden_ratio, 1) * 2 * np.pi
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Add some randomness to avoid local minima
        points.append([x + np.random.normal(0, 0.01), 
                      y + np.random.normal(0, 0.01),
                      z + np.random.normal(0, 0.01)])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / np.max(norms) * 0.9
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # Define constraints
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Run optimization
    try:
        result = minimize(objective, x0, method='SLSQP', 
                         constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-8},
                         bounds=[(-1, 1)] * 42)
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            # Normalize to unit sphere if needed
            norms_final = np.linalg.norm(final_points, axis=1)
            if np.max(norms_final) > 1:
                final_points = final_points / np.max(norms_final) * 0.99
            
            return final_points
        else:
            # If optimization fails, return the initial points
            return points
            
    except Exception as e:
        # Fallback to initial points if anything goes wrong
        return points


# EVOLVE-BLOCK-END
