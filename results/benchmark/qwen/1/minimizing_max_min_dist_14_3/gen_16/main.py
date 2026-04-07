# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio since we want to maximize
        # We add a small epsilon to avoid division by zero
        if d_max < 1e-12:
            return -1e10
        return -d_min / d_max

    # Create a better initial guess using a structured approach
    # Start with a simple grid-like distribution
    np.random.seed(42)
    
    # Generate points in a way that avoids very close clustering
    initial_points = []
    
    # Distribute points more evenly in 3D space
    # Using a modified Fibonacci spiral approach for 3D
    n = 14
    points = np.zeros((n, 3))
    
    # Golden angle in 3D
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    for i in range(n):
        # Distribute points along a spiral pattern in 3D
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = np.arccos(y)  # angle from z-axis
        
        # Add some randomness while maintaining structure
        # Use Fibonacci-like spacing with jitter
        theta = i * 2.399963229728653  # slightly adjusted for better spread
        theta += np.random.normal(0, 0.1)  # add small randomization
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points[i] = [x, y, z]
        
    # Normalize to [0,1]^3 for optimization
    # Find the bounding box and scale appropriately
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    
    # Scale to [0,1] range
    scaled_points = (points - min_vals) / ranges
    
    # Flatten for optimization
    x0 = scaled_points.flatten()
    
    # Set up bounds for optimization (0 to 1 for all coordinates)
    bounds = [(0.0, 1.0)] * 14 * 3

    # Use differential evolution for global optimization
    # This is more robust for this type of problem
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=500,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=False,
        init=x0  # Start with our structured initial guess
    )

    # Return the optimized points
    optimized_points = result.x.reshape(-1, 3)
    
    # Convert back to original range if needed
    # But since we're working in [0,1], we can return directly
    return optimized_points


# EVOLVE-BLOCK-END