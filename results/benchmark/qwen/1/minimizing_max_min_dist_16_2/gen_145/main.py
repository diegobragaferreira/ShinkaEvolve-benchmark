# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x):
        # Reshape x into points
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances using squareform for better numerical stability
        distances = squareform(pdist(points))
        
        # Zero out diagonal elements (distance to self)
        np.fill_diagonal(distances, np.inf)
        
        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing the negative)
        if d_max == 0:
            return -1.0
        return -d_min / d_max
    
    def constraint_func(x):
        """Constraint function to ensure points stay within bounds"""
        points = x.reshape(-1, 2)
        # Add small padding to avoid boundary issues
        padding = 1e-8
        return np.concatenate([
            points[:, 0] - padding,           # x >= padding
            1 - points[:, 0] - padding,       # x <= 1 - padding
            points[:, 1] - padding,           # y >= padding
            1 - points[:, 1] - padding        # y <= 1 - padding
        ])
    
    # Create initial configuration using hexagonal packing with better spacing
    np.random.seed(42)
    
    # Generate a hexagonal-like pattern that fits well in the unit square
    points = np.zeros((16, 2))
    
    # Create a 4x4 grid with hexagonal offset pattern
    rows, cols = 4, 4
    
    # Calculate optimal spacing to fit in unit square
    # Use slightly smaller spacing to allow for boundary padding
    spacing_x = 0.9 / (cols - 1) if cols > 1 else 0.9
    spacing_y = 0.9 / (rows - 1) if rows > 1 else 0.9
    
    # Apply hexagonal offset for better packing
    for i in range(rows):
        for j in range(cols):
            if i * cols + j >= 16:
                break
            x = j * spacing_x + (i % 2) * spacing_x * 0.5
            y = i * spacing_y
            
            # Scale to [0.05, 0.95] range to maintain padding
            x_scaled = 0.05 + x * 0.9
            y_scaled = 0.05 + y * 0.9
            
            # Add slight random perturbation to avoid perfect symmetry
            x_scaled += np.random.normal(0, 0.005)
            y_scaled += np.random.normal(0, 0.005)
            
            points[i * cols + j] = [x_scaled, y_scaled]
    
    # Ensure points are within bounds with padding
    points = np.clip(points, 0.05, 0.95)
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # Define bounds for each coordinate with stricter padding
    bounds = [(0.05, 0.95) for _ in range(32)]
    
    # Optimization parameters
    de_options = {
        'seed': 42,
        'maxiter': 100,
        'popsize': 20,
        'tol': 1e-9,
        'recombination': 0.9,
        'mutation': (0.8, 1.0),
        'disp': False
    }
    
    # First stage: Use differential evolution for global optimization
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            **de_options
        )
        x0 = de_result.x.copy()
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
        pass
    
    # Second stage: Local refinement with L-BFGS-B
    try:
        # Use tighter tolerances for better convergence
        lbfgs_result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-13},
            callback=None
        )
        
        if lbfgs_result.success:
            x0 = lbfgs_result.x.copy()
        else:
            # Fallback to SLSQP if L-BFGS-B fails
            slsqp_result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                callback=None
            )
            if slsqp_result.success:
                x0 = slsqp_result.x.copy()
                
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
        pass
    
    # Final validation and refinement
    try:
        final_result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
            callback=None
        )
        
        if final_result.success:
            points = final_result.x.reshape(-1, 2)
        else:
            points = x0.reshape(-1, 2)
    except Exception as e:
        warnings.warn(f"Final optimization failed: {e}")
        points = x0.reshape(-1, 2)
    
    # Ensure final points are within strict bounds
    points = np.clip(points, 0.05, 0.95)
    
    return points

# EVOLVE-BLOCK-END