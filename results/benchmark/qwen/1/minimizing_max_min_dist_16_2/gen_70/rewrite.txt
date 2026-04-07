# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Implements a novel geometric optimization approach with Voronoi-based initialization.

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
    
    def adaptive_objective_with_penalty(x, penalty_weight=1000.0):
        """
        Enhanced objective with built-in geometric penalties for better convergence
        """
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # If all points are identical or near identical, penalize heavily
        if d_max == 0:
            return -1.0
        
        # Calculate ratio to maximize
        ratio = d_min / d_max
        
        # Add penalty for points near boundaries to avoid numerical issues
        boundary_penalty = 0.0
        margin = 0.01
        for point in points:
            if (point[0] < margin or point[0] > 1-margin or 
                point[1] < margin or point[1] > 1-margin):
                boundary_penalty += penalty_weight * (margin - min(point[0], 1-point[0], point[1], 1-point[1]))
        
        # Add penalty for very small distances (close point clustering)
        min_distance_penalty = 0.0
        if d_min < 0.05:  # Threshold for clustering penalty
            min_distance_penalty = penalty_weight * (0.05 - d_min)
        
        total_penalty = boundary_penalty + min_distance_penalty
        return -(ratio - total_penalty / penalty_weight)
    
    # Phase 1: Voronoi-inspired geometric initialization
    np.random.seed(42)
    
    # Create initial configuration based on hexagonal tiling with perturbations
    # This creates a more uniform initial distribution than simple grids
    points = []
    
    # Generate points in a hexagonal pattern with slight randomness
    rows, cols = 4, 4
    sqrt3 = np.sqrt(3)
    spacing = 0.8  # Adjust spacing to fit better in [0,1] square
    row_spacing = spacing / sqrt3
    col_spacing = spacing
    
    for i in range(rows):
        for j in range(cols):
            if len(points) >= 16:
                break
            # Offset every other row for hexagonal packing
            x = j * col_spacing + (i % 2) * col_spacing * 0.5
            y = i * row_spacing
            
            # Scale to fit within unit square [0,1]
            x_scaled = 0.1 + (x / (col_spacing * cols)) * 0.8
            y_scaled = 0.1 + (y / (row_spacing * rows)) * 0.8
            
            # Add slight random perturbation
            x_scaled += np.random.normal(0, 0.02)
            y_scaled += np.random.normal(0, 0.02)
            
            points.append([x_scaled, y_scaled])
    
    points = np.array(points[:16])
    
    # Ensure points are within bounds
    points = np.clip(points, 0.01, 0.99)
    
    # Phase 2: Hybrid optimization approach
    x0 = points.flatten()
    
    # Define bounds with extra margin for better numerical behavior
    bounds = [(0.01, 0.99) for _ in range(32)]
    
    # First stage: Differential Evolution with adaptive parameters
    try:
        de_result = differential_evolution(
            adaptive_objective_with_penalty,  # Use enhanced objective with penalties
            bounds,
            seed=42,
            maxiter=250,
            popsize=25,
            tol=1e-9,
            recombination=0.9,
            mutation=(0.8, 1.0),
            disp=False
        )
        
        # Update x0 with better solution from DE
        x0 = de_result.x.copy()
        
    except Exception as e:
        warnings.warn(f"Differential evolution failed: {e}")
        pass
    
    # Second stage: Local optimization with enhanced convergence control
    try:
        # Try L-BFGS-B first with strict tolerances
        lbfgs_result = minimize(
            adaptive_objective_with_penalty,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15},
            callback=None
        )
        
        # If L-BFGS-B doesn't work, fallback to SLSQP
        if lbfgs_result.success:
            x0 = lbfgs_result.x.copy()
        else:
            slsqp_result = minimize(
                adaptive_objective_with_penalty,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
                callback=None
            )
            if slsqp_result.success:
                x0 = slsqp_result.x.copy()
                
    except Exception as e:
        warnings.warn(f"Local optimization failed: {e}")
        pass
    
    # Final refinement with standard objective (to ensure proper output)
    try:
        final_result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
            callback=None
        )
        
        if final_result.success:
            points = final_result.x.reshape(-1, 2)
        else:
            points = x0.reshape(-1, 2)
    except Exception as e:
        warnings.warn(f"Final optimization failed: {e}")
        points = x0.reshape(-1, 2)
    
    # Ensure final points are within bounds
    points = np.clip(points, 0.01, 0.99)
    
    return points

# EVOLVE-BLOCK-END