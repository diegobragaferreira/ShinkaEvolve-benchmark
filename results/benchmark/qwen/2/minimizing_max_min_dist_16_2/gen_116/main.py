# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given point configuration."""
        # Ensure points are within unit square
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = squareform(pdist(points))
        
        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)
        
        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return ratio
        if max_dist > 0:
            return min_dist / max_dist
        else:
            return 0.0
    
    def objective_function(points_flat):
        # Reshape flat array back to 16x2 points
        points = points_flat.reshape(-1, 2)
        
        # Ensure points are within unit square
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = squareform(pdist(points))
        
        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)
        
        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio since we want to maximize
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return -np.inf
    
    # Initialize with hexagonal grid pattern plus adaptive perturbations
    np.random.seed(42)
    
    # Create a hexagonal lattice pattern (better than simple grid)
    points = np.zeros((16, 2))
    rows = 4
    cols = 4
    
    # Hexagonal grid with slight perturbation
    for i in range(rows):
        for j in range(cols):
            x = j * 0.25 + (i % 2) * 0.125
            y = i * 0.25
            # Add small random perturbation to break symmetry
            x += (np.random.random() - 0.5) * 0.05
            y += (np.random.random() - 0.5) * 0.05
            points[i*4 + j] = [x, y]
    
    # Clip to unit square
    points = np.clip(points, 0, 1)
    
    # Fix corner points to break symmetry and avoid degenerate solutions
    points[0] = [0.0, 0.0]  # Bottom-left corner
    points[3] = [1.0, 0.0]  # Bottom-right corner  
    points[12] = [0.0, 1.0] # Top-left corner
    points[15] = [1.0, 1.0] # Top-right corner
    
    # Multi-start optimization with differential evolution
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(points)
    
    # Try multiple differential evolution restarts with varying perturbations
    bounds = [(0, 1) for _ in range(32)]
    
    # Run multiple optimization restarts with different seeds
    for restart in range(5):
        # Create slightly different initial configuration for each restart
        restart_points = points.copy()
        
        # Add small random perturbations to all points
        for i in range(16):
            restart_points[i, 0] += (np.random.random() - 0.5) * 0.02
            restart_points[i, 1] += (np.random.random() - 0.5) * 0.02
        
        # Clip to unit square
        restart_points = np.clip(restart_points, 0, 1)
        
        # Fix corners again after perturbations
        restart_points[0] = [0.0, 0.0]
        restart_points[3] = [1.0, 0.0]
        restart_points[12] = [0.0, 1.0]
        restart_points[15] = [1.0, 1.0]
        
        # Run differential evolution
        try:
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42 + restart,
                disp=False
            )
            
            # Evaluate DE result
            de_ratio = -objective_function(de_result.x)
            if de_ratio > best_ratio:
                best_ratio = de_ratio
                best_points = de_result.x.reshape(-1, 2)
                best_points = np.clip(best_points, 0, 1)
                
                # Fix corners in the best solution
                best_points[0] = [0.0, 0.0]
                best_points[3] = [1.0, 0.0]
                best_points[12] = [0.0, 1.0]
                best_points[15] = [1.0, 1.0]
        except:
            continue
    
    # Refine with local optimization using L-BFGS-B
    try:
        refined_result = minimize(
            objective_function,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        final_points = refined_result.x.reshape(-1, 2)
        final_points = np.clip(final_points, 0, 1)
        
        # Fix corners in final solution
        final_points[0] = [0.0, 0.0]
        final_points[3] = [1.0, 0.0]
        final_points[12] = [0.0, 1.0]
        final_points[15] = [1.0, 1.0]
        
        final_ratio = compute_min_max_ratio(final_points)
        if final_ratio > best_ratio:
            best_points = final_points
    except:
        pass
    
    return best_points

# EVOLVE-BLOCK-END