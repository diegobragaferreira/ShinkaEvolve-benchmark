# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0
        
        # Compute pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Avoid division by zero
        if dmax == 0:
            return 0
            
        return dmin / dmax
    
    def generate_hexagon_spiral_initialization():
        """Generate a hexagonal spiral initialization that balances uniformity and spread."""
        np.random.seed(42)
        points = []
        
        # Create hexagonal-like structure but with spiral variation
        # Use 4 rings of points with varying angular positions
        ring_radii = [0.15, 0.35, 0.55, 0.75]
        points_per_ring = [4, 8, 12, 16]  # Each ring increases in density
        
        for ring_idx, (radius, num_points) in enumerate(zip(ring_radii, points_per_ring)):
            # Distribute points around the ring
            for i in range(num_points):
                # Add slight randomness to prevent perfect symmetry
                angle_offset = np.random.uniform(-0.1, 0.1)
                angle = (2 * np.pi * i / num_points) + angle_offset + ring_idx * 0.2
                
                # Vary radius slightly for better distribution
                radius_variation = np.random.uniform(-0.03, 0.03)
                x = 0.5 + (radius + radius_variation) * np.cos(angle)
                y = 0.5 + (radius + radius_variation) * np.sin(angle)
                
                # Ensure points are within [0,1] bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                
                points.append([x, y])
        
        # If we have more than 16 points, take the first 16
        # If we have fewer, pad with random points in the center
        if len(points) >= 16:
            points = points[:16]
        else:
            # Fill remaining spots with random points near center
            for i in range(16 - len(points)):
                x = 0.5 + np.random.normal(0, 0.05)
                y = 0.5 + np.random.normal(0, 0.05)
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                points.append([x, y])
        
        return np.array(points)
    
    def optimized_objective(x_flat):
        """Optimized objective function with custom improvements."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Ensure points are within bounds with epsilon padding
        points = np.clip(points, 1e-6, 1-1e-6)
        
        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        
        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing)
        # Add penalty for extreme distances to encourage balanced distribution
        if d_max <= 0:
            return -1.0
            
        # Add regularization term to avoid degenerate solutions
        ratio = d_min / d_max
        if ratio < 1e-10:  # Very small ratios penalize heavily
            return -1.0
        
        # Use log transformation to make optimization more stable
        return -ratio
    
    def adaptive_gradient_descent(x_start, max_iter=500):
        """Perform adaptive gradient descent with smart restarts."""
        # Use L-BFGS-B for smooth optimization
        bounds = [(1e-6, 1-1e-6) for _ in range(32)]
        
        # Try multiple restarts with different tolerances
        for restart in range(3):
            try:
                # Configure different tolerance settings for retries
                ftol = 1e-12 if restart == 0 else 1e-10
                gtol = 1e-12 if restart == 0 else 1e-10
                maxiter = max_iter if restart == 0 else max_iter // 2
                
                result = minimize(
                    optimized_objective,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
                    tol=1e-12
                )
                
                if result.success:
                    return result.x
            except Exception:
                pass
        
        # If L-BFGS fails, return original points
        return x_start
    
    # Phase 1: Generate high-quality initial configuration
    initial_points = generate_hexagon_spiral_initialization()
    
    # Phase 2: Multi-scale refinement
    current_solution = initial_points.flatten()
    
    # First coarse refinement
    try:
        coarse_result = minimize(
            optimized_objective,
            current_solution,
            method='L-BFGS-B',
            bounds=[(1e-6, 1-1e-6) for _ in range(32)],
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        if coarse_result.success:
            current_solution = coarse_result.x
    except Exception:
        pass
    
    # Second fine-grained refinement
    fine_result = adaptive_gradient_descent(current_solution, max_iter=300)
    current_solution = fine_result
    
    # Phase 3: Final validation and boundary correction
    final_points = current_solution.reshape(-1, 2)
    final_points = np.clip(final_points, 1e-6, 1-1e-6)
    
    # Double-check final ratio
    final_ratio = compute_min_max_ratio(final_points)
    
    # If we got a very poor result, fallback to initial configuration
    if final_ratio < 1e-8:
        final_points = initial_points
        
    return final_points

# EVOLVE-BLOCK-END