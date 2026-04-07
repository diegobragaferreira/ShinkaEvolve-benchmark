# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n_points = 16
    dimension = 2
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0
        
        # Compute pairwise distances with numerical stability
        distances = pdist(points)
        
        # Filter out very small distances that might be numerical artifacts
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return 0
        
        # Get min and max distances
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Avoid division by zero
        if dmax == 0:
            return 0
            
        return dmin / dmax
    
    def generate_hexagonal_initial():
        """Generate initial points arranged in a hexagonal lattice structure."""
        # Create a 4x4 grid with alternating rows offset
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25 * np.sqrt(3) / 2
                
                # Add small random perturbation
                x += np.random.normal(0, 0.005)
                y += np.random.normal(0, 0.005)
                
                points.append([x, y])
        
        points = np.array(points[:n_points])
        
        # Normalize to [0,1] bounds
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if x_range > 0:
            points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range * 0.9 + 0.05
        if y_range > 0:
            points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range * 0.9 + 0.05
            
        return points
    
    def compute_voronoi_energy(points):
        """Compute an energy-like metric based on Voronoi cell areas for regularization."""
        try:
            vor = Voronoi(points)
            areas = []
            
            # Get Voronoi cell areas
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    # Calculate area of polygon
                    poly_points = vor.vertices[region]
                    if len(poly_points) >= 3:
                        # Simple polygon area calculation using shoelace formula
                        x_vals = poly_points[:, 0]
                        y_vals = poly_points[:, 1]
                        area = 0.5 * np.abs(np.dot(x_vals, np.roll(y_vals, 1)) - np.dot(y_vals, np.roll(x_vals, 1)))
                        areas.append(area)
            
            if not areas:
                return 0
                
            # Return variance of areas (lower variance means more uniformity)
            return 1.0 / (1.0 + np.var(areas))
        except:
            return 0
    
    def combined_objective(x_flat):
        """Combined objective function that balances distance ratio and geometric uniformity."""
        points = x_flat.reshape(-1, 2)
        
        # Ensure bounds
        points = np.clip(points, 0, 1)
        
        # Compute distance ratio
        ratio = compute_min_max_ratio(points)
        
        # Compute Voronoi-based uniformity energy
        voronoi_energy = compute_voronoi_energy(points)
        
        # Weighted combination: prioritize distance ratio but penalize irregular Voronoi cells
        # This helps avoid extreme clustering while maximizing the desired ratio
        combined = ratio * 0.8 + voronoi_energy * 0.2
        
        # Return negative for minimization
        return -combined
    
    def constrained_optimization(initial_points, max_iter=1000):
        """Perform constrained optimization with better convergence."""
        bounds = [(0, 1) for _ in range(n_points * dimension)]
        
        # Use L-BFGS-B with enhanced parameters
        result = minimize(
            combined_objective,
            initial_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': max_iter,
                'ftol': 1e-12,
                'gtol': 1e-12,
                'maxcor': 30
            }
        )
        
        # Return optimized points
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
        else:
            optimized_points = initial_points
            
        return optimized_points
    
    def local_refinement_step(points, iterations=5):
        """Apply local refinement using conjugate gradient method."""
        # Start with the best known configuration
        current_points = points.copy()
        
        # Perform several rounds of refinement
        for i in range(iterations):
            # Apply constrained optimization step
            refined_points = constrained_optimization(current_points, max_iter=200)
            
            # Check if there's significant improvement
            old_ratio = compute_min_max_ratio(current_points)
            new_ratio = compute_min_max_ratio(refined_points)
            
            if new_ratio > old_ratio:
                current_points = refined_points
            else:
                # If no improvement, add small random perturbations to escape local minima
                current_points += np.random.normal(0, 0.001, current_points.shape)
                current_points = np.clip(current_points, 0, 1)
        
        return current_points
    
    # Generate initial configuration
    initial_points = generate_hexagonal_initial()
    
    # Multi-start approach for better exploration
    best_points = initial_points.copy()
    best_ratio = compute_min_max_ratio(initial_points)
    
    # Try different initialization strategies
    strategies = [
        generate_hexagonal_initial,  # Original hexagonal approach
        lambda: np.random.rand(n_points, dimension) * 0.9 + 0.05,  # Random in [0.05, 0.95]
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            # Generate initial points
            strategy_points = strategy()
            
            # Apply local refinement
            refined_points = local_refinement_step(strategy_points, iterations=3)
            
            # Evaluate
            ratio = compute_min_max_ratio(refined_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
                
        except Exception:
            continue
    
    # Final optimization of best configuration
    final_points = local_refinement_step(best_points, iterations=8)
    
    # Ensure final bounds
    final_points = np.clip(final_points, 0, 1)
    
    return final_points

# EVOLVE-BLOCK-END