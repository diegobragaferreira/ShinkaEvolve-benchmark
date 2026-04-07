# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a novel geometric approach combining:
    1. Initial placement based on circle packing principles
    2. Multi-start optimization with different initialization strategies
    3. Adaptive optimization algorithm selection
    4. Geometric constraints enforcement
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        # Efficiently compute all pairwise distances
        distances = squareform(pdist(points))
        
        # Mask diagonal (self-distances)
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def create_circle_packing_initialization():
        """Initialize points using circle packing principles for uniform distribution."""
        # Create points arranged in a circle with radial spacing
        n = 16
        points = np.zeros((n, 2))
        
        # Place points evenly around a circle, but with some perturbation for better spread
        radius = 0.4
        center = np.array([0.5, 0.5])
        
        # Angle spacing
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        
        # Add some variation to prevent perfect symmetry
        angle_noise = np.random.normal(0, 0.1, n)
        
        for i in range(n):
            angle = angles[i] + angle_noise[i]
            points[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
            
        # Add slight random perturbations to avoid degenerate cases
        points += np.random.normal(0, 0.02, points.shape)
        
        # Ensure all points remain in [0.1, 0.9] x [0.1, 0.9]
        points = np.clip(points, 0.1, 0.9)
        
        return points
    
    def create_hexagonal_grid_initialization():
        """Initialize using a hexagonal grid pattern."""
        n = 16
        points = np.zeros((n, 2))
        
        # Create a 4x4 hexagonal grid
        rows = 4
        cols = 4
        spacing = 0.25
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < n:
                    # Hexagonal offset
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * math.sqrt(3) / 2
                    points[idx] = [x, y]
                    idx += 1
        
        # Scale and shift to [0.1, 0.9] range
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1
        
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        
        # Clamp to bounds
        points = np.clip(points, 0.01, 0.99)
        
        return points
    
    def create_fibonacci_spiral_initialization():
        """Initialize using Fibonacci spiral for even distribution."""
        n = 16
        points = np.zeros((n, 2))
        center = np.array([0.5, 0.5])
        radius = 0.4
        
        # Fibonacci spiral approach
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            r = radius * np.sqrt(i / (n - 1)) if n > 1 else 0
            theta = i * golden_angle
            
            points[i] = center + r * np.array([np.cos(theta), np.sin(theta)])
            
        # Add noise for better optimization
        points += np.random.normal(0, 0.015, points.shape)
        
        # Clamp to bounds
        points = np.clip(points, 0.05, 0.95)
        
        return points
    
    def constraint_bound_check(points):
        """Check if all points are within [0,1] bounds."""
        return np.all((points >= 0) & (points <= 1))
    
    def objective_function(points_flat):
        """Objective function for optimization - minimize negative ratio."""
        points = points_flat.reshape(-1, 2)
        
        # Enforce boundary constraints in objective
        if not constraint_bound_check(points):
            # High penalty for out-of-bounds points
            return 1e10
        
        # Compute negative ratio (since we want to maximize ratio)
        ratio = compute_min_max_ratio(points)
        return -ratio
    
    # Try different initialization strategies
    initializations = [
        create_circle_packing_initialization(),
        create_hexagonal_grid_initialization(),  
        create_fibonacci_spiral_initialization()
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Run optimization from each initialization
    for init_points in initializations:
        # First try L-BFGS-B for fast local optimization
        try:
            # Flatten points for optimization
            x0 = init_points.flatten()
            
            # Define bounds for optimization
            bounds = [(0, 1) for _ in range(32)]
            
            # Try L-BFGS-B first (faster for smooth functions)
            result1 = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if result1.success:
                optimized_points = result1.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except:
            pass
        
        # If L-BFGS-B failed, try a more robust method
        try:
            # Flatten points for optimization
            x0 = init_points.flatten()
            
            # Define bounds for optimization
            bounds = [(0, 1) for _ in range(32)]
            
            # Try Nelder-Mead as fallback
            result2 = minimize(
                objective_function,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'adaptive': True}
            )
            
            if result2.success:
                optimized_points = result2.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except:
            pass
    
    # Fallback to the best initialization if no optimization worked
    if best_points is None:
        # Take the best of the initializations
        ratios = [compute_min_max_ratio(init) for init in initializations]
        best_idx = np.argmax(ratios)
        best_points = initializations[best_idx]
    
    return best_points

# EVOLVE-BLOCK-END
