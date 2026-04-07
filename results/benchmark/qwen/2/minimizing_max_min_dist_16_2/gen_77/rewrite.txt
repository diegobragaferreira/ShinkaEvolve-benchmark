# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the minimum to maximum distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def spherical_voronoi_initialization():
        """Initialize points using a spherical Voronoi-inspired approach combined with structured grid."""
        # Start with a structured 4x4 grid
        grid_size = 4
        x_vals = np.linspace(0.1, 0.9, grid_size)
        y_vals = np.linspace(0.1, 0.9, grid_size)
        points = np.array([[x, y] for x in x_vals for y in y_vals])[:16]
        
        # Apply adaptive perturbations with higher variance at edges to promote spread
        np.random.seed(42)
        for i in range(16):
            row, col = i // 4, i % 4
            
            # Apply higher perturbation to boundary points
            if row in [0, 3] or col in [0, 3]:
                perturbation_magnitude = 0.03
            else:
                perturbation_magnitude = 0.01
                
            noise = np.random.normal(0, perturbation_magnitude, 2)
            points[i] += noise
            
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    def distance_balance_step(points, learning_rate=0.1):
        """Apply a distance balance step to improve the min/max distance ratio."""
        n = len(points)
        
        # Compute all pairwise distances
        distances = pdist(points)
        distance_matrix = np.zeros((n, n))
        distance_matrix[np.triu_indices(n, k=1)] = distances
        distance_matrix += distance_matrix.T
        
        # For each point, calculate how much moving it would improve the ratio
        # This is a simplified gradient-based update
        updated_points = points.copy()
        
        for i in range(n):
            # Calculate average distance to neighbors
            avg_distance = np.mean(distance_matrix[i][distance_matrix[i] > 0])
            
            # Find nearest and farthest neighbors
            non_zero_dists = distance_matrix[i][distance_matrix[i] > 0]
            if len(non_zero_dists) > 0:
                nearest_idx = np.argmin(non_zero_dists)
                farthest_idx = np.argmax(non_zero_dists)
                
                # Simple heuristic: move point to reduce extreme differences
                # Move towards neighbors that are closer than average (if they exist)
                # but also away from very distant points
                target_direction = np.zeros(2)
                
                # Get the nearest neighbor
                if len(non_zero_dists) > 0:
                    nearest_point = points[nearest_idx]
                    target_direction += (nearest_point - points[i]) * 0.5
                
                # Get the farthest neighbor
                if len(non_zero_dists) > 0:
                    farthest_point = points[farthest_idx]
                    target_direction -= (farthest_point - points[i]) * 0.3
                
                # Apply movement with learning rate
                updated_points[i] += target_direction * learning_rate
                
        # Clip points to remain within bounds
        updated_points = np.clip(updated_points, 0, 1)
        return updated_points
    
    def multi_scale_refinement(points, scales=[1.0, 0.5, 0.1]):
        """Refine the point configuration using multiple scales."""
        current_points = points.copy()
        
        for scale in scales:
            # Reduce learning rate for finer scales
            lr = 0.05 * scale
            
            # Apply several steps of distance balancing
            for _ in range(20):
                current_points = distance_balance_step(current_points, learning_rate=lr)
                
        return current_points
    
    def geometric_optimization_step(points):
        """Perform a simple geometric optimization step."""
        # This approach uses a combination of:
        # 1. Gradient estimation based on distance ratios
        # 2. Boundary constraint handling
        # 3. Symmetry breaking
        
        # Try to improve current configuration by small steps
        improved_points = points.copy()
        
        # First, get current metrics
        current_ratio = compute_min_max_ratio(improved_points)
        
        # Try small random perturbations to find improvements
        np.random.seed(42)
        best_points = improved_points.copy()
        best_ratio = current_ratio
        
        # Test multiple small perturbations
        for _ in range(50):
            test_points = improved_points.copy()
            
            # Select random point to perturb
            idx = np.random.randint(len(test_points))
            
            # Small random perturbation
            delta = np.random.normal(0, 0.005, 2)
            test_points[idx] += delta
            
            # Keep within bounds
            test_points[idx] = np.clip(test_points[idx], 0, 1)
            
            # Test the new configuration
            test_ratio = compute_min_max_ratio(test_points)
            
            if test_ratio > best_ratio:
                best_ratio = test_ratio
                best_points = test_points.copy()
                
        return best_points
    
    # Main optimization loop
    # Step 1: Generate initial configuration using spherical Voronoi-inspired approach
    points = spherical_voronoi_initialization()
    
    # Step 2: Multi-scale refinement to improve distribution
    points = multi_scale_refinement(points)
    
    # Step 3: Geometric optimization to fine-tune
    points = geometric_optimization_step(points)
    
    # Step 4: Final optimization using gradient-like approach
    # We'll do one final pass of our distance balance
    points = distance_balance_step(points, learning_rate=0.02)
    
    # Step 5: One more geometric optimization pass
    points = geometric_optimization_step(points)
    
    return points

# EVOLVE-BLOCK-END