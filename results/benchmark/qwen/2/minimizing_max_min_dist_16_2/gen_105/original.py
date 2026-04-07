# EVOLVE-BLOCK-START
import numpy as np
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
    
    # Initialize with grid pattern
    np.random.seed(42)
    
    # Create a 4x4 grid pattern
    points = np.zeros((16, 2))
    for i in range(4):
        for j in range(4):
            # Grid positions
            x = j * 0.25 + (i % 2) * 0.125
            y = i * 0.25
            points[i*4 + j] = [x, y]
    
    # Add slight perturbations to break symmetry
    for i in range(16):
        points[i, 0] += (np.random.random() - 0.5) * 0.05
        points[i, 1] += (np.random.random() - 0.5) * 0.05
    
    # Clip to unit square
    points = np.clip(points, 0, 1)
    
    # Apply progressive refinement with adaptive perturbations
    current_ratio = compute_min_max_ratio(points)
    best_points = points.copy()
    best_ratio = current_ratio
    
    # Main refinement loop with multiple passes
    for iteration in range(100):
        # Track improvement
        improved = False
        
        # For each point, try moving it to improve the ratio
        for point_idx in range(16):
            # Save current point
            original_point = points[point_idx].copy()
            
            # Try several nearby positions to find a better one
            best_local_ratio = current_ratio
            best_local_position = original_point.copy()
            
            # Try a dense neighborhood of possible moves
            for dx in [-0.05, -0.025, 0, 0.025, 0.05]:
                for dy in [-0.05, -0.025, 0, 0.025, 0.05]:
                    # Calculate new position
                    new_x = original_point[0] + dx
                    new_y = original_point[1] + dy
                    
                    # Ensure within bounds
                    new_x = np.clip(new_x, 0, 1)
                    new_y = np.clip(new_y, 0, 1)
                    
                    # Temporarily move point
                    points[point_idx] = [new_x, new_y]
                    
                    # Calculate new ratio
                    new_ratio = compute_min_max_ratio(points)
                    
                    # If better, remember it
                    if new_ratio > best_local_ratio:
                        best_local_ratio = new_ratio
                        best_local_position = [new_x, new_y]
                        
                    # Restore original position
                    points[point_idx] = original_point
            
            # Apply the best local move if it improved the ratio
            if best_local_ratio > current_ratio:
                points[point_idx] = best_local_position
                current_ratio = best_local_ratio
                improved = True
                
                # Update best overall
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = points.copy()
        
        # Early stopping if no improvement for several iterations
        if not improved and iteration > 20:
            break
    
    # Final optimization pass using a greedy hill-climbing approach
    for _ in range(50):
        improved = False
        for point_idx in range(16):
            original_point = points[point_idx].copy()
            
            # Try a finer grid of moves
            best_move_ratio = current_ratio
            best_move_position = original_point.copy()
            
            # Sample potential moves more densely
            for dx in np.linspace(-0.03, 0.03, 7):
                for dy in np.linspace(-0.03, 0.03, 7):
                    new_x = original_point[0] + dx
                    new_y = original_point[1] + dy
                    
                    new_x = np.clip(new_x, 0, 1)
                    new_y = np.clip(new_y, 0, 1)
                    
                    points[point_idx] = [new_x, new_y]
                    
                    new_ratio = compute_min_max_ratio(points)
                    
                    if new_ratio > best_move_ratio:
                        best_move_ratio = new_ratio
                        best_move_position = [new_x, new_y]
                    
                    points[point_idx] = original_point
            
            if best_move_ratio > current_ratio:
                points[point_idx] = best_move_position
                current_ratio = best_move_ratio
                improved = True
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = points.copy()
        
        if not improved:
            break
    
    return best_points

# EVOLVE-BLOCK-END