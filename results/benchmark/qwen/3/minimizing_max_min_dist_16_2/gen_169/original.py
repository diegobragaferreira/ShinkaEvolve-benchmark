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
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def perturb_points(points, magnitude=0.01):
        """Apply small perturbations to points with boundary checking."""
        new_points = points.copy()
        
        # Select random subset of points to perturb
        num_perturb = max(1, len(points) // 4)
        indices = np.random.choice(len(points), size=num_perturb, replace=False)
        
        for idx in indices:
            # Apply perturbation
            delta = np.random.uniform(-magnitude, magnitude, 2)
            new_points[idx] += delta
            
            # Keep within bounds
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
        return new_points
    
    def optimize_with_simulated_annealing():
        """Optimize point configuration using simulated annealing."""
        
        # Initialize with hexagonal grid
        points = np.zeros((16, 2))
        row_spacing = 1.0
        col_spacing = np.sqrt(3) / 2.0
        
        idx = 0
        for row in range(4):
            for col in range(4):
                x = col * 1.0 + (row % 2) * 0.5  # Offset every other row
                y = row * col_spacing
                points[idx, 0] = x
                points[idx, 1] = y
                idx += 1
        
        # Normalize to unit square [0,1] x [0,1]
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        if x_max > x_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        if y_max > y_min:
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        points += np.random.uniform(-0.02, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        
        # Simulated Annealing parameters
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Adaptive cooling schedule
        temperature = 0.1
        cooling_rate = 0.9995
        min_temp = 1e-6
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        max_recent = 100
        
        # Optimization loop
        max_iterations = 5000
        for iteration in range(max_iterations):
            # Perturb points
            new_points = perturb_points(current_points, magnitude=temperature * 0.5)
            
            # Evaluate new configuration
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temperature):
                current_points = new_points
                current_ratio = new_ratio
                
                # Update best solution
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
                    
                    # Reset improvement tracking
                    recent_improvements = []
                else:
                    # Track recent improvements
                    if len(recent_improvements) < max_recent:
                        recent_improvements.append(current_ratio)
                    else:
                        recent_improvements.pop(0)
                        recent_improvements.append(current_ratio)
            
            # Adaptive cooling based on recent improvement
            if len(recent_improvements) >= 10:
                recent_avg = np.mean(recent_improvements[-10:])
                if recent_avg < 0.01 * best_ratio:
                    cooling_rate = min(cooling_rate * 0.99, 0.9999)
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
                
            # Periodic reporting
            if iteration % 1000 == 0:
                pass  # Could add progress logging here
        
        return best_points
    
    # Run optimization
    result = optimize_with_simulated_annealing()
    
    return result


# EVOLVE-BLOCK-END