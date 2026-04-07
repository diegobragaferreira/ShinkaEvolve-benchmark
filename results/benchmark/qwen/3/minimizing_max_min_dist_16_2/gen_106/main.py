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
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return np.min(distances) / max_dist

    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        return -compute_min_max_ratio(points)

    def hexagonal_lattice_points():
        """Generate points in a precise hexagonal lattice pattern"""
        # Create a 4x4 hexagonal grid properly spaced
        points = []
        
        # Perfect hexagonal packing parameters
        row_spacing = np.sqrt(3) / 2.0
        col_spacing = 1.0
        
        # Generate points in hexagonal pattern
        for row in range(4):
            for col in range(4):
                if len(points) >= 16:
                    break
                x = col * col_spacing
                # Offset odd rows  
                if row % 2 == 1:
                    x += col_spacing / 2.0
                y = row * row_spacing
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to fit in [0,1]x[0,1] while maintaining hexagonal properties
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        
        if max_x <= min_x or max_y <= min_y:
            # Fallback for degenerate case
            return np.random.rand(16, 2)
        
        # Scale to fit within bounds
        scale_x = 1.0 / (max_x - min_x)
        scale_y = 1.0 / (max_y - min_y)
        scale = min(scale_x, scale_y, 0.9)  # Leave margin
        
        points[:, 0] = (points[:, 0] - min_x) * scale
        points[:, 1] = (points[:, 1] - min_y) * scale
        
        # Center in unit square
        center_shift_x = 0.5 - (np.max(points[:, 0]) + np.min(points[:, 0])) / 2.0
        center_shift_y = 0.5 - (np.max(points[:, 1]) + np.min(points[:, 1])) / 2.0
        
        points[:, 0] += center_shift_x
        points[:, 1] += center_shift_y
        
        return points

    def generate_initial_configuration():
        """Create high-quality initial configuration using hexagonal lattice with symmetry breaking"""
        # Start with precise hexagonal lattice
        points = hexagonal_lattice_points()
        
        # Apply sophisticated symmetry breaking
        center = np.mean(points, axis=0)
        
        # Apply deterministic rotation to break rotational symmetry
        # Use irrational rotation angle to avoid periodic patterns
        rotation_angle = np.pi * (3 - np.sqrt(5))  # Golden angle
        cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply to strategic points to break symmetry
        strategy_indices = [0, 3, 5, 8, 10, 13, 15]  # Key positions
        for i in strategy_indices:
            if i < len(points):
                points[i] = rotation_matrix @ (points[i] - center) + center
        
        # Apply adaptive perturbations that depend on position
        distances_from_center = np.sqrt(np.sum((points - center)**2, axis=1))
        max_dist = np.max(distances_from_center)
        
        if max_dist > 0:
            normalized_dists = distances_from_center / max_dist
            # Points farther from center get smaller perturbations, closer get larger
            perturbation_magnitude = 0.008 * (1 - normalized_dists) + 0.002
            
            np.random.seed(42)
            perturbations = np.random.normal(0, 0.005, points.shape)
            perturbations *= perturbation_magnitude.reshape(-1, 1)
            points += perturbations
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        
        return points

    def local_improvement(points, max_iter=100):
        """Apply local improvements to enhance the configuration"""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Iterative local search with adaptive step sizes
        for iteration in range(max_iter):
            # Try small perturbations to each point
            np.random.seed(iteration + 42)
            improved = False
            
            for i in range(len(current_points)):
                # Save current state
                old_point = current_points[i].copy()
                
                # Small random perturbation
                delta = np.random.normal(0, 0.002, 2)
                current_points[i] += delta
                
                # Ensure within bounds
                current_points[i] = np.clip(current_points[i], 0, 1)
                
                # Test improvement
                new_ratio = compute_min_max_ratio(current_points)
                
                if new_ratio > current_ratio:
                    current_ratio = new_ratio
                    improved = True
                else:
                    # Revert if no improvement
                    current_points[i] = old_point
            
            # If no improvement in this iteration, reduce step size
            if not improved:
                break
                
        return current_points

    def constrained_optimization(points):
        """Refine using constrained optimization with better tolerance"""
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        # Simple local optimization approach
        try:
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                
                # Only accept if actually better
                if ratio > compute_min_max_ratio(points):
                    return final_points
        except:
            pass
            
        return points

    # Main algorithm
    best_points = None
    best_ratio = -np.inf
    
    # Try multiple initialization strategies with different seeds
    for seed_val in [42, 123, 456, 789, 999]:
        np.random.seed(seed_val)
        
        # Generate initial configuration
        points = generate_initial_configuration()
        
        # Apply local improvements
        improved_points = local_improvement(points)
        
        # Final optimization
        optimized_points = constrained_optimization(improved_points)
        
        # Evaluate
        ratio = compute_min_max_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # Fallback if everything failed
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.rand(16, 2)
    
    return best_points

# EVOLVE-BLOCK-END