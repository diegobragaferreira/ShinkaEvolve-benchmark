# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    np.random.seed(42)
    
    def generate_precise_hexagonal_grid(n_points=16):
        """Generate mathematically precise hexagonal grid with optimal packing."""
        # Create a hexagonal arrangement that naturally fits 16 points
        # Using sqrt(3)/2 spacing for true hexagonal packing
        
        # Determine grid dimensions
        rows = int(math.ceil(math.sqrt(n_points)))
        cols = int(math.ceil(n_points / rows))
        
        # Ensure we have enough space for all points
        if rows * cols < n_points:
            cols += 1
            
        # Calculate spacing for perfect hexagonal packing
        row_spacing = np.sqrt(3) / 2.0
        col_spacing = 1.0
        
        # Generate base hexagonal grid with proper offsets
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                # Apply hexagonal offset pattern
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                points.append([x, y])
        
        points = np.array(points[:n_points])
        
        # Normalize to [0,1] x [0,1] with proper aspect ratio preservation
        if len(points) > 1:
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            
            # Handle case where there's no spread
            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
            elif x_max > x_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            elif y_max > y_min:
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        
        return points
    
    def apply_deterministic_symmetry_breaking(points):
        """Apply mathematically derived perturbations to break all symmetries."""
        # Use golden ratio and Fibonacci sequences for deterministic yet diverse perturbations
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Apply perturbations with Fibonacci-derived indexing for variety
        for i in range(len(points)):
            # Fibonacci-like indexing for deterministic but non-repeating pattern
            fib_idx = int(((golden_ratio ** i) / np.sqrt(5)) % len(points))
            
            # Different perturbation strengths based on position
            if i % 7 == 0:  # Every 7th point - largest perturbation
                scale = 0.025
            elif i % 5 == 0:  # Every 5th point - medium perturbation  
                scale = 0.015
            else:  # Others - small perturbation
                scale = 0.008
                
            # Apply directional perturbation with golden ratio phase
            phase = (i * golden_ratio) % 1
            theta = phase * 2 * np.pi
            
            # Directional perturbation that breaks rotational symmetry
            dx = scale * np.cos(theta) * (0.7 + np.random.random() * 0.3)
            dy = scale * np.sin(theta) * (0.7 + np.random.random() * 0.3)
            
            points[i, 0] += dx
            points[i, 1] += dy
            
        # Ensure bounds remain valid
        points = np.clip(points, 0, 1)
        return points
    
    def compute_min_max_ratio_efficient(points):
        """Efficiently compute the min/max distance ratio using KDTree."""
        if len(points) < 2:
            return 0.0
            
        # Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(points)
        
        # Find minimum distance (excluding self-distance)
        distances, indices = tree.query(points, k=2)
        d_min = np.min(distances[:, 1])  # Second closest (not self)
        
        # Find maximum distance using direct computation for small sets
        distances_matrix = cdist(points, points)
        np.fill_diagonal(distances_matrix, np.inf)
        d_max = np.max(distances_matrix)
        
        # Avoid division by zero
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def transform_to_polar_and_optimize(points):
        """Transform to polar coordinates and optimize angular distribution."""
        # Convert to polar coordinates
        def to_polar(points):
            # Center the points
            center = np.mean(points, axis=0)
            centered = points - center
            
            # Convert to polar
            r = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
            theta = np.arctan2(centered[:, 1], centered[:, 0])
            
            # Normalize angles to [0, 2π]
            theta = (theta + 2*np.pi) % (2*np.pi)
            
            return r, theta, center
            
        def from_polar(r, theta, center):
            # Convert back to cartesian coordinates
            x = r * np.cos(theta) + center[0]
            y = r * np.sin(theta) + center[1]
            return np.column_stack([x, y])
        
        # Convert to polar coordinates
        r, theta, center = to_polar(points)
        
        # Optimize radial positions to avoid clustering
        def optimize_radial_distribution(r_values):
            r_opt = r_values.copy()
            mean_r = np.mean(r_opt)
            
            # Adjust radial positions to prevent clustering
            for i in range(len(r_opt)):
                # If point is too close to center, push outward
                if r_opt[i] < mean_r * 0.6:
                    r_opt[i] = min(r_opt[i] * 1.2, mean_r * 0.8)
                # If point is too far from center, pull inward
                elif r_opt[i] > mean_r * 1.4:
                    r_opt[i] = max(r_opt[i] * 0.8, mean_r * 1.2)
                    
            return np.clip(r_opt, 0.001, 0.5)  # Ensure valid radial bounds
        
        # Optimize angular distribution for uniformity
        def optimize_angular_distribution(theta_values):
            # Sort angles for easier manipulation
            sorted_indices = np.argsort(theta_values)
            sorted_thetas = theta_values[sorted_indices]
            
            # Distribute angles more uniformly
            n = len(sorted_thetas)
            target_angle = 2 * np.pi / n
            
            # Distribute angles more evenly
            for i in range(n):
                # Position this angle to maintain spacing around the circle
                expected_pos = i * target_angle
                current_pos = sorted_thetas[i]
                
                # Make small adjustments to distribute angles more evenly
                theta_diff = expected_pos - current_pos
                # Only adjust by small amounts to preserve structure
                sorted_thetas[i] += theta_diff * 0.15
                
            # Keep angles in [0, 2π]
            sorted_thetas = (sorted_thetas + 2*np.pi) % (2*np.pi)
            
            # Reorder back to original order
            theta_opt = np.empty_like(theta_values)
            theta_opt[sorted_indices] = sorted_thetas
            
            return theta_opt
        
        # Apply radial and angular optimizations
        r_opt = optimize_radial_distribution(r)
        theta_opt = optimize_angular_distribution(theta)
        
        # Convert back to Cartesian coordinates
        refined_points = from_polar(r_opt, theta_opt, center)
        
        # Ensure all points are within bounds
        refined_points = np.clip(refined_points, 0, 1)
        
        return refined_points
    
    def multi_scale_refinement(initial_points):
        """Perform multi-resolution refinement using different optimization scales."""
        points = initial_points.copy()
        
        # Multi-scale approach: coarse to fine refinement
        scales = [0.05, 0.02, 0.005]  # Different optimization scales
        
        for scale in scales:
            # For each scale, perform several iterations
            for _ in range(5):
                # Apply polar coordinate optimization
                points = transform_to_polar_and_optimize(points)
                
                # Apply small random perturbations
                np.random.seed(42)
                indices = np.random.choice(len(points), size=max(1, len(points)//4), replace=False)
                for idx in indices:
                    delta = np.random.normal(0, scale, 2)
                    points[idx] += delta
                    points[idx] = np.clip(points[idx], 0, 1)
        
        return points
    
    # Main optimization workflow
    # Step 1: Generate precise hexagonal grid
    base_points = generate_precise_hexagonal_grid(16)
    
    # Step 2: Apply deterministic symmetry breaking
    points = apply_deterministic_symmetry_breaking(base_points)
    
    # Step 3: Multi-scale refinement
    refined_points = multi_scale_refinement(points)
    
    # Step 4: Final polar space optimization
    final_points = transform_to_polar_and_optimize(refined_points)
    
    # Step 5: Final validation and boundary correction
    final_points = np.clip(final_points, 0, 1)
    
    return final_points

# EVOLVE-BLOCK-END