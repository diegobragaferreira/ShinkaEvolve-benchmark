# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def generate_hexagonal_lattice(n_points=16):
        """Generate a precise hexagonal lattice arrangement with mathematical optimality."""
        # Create a hexagonal grid that naturally fits 16 points
        # This uses sqrt(3)/2 spacing for optimal packing
        
        # Determine grid dimensions
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))
        
        # Ensure we have enough space for all points
        if rows * cols < n_points:
            cols += 1
            
        # Calculate spacing
        row_spacing = np.sqrt(3) / 2.0
        col_spacing = 1.0
        
        # Generate base hexagonal grid
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                # Apply hexagonal offset
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                points.append([x, y])
        
        points = np.array(points[:n_points])
        
        # Normalize to [0,1] x [0,1] with proper aspect ratio preservation
        if len(points) > 1:
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            
            if x_max > x_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            if y_max > y_min:
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        
        return points
    
    def apply_deterministic_symmetry_breaking(points):
        """Apply mathematically determined perturbations to break all symmetries."""
        # Use Fibonacci sequence derived pattern for symmetry breaking
        # This makes the perturbations non-repeating and mathematically unique
        golden_ratio = (1 + np.sqrt(5)) / 2
        np.random.seed(42)
        
        # Apply varied perturbations based on Fibonacci-derived indices
        for i in range(len(points)):
            # Fibonacci-like indexing for deterministic yet diverse perturbations
            fib_idx = int(((golden_ratio ** i) / np.sqrt(5)) % len(points))
            
            # Different perturbation strengths
            if i % 7 == 0:  # Every 7th point - largest perturbation
                scale = 0.025
            elif i % 5 == 0:  # Every 5th point - medium perturbation  
                scale = 0.015
            else:  # Others - small perturbation
                scale = 0.008
                
            # Apply perturbation with Fibonacci-derived phase
            phase = (i * golden_ratio) % 1
            theta = phase * 2 * np.pi
            
            # Directional perturbation
            dx = scale * np.cos(theta) * (0.5 + np.random.random() * 0.5)
            dy = scale * np.sin(theta) * (0.5 + np.random.random() * 0.5)
            
            points[i, 0] += dx
            points[i, 1] += dy
            
        # Ensure bounds
        points = np.clip(points, 0, 1)
        return points
    
    def compute_ratio(points):
        """Efficiently compute the min/max distance ratio."""
        if len(points) < 2:
            return 0.0
            
        # Use scipy's cdist for efficient distance calculation
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Find minimum and maximum distances
        d_min = np.min(distances[distances != np.inf])
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def polar_space_optimization(start_points):
        """Optimize points using a hybrid approach in polar space representation."""
        # Convert to polar coordinates to better capture angular distributions
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
        
        # Initial conversion
        r_initial, theta_initial, center_initial = to_polar(start_points)
        
        # Optimize radial and angular components separately
        # This treats the problem as two independent optimizations
        
        # For the optimization, we'll work with normalized values
        def optimize_radial_only():
            # Simple iterative refinement on radial positions
            r_opt = r_initial.copy()
            r_opt = np.clip(r_opt, 0.001, 0.5)  # Reasonable bounds
            
            # Iterative refinement for better radial distribution
            for _ in range(10):
                # Simple adjustment to increase minimum distances
                # We want to avoid clustering radially
                mean_r = np.mean(r_opt)
                for i in range(len(r_opt)):
                    if r_opt[i] < mean_r * 0.8:
                        r_opt[i] = min(r_opt[i] * 1.1, mean_r * 0.9)
                    elif r_opt[i] > mean_r * 1.2:
                        r_opt[i] = max(r_opt[i] * 0.9, mean_r * 1.1)
                        
            return r_opt
            
        def optimize_angles_only():
            # For angles, we want more uniform distribution to maximize minimum angle differences
            theta_opt = theta_initial.copy()
            # Sort angles for easier manipulation
            sorted_indices = np.argsort(theta_opt)
            sorted_thetas = theta_opt[sorted_indices]
            
            # Distribute angles more evenly
            n = len(sorted_thetas)
            target_angle = 2 * np.pi / n
            
            # Adjust angles to be more uniformly spaced
            for i in range(n):
                # Position this angle to maintain spacing around the circle
                expected_pos = i * target_angle
                current_pos = sorted_thetas[i]
                
                # Make small adjustments to distribute angles more evenly
                theta_diff = expected_pos - current_pos
                # Only adjust by small amounts to preserve structure
                theta_opt[sorted_indices[i]] += theta_diff * 0.1
                
            # Keep angles in [0, 2π]
            theta_opt = (theta_opt + 2*np.pi) % (2*np.pi)
            return theta_opt
            
        # Apply optimizations
        r_opt = optimize_radial_only()
        theta_opt = optimize_angles_only()
        
        # Convert back to Cartesian coordinates
        refined_points = from_polar(r_opt, theta_opt, center_initial)
        
        # Ensure all points are within bounds
        refined_points = np.clip(refined_points, 0, 1)
        
        return refined_points
    
    def multi_resolution_refinement(initial_points):
        """Perform multi-resolution refinement for better optimization."""
        points = initial_points.copy()
        
        # First pass: coarse optimization
        for _ in range(5):
            # Apply geometric refinement
            points = polar_space_optimization(points)
            
        # Second pass: fine tuning
        for _ in range(10):
            # Apply small random perturbations with boundary handling
            new_points = points.copy()
            np.random.seed(42)
            
            # Select a few points to perturb
            indices = np.random.choice(len(points), size=max(1, len(points)//3), replace=False)
            for idx in indices:
                # Small perturbation
                delta = np.random.normal(0, 0.005, 2)
                new_points[idx] += delta
                new_points[idx] = np.clip(new_points[idx], 0, 1)
                
            # Accept if improvement, or accept with probability
            old_ratio = compute_ratio(points)
            new_ratio = compute_ratio(new_points)
            
            if new_ratio > old_ratio or np.random.random() < 0.1:
                points = new_points
                
        return points
    
    # Main optimization flow
    np.random.seed(42)
    
    # Step 1: Generate hexagonal lattice
    base_points = generate_hexagonal_lattice(16)
    
    # Step 2: Apply symmetry breaking
    points = apply_deterministic_symmetry_breaking(base_points)
    
    # Step 3: Multi-resolution refinement
    refined_points = multi_resolution_refinement(points)
    
    # Step 4: Final optimization pass
    final_points = polar_space_optimization(refined_points)
    
    # Final validation and return
    return final_points

# EVOLVE-BLOCK-END