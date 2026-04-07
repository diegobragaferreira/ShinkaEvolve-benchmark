# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sphere-packing inspired initialization combined with hybrid evolutionary-local optimization.
    """
    
    def objective(x):
        # Reshape flat array back to points
        pts = x.reshape(16, 2)

        # Calculate all pairwise distances efficiently using scipy
        distances = pdist(pts)

        # Handle case where there are no distances (shouldn't happen)
        if len(distances) == 0:
            return 0

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize the ratio
        if max_dist <= 1e-12:  # Avoid division by zero
            return 0
        return -min_dist / max_dist
    
    def sphere_packing_initialization():
        """Initialize points using a sphere-packing inspired approach."""
        points = []
        # Start with the center point
        points.append([0.5, 0.5])
        
        # Add remaining 15 points using greedy placement
        # Place each new point to maximize its minimum distance to existing points
        for i in range(15):
            best_point = None
            best_min_dist = -1
            
            # Sample many candidate locations to find good placement
            for _ in range(1000):
                # Random sample in [0.05, 0.95] to avoid boundary issues
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                
                # Calculate minimum distance to all existing points
                min_dist = float('inf')
                for px, py in points:
                    dist_sq = (x - px)**2 + (y - py)**2
                    min_dist = min(min_dist, dist_sq)
                
                # Prefer points that are farther from existing points
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_point = [x, y]
            
            if best_point is not None:
                points.append(best_point)
        
        return np.array(points)
    
    def fibonacci_spiral_initialization():
        """Alternative initialization using Fibonacci spiral."""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            theta = math.acos(-1 + (2 * i) / 15)  # elevation angle
            phi_angle = (i * 2 * math.pi) / (phi * phi)  # azimuthal angle
            
            # Convert to cartesian coordinates
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)
            
            # Map to [0.05, 0.95] range to avoid boundaries
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def grid_initialization():
        """Initialize using a 4x4 grid pattern."""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])
        return np.array(points)
    
    def generate_initial_configs():
        """Generate multiple high-quality initial configurations."""
        configs = []
        
        # Sphere packing initialization
        np.random.seed(42)
        configs.append(sphere_packing_initialization())
        
        # Fibonacci spiral
        configs.append(fibonacci_spiral_initialization())
        
        # Regular grid
        configs.append(grid_initialization())
        
        # Perturbed versions of grid
        for i in range(3):
            np.random.seed(42 + i)
            perturbed = configs[2] + np.random.normal(0, 0.02, configs[2].shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            configs.append(perturbed)
        
        return configs
    
    # Generate multiple initial configurations
    initial_configs = generate_initial_configs()
    
    best_ratio = -np.inf
    best_points = None
    
    # Hyperparameters for hybrid approach
    max_global_iter = 20
    max_local_iter = 50
    tolerance = 1e-10
    
    # Try optimization from different starting points
    for i, initial_config in enumerate(initial_configs):
        # Define bounds (points must be in [0.001, 0.999] x [0.001, 0.999] to avoid edge issues)
        bounds = [(0.001, 0.999) for _ in range(32)]
        
        # Hybrid approach: Global exploration followed by local refinement
        try:
            # Step 1: Global search with coarse optimization
            x0 = initial_config.flatten()
            
            # Use a modified L-BFGS-B approach for initial coarse optimization
            coarse_result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_global_iter, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if coarse_result.success:
                # Step 2: Refinement with tighter tolerances
                refined_result = minimize(
                    objective,
                    coarse_result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_local_iter, 'ftol': tolerance, 'gtol': tolerance}
                )
                
                if refined_result.success:
                    optimized_points = refined_result.x.reshape(16, 2)
                    
                    # Calculate the actual ratio for this solution
                    distances = pdist(optimized_points)
                    if len(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        if max_dist > 0:
                            ratio = min_dist / max_dist
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
            
        except Exception as e:
            continue  # Skip this configuration if optimization fails
    
    # If no good solution was found, return the best of our initial configurations
    if best_points is None:
        # Evaluate all initial configurations to select the best one
        max_ratio = -np.inf
        for config in initial_configs:
            distances = pdist(config)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_points = config.copy()
    
    # Ensure we have a valid result
    if best_points is None:
        # Fallback to the sphere packing initialization
        np.random.seed(42)
        best_points = sphere_packing_initialization()
    
    return best_points

# EVOLVE-BLOCK-END