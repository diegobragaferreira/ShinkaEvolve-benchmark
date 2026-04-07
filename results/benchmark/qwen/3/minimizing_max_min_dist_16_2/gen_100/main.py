# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
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
        ratio = compute_min_max_ratio(points)
        
        # Add penalty for boundary violations
        penalty = 0
        for point in points:
            if point[0] < 0.01 or point[0] > 0.99 or point[1] < 0.01 or point[1] > 0.99:
                penalty -= 1000  # Heavy penalty for being close to boundary
                
        return -ratio + penalty

    def initialize_spherical_projection():
        """Initialize points using spherical code approach with stereographic projection"""
        # Generate points on a sphere using fibonacci spiral method
        n = 16
        points_sphere = []
        
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points_sphere.append([x, y, z])
        
        points_sphere = np.array(points_sphere)
        
        # Apply stereographic projection from south pole to plane
        # Projection formula: P'(x,y,z) = (x/(1-z), y/(1-z))
        points_2d = []
        for point in points_sphere:
            x, y, z = point
            if abs(1 - z) < 1e-10:  # Avoid division by zero
                # Project to infinity - use a large number
                points_2d.append([0, 0])
            else:
                x_proj = x / (1 - z)
                y_proj = y / (1 - z)
                points_2d.append([x_proj, y_proj])
        
        points_2d = np.array(points_2d)
        
        # Normalize to fit in [0,1] x [0,1]
        if len(points_2d) > 0:
            min_x, max_x = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
            min_y, max_y = np.min(points_2d[:, 1]), np.max(points_2d[:, 1])
            
            if max_x > min_x and max_y > min_y:
                scale_x = 1.0 / (max_x - min_x)
                scale_y = 1.0 / (max_y - min_y)
                scale = min(scale_x, scale_y) * 0.8  # Leave margin
                
                points_2d[:, 0] = (points_2d[:, 0] - min_x) * scale
                points_2d[:, 1] = (points_2d[:, 1] - min_y) * scale
                
                # Center in [0,1] x [0,1]
                center_shift_x = 0.5 - (np.max(points_2d[:, 0]) + np.min(points_2d[:, 0])) / 2.0
                center_shift_y = 0.5 - (np.max(points_2d[:, 1]) + np.min(points_2d[:, 1])) / 2.0
                
                points_2d[:, 0] += center_shift_x
                points_2d[:, 1] += center_shift_y
                
                # Ensure within bounds
                points_2d = np.clip(points_2d, 0, 1)
        
        return points_2d

    def initialize_regular_hex():
        """Initialize with regular hexagonal pattern"""
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points.append([x, y])
        
        points = np.array(points[:16])
        points = np.clip(points, 0, 1)
        return points

    def initialize_perturbed_hex():
        """Initialize with perturbed hexagonal pattern"""
        points = initialize_regular_hex()
        np.random.seed(42)
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def initialize_random():
        """Initialize with random points"""
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return points

    def initialize_clustered_avoidance():
        """Initialize with clustered avoidance strategy"""
        points = []
        # Start with 4 points at corners and center
        corners = [[0,0], [1,0], [0,1], [1,1]]
        center = [0.5, 0.5]
        points.extend(corners)
        points.append(center)
        
        # Add remaining points randomly but avoid clustering
        np.random.seed(42)
        for i in range(16 - len(points)):
            # Add a point with some clustering avoidance
            attempt = 0
            while attempt < 100:
                candidate = np.random.rand(2)
                # Check distance to existing points
                min_dist = float('inf')
                for pt in points:
                    dist = np.sqrt(np.sum((candidate - pt)**2))
                    min_dist = min(min_dist, dist)
                if min_dist > 0.1:  # Ensure minimum distance
                    points.append(candidate)
                    break
                attempt += 1
        
        # If we couldn't place enough points, fill with random
        while len(points) < 16:
            points.append(np.random.rand(2))
        
        return np.array(points[:16])

    initial_strategies = [
        initialize_spherical_projection,
        initialize_regular_hex,
        initialize_perturbed_hex,
        initialize_random,
        initialize_clustered_avoidance
    ]

    best_ratio = -float('inf')
    best_points = None

    # Multi-start optimization with different initialization strategies
    num_restarts = 5
    
    for restart in range(num_restarts):
        # Select initialization strategy
        init_func = initial_strategies[restart % len(initial_strategies)]
        points = init_func()
        
        # Flatten for optimization
        x0 = points.flatten()
        
        # Set up bounds
        bounds = [(0, 1) for _ in range(32)]
        
        # Stage 1: Use differential evolution for global search
        try:
            # Adaptive population size based on restart
            popsize = 10 + restart * 2
            
            result_de = differential_evolution(
                objective_function,
                bounds,
                maxiter=300 + restart * 100,
                popsize=min(popsize, 30),
                tol=1e-8,
                seed=42 + restart,
                callback=None
            )
            
            if result_de.success:
                final_points = result_de.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
        except Exception as e:
            pass

    # Stage 2: If no good result from DE, try different approach
    if best_points is None:
        # Try simpler approach with fewer restarts
        for restart in range(3):
            init_func = initial_strategies[restart % len(initial_strategies)]
            points = init_func()
            
            x0 = points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            try:
                # Use SLSQP with tighter tolerances
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-4}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
            except Exception as e:
                continue

    # Stage 3: Final refinement
    if best_points is not None:
        # Do one final refinement with a more thorough local search
        try:
            # Refine with additional optimization
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Try a different optimizer for refinement
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
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
        except Exception as e:
            pass

    # Final fallback if nothing worked
    if best_points is None:
        # Fallback to the spherical projection approach
        points = initialize_spherical_projection()
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-6, 'eps': 1e-4}
            )

            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                # Final fallback to random points
                np.random.seed(42)
                best_points = np.random.rand(16, 2)
        except:
            # Final fallback
            np.random.seed(42)
            best_points = np.random.rand(16, 2)

    return best_points

# EVOLVE-BLOCK-END