# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently using cdist
        distances = cdist(points, points)
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def objective(x):
        """Objective function to maximize min/max distance ratio (minimize negative ratio)."""
        # Reshape flat array back to points
        points = x.reshape(-1, 2)
        # Return negative ratio since we want to maximize
        return -compute_min_max_ratio(points)

    def adaptive_hexagonal_init():
        """Initialize points using hexagonal packing pattern with adaptive perturbations."""
        # Create a hexagonal lattice pattern
        rows = 4
        cols = 4
        points = []
        
        for i in range(rows):
            for j in range(cols):
                # offset every other row
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125  # Scale and shift to [0.125, 0.875]
                y = i * 0.25 + 0.125
                
                # Add adaptive perturbation based on position
                if (i == 0 or i == rows-1) and (j == 0 or j == cols-1):
                    # Corner points - smallest perturbation
                    perturbation = 0.005
                elif i == 0 or i == rows-1 or j == 0 or j == cols-1:
                    # Edge points - medium perturbation  
                    perturbation = 0.01
                else:
                    # Interior points - largest perturbation
                    perturbation = 0.015
                    
                x += np.random.normal(0, perturbation)
                y += np.random.normal(0, perturbation)
                points.append([x, y])
        
        # Ensure all points are within bounds
        result = np.array(points)
        result = np.clip(result, 0, 1)
        return result[:16]

    def perturbed_grid_init():
        """Initialize with grid points plus adaptive random perturbations."""
        # Start with a regular grid
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
        points = grid_points.astype(float) / 3.0  # Normalize to [0,1] range
        
        # Adaptive perturbation based on point position
        for i in range(len(points)):
            # Determine perturbation strength based on location
            x, y = points[i]
            # Points near corners should have smaller perturbations
            if (abs(x - 0) < 0.1 or abs(x - 1) < 0.1) and (abs(y - 0) < 0.1 or abs(y - 1) < 0.1):
                perturbation_magnitude = 0.008
            elif abs(x - 0.5) < 0.2 and abs(y - 0.5) < 0.2:
                # Center points allow more variation
                perturbation_magnitude = 0.02
            else:
                # Edge points
                perturbation_magnitude = 0.015
                
            points[i] += np.random.uniform(-perturbation_magnitude, perturbation_magnitude, 2)
        
        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)
        return points

    def random_spread_init():
        """Initialize with random points that are intentionally spread out."""
        np.random.seed(42)
        points = np.random.rand(16, 2)
        
        # Apply some basic spacing to prevent clustering with adaptive factors
        for i in range(16):
            # Move points away from center slightly with adaptive factor
            center_vec = points[i] - [0.5, 0.5]
            center_distance = np.linalg.norm(center_vec)
            if center_distance > 0:
                # Adaptive movement based on distance from center
                adaptive_factor = min(0.2, 0.1 / (center_distance + 0.01))
                points[i] += center_vec * adaptive_factor
                
        # Clip to ensure within bounds
        points = np.clip(points, 0, 1)
        return points

    def smart_optimization(initial_points, time_limit):
        """Apply intelligent optimization with adaptive parameters."""
        n = 16
        d = 2
        start_time = time.time()
        
        # Define bounds (points must be in [0,1] x [0,1])
        bounds = [(0, 1) for _ in range(n * d)]
        
        # Try multiple optimization methods with appropriate settings
        best_result = None
        best_value = np.inf
        
        # Method 1: L-BFGS-B with conservative settings for reliable convergence
        if time_limit > 30:
            try:
                result1 = minimize(
                    objective,
                    initial_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if result1.fun < best_value and result1.success:
                    best_value = result1.fun
                    best_result = result1
            except:
                pass
        
        # Method 2: SLSQP for tighter constraint handling
        if time_limit > 45 and best_result is None:
            try:
                result2 = minimize(
                    objective,
                    initial_points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-11}
                )
                if result2.fun < best_value and result2.success:
                    best_value = result2.fun
                    best_result = result2
            except:
                pass
        
        # Method 3: Nelder-Mead as fallback
        if time_limit > 20 and best_result is None:
            try:
                result3 = minimize(
                    objective,
                    initial_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'fatol': 1e-11, 'xatol': 1e-11}
                )
                if result3.fun < best_value and result3.success:
                    best_value = result3.fun
                    best_result = result3
            except:
                pass
        
        # Extract optimized points or return original if optimization failed
        if best_result is not None:
            optimized_points = best_result.x.reshape(n, d)
        else:
            optimized_points = initial_points.copy()
        
        # Ensure points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)
        return optimized_points

    # Multi-start optimization with strategic initialization
    best_ratio = -np.inf
    best_points = None
    start_time = time.time()
    time_limit = 180.0
    
    # Strategy 1: Adaptive hexagonal pattern
    try:
        np.random.seed(42)
        points = adaptive_hexagonal_init()
        optimized_points = smart_optimization(points, time_limit - (time.time() - start_time))
        ratio = compute_min_max_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Perturbed grid
    try:
        points = perturbed_grid_init()
        optimized_points = smart_optimization(points, time_limit - (time.time() - start_time))
        ratio = compute_min_max_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except Exception as e:
        pass
    
    # Strategy 3: Random spread
    try:
        points = random_spread_init()
        optimized_points = smart_optimization(points, time_limit - (time.time() - start_time))
        ratio = compute_min_max_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except Exception as e:
        pass
    
    # Strategy 4: Deterministic fallback
    if best_points is None:
        fallback_points = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.25, 0.75], [0.75, 0.75],
            [0.1, 0.1], [0.9, 0.1],
            [0.1, 0.9], [0.9, 0.9],
            [0.3, 0.5], [0.7, 0.5],
            [0.5, 0.3], [0.5, 0.7],
            [0.4, 0.4], [0.6, 0.6],
            [0.4, 0.6], [0.6, 0.4]
        ])
        return fallback_points
    
    return best_points

# EVOLVE-BLOCK-END