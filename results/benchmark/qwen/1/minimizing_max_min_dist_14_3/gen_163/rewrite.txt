# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
import random
from scipy.spatial.transform import Rotation as R

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    random.seed(42)

    n = 14
    d = 3
    
    # Generate points on torus surface for better distribution
    def torus_points(samples=14, R=1.0, r=0.3):
        """Generate points on a torus surface"""
        points = []
        for i in range(samples):
            # Parameterize torus
            u = 2 * np.pi * i / samples
            v = 2 * np.pi * np.random.random()
            
            # Torus equation
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            
            points.append([x, y, z])
        return np.array(points)
    
    # Quaternion-based symmetry breaking
    def quaternion_rotate_point(point, angle_degrees=15):
        """Rotate a point using quaternion for symmetry breaking"""
        # Create random rotation quaternion
        rotation = R.from_euler('xyz', [angle_degrees, angle_degrees*0.5, angle_degrees*1.5], degrees=True)
        return rotation.apply(point)
    
    # Enhanced initialization with torus and quaternion perturbation
    def enhanced_init(samples=14):
        # Start with torus points
        points = torus_points(samples)
        
        # Apply quaternion-based perturbations to break symmetries
        for i in range(samples):
            # Apply small random rotations
            angle = np.random.uniform(0, 30)
            points[i] = quaternion_rotate_point(points[i], angle)
            
        return points
    
    # Multi-objective optimization function
    def multi_objective(x_flat):
        points = x_flat.reshape((n, d))
        
        # Calculate pairwise distances
        distances = pdist(points)
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return np.inf
            
        # Primary objective: maximize min/max ratio
        ratio = d_min / d_max
        
        # Secondary objective: minimize distance variance (encourage uniform distribution)
        if len(distances) > 1:
            var_distance = np.var(distances)
            # Penalize high variance with a factor
            variance_penalty = var_distance / (d_max * d_max) * 0.1
        else:
            variance_penalty = 0.0
            
        # Combined objective: minimize negative ratio plus variance penalty
        return -(ratio - variance_penalty)
    
    # Normalization to unit cube [0,1]^3
    def normalize_to_cube(points):
        centered = points - np.mean(points, axis=0)
        max_coord = np.max(np.abs(centered))
        if max_coord > 0:
            scaled = centered / max_coord * 0.5
        else:
            scaled = centered
        normalized = scaled + 0.5
        return normalized
    
    # Adaptive optimization parameters
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with varying strategies
    num_starts = 35
    
    # Different initialization strategies with varying levels of diversity
    init_strategies = [
        enhanced_init,
        lambda s: torus_points(s),
        lambda s: np.random.rand(s, 3) * 2 - 1,  # Random in [-1,1]^3
        lambda s: np.random.rand(s, 3),  # Random in [0,1]^3
    ]
    
    for start_idx in range(num_starts):
        # Select initialization strategy
        strategy_idx = start_idx % len(init_strategies)
        init_func = init_strategies[strategy_idx]
        
        # Get initial points
        initial_points = init_func(n)
        
        # Normalize to unit cube [0,1]^3
        initial_points = normalize_to_cube(initial_points)
        
        # Add controlled perturbation based on iteration
        if start_idx > 0:
            perturbation_magnitude = max(0.01, 0.05 * np.exp(-start_idx * 0.1))
            perturbation = np.random.normal(0, perturbation_magnitude, initial_points.shape)
            initial_points += perturbation
            initial_points = np.clip(initial_points, 0, 1)
        
        # Flatten for optimization
        initial_flat = initial_points.flatten()
        
        # Optimization bounds: [0,1] for all coordinates
        bounds = [(0, 1) for _ in range(n * d)]
        
        # Adaptive optimization parameters
        maxiter = 1200
        ftol = 1e-12
        gtol = 1e-12
        
        # Try different optimization methods in order of preference
        result = None
        
        try:
            # Try L-BFGS-B first (fast convergence)
            result = minimize(
                multi_objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )
            
            # If L-BFGS-B fails, try SLSQP
            if not result.success:
                result = minimize(
                    multi_objective,
                    initial_flat,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': maxiter//2, 'ftol': ftol, 'gtol': gtol}
                )
                
            # If still failing, try Trust-Region Constrained
            if not result.success:
                result = minimize(
                    multi_objective,
                    initial_flat,
                    method='trust-constr',
                    bounds=bounds,
                    options={'maxiter': maxiter//2, 'ftol': ftol, 'gtol': gtol}
                )
                
        except Exception:
            continue
        
        # Extract optimized points
        if result is not None and result.success:
            optimized_points = result.x.reshape((n, d))
            optimized_points = np.clip(optimized_points, 0, 1)
            
            # Calculate actual ratio
            final_distances = pdist(optimized_points)
            final_distances = final_distances[final_distances > 1e-12]
            
            if len(final_distances) > 0:
                d_min = np.min(final_distances)
                d_max = np.max(final_distances)
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
    
    # If no good solution found, fallback to enhanced initialization
    if best_points is None:
        initial_points = enhanced_init(n)
        initial_points = normalize_to_cube(initial_points)
        return initial_points
    
    # Apply final multi-stage refinement
    refined_points = best_points.copy()
    
    # Stage 1: Very tight L-BFGS-B optimization
    try:
        final_flat = refined_points.flatten()
        refined_result = minimize(
            multi_objective,
            final_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2500, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if refined_result.success:
            candidate_points = refined_result.x.reshape((n, d))
            candidate_points = np.clip(candidate_points, 0, 1)
            
            # Validate improvement
            final_distances_new = pdist(candidate_points)
            final_distances_new = final_distances_new[final_distances_new > 1e-12]
            
            if len(final_distances_new) > 0:
                new_min = np.min(final_distances_new)
                new_max = np.max(final_distances_new)
                if new_max > 0:
                    new_ratio = new_min / new_max
                    if new_ratio > best_ratio:
                        refined_points = candidate_points
    except Exception:
        pass
    
    # Stage 2: Additional trust-constr refinement
    try:
        final_flat = refined_points.flatten()
        refined_result = minimize(
            multi_objective,
            final_flat,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-13}
        )
        
        if refined_result.success:
            candidate_points = refined_result.x.reshape((n, d))
            candidate_points = np.clip(candidate_points, 0, 1)
            
            # Validate improvement
            final_distances_new = pdist(candidate_points)
            final_distances_new = final_distances_new[final_distances_new > 1e-12]
            
            if len(final_distances_new) > 0:
                new_min = np.min(final_distances_new)
                new_max = np.max(final_distances_new)
                if new_max > 0:
                    new_ratio = new_min / new_max
                    if new_ratio > best_ratio:
                        refined_points = candidate_points
    except Exception:
        pass
    
    # Final validation
    final_distances = pdist(refined_points)
    final_distances = final_distances[final_distances > 1e-12]
    
    if len(final_distances) > 0 and np.min(final_distances) > 1e-12:
        return refined_points
    else:
        return best_points

# EVOLVE-BLOCK-END