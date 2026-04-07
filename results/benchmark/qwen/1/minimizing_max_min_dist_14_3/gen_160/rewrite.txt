# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
from scipy.spatial.transform import Rotation as R

def generate_spherical_voronoi_initialization(n_points):
    """Generate initial point distribution using Spherical Voronoi concepts"""
    # Create a coarse distribution using Fibonacci-like approach but more evenly spaced
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    
    # Generate points that are more evenly distributed initially
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i + np.random.uniform(-0.1, 0.1)  # Add some randomness
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-12)
    
    return points

def generate_quaternion_rotated_points(base_points, num_rotations=5):
    """Generate multiple rotated versions of base points using quaternions"""
    rotated_sets = []
    
    for i in range(num_rotations):
        # Generate random rotation quaternion
        random_rotation = R.from_euler('xyz', np.random.uniform(0, 2*np.pi, 3))
        rotated_points = random_rotation.apply(base_points)
        rotated_sets.append(rotated_points)
    
    return rotated_sets

def spherical_voronoi_objective(points, alpha=0.1, beta=0.1):
    """Modified objective function using Spherical Voronoi principles"""
    if len(points) < 2:
        return 0.0
    
    # Calculate pairwise distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist < 1e-12:
        return 0.0
    
    # Ratio to maximize
    ratio = min_dist / max_dist
    
    # Additional penalty for highly non-uniform distributions (variance in distances)
    if len(distances) > 0:
        dist_variance = np.var(distances)
        dist_mean = np.mean(distances)
        if dist_mean > 1e-12:
            uniformity_penalty = dist_variance / (dist_mean ** 2)
            # Combine ratio with penalty (we want to maximize ratio and minimize penalty)
            return ratio - alpha * uniformity_penalty
    
    return ratio

def adaptive_constraint_tightening(current_iter, max_iter, initial_radius=1.0):
    """Adaptively tighten the constraint during optimization"""
    # Start with a slightly larger radius and gradually tighten
    return initial_radius * (0.9 + 0.1 * (current_iter / max_iter))

def constrained_optimization_pipeline(points, max_iter=1000):
    """Enhanced constrained optimization using multiple methods"""
    n, d = points.shape
    
    def objective(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points_reshaped, axis=1, keepdims=True)
        normalized_points = points_reshaped / np.maximum(norms, 1e-12)
        return -spherical_voronoi_objective(normalized_points)
    
    def constraint_sphere(x_flat):
        points_reshaped = x_flat.reshape(n, d)
        norms = np.linalg.norm(points_reshaped, axis=1)
        return norms - 1.0
    
    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(n * d)]
    
    best_points = points.copy()
    best_ratio = spherical_voronoi_objective(points)
    
    # Stage 1: L-BFGS-B with moderate constraints for quick convergence
    try:
        result1 = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter//3, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        if result1.success:
            optimized_points = result1.x.reshape(n, d)
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            optimized_points = optimized_points / np.maximum(norms, 1e-12)
            ratio = spherical_voronoi_objective(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Stage 2: Trust-Constr for better local refinement
    try:
        result2 = minimize(
            objective,
            best_points.flatten(),
            method='trust-constr',
            constraints=constraints,
            options={'maxiter': max_iter//3, 'xtol': 1e-10, 'gtol': 1e-10}
        )
        
        if result2.success:
            optimized_points = result2.x.reshape(n, d)
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            optimized_points = optimized_points / np.maximum(norms, 1e-12)
            ratio = spherical_voronoi_objective(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    # Stage 3: SLSQP for final tuning with constraints
    try:
        result3 = minimize(
            objective,
            best_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter//3, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result3.success:
            optimized_points = result3.x.reshape(n, d)
            norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
            optimized_points = optimized_points / np.maximum(norms, 1e-12)
            ratio = spherical_voronoi_objective(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Spherical Voronoi Evolution methodology.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    max_iter = 800
    
    # Generate initial point set using spherical Voronoi-inspired approach
    initial_points = generate_spherical_voronoi_initialization(n)
    
    # Generate multiple rotated variants to explore different regions of the solution space
    rotated_sets = generate_quaternion_rotated_points(initial_points, num_rotations=3)
    
    best_ratio = -np.inf
    best_points = None
    
    # Try each rotated variant
    for i, rotated_set in enumerate(rotated_sets):
        # Add slight random noise to break any remaining symmetries
        np.random.seed(42 + i)
        noisy_points = rotated_set + np.random.normal(0, 0.02, (n, d))
        
        # Normalize to unit sphere
        norms = np.linalg.norm(noisy_points, axis=1, keepdims=True)
        normalized_points = noisy_points / np.maximum(norms, 1e-12)
        
        # Apply constrained optimization pipeline
        optimized_points = constrained_optimization_pipeline(normalized_points, max_iter=max_iter)
        
        # Evaluate result
        ratio = spherical_voronoi_objective(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # If no good solution found, try random initialization with optimization
    if best_points is None:
        np.random.seed(42)
        random_points = np.random.randn(n, d)
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        random_points = random_points / np.maximum(norms, 1e-12)
        optimized_points = constrained_optimization_pipeline(random_points, max_iter=max_iter)
        ratio = spherical_voronoi_objective(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # Final fallback to random points
    if best_points is None:
        np.random.seed(42)
        points = np.random.rand(n, d) * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        best_points = points / np.maximum(norms, 1e-12)
    
    return best_points

# EVOLVE-BLOCK-END