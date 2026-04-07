# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def compute_voronoi_energy(points):
    """Compute electrostatic energy of points on sphere"""
    distances = pdist(points)
    # Energy is sum of inverse distances (electrostatic repulsion)
    energy = np.sum(1.0 / (distances + 1e-12))
    return energy

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance."""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist < 1e-12:
        return 0.0
    return min_dist / max_dist

def spherical_voronoi_initialization(n_points):
    """Initialize points using spherical Voronoi construction approach"""
    # Start with a simple geometric configuration
    points = np.zeros((n_points, 3))
    
    # Use a modified Fibonacci approach with better coverage
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    # Distribute points more evenly using modified spiral
    for i in range(n_points):
        # Modified distribution to avoid clustering
        y = 1 - (i / max(1, n_points - 1)) * 2  # y from 1 to -1
        radius = np.sqrt(max(0, 1 - y * y))
        
        # Add perturbation to avoid perfect symmetry
        theta = golden_angle * i + np.sin(i * 0.5) * 0.1
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points[i] = [x, y, z]
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-12)
    
    # Slight random perturbation to break symmetries
    np.random.seed(42)
    perturbation = np.random.normal(0, 0.03, (n_points, 3))
    points += perturbation
    
    # Project back to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.maximum(norms, 1e-12)
    
    return points

def spherical_gradient_descent(initial_points, max_iter=1000):
    """Optimize points on sphere using gradient descent with spherical projection"""
    
    def objective(x_flat):
        points = x_flat.reshape(-1, 3)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / np.maximum(norms, 1e-12)
        
        # Maximize min/max ratio (minimize negative)
        ratio = min_max_dist_ratio(normalized_points)
        return -ratio if ratio > 0 else 1e10
    
    def constraint_sphere(x_flat):
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Use trust-constr method which handles constraints well
    constraints = {'type': 'eq', 'fun': constraint_sphere}
    
    # Bounds are not needed as we're using constraint
    bounds = [(-2, 2) for _ in range(len(initial_points) * 3)]
    
    # First pass with L-BFGS-B for coarse optimization
    result = minimize(
        objective,
        initial_points.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter//2, 'ftol': 1e-8, 'gtol': 1e-8}
    )
    
    if result.success:
        # Refine with trust-constr for better constraint handling
        result = minimize(
            objective,
            result.x,
            method='trust-constr',
            constraints=constraints,
            options={'maxiter': max_iter//2, 'xtol': 1e-10, 'gtol': 1e-10}
        )
    
    optimized_points = result.x.reshape(-1, 3)
    
    # Ensure final points are on sphere
    norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
    final_points = optimized_points / np.maximum(norms, 1e-12)
    
    return final_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical Voronoi and energy-based optimization approach.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Multi-start approach with Voronoi-based initialization
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Voronoi-inspired initialization with multiple restarts
    for restart in range(10):
        np.random.seed(restart * 100)  # Different seed for each restart
        
        # Initialize with spherical Voronoi-inspired approach
        initial_points = spherical_voronoi_initialization(n)
        
        # Optimize using our specialized spherical gradient descent
        optimized_points = spherical_gradient_descent(initial_points, max_iter=800)
        
        # Evaluate the result
        ratio = min_max_dist_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # Strategy 2: If first strategy failed, try with completely random initialization
    if best_points is None or best_ratio < 0.01:
        np.random.seed(42)
        random_points = np.random.rand(n, d) * 2 - 1  # [-1, 1]
        norms = np.linalg.norm(random_points, axis=1, keepdims=True)
        normalized_points = random_points / np.maximum(norms, 1e-12)
        
        # Optimize this random configuration
        optimized_points = spherical_gradient_descent(normalized_points, max_iter=600)
        
        ratio = min_max_dist_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    
    # Final safeguard - return random points if nothing worked
    if best_points is None:
        np.random.seed(42)
        points = np.random.rand(n, d) * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        best_points = points / np.maximum(norms, 1e-12)
    
    return best_points

# EVOLVE-BLOCK-END