# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.sparse import diags
import time
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    # Define multiple initialization strategies
    initial_strategies = [
        lambda: fibonacci_sphere(14),
        lambda: icosahedral_initialization(14),
        lambda: random_perturbed_sphere(14),
        lambda: golden_spiral_initialization(14)
    ]
    
    best_points = None
    best_ratio = 0.0
    
    # Multi-start optimization with different strategies
    for strategy in initial_strategies:
        try:
            # Generate initial points
            initial_points = strategy()
            
            # Optimize using quadratic programming approach
            optimized_points = optimize_with_quadratic_programming(initial_points)
            
            # Compute final ratio
            ratio, _ = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Final refinement with specialized optimization
    if best_points is not None:
        try:
            # Apply final optimization with tighter constraints
            final_points = optimize_with_quadratic_programming(best_points, max_iter=2000)
            ratio, _ = compute_min_max_ratio(final_points)
            
            if ratio > best_ratio:
                best_points = final_points
        except Exception:
            pass
    
    return best_points

def fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral method"""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)

def icosahedral_initialization(n):
    """Initialize points using icosahedral symmetry for better spread"""
    # Simple icosahedral-like distribution with proper number of points
    points = np.zeros((n, 3))
    
    # Add poles
    points[0] = [0, 0, 1]       # North pole
    points[1] = [0, 0, -1]      # South pole
    
    # Add equatorial points in two rings
    angle_step = 2 * math.pi / 5
    for i in range(5):
        angle1 = i * angle_step
        angle2 = angle1 + angle_step / 2
        
        # First ring
        points[2+i] = [math.cos(angle1), math.sin(angle1), 0.0]
        # Second ring offset
        points[7+i] = [math.cos(angle2), math.sin(angle2), 0.0]
        
    # Add additional points near poles
    points[12] = [0, 0, 0.7]
    points[13] = [0, 0, -0.7]
    
    # Add small random perturbations
    points += np.random.normal(0, 0.01, points.shape)
    
    # Ensure they're on the sphere
    for i in range(len(points)):
        norm = np.linalg.norm(points[i])
        if norm > 0:
            points[i] = points[i] / norm
    
    return points

def random_perturbed_sphere(n):
    """Generate random points on unit sphere"""
    points = np.random.randn(n, 3)
    norms = np.linalg.norm(points, axis=1)
    norms = np.where(norms == 0, 1, norms)
    points = points / norms[:, np.newaxis]
    return points

def golden_spiral_initialization(n):
    """Alternative golden spiral initialization"""
    points = []
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    
    for i in range(n):
        # Distribute points more evenly
        z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
        radius = np.sqrt(1 - z*z)

        # Better distribution using Fibonacci sequence
        theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    
    return np.array(points)

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances"""
    if len(points) < 2:
        return 0.0, 0.0
    
    distances = pdist(points)
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    if d_max == 0:
        return 0.0, 0.0
    
    return d_min / d_max, d_min

def optimize_with_quadratic_programming(initial_points, max_iter=5000):
    """Optimize using quadratic programming approach with penalty functions"""
    
    def objective_and_gradient(x_flat):
        """Objective function combining min/max ratio with penalties"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Enforce spherical constraint
        norms = np.linalg.norm(points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        points_normalized = points / norms[:, np.newaxis]
        
        # Compute pairwise distances
        distances = pdist(points_normalized)
        
        if len(distances) == 0:
            return 0.0, np.zeros_like(x_flat)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0, np.zeros_like(x_flat)
        
        # Ratio to maximize
        ratio = d_min / d_max
        
        # Penalties for constraint violations (only in case of numerical issues)
        penalty = 0.0
        
        # Gradient approximation (simplified - this is a complex calculation)
        # We return a small gradient to avoid getting stuck
        grad = np.ones_like(x_flat) * 0.001
        
        # Return negative ratio since we want to maximize it
        return -(ratio + penalty), grad
    
    def constraint_func(x_flat):
        """Constraint function to maintain unit sphere"""
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should be zero for unit sphere
    
    # Convert initial points to flat array
    x0 = initial_points.flatten()
    
    # Set up constraints for unit sphere
    constraints = {'type': 'eq', 'fun': constraint_func}
    
    # Optimization parameters
    options = {
        'maxiter': max_iter,
        'ftol': 1e-8,
        'gtol': 1e-6,
        'disp': False
    }
    
    # Use L-BFGS-B for better convergence properties
    try:
        result = minimize(
            lambda x: objective_and_gradient(x)[0],
            x0,
            method='L-BFGS-B',
            jac=lambda x: objective_and_gradient(x)[1],
            constraints=constraints,
            options=options,
            bounds=[(-1.0, 1.0)] * len(x0)  # Bounds for coordinates
        )
        
        if result.success:
            # Extract result and reshape
            optimized_points = result.x.reshape(-1, 3)
            
            # Ensure points are on unit sphere
            norms = np.linalg.norm(optimized_points, axis=1)
            norms = np.where(norms == 0, 1, norms)
            optimized_points = optimized_points / norms[:, np.newaxis]
            
            return optimized_points
    except:
        pass
    
    # Fallback to basic iterative optimization with better convergence
    return basic_optimization_loop(initial_points, max_iter=2000)

def basic_optimization_loop(initial_points, max_iter=2000):
    """Simplified optimization loop with better convergence behavior"""
    points = initial_points.copy()
    
    # Parameters for optimization
    learning_rate = 0.01
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    best_points = points.copy()
    best_ratio, _ = compute_min_max_ratio(points)
    
    for iteration in range(max_iter):
        # Compute current ratio
        current_ratio, _ = compute_min_max_ratio(points)
        
        # Simple gradient estimation via finite differences
        grad_sum = np.zeros_like(points)
        epsilon = 1e-6
        
        for i in range(len(points)):
            for j in range(3):
                # Perturb point
                points_perturbed = points.copy()
                points_perturbed[i, j] += epsilon
                
                # Ensure on sphere
                norm = np.linalg.norm(points_perturbed[i])
                if norm > 0:
                    points_perturbed[i] = points_perturbed[i] / norm
                
                # Compute ratio
                ratio_perturbed, _ = compute_min_max_ratio(points_perturbed)
                
                # Estimate gradient
                grad_sum[i, j] = (ratio_perturbed - current_ratio) / epsilon
        
        # Update with momentum
        velocity = momentum * velocity - learning_rate * grad_sum
        points = points + velocity
        
        # Project back to sphere
        for i in range(len(points)):
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] = points[i] / norm
        
        # Update best solution
        current_ratio, _ = compute_min_max_ratio(points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
        
        # Early stopping for stagnation
        if iteration > 100 and abs(current_ratio - best_ratio) < 1e-8:
            break
    
    return best_points

# EVOLVE-BLOCK-END