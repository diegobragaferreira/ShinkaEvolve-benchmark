# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
from copy import deepcopy

def sobol_points_sphere(n_points, seed=42):
    """Generate points on sphere using 3D Sobol sequence for superior space-filling properties"""
    try:
        from sobol_seq import i4_sobol_generate
        np.random.seed(seed)
        sobol_points = i4_sobol_generate(3, n_points)
        points = np.zeros((n_points, 3))
        for i in range(n_points):
            u = sobol_points[i, 0]
            v = sobol_points[i, 1]
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points[i] = [x, y, z]
        return points
    except ImportError:
        # Fallback to fibonacci if sobol not available
        return fibonacci_spiral_sphere(n_points, seed)

def fibonacci_spiral_sphere(n_points, seed=42):
    """Generate points on a sphere using Fibonacci spiral method."""
    np.random.seed(seed)
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

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

def quasirandom_evolutionary_optimization():
    """
    Quasi-random evolutionary optimization for 14-point 3D dispersion problem.
    Uses Sobol sequences for initial sampling and gradient-free optimization.
    """
    n = 14
    d = 3
    
    def objective(x_flat):
        points = x_flat.reshape(n, d)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / np.maximum(norms, 1e-12)
        
        # Calculate distances
        distances = pdist(normalized_points)
        if len(distances) == 0:
            return 1e10
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max < 1e-12:
            return 1e10
            
        # Return negative ratio since we want to maximize
        ratio = d_min / d_max
        return -ratio
    
    def constraint_sphere(x_flat):
        points = x_flat.reshape(n, d)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(n * d)]
    
    best_ratio = -np.inf
    best_points = None
    
    # Generate multiple quasirandom initial samples using Sobol sequences
    initial_samples = []
    
    # Generate 20 diverse Sobol-based initial configurations
    for i in range(20):
        np.random.seed(i * 100 + 42)
        # Generate Sobol points and project to sphere
        sobol_points = sobol_points_sphere(n, i * 100 + 42)
        # Add slight perturbation to break symmetry
        noise = np.random.normal(0, 0.02, (n, d))
        sobol_points += noise
        # Project back to sphere
        norms = np.linalg.norm(sobol_points, axis=1, keepdims=True)
        sobol_points = sobol_points / np.maximum(norms, 1e-12)
        initial_samples.append(sobol_points)
    
    # Generate additional diversified samples
    for i in range(10):
        np.random.seed(i * 50 + 123)
        # Generate points using different approach
        points = np.random.rand(n, d) * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-12)
        initial_samples.append(points)
    
    # Now run optimization on each sample with different strategies
    for idx, initial_points in enumerate(initial_samples):
        try:
            # First stage: Coarse optimization with DE for global exploration
            x0 = initial_points.flatten()
            
            # Use differential evolution with reduced population for faster convergence
            de_result = differential_evolution(
                objective,
                bounds,
                seed=idx,
                maxiter=150,
                popsize=8,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            # Second stage: Local refinement with trust-constr for fine-tuning
            refined_result = minimize(
                objective,
                de_result.x,
                method='trust-constr',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'xtol': 1e-12, 'gtol': 1e-12},
                callback=None
            )
            
            # Evaluate final result
            final_points = refined_result.x.reshape(n, d)
            # Project back to sphere for final verification
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            final_points = final_points / np.maximum(norms, 1e-12)
            
            ratio = min_max_dist_ratio(final_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
                
        except Exception as e:
            continue
    
    # Additional fallback optimizations
    if best_points is None:
        # Try one final comprehensive optimization
        try:
            np.random.seed(42)
            # Use Sobol-based initialization
            sobol_init = sobol_points_sphere(n, 42)
            noise = np.random.normal(0, 0.01, (n, d))
            sobol_init += noise
            norms = np.linalg.norm(sobol_init, axis=1, keepdims=True)
            sobol_init = sobol_init / np.maximum(norms, 1e-12)
            
            x0 = sobol_init.flatten()
            
            # Direct optimization without evolution
            result = minimize(
                objective,
                x0,
                method='trust-constr',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'xtol': 1e-12, 'gtol': 1e-12},
                callback=None
            )
            
            final_points = result.x.reshape(n, d)
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            final_points = final_points / np.maximum(norms, 1e-12)
            
            ratio = min_max_dist_ratio(final_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
                
        except Exception as e:
            pass
    
    # Final safeguard with random initialization
    if best_points is None:
        np.random.seed(42)
        points = np.random.rand(n, d) * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        best_points = points / np.maximum(norms, 1e-12)
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses quasirandom evolutionary optimization approach.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Time tracking to respect computational budget
    start_time = time.time()
    
    # Run the quasirandom evolutionary optimization
    best_points = quasirandom_evolutionary_optimization()
    
    # Convert to unit cube [0,1]^3
    centered = best_points - np.mean(best_points, axis=0)
    max_coord = np.max(np.abs(centered))
    if max_coord > 0:
        scaled = centered / max_coord * 0.5
    else:
        scaled = centered
    # Shift to [0,1]^3
    final_points = scaled + 0.5
    
    return final_points

# EVOLVE-BLOCK-END