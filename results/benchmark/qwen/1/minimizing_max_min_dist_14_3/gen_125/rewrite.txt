# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def fibonacci_spiral_on_sphere(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    def sobol_initialization(n: int, seed: int = 42) -> np.ndarray:
        """Generate points using Sobol sequence for better space-filling properties"""
        # Create Sobol sequence sampler
        sampler = qmc.Sobol(d=3, seed=seed)
        # Generate points
        points = sampler.random(n)
        # Scale to [-1, 1]^3
        points = points * 2 - 1
        return points
    
    def normalize_to_unit_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms
    
    def calculate_min_max_ratio(points):
        """Calculate the minimum-to-maximum distance ratio"""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist
    
    def objective_function(points_flat):
        """Objective function to maximize min/max distance ratio"""
        points = points_flat.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return -np.inf
        return min_dist / max_dist
    
    def optimize_with_de_and_refinement(initial_points):
        """Optimize using differential evolution followed by L-BFGS-B refinement"""
        initial_flat = initial_points.flatten()
        bounds = [(-1, 1) for _ in range(42)]
        
        # Differential evolution with aggressive parameters for global search
        de_result = differential_evolution(
            lambda x: -objective_function(x),  # Minimize negative to maximize
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            mutation=(0.8, 1),
            recombination=0.9,
            disp=False
        )
        
        # Extract optimized points from DE
        de_points = de_result.x.reshape(-1, 3)
        
        # Refinement with L-BFGS-B
        lbfgs_result = minimize(
            lambda x: -objective_function(x),
            de_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        # Extract refined points
        refined_points = lbfgs_result.x.reshape(-1, 3)
        
        # Final normalization
        final_points = normalize_to_unit_sphere(refined_points)
        
        return final_points
    
    # Multi-start approach with diverse initializations
    best_ratio = -np.inf
    best_points = None
    
    # Create diverse initial configurations
    initializations = []
    
    # Fibonacci spiral with different seeds
    initializations.append(fibonacci_spiral_on_sphere(14))
    
    # Sobol sequence with different seeds
    initializations.append(sobol_initialization(14, 42))
    initializations.append(sobol_initialization(14, 123))
    initializations.append(sobol_initialization(14, 456))
    
    # Perturbed versions of base configurations
    for i, base_points in enumerate(initializations[:2]):  # Only perturb first 2
        perturbed = base_points + np.random.normal(0, 0.05, base_points.shape)
        initializations.append(perturbed)
    
    # Try each initialization with optimization
    for i, initial_points in enumerate(initializations):
        try:
            # Optimization with DE + L-BFGS-B
            optimized_points = optimize_with_de_and_refinement(initial_points)
            
            # Calculate final ratio
            final_ratio = calculate_min_max_ratio(optimized_points)
            
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            # If optimization fails, continue with other initializations
            continue
    
    # Fallback to basic approach if nothing worked
    if best_points is None:
        # Start with Fibonacci spiral
        initial_points = fibonacci_spiral_on_sphere(14)
        
        # Direct optimization with L-BFGS-B for speed
        initial_flat = initial_points.flatten()
        bounds = [(-1, 1) for _ in range(42)]
        
        result = minimize(
            lambda x: -objective_function(x),
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        best_points = result.x.reshape(-1, 3)
        best_points = normalize_to_unit_sphere(best_points)
    
    return best_points

# EVOLVE-BLOCK-END