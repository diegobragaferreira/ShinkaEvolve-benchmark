# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
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
        try:
            # Create Sobol sequence sampler
            sampler = qmc.Sobol(d=3, seed=seed)
            # Generate points
            points = sampler.random(n)
            # Scale to [-1, 1]^3
            points = points * 2 - 1
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1, norms)
            return points / safe_norms
        except ImportError:
            # Fallback to random initialization if qmc not available
            points = np.random.uniform(-1, 1, (n, 3))
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            safe_norms = np.where(norms == 0, 1, norms)
            return points / safe_norms
    
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
    
    def spherical_voronoi_quality(points):
        """Evaluate quality based on spherical Voronoi diagram properties"""
        # Normalize points
        points = normalize_to_unit_sphere(points)
        
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
            
            # Calculate Voronoi cell areas
            cell_areas = sv.voronoi_regions_area()
            
            # Return variance of cell areas (lower variance = more uniform distribution)
            return np.var(cell_areas)
        except:
            # Fallback if Voronoi computation fails
            return np.inf
    
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
    
    def objective_with_regularization(points_flat):
        """Objective function with regularization for better optimization"""
        points = points_flat.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        
        # Standard distance ratio
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            ratio = 0.0
        else:
            ratio = min_dist / max_dist
            
        # Add regularization term based on Voronoi quality
        voronoi_penalty = spherical_voronoi_quality(points)
        
        # Combine objective (minimize negative ratio + regularization)
        return -(ratio - 0.01 * voronoi_penalty)
    
    def constraint_sphere(points_flat):
        """Keep points on unit sphere constraint"""
        points = points_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should be zero for unit sphere
    
    # Multi-start approach with different initializations
    best_ratio = -np.inf
    best_points = None
    
    # Create diverse initial configurations
    initializations = []
    
    # 1. Base Fibonacci spiral
    initializations.append(("fibonacci_base", fibonacci_spiral_on_sphere(14)))
    
    # 2. Sobol sequence
    initializations.append(("sobol_base", sobol_initialization(14, 42)))
    
    # 3. Perturbed Fibonacci
    fib_perturbed = fibonacci_spiral_on_sphere(14) + np.random.normal(0, 0.05, (14, 3))
    initializations.append(("fibonacci_perturbed", fib_perturbed))
    
    # 4. Perturbed Sobol
    sobol_perturbed = sobol_initialization(14, 123) + np.random.normal(0, 0.05, (14, 3))
    initializations.append(("sobol_perturbed", sobol_perturbed))
    
    # 5. Another Fibonacci with different seed
    initializations.append(("fibonacci_alt", fibonacci_spiral_on_sphere(14)))
    
    # 6. Another Sobol with different seed
    initializations.append(("sobol_alt", sobol_initialization(14, 456)))
    
    # Optimization parameters for differential evolution
    de_params = [
        {"maxiter": 75, "popsize": 20, "mutation": (0.8, 1), "recombination": 0.9},  # Aggressive global search
        {"maxiter": 100, "popsize": 15, "mutation": (0.6, 1), "recombination": 0.8}, # Balanced
        {"maxiter": 125, "popsize": 10, "mutation": (0.5, 1), "recombination": 0.7}   # Fine tuning
    ]
    
    # Optimization with multi-start approach
    for init_name, initial_points in initializations:
        try:
            # Run multiple DE configurations for each initialization
            for i, config in enumerate(de_params):
                try:
                    # Stage 1: Coarse optimization with Differential Evolution
                    initial_flat = initial_points.flatten()
                    bounds = [(-1, 1) for _ in range(42)]
                    
                    # Run differential evolution with constraints
                    result = differential_evolution(
                        lambda x: -objective_function(x),  # Minimize negative to maximize
                        bounds,
                        maxiter=config["maxiter"],
                        popsize=config["popsize"],
                        tol=1e-6,
                        seed=42,
                        mutation=config["mutation"],
                        recombination=config["recombination"],
                        disp=False
                    )
                    
                    # Extract coarse optimized points
                    coarse_points = result.x.reshape(-1, 3)
                    
                    # Stage 2: Fine optimization with L-BFGS-B using regularization
                    bounds = [(-1, 1) for _ in range(42)]
                    
                    # Refinement with L-BFGS-B
                    result_fine = minimize(
                        lambda x: -objective_with_regularization(x),
                        coarse_points.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
                        tol=1e-14
                    )
                    
                    # Extract refined points
                    refined_points = result_fine.x.reshape(-1, 3)
                    
                    # Ensure points are normalized to unit sphere
                    norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
                    refined_points = refined_points / norms
                    
                    # Calculate final ratio
                    final_ratio = calculate_min_max_ratio(refined_points)
                    
                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = refined_points.copy()
                        
                except Exception as e:
                    # Skip this configuration if it fails
                    continue
                    
        except Exception as e:
            # If optimization fails completely, continue with next initialization
            continue
    
    # If no good solution found, fallback to basic approach
    if best_points is None:
        # Fallback to basic Fibonacci + DE + L-BFGS approach
        initial_points = fibonacci_spiral_on_sphere(14)
        initial_flat = initial_points.flatten()
        bounds = [(-1, 1) for _ in range(42)]
        
        # Coarse optimization with DE
        result = differential_evolution(
            lambda x: -objective_function(x),
            bounds,
            maxiter=150,
            popsize=20,
            tol=1e-6,
            seed=42,
            mutation=(0.7, 1),
            recombination=0.8,
            disp=False
        )
        
        coarse_points = result.x.reshape(-1, 3)
        
        # Fine optimization with L-BFGS-B
        result_fine = minimize(
            lambda x: -objective_with_regularization(x),
            coarse_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
            tol=1e-14
        )
        
        best_points = result_fine.x.reshape(-1, 3)
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END