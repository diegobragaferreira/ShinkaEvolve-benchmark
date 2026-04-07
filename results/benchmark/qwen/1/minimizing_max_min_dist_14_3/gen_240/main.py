# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from scipy.stats import qmc
import time
from typing import Tuple, Optional

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    n = 14
    d = 3

    def fibonacci_sphere(samples=14):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def sobol_points(samples=14, seed=42):
        """Generate points using Sobol sequence for better space-filling properties"""
        try:
            sampler = qmc.Sobol(d=d, seed=seed)
            points = sampler.random(samples)
            # Scale to [-1, 1]^3 then normalize to unit sphere
            points = points * 2 - 1
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return points / norms
        except:
            # Fallback to random if qmc fails
            return np.random.randn(samples, 3)
    
    def enhanced_sobol_points(samples=14, seed=42):
        """Enhanced Sobol points with better sphere distribution"""
        # Generate Sobol points in [0,1]^3
        try:
            sampler = qmc.Sobol(d=d, seed=seed)
            points = sampler.random(samples)
            # Transform to [-1, 1]^3
            points = points * 2 - 1
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return points / norms
        except:
            return np.random.randn(samples, 3)

    def icosahedron_points():
        """Generate points from regular icosahedron"""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices / norms

    def perturbed_points(base_points, sigma=0.03):
        """Add perturbation to points"""
        noise = np.random.normal(0, sigma, base_points.shape)
        perturbed = base_points + noise
        # Normalize to unit sphere again
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return perturbed / norms

    def calculate_min_max_ratio(points):
        """Calculate the minimum-to-maximum distance ratio"""
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def calculate_voronoi_uniformity(points):
        """Calculate uniformity based on spherical Voronoi diagram"""
        try:
            sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
            cell_areas = sv.voronoi_regions_area()
            if len(cell_areas) > 1:
                mean_area = np.mean(cell_areas)
                if mean_area > 0:
                    cv = np.std(cell_areas) / mean_area
                    return cv
            return 0.0
        except:
            return 1.0

    def combined_objective(points_flat, iteration=0):
        """
        Combined objective function that balances distance ratio with uniformity
        """
        points = points_flat.reshape(-1, 3)
        
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_points = points / norms
        
        # Calculate distance ratio
        dist_ratio = calculate_min_max_ratio(normalized_points)
        
        # Calculate uniformity (lower is better)
        uniformity = calculate_voronoi_uniformity(normalized_points)
        
        # Combine objectives with dynamic weights
        progress = min(iteration / 100, 1.0)
        
        # Early iterations favor distance ratio, later iterations favor uniformity
        weight_ratio = 0.7 * (1 - progress) + 0.3 * progress
        weight_uniformity = 0.3 * (1 - progress) + 0.7 * progress
        
        # Objective: maximize distance ratio while maintaining uniformity
        # Use negative because we minimize in scipy.optimize
        objective_value = -(weight_ratio * dist_ratio - weight_uniformity * uniformity)
        
        return objective_value

    def project_to_sphere(points):
        """Project points to unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def normalize_to_unit_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms

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

    def optimize_with_differential_evolution(initial_points, maxiter=100, popsize=15):
        """Optimize using differential evolution"""
        initial_flat = initial_points.flatten()
        bounds = [(-1, 1) for _ in range(n * d)]
        
        # Run differential evolution with constraints
        result = differential_evolution(
            lambda x: -objective_function(x),  # Minimize negative to maximize
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            mutation=(0.7, 1),
            recombination=0.9,
            disp=False
        )
        
        # Extract optimized points
        final_points = result.x.reshape(-1, 3)
        
        # Ensure points are normalized to unit sphere
        norms = np.linalg.norm(final_points, axis=1, keepdims=True)
        final_points = final_points / norms
        
        return final_points, result.fun

    def refine_with_lbfgs(initial_points):
        """Refine solution using L-BFGS-B"""
        initial_flat = initial_points.flatten()
        bounds = [(-1, 1) for _ in range(n * d)]
        
        # Refinement with L-BFGS-B
        result = minimize(
            lambda x: -objective_function(x),
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        # Extract refined points
        refined_points = result.x.reshape(-1, 3)
        
        # Ensure points are normalized to unit sphere
        norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
        refined_points = refined_points / norms
        
        return refined_points

    # Multi-start approach with diverse initializations
    initializations = []

    # Fibonacci spiral points
    fib_points = fibonacci_sphere(n)
    initializations.append(("fibonacci", fib_points.copy()))

    # Sobol points
    sobol_points_gen = sobol_points(n)
    initializations.append(("sobol", sobol_points_gen.copy()))

    # Icosahedron points
    ico_points = icosahedron_points()
    initializations.append(("icosahedron", ico_points.copy()))

    # Perturbed Fibonacci
    perturbed_fib = perturbed_points(fib_points, 0.05)
    initializations.append(("perturbed_fibonacci", perturbed_fib.copy()))

    # Perturbed Sobol
    perturbed_sobol = perturbed_points(sobol_points_gen, 0.05)
    initializations.append(("perturbed_sobol", perturbed_sobol.copy()))

    # Enhanced Sobol points
    enhanced_sobol = enhanced_sobol_points(n)
    initializations.append(("enhanced_sobol", enhanced_sobol.copy()))

    # Random points on sphere
    random_points = np.random.randn(n, 3)
    random_points = project_to_sphere(random_points)
    initializations.append(("random_sphere", random_points.copy()))

    # Combining icosahedron with extra points
    # Add 2 more points at poles
    extended_points = np.vstack([ico_points, [[0,0,1], [0,0,-1]]])
    # Remove one icosahedron point to make exactly 14
    extended_points = extended_points[:14]
    # Normalize again
    extended_points = project_to_sphere(extended_points)
    initializations.append(("extended_icosahedron", extended_points.copy()))

    best_ratio = -np.inf
    best_points = None

    # Run multi-start optimization with 3-stage approach
    for init_name, initial_points in initializations:
        try:
            # Stage 1: Coarse optimization with Differential Evolution
            coarse_points, _ = optimize_with_differential_evolution(
                initial_points, 
                maxiter=50, 
                popsize=20
            )
            
            # Stage 2: Medium refinement with L-BFGS-B
            medium_points = refine_with_lbfgs(coarse_points)
            
            # Stage 3: Fine optimization with multiple methods
            refinement_methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            for method in refinement_methods:
                try:
                    # Use the medium result as starting point
                    medium_flat = medium_points.flatten()
                    bounds = [(-1, 1) for _ in range(n * d)]
                    
                    # Refinement with selected method
                    result = minimize(
                        lambda x: -objective_function(x),
                        medium_flat,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                        tol=1e-12
                    )
                    
                    # Extract final points
                    final_points = result.x.reshape(n, d)
                    
                    # Ensure final points are on unit sphere
                    final_points = normalize_to_unit_sphere(final_points)
                    
                    # Calculate final performance
                    final_ratio = calculate_min_max_ratio(final_points)
                    
                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = final_points.copy()
                        
                except Exception:
                    continue
                    
        except Exception as e:
            continue

    # Fallback to best initialization if no optimization succeeded
    if best_points is None:
        # Use the Fibonacci approach as fallback
        initial_points = fibonacci_sphere(n)
        initial_points = project_to_sphere(initial_points)
        return initial_points

    return best_points

# EVOLVE-BLOCK-END