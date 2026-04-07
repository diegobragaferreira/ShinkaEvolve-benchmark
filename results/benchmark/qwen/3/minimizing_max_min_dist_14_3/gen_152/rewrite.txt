# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from scipy.optimize import differential_evolution, minimize
import time
from scipy.spatial.transform import Rotation as R

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel spherical Voronoi-inspired approach with multi-scale optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3

    def generate_voronoi_like_initialization():
        """Generate points using spherical Voronoi-inspired distribution with better spatial properties"""
        # Create points using a modified Fibonacci spiral with spherical harmonics influence
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        # Generate points distributed like Fibonacci but with added harmonic influence
        for i in range(n):
            # Base Fibonacci spiral
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            # Add spherical harmonic influence to improve distribution
            # This helps prevent clustering and creates more uniform spacing
            harmonic_factor = 0.1 * np.sin(3 * theta) * np.cos(2 * y * np.pi)
            x += harmonic_factor * radius * 0.1
            
            points.append([x, y, z])
        
        return np.array(points)

    def project_to_cube(points):
        """Project spherical points to unit cube [0,1]^3"""
        # Normalize to unit sphere first
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized = points / np.maximum(norms, 1e-10)
        
        # Map to [0,1]^3 using stereographic projection approximation
        cube_points = (normalized + 1) / 2
        return np.clip(cube_points, 0, 1)

    def compute_min_max_ratio(points):
        """Compute negative of min/max distance ratio"""
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -np.inf
            
        return -min_dist / max_dist

    def adaptive_penalty_objective(points_flat):
        """Objective function with adaptive penalty for boundary violations"""
        points = points_flat.reshape(n, d)
        
        # Apply soft penalty for boundary violations
        penalty = 0
        for i in range(n):
            for j in range(d):
                if points[i,j] < 0:
                    penalty += 1e6 * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += 1e6 * (points[i,j] - 1)**2
        
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -np.inf + penalty
            
        return -(min_dist / max_dist) + penalty

    def hierarchical_refinement(points):
        """Apply multi-scale refinement to improve solution quality"""
        current_points = points.copy()
        
        # Coarse refinement level - larger steps
        def coarse_refinement():
            def obj_coarse(x):
                points_temp = x.reshape(n, d)
                distances = cdist(points_temp, points_temp, 'euclidean')
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 1e10
                return -min_dist / max_dist
            
            # Use L-BFGS-B for coarse refinement
            bounds = [(0, 1) for _ in range(n * d)]
            try:
                res = minimize(obj_coarse, current_points.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6})
                if res.success:
                    return res.x.reshape(n, d)
            except:
                pass
            return current_points
        
        # Fine refinement level - smaller steps
        def fine_refinement():
            def obj_fine(x):
                points_temp = x.reshape(n, d)
                distances = cdist(points_temp, points_temp, 'euclidean')
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 1e10
                return -min_dist / max_dist
            
            # Use L-BFGS-B with stricter tolerances
            bounds = [(0, 1) for _ in range(n * d)]
            try:
                res = minimize(obj_fine, current_points.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9})
                if res.success:
                    return res.x.reshape(n, d)
            except:
                pass
            return current_points
        
        # Apply coarse refinement first
        coarse_result = coarse_refinement()
        # Then fine refinement
        fine_result = fine_refinement()
        
        return fine_result

    def symmetry_breaking_transformation(points):
        """Apply transformations that break symmetries while preserving geometry"""
        # Create multiple transformed versions
        transformed_versions = []
        
        # Rotation around all axes
        rotations = [
            R.from_euler('x', np.pi/4).as_matrix(),
            R.from_euler('y', np.pi/3).as_matrix(),
            R.from_euler('z', np.pi/6).as_matrix(),
            np.eye(3)
        ]
        
        for rot in rotations:
            rotated = points @ rot.T
            transformed_versions.append(rotated)
        
        # Add small noise to create diversity
        for i in range(len(transformed_versions)):
            noisy = transformed_versions[i] + np.random.normal(0, 0.01, transformed_versions[i].shape)
            transformed_versions[i] = np.clip(noisy, 0, 1)
        
        return transformed_versions

    # Phase 1: Generate high-quality Voronoi-like initialization
    np.random.seed(42)
    sphere_points = generate_voronoi_like_initialization()
    cube_points = project_to_cube(sphere_points)
    
    # Add structured perturbation to break perfect symmetry
    perturbation = np.random.normal(0, 0.015, cube_points.shape)
    cube_points = np.clip(cube_points + perturbation, 0, 1)

    # Phase 2: Multi-phase optimization
    # Global optimization using differential evolution
    bounds = [(0, 1) for _ in range(n * d)]
    
    # Run differential evolution for broad exploration
    try:
        result = differential_evolution(
            adaptive_penalty_objective,
            bounds,
            seed=42,
            maxiter=300,
            popsize=25,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False,
            tol=1e-6
        )
        de_result = result.x.reshape(n, d)
    except:
        de_result = cube_points.copy()

    # Phase 3: Hierarchical refinement
    refined_points = hierarchical_refinement(de_result)

    # Phase 4: Symmetry breaking and ensemble optimization
    best_points = refined_points.copy()
    best_ratio = compute_min_max_ratio(best_points.flatten())
    
    # Generate and evaluate transformed versions
    transformed_versions = symmetry_breaking_transformation(best_points)
    
    for version in transformed_versions:
        ratio = compute_min_max_ratio(version.flatten())
        if ratio < best_ratio:
            best_ratio = ratio
            best_points = version.copy()

    # Phase 5: Final multistart refinement
    for attempt in range(3):
        np.random.seed(attempt * 1000 + 42)
        
        # Create perturbed starting point
        perturbed = best_points + np.random.normal(0, 0.005, best_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        
        # Refine
        refined = hierarchical_refinement(perturbed)
        ratio = compute_min_max_ratio(refined.flatten())
        
        if ratio < best_ratio:
            best_ratio = ratio
            best_points = refined.copy()

    # Final validation
    final_points = best_points
    
    # Ensure bounds
    final_points = np.clip(final_points, 0, 1)
    
    # Validate output shape
    assert final_points.shape == (14, 3), f"Expected shape (14, 3), got {final_points.shape}"
    
    return final_points

# EVOLVE-BLOCK-END