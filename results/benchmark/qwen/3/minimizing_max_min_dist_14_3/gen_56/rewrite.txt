# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time
from sklearn.cluster import KMeans


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical Voronoi optimization approach for superior convergence.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    
    def generate_initial_spherical_points():
        """Generate initial points on unit sphere using Fibonacci-based approach"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def project_to_cube(points):
        """Project points from unit sphere to unit cube uniformly"""
        # Normalize to unit sphere first
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points_normalized = points / norms
        
        # Project to cube keeping direction but adjusting magnitude
        # Map to [0,1]^3 using the direction and appropriate scaling
        # This is a more geometric approach than simple linear mapping
        points_cube = (points_normalized + 1) / 2
        return np.clip(points_cube, 0, 1)
    
    def compute_ratio(points):
        """Compute min/max distance ratio efficiently"""
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def spherical_voronoi_objective(points_flat):
        """Objective function using spherical Voronoi properties"""
        points = points_flat.reshape(n, d)
        
        # Ensure points are normalized to unit sphere for Voronoi computation
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points_unit = points / np.maximum(norms, 1e-10)
        
        # Use Voronoi area approximation as proxy for uniformity
        try:
            sv = SphericalVoronoi(points_unit)
            areas = sv.calculate_areas()
            # Penalize non-uniform distribution
            area_variance = np.var(areas)
            return area_variance
        except:
            # Fallback to distance-based objective
            distances = cdist(points, points, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 1e10
                
            # We want to maximize min/max ratio, so minimize the negative ratio
            return -min_dist / max_dist
    
    def constrain_to_cube(points):
        """Constrain points to [0,1]^3 cube"""
        return np.clip(points, 0, 1)
    
    def optimize_on_sphere(points_start):
        """Optimize points on unit sphere using gradient-based method"""
        # Start with spherical points
        points = points_start.copy()
        
        # Convert to optimization variables (normalized)
        points_normalized = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        def objective(x):
            # Reshape and normalize
            points_temp = x.reshape(n, d)
            norms = np.linalg.norm(points_temp, axis=1, keepdims=True)
            points_unit = points_temp / np.maximum(norms, 1e-10)
            
            # Compute distances on sphere (great circle distances)
            distances = cdist(points_unit, points_unit, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            
            # Convert to actual great circle distances for better metric
            # But work with chordal distances for simplicity
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 1e10
                
            # Minimize negative ratio to maximize ratio
            return -min_dist / max_dist
        
        # Use L-BFGS-B on sphere-constrained manifold
        try:
            result = minimize(objective, points_normalized.flatten(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9})
            
            if result.success:
                optimized = result.x.reshape(n, d)
                # Normalize back to unit sphere
                norms = np.linalg.norm(optimized, axis=1, keepdims=True)
                optimized = optimized / np.maximum(norms, 1e-10)
                return optimized
        except:
            pass
            
        return points_normalized
    
    # Phase 1: Generate high-quality initial configuration
    np.random.seed(42)
    
    # Method 1: Generate points on sphere with good distribution
    sphere_points = generate_initial_spherical_points()
    
    # Method 2: Apply optimization to improve sphere configuration
    sphere_points_optimized = optimize_on_sphere(sphere_points)
    
    # Phase 2: Project to cube and fine-tune
    cube_points = project_to_cube(sphere_points_optimized)
    
    # Phase 3: Fine-tune using local optimization
    def objective_final(x):
        points = x.reshape(n, d)
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 1e10
            
        # Minimize negative ratio to maximize ratio
        return -min_dist / max_dist
    
    # Multi-start local optimization with better initial points
    best_points = cube_points.copy()
    best_ratio = compute_ratio(best_points)
    
    # Try different starting points around the initial solution
    for i in range(5):
        np.random.seed(i * 100 + 42)
        
        # Add small perturbation
        perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
        perturbed = constrain_to_cube(perturbed)
        
        # Optimize around this point
        try:
            result = minimize(objective_final, perturbed.flatten(), 
                            method='L-BFGS-B',
                            bounds=[(0, 1) for _ in range(n * d)],
                            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9})
            
            if result.success:
                refined = result.x.reshape(n, d)
                refined = constrain_to_cube(refined)
                
                current_ratio = compute_ratio(refined)
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = refined
                    
        except:
            continue
    
    # Phase 4: Final refinement with multiple strategies
    # Try Nelder-Mead as fallback for robustness
    try:
        result_nm = minimize(objective_final, best_points.flatten(), 
                           method='Nelder-Mead',
                           options={'maxiter': 300, 'disp': False})
        
        if result_nm.success:
            nm_points = result_nm.x.reshape(n, d)
            nm_points = constrain_to_cube(nm_points)
            
            nm_ratio = compute_ratio(nm_points)
            if nm_ratio > best_ratio:
                best_points = nm_points
                
    except:
        pass
    
    # Final validation
    final_points = constrain_to_cube(best_points)
    
    # Ensure exact shape
    assert final_points.shape == (14, 3), f"Expected shape (14, 3), got {final_points.shape}"
    
    return final_points


# EVOLVE-BLOCK-END