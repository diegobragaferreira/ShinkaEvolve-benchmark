# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
from scipy.spatial import SphericalVoronoi
from scipy.spatial.transform import Rotation as R


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical Voronoi geometry and targeted geometric optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    
    def generate_voronoi_initialization():
        """Generate initial points using spherical Voronoi principles"""
        # Generate points using Fibonacci spiral method on sphere
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
        """Project spherical points to unit cube"""
        # Normalize to unit sphere first
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized = points / np.maximum(norms, 1e-10)
        
        # Map to [0,1]^3
        cube_points = (normalized + 1) / 2
        return np.clip(cube_points, 0, 1)
    
    def compute_min_max_ratio(points):
        """Compute the negative of min/max distance ratio"""
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return -np.inf
            
        return -min_dist / max_dist
    
    def spherical_energy_objective(points_flat):
        """Energy-based objective using spherical Voronoi properties"""
        points = points_flat.reshape(n, d)
        
        # Project to unit sphere to enforce constraints
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized = points / np.maximum(norms, 1e-10)
        
        # Calculate distances on sphere (great circle distances)
        dot_products = np.dot(normalized, normalized.T)
        # Clamp to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angular_distances = np.arccos(dot_products)
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(angular_distances, np.inf)
        
        min_angle = np.min(angular_distances)
        max_angle = np.max(angular_distances)
        
        if max_angle == 0:
            return 1e10
            
        # Convert to linear distance ratio (min/mean) for better scaling
        # Use actual chordal distances
        chordal_distances = 2 * np.sin(angular_distances / 2)
        np.fill_diagonal(chordal_distances, np.inf)
        
        min_chord = np.min(chordal_distances)
        max_chord = np.max(chordal_distances)
        
        if max_chord == 0:
            return 1e10
            
        return -min_chord / max_chord
    
    def voronoi_based_refinement(points):
        """Refine using Voronoi-based geometric operations"""
        def objective(x):
            points_temp = x.reshape(n, d)
            
            # Ensure points are valid (project to unit sphere)
            norms = np.linalg.norm(points_temp, axis=1, keepdims=True)
            normalized = points_temp / np.maximum(norms, 1e-10)
            
            # Calculate chordal distances
            distances = cdist(normalized, normalized, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return 1e10
                
            return -min_dist / max_dist
            
        # Use L-BFGS-B for refinement
        bounds = [(0, 1) for _ in range(n * d)]
        try:
            result = minimize(objective, points.flatten(), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9})
            if result.success:
                refined = result.x.reshape(n, d)
                # Project back to unit cube
                norms = np.linalg.norm(refined, axis=1, keepdims=True)
                return (refined / np.maximum(norms, 1e-10) + 1) / 2
        except:
            pass
            
        return points
    
    def apply_symmetry_breaking(points):
        """Apply random rotations to break symmetries"""
        # Generate random rotation
        rotation = R.from_euler('xyz', np.random.uniform(0, 2*np.pi, 3)).as_matrix()
        rotated = points @ rotation.T
        
        # Project back to cube
        norms = np.linalg.norm(rotated, axis=1, keepdims=True)
        projected = (rotated / np.maximum(norms, 1e-10) + 1) / 2
        return np.clip(projected, 0, 1)
    
    def adaptive_local_search(points, max_iterations=5):
        """Adaptive local search with multiple refinement strategies"""
        current_points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        for iteration in range(max_iterations):
            # Strategy 1: Voronoi-based refinement
            refined_voronoi = voronoi_based_refinement(current_points)
            ratio_voronoi = compute_min_max_ratio(refined_voronoi)
            
            # Strategy 2: Symmetry breaking with random rotation
            broken_symmetry = apply_symmetry_breaking(current_points)
            ratio_broken = compute_min_max_ratio(broken_symmetry)
            
            # Choose best of current strategies
            candidates = [(refined_voronoi, ratio_voronoi), (broken_symmetry, ratio_broken)]
            best_candidate = min(candidates, key=lambda x: x[1])
            
            if best_candidate[1] < best_ratio:
                best_points = best_candidate[0].copy()
                best_ratio = best_candidate[1]
                
            current_points = best_candidate[0].copy()
            
        return best_points
    
    # Phase 1: Generate spherical Voronoi-inspired initial configuration
    np.random.seed(42)
    sphere_points = generate_voronoi_initialization()
    cube_points = project_to_cube(sphere_points)
    
    # Add structured perturbation to break perfect symmetry
    perturbation = np.random.normal(0, 0.02, cube_points.shape)
    cube_points = np.clip(cube_points + perturbation, 0, 1)
    
    # Phase 2: Adaptive local search with geometric constraints
    adapted_points = adaptive_local_search(cube_points, max_iterations=3)
    
    # Phase 3: Multi-start optimization with different perturbation scales
    best_final_points = adapted_points.copy()
    best_final_ratio = compute_min_max_ratio(adapted_points)
    
    # Try different perturbation scales
    scales = [0.01, 0.02, 0.03]
    
    for attempt in range(4):
        np.random.seed(attempt * 1000 + 42)
        
        # Select scale
        scale = scales[attempt % len(scales)]
        
        # Create perturbed version
        perturbed = adapted_points + np.random.normal(0, scale, adapted_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        
        # Apply refinement
        refined = adaptive_local_search(perturbed, max_iterations=2)
        ratio = compute_min_max_ratio(refined)
        
        if ratio < best_final_ratio:
            best_final_ratio = ratio
            best_final_points = refined.copy()
    
    # Final validation
    final_points = best_final_points
    
    # Ensure bounds
    final_points = np.clip(final_points, 0, 1)
    
    # Validate output shape
    assert final_points.shape == (14, 3), f"Expected shape (14, 3), got {final_points.shape}"
    
    return final_points


# EVOLVE-BLOCK-END