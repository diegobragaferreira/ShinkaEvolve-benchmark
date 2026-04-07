# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
from sklearn.metrics.pairwise import euclidean_distances
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical coding evolution approach with stereographic projection.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def stereographic_project(points_2d):
        """Project 2D points to 3D sphere using stereographic projection."""
        # Map [0,1] x [0,1] to [-1,1] x [-1,1] 
        points_norm = points_2d * 2 - 1
        
        # Stereographic projection from plane to sphere
        # Using projection from point (0,0,-1) to unit sphere
        x, y = points_norm[:, 0], points_norm[:, 1]
        
        # Convert to 3D using stereographic projection from south pole
        denominator = 1 + x*x + y*y
        x_3d = 2 * x / denominator
        y_3d = 2 * y / denominator 
        z_3d = (x*x + y*y - 1) / denominator
        
        return np.column_stack([x_3d, y_3d, z_3d])
    
    def stereographic_unproject(points_3d):
        """Unproject 3D sphere points back to 2D plane."""
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        # Inverse stereographic projection from north pole
        denominator = 1 - z
        x_2d = x / denominator
        y_2d = y / denominator
        
        # Map back from [-1,1] to [0,1]
        points_2d = np.column_stack([x_2d, y_2d]) * 0.5 + 0.5
        
        return points_2d
    
    def sphere_distance_matrix(points_sphere):
        """Compute distance matrix on sphere using great circle distances."""
        # For points on unit sphere, we can use dot products
        # Distance = arccos(dot_product) for unit vectors
        dot_products = np.dot(points_sphere, points_sphere.T)
        # Clamp to avoid numerical errors
        dot_products = np.clip(dot_products, -1.0, 1.0)
        distances = np.arccos(dot_products)
        # Set diagonal to inf
        np.fill_diagonal(distances, np.inf)
        return distances
    
    def spherical_min_max_ratio(points_sphere):
        """Compute min/max ratio in spherical space."""
        if len(points_sphere) < 2:
            return 0.0
            
        distances = sphere_distance_matrix(points_sphere)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def spherical_energy_objective(points_sphere):
        """Energy function based on spherical distances."""
        if len(points_sphere) < 2:
            return 0.0
            
        distances = sphere_distance_matrix(points_sphere)
        # Sum of inverse squared distances (repulsive potential)
        distances = np.clip(distances, 1e-10, np.inf)  # Avoid division by zero
        energy = np.sum(1.0 / (distances * distances))
        return energy
    
    def spherical_objective(x):
        """Objective function in spherical coordinates."""
        # Convert flat array to 3D points
        points_sphere = x.reshape(-1, 3)
        
        # Calculate min/max ratio
        ratio = spherical_min_max_ratio(points_sphere)
        
        # Return negative ratio for maximization
        return -ratio
    
    def constraint_spherical_bounds(x):
        """Constraints for spherical coordinates (point must be on unit sphere)."""
        points_sphere = x.reshape(-1, 3)
        norms = np.linalg.norm(points_sphere, axis=1)
        # Penalty for deviation from unit sphere
        penalties = np.abs(norms - 1.0)
        return np.sum(penalties) * 1e6
    
    def adaptive_spherical_optimization(initial_points):
        """Perform adaptive spherical optimization."""
        # Project to sphere
        points_sphere = stereographic_project(initial_points)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points_sphere, axis=1, keepdims=True)
        points_sphere = points_sphere / norms
        
        # Flatten for optimization
        x0 = points_sphere.flatten()
        
        # Define bounds (but we'll handle sphere constraint differently)
        bounds = [(-1.5, 1.5) for _ in range(48)]  # 16 points * 3 coordinates
        
        # Try multiple approaches
        best_ratio = -np.inf
        best_solution = x0.copy()
        
        # Approach 1: Differential evolution
        try:
            de_result = differential_evolution(
                spherical_objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                tol=1e-6
            )
            if -de_result.fun > best_ratio:
                best_ratio = -de_result.fun
                best_solution = de_result.x.copy()
        except:
            pass
            
        # Approach 2: Local optimization refinement
        try:
            # Refine with L-BFGS-B
            result = minimize(
                spherical_objective,
                best_solution,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 300}
            )
            if result.success and -result.fun > best_ratio:
                best_ratio = -result.fun
                best_solution = result.x.copy()
        except:
            pass
            
        # Unproject and return
        final_points_sphere = best_solution.reshape(-1, 3)
        # Ensure unit sphere constraint
        norms = np.linalg.norm(final_points_sphere, axis=1, keepdims=True)
        final_points_sphere = final_points_sphere / norms
        
        final_points_2d = stereographic_unproject(final_points_sphere)
        
        # Ensure within bounds
        final_points_2d = np.clip(final_points_2d, 0, 1)
        
        return final_points_2d
    
    def create_good_initialization():
        """Create high-quality initial configuration."""
        # Start with hexagonal pattern
        rows, cols = 4, 4
        points = []
        
        for i in range(rows):
            for j in range(cols):
                # offset every other row
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])
        
        # Add small random perturbations
        points = np.array(points[:16])
        np.random.seed(42)
        noise_magnitude = 0.02
        noise = np.random.normal(0, noise_magnitude, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        
        return points
    
    def hybrid_optimization():
        """Main hybrid optimization routine."""
        best_points = None
        best_ratio = -np.inf
        
        # Try multiple initializations
        for seed_val in [42, 123, 456]:
            np.random.seed(seed_val)
            
            # Create initial configuration
            initial_points = create_good_initialization()
            
            # Optimize using spherical approach
            optimized_points = adaptive_spherical_optimization(initial_points)
            
            # Evaluate
            distances = cdist(optimized_points, optimized_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        
        # Final refinement if we have a good solution
        if best_points is not None:
            # Try additional refinement with local method
            try:
                # Simple gradient descent refinement
                def refined_objective(x):
                    points = x.reshape(-1, 2)
                    distances = cdist(points, points)
                    np.fill_diagonal(distances, np.inf)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist <= 0:
                        return 0
                    return -min_dist / max_dist
                
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(
                    refined_objective,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    
                    # Recalculate ratio
                    distances = cdist(final_points, final_points)
                    np.fill_diagonal(distances, np.inf)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    
                    if max_dist > 0:
                        final_ratio = min_dist / max_dist
                        if final_ratio > best_ratio:
                            best_points = final_points
            
            except:
                pass
        
        return best_points if best_points is not None else create_good_initialization()
    
    # Execute main optimization
    result = hybrid_optimization()
    return result

# EVOLVE-BLOCK-END