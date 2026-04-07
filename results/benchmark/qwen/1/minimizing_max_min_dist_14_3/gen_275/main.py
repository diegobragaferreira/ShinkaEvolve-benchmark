# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def generate_icosahedron_points():
        """Generate points of a regular icosahedron"""
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
    
    def refine_to_14_points(ico_points):
        """Refine icosahedron to get 14 points using barycentric subdivision and normalization"""
        # Start with 12 vertices of icosahedron
        points = ico_points.copy()
        
        # Add 2 more points by placing them at specific locations to create 14 total
        # These are chosen to maintain good distribution properties
        extra_point1 = np.array([0, 0, 1])  # North pole
        extra_point2 = np.array([0, 0, -1])  # South pole
        
        # Normalize the extra points
        extra_point1 = extra_point1 / np.linalg.norm(extra_point1)
        extra_point2 = extra_point2 / np.linalg.norm(extra_point2)
        
        points = np.vstack([points, extra_point1, extra_point2])
        
        # Apply a small random perturbation to break symmetries and improve optimization
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.01, points.shape)
        points = points + perturbation
        
        # Normalize again to maintain unit sphere constraint
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        return points
    
    def normalize_to_unit_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        safe_norms = np.where(norms == 0, 1, norms)
        return points / safe_norms
    
    def spherical_energy_objective(points):
        """Energy-based objective function that promotes uniformity and maximizes min/max ratio"""
        # Normalize points to unit sphere
        points = normalize_to_unit_sphere(points)
        
        # Compute pairwise distances
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        # Calculate minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            ratio = 0.0
        else:
            ratio = min_dist / max_dist
            
        # Energy components:
        # 1. Minimize energy due to large distances (try to keep points well-separated)
        # 2. Maximize ratio of minimum to maximum distance
        # 3. Penalize irregular distribution via variance of distances
        
        # Variance of distances as a measure of irregularity
        valid_distances = distances[distances != np.inf]
        if len(valid_distances) > 0:
            mean_dist = np.mean(valid_distances)
            var_dist = np.var(valid_distances)
            uniformity_penalty = var_dist / (mean_dist + 1e-10)
        else:
            uniformity_penalty = 0.0
        
        # Return negative energy (we want to minimize this)
        # The energy is composed of:
        # - Negative ratio (to maximize ratio)  
        # - Uniformity penalty (to avoid clustering)
        # - Preferential attraction for medium distances (not too close, not too far)
        energy = -ratio + 0.2 * uniformity_penalty
        
        return energy
    
    def spherical_gradient(points):
        """Compute gradient on sphere that respects the constraint"""
        points = normalize_to_unit_sphere(points)
        n = len(points)
        
        # Initialize gradient
        grad = np.zeros_like(points)
        
        # Compute pairwise distances and gradients
        for i in range(n):
            for j in range(i+1, n):  # Only compute upper triangle for efficiency
                diff = points[i] - points[j]
                dist_ij = np.linalg.norm(diff)
                
                if dist_ij > 1e-10:
                    # Unit vector pointing from j to i
                    unit_vec = diff / dist_ij
                    
                    # Gradient contribution from distance (repulsion when too close)
                    # Inverse square law for repulsion
                    force_magnitude = 1.0 / (dist_ij * dist_ij + 1e-10)
                    
                    # Add to both points' gradients (Newton's third law)
                    grad[i] += force_magnitude * unit_vec
                    grad[j] -= force_magnitude * unit_vec
        
        # Project gradient onto tangent space of sphere
        # This ensures gradient flows along the surface, not through it
        for i in range(n):
            grad[i] = grad[i] - np.dot(grad[i], points[i]) * points[i]
            
        return grad
    
    def voronoi_based_regularization(points):
        """Regularization based on Voronoi diagram properties for uniform distribution"""
        points = normalize_to_unit_sphere(points)
        
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
            
            # Calculate Voronoi cell areas
            cell_areas = sv.voronoi_regions_area()
            
            # Regularization term: variance of areas (lower variance = more uniform)
            if len(cell_areas) > 1:
                area_mean = np.mean(cell_areas)
                area_var = np.var(cell_areas)
                uniformity_penalty = area_var / (area_mean + 1e-10)
            else:
                uniformity_penalty = 0.0
                
            return uniformity_penalty
        except:
            # Fallback if Voronoi computation fails
            return 0.0
    
    def energy_minimization_step(points, step_size=0.02):
        """Perform one step of energy minimization using gradient descent"""
        points = normalize_to_unit_sphere(points)
        
        # Compute gradient
        grad = spherical_gradient(points)
        
        # Update points
        new_points = points - step_size * grad
        
        # Project back to sphere
        new_points = normalize_to_unit_sphere(new_points)
        
        return new_points
    
    def adaptive_optimization(points, max_iterations=200):
        """Adaptive optimization with variable step sizes and convergence checking"""
        current_points = points.copy()
        
        # Initial step size
        step_size = 0.02
        
        for iteration in range(max_iterations):
            # Store previous points for convergence check
            prev_points = current_points.copy()
            
            # Perform optimization step
            current_points = energy_minimization_step(current_points, step_size)
            
            # Adapt step size based on convergence
            if iteration > 10:
                # Check convergence
                diff_norm = np.linalg.norm(current_points - prev_points)
                if diff_norm < 1e-6:
                    # Converged, reduce step size
                    step_size *= 0.9
                elif diff_norm > 1e-3:
                    # Diverging, increase step size
                    step_size *= 1.1
            
            # Ensure reasonable step size bounds
            step_size = np.clip(step_size, 0.001, 0.1)
            
            # Periodic convergence check
            if iteration % 20 == 0:
                energy_before = spherical_energy_objective(prev_points)
                energy_after = spherical_energy_objective(current_points)
                if abs(energy_before - energy_after) < 1e-8:
                    break
                    
        return current_points
    
    def hybrid_refinement(points):
        """Combine fast energy-based refinement with precise optimization"""
        # Fast refinement with energy minimization
        fast_refined = adaptive_optimization(points, max_iterations=100)
        
        # Final precise optimization using scipy
        def refined_objective(x_flat):
            points_reshaped = x_flat.reshape(-1, 3)
            return spherical_energy_objective(points_reshaped)
        
        # Flatten and optimize
        x0 = fast_refined.flatten()
        bounds = [(-1, 1) for _ in range(42)]
        
        result = minimize(
            refined_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        final_points = result.x.reshape(-1, 3)
        return normalize_to_unit_sphere(final_points)
    
    # Initialize with icosahedron-based approach
    ico_points = generate_icosahedron_points()
    points = refine_to_14_points(ico_points)
    
    # Apply hybrid refinement approach
    final_points = hybrid_refinement(points)
    
    return final_points

# EVOLVE-BLOCK-END
