# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Voronoi-based optimization on the sphere for improved convergence.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def fibonacci_sphere(n_points):
        """Generate points on sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)
    
    def voronoi_uniformity_objective(cell_areas):
        """Objective function to minimize variance in cell areas."""
        # Target area is total surface area divided by number of cells (4π/n)
        target_area = 4 * np.pi / len(cell_areas)
        # We want to minimize variance from target area  
        variance = np.var(cell_areas)
        # Also penalize areas that are too small (close to zero)
        penalty = np.sum(np.maximum(0, target_area/10 - cell_areas)**2)
        return variance + penalty
    
    def compute_voronoi_on_sphere(points):
        """Compute spherical Voronoi diagram and return cell areas."""
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points_normalized = points / np.where(norms > 0, norms, 1)
        
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points_normalized, radius=1.0)
        sv.sort_vertices_of_regions()
        
        # Compute areas of Voronoi cells
        areas = []
        for region in sv.regions:
            # Approximate area using spherical triangle areas
            if len(region) >= 3:
                # Simple approximation using centroid method
                vertices = sv.vertices[region]
                # For simplicity, use approximation formula
                area = np.sum(np.linalg.norm(np.cross(vertices[:-1], vertices[1:]), axis=1))
                areas.append(area)
            else:
                areas.append(0)
        
        return np.array(areas)
    
    def project_point_to_sphere(point):
        """Project a point to unit sphere."""
        norm = np.linalg.norm(point)
        if norm > 0:
            return point / norm
        return point
    
    def gradient_descent_voronoi(points, max_iter=100, learning_rate=0.01):
        """Optimize points to achieve uniform Voronoi cell areas."""
        current_points = points.copy()
        
        for iteration in range(max_iter):
            # Compute current Voronoi diagram and areas
            try:
                cell_areas = compute_voronoi_on_sphere(current_points)
                target_area = 4 * np.pi / len(cell_areas)
                
                # Compute gradient with respect to cell area uniformity
                # Simple gradient: derivative of variance w.r.t. each point
                # For this simplified version, we'll use a rough approximation
                
                # Create small perturbations to estimate gradients
                grad_sum = np.zeros_like(current_points)
                epsilon = 1e-6
                
                for i in range(len(current_points)):
                    for j in range(3):  # x, y, z components
                        # Forward difference approximation
                        perturbed = current_points.copy()
                        perturbed[i, j] += epsilon
                        
                        perturbed_areas = compute_voronoi_on_sphere(perturbed)
                        # Simple finite difference for gradient estimation
                        if len(perturbed_areas) == len(cell_areas):
                            diff = (voronoi_uniformity_objective(perturbed_areas) - 
                                   voronoi_uniformity_objective(cell_areas)) / epsilon
                            grad_sum[i, j] = diff
                
                # Update points using gradient descent
                current_points -= learning_rate * grad_sum
                # Project back to sphere
                for i in range(len(current_points)):
                    current_points[i] = project_point_to_sphere(current_points[i])
                    
            except Exception:
                # If Voronoi computation fails, continue with current points
                pass
                
        return current_points
    
    # Start with Fibonacci points on sphere
    n = 14
    points = fibonacci_sphere(n)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.where(norms > 0, norms, 1)
    
    # Multi-start optimization
    best_ratio = -np.inf
    best_points = points.copy()
    
    # Try several different initial configurations
    for restart in range(5):
        np.random.seed(42 + restart)
        
        # Start with Fibonacci points
        if restart == 0:
            current_points = points.copy()
        else:
            # Add slight random perturbation
            perturbation = np.random.normal(0, 0.05, points.shape)
            current_points = points + perturbation
            # Project back to sphere
            for i in range(len(current_points)):
                current_points[i] = project_point_to_sphere(current_points[i])
        
        # Apply Voronoi-based optimization
        optimized_points = gradient_descent_voronoi(current_points, max_iter=50, learning_rate=0.05)
        
        # Calculate the resulting ratio
        distances = cdist(optimized_points, optimized_points)
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist > 0:
            ratio = min_dist / max_dist
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    
    # Final refinement using constrained optimization
    def objective_ratio(x):
        points = x.reshape(-1, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points_normalized = points / np.where(norms > 0, norms, 1)
        
        # Compute distance matrix
        dist_matrix = cdist(points_normalized, points_normalized)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_dist = np.min(dist_matrix)
        max_dist = np.max(dist_matrix)
        
        if max_dist == 0:
            return -1.0
            
        # We want to maximize min_dist / max_dist, so minimize -min_dist / max_dist
        return -min_dist / max_dist
    
    # Define constraint: points must be on unit sphere
    def constraint_sphere(x):
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Use the best points found so far as starting point for final optimization
    x0 = best_points.flatten()
    cons = {'type': 'eq', 'fun': constraint_sphere}
    
    try:
        result = minimize(
            objective_ratio,
            x0,
            method='SLSQP',
            constraints=cons,
            options={'ftol': 1e-12, 'maxiter': 300}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            # Final normalization
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            final_points = final_points / np.where(norms > 0, norms, 1)
            
            # Recalculate ratio
            distances = cdist(final_points, final_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_points = final_points.copy()
    except Exception:
        pass
    
    return best_points

# EVOLVE-BLOCK-END