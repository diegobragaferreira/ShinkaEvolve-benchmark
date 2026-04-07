# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def initialize_points(n):
        """Initialize points using icosahedron-based method with spherical Voronoi refinement."""
        # Generate icosahedron vertices and normalize
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        
        points = np.array(vertices, dtype=float)
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Add remaining points using Fibonacci-like method
        remaining = n - len(points)
        for i in range(remaining):
            theta = math.acos(1 - 2 * (i / (remaining - 1)))
            phi_coord = math.sqrt(n * math.pi) * theta
            
            x = math.sin(theta) * math.cos(phi_coord)
            y = math.sin(theta) * math.sin(phi_coord)
            z = math.cos(theta)
            points = np.vstack([points, [x, y, z]])
        
        # Add jitter to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        
        # Normalize back to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        return points
    
    def calculate_distance_ratio(points):
        """Calculate the ratio of minimum to maximum distance."""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def objective_function(points_flat):
        """Minimize negative of distance ratio to maximize the ratio."""
        points = points_flat.reshape(-1, 3)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1, norms)
        normalized_points = points / safe_norms
        return -calculate_distance_ratio(normalized_points)
    
    def spherical_constraint_func(points_flat):
        """Constraint function ensuring all points lie on unit sphere."""
        points = points_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    def setup_constraints():
        """Setup equality constraints for spherical constraint."""
        constraints = []
        for i in range(14):
            constraints.append({
                'type': 'eq', 
                'fun': lambda x, i=i: spherical_constraint_func(x)[i]
            })
        return constraints
    
    def optimize_with_adaptive_tolerance(x0, maxiter=500):
        """Optimize with adaptive tolerance settings based on convergence."""
        constraints = setup_constraints()
        bounds = [(-1.1, 1.1)] * len(x0)
        
        # Track convergence
        old_value = np.inf
        history = []
        
        for i in range(3):
            # Adjust tolerance based on iteration phase
            if i == 0:
                options = {'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-4}
            elif i == 1:
                options = {'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6}
            else:
                options = {'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options=options,
                tol=options['ftol']
            )
            
            if not result.success:
                break
                
            x0 = result.x
            current_value = -objective_function(x0)
            history.append(current_value)
            
            # Check for convergence
            if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-8:
                break
                
            # Early stopping if improvement is negligible
            if abs(old_value - current_value) < 1e-8:
                break
                
            old_value = current_value
            
        return x0
    
    def voronoi_refinement(points, iterations=20):
        """Refine point distribution using spherical Voronoi relaxation."""
        refined_points = points.copy()
        
        for _ in range(iterations):
            try:
                # Create spherical Voronoi diagram
                sv = SphericalVoronoi(refined_points, radius=1.0, center=np.zeros(3))
                
                # Get the Voronoi cell centers
                cell_centers = sv.vertices
                
                # Normalize to unit sphere
                norms = np.linalg.norm(cell_centers, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                refined_points = cell_centers / safe_norms
                
                # Add small noise to prevent stagnation
                noise = np.random.normal(0, 0.005, refined_points.shape)
                refined_points += noise
                
            except:
                # If Voronoi fails, fall back to original points
                break
                
        return refined_points
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with improved initialization
    for restart in range(5):
        np.random.seed(42 + restart)
        
        # Initialize points
        initial_points = initialize_points(14)
        
        # Refine using Voronoi relaxation
        refined_points = voronoi_refinement(initial_points, iterations=10)
        
        # Flatten for optimization
        x0 = refined_points.flatten()
        
        # Optimize with adaptive tolerance
        try:
            optimized_points = optimize_with_adaptive_tolerance(x0, maxiter=500)
            
            # Evaluate final solution
            final_points = optimized_points.reshape(-1, 3)
            ratio = calculate_distance_ratio(final_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Fallback to initial points if no improvement found
    if best_points is None:
        initial_points = initialize_points(14)
        best_points = initial_points.flatten()
    
    # Return final result
    return best_points.reshape(14, 3)

# EVOLVE-BLOCK-END
