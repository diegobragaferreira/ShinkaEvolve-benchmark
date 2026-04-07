# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def initialize_on_sphere(n_points):
        """Initialize points using spherical coordinates on unit sphere"""
        # Use a variant of Fibonacci distribution for good spreading
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        
        for i in range(n_points):
            # Distribute points more uniformly using modified Fibonacci approach
            y = 1 - (i / float(n_points - 1)) * 2  # y from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            # Use golden angle for azimuthal distribution
            theta = math.acos(y)  # polar angle
            phi_angle = (i * 2.414213562) % (2 * math.pi)  # golden ratio multiple
            
            x = radius * math.cos(phi_angle)
            z = radius * math.sin(phi_angle)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def spherical_coordinate_transform(points):
        """Convert 3D Cartesian points to spherical coordinates"""
        # Convert cartesian to spherical coordinates
        r = np.linalg.norm(points, axis=1)
        theta = np.arccos(points[:, 2] / (r + 1e-12))  # polar angle
        phi = np.arctan2(points[:, 1], points[:, 0])   # azimuthal angle
        
        return np.column_stack([theta, phi])
    
    def cartesian_from_spherical(spherical_coords):
        """Convert spherical coordinates back to Cartesian"""
        theta = spherical_coords[:, 0]
        phi = spherical_coords[:, 1]
        
        # Convert to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return np.column_stack([x, y, z])
    
    def distance_ratio(points):
        """Calculate the ratio of minimum to maximum distance"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def distance_variance_penalty(points):
        """Penalize high variance in distances to encourage uniform distribution"""
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        if len(distances[distances != np.inf]) == 0:
            return 0
        mean_dist = np.mean(distances[distances != np.inf])
        var_dist = np.var(distances[distances != np.inf])
        # Penalty increases with distance variance
        return var_dist / (mean_dist * mean_dist + 1e-10)
    
    def objective_with_penalty(points_flat):
        """Objective function with penalty for non-uniform distribution"""
        points = points_flat.reshape(-1, 3)
        
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / np.maximum(norms, 1e-12)
        
        # Calculate base ratio
        ratio = distance_ratio(normalized_points)
        
        # Add penalty for variance in distances
        variance_penalty = distance_variance_penalty(normalized_points)
        
        # Combine: maximize ratio with penalty for non-uniformity
        # Using a scaling factor to balance both considerations
        alpha = 0.8  # weight for ratio
        beta = 0.2   # weight for penalty
        
        # Return negative because we want to minimize negative values (maximize positive)
        return -(alpha * ratio - beta * variance_penalty)
    
    def constraint_sphere(x):
        """Constraint that all points lie on unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    def constrained_optimization(x0, maxiter=1000):
        """Optimize with spherical constraints using L-BFGS-B"""
        # Define constraints for unit sphere
        constraints = [{'type': 'eq', 'fun': constraint_sphere}]
        
        # Bounds for coordinates in [-1.5, 1.5] to allow some flexibility
        bounds = [(-1.5, 1.5)] * len(x0)
        
        try:
            result = minimize(
                objective_with_penalty,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                return result.x
        except:
            pass
        
        return x0
    
    def spherical_evolution_optimization(x0, max_iter=100):
        """Main optimization loop using spherical evolution approach"""
        current_solution = x0.copy()
        best_ratio = -np.inf
        best_solution = x0.copy()
        
        # Progressive refinement approach
        for iteration in range(max_iter):
            # Apply constraint satisfaction and normalization
            points_current = current_solution.reshape(-1, 3)
            norms = np.linalg.norm(points_current, axis=1, keepdims=True)
            points_current = points_current / np.maximum(norms, 1e-12)
            
            # Calculate current ratio
            current_ratio = distance_ratio(points_current)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_solution = current_solution.copy()
            
            # Apply optimized spherical optimization
            # Use progressively tighter constraints for better convergence
            refined_solution = constrained_optimization(current_solution, maxiter=50)
            
            # Update current solution with refinement
            current_solution = refined_solution.copy()
            
            # Add small perturbations to prevent stagnation
            if iteration % 5 == 0:
                np.random.seed(iteration)
                noise = np.random.normal(0, 0.001, current_solution.shape)
                current_solution += noise
                
                # Re-normalize
                points_perturbed = current_solution.reshape(-1, 3)
                norms = np.linalg.norm(points_perturbed, axis=1, keepdims=True)
                points_perturbed = points_perturbed / np.maximum(norms, 1e-12)
                current_solution = points_perturbed.flatten()
        
        return best_solution
    
    def generate_multiple_initializations(n_initializations=10):
        """Generate multiple good initial points using different strategies"""
        initial_sets = []
        
        for i in range(n_initializations):
            np.random.seed(42 + i)
            
            # Strategy 1: Fibonacci spiral-based initialization
            points = initialize_on_sphere(14)
            
            # Apply small jitter to break symmetry
            noise = np.random.normal(0, 0.02, points.shape)
            points += noise
            
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / np.maximum(norms, 1e-12)
            
            initial_sets.append(points.flatten())
        
        return initial_sets
    
    # Main algorithm
    best_ratio = -np.inf
    best_points = None
    
    # Try multiple initializations
    initial_sets = generate_multiple_initializations(15)
    
    for i, x0 in enumerate(initial_sets):
        try:
            # Apply spherical evolution optimization
            optimized_solution = spherical_evolution_optimization(x0.copy(), max_iter=50)
            
            # Extract points
            final_points = optimized_solution.reshape(-1, 3)
            
            # Ensure all points are on unit sphere
            norms = np.linalg.norm(final_points, axis=1, keepdims=True)
            final_points = final_points / np.maximum(norms, 1e-12)
            
            # Calculate final ratio
            ratio = distance_ratio(final_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
                
        except Exception as e:
            continue
    
    # Fallback to best initialization if optimization failed
    if best_points is None:
        # Use a simple Fibonacci-based initialization
        points = initialize_on_sphere(14)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-12)
        best_points = points
    
    return best_points

# EVOLVE-BLOCK-END