# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import math

def icosahedron_points():
    """Generate points of a regular icosahedron inscribed in unit sphere"""
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    points = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]
    # Normalize to unit sphere
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def subdivision_refinement(points, iterations=2):
    """Refine point set by subdividing edges and projecting to sphere"""
    if iterations <= 0:
        return points
    
    # Create a mapping from edge indices to new points (midpoints)
    edges = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            edges.append((i, j))
    
    # For each edge, add midpoint and project to sphere
    new_points = list(points)
    for _ in range(iterations):
        old_points = new_points[:]
        new_points = list(old_points)
        edge_map = {}
        
        # Generate new points at midpoints of edges
        for i, j in edges:
            if i < j:
                midpoint = (old_points[i] + old_points[j]) / 2
                # Project to sphere
                norm = np.linalg.norm(midpoint)
                if norm > 0:
                    midpoint = midpoint / norm
                new_points.append(midpoint)
    
    return np.array(new_points)

def spherical_projection(points):
    """Project points onto unit sphere"""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def compute_distance_matrix(points):
    """Compute distance matrix efficiently"""
    return cdist(points, points)

def local_optimization_step(points, max_iter=50):
    """Apply local optimization to improve distance distribution"""
    def objective(x):
        # Reshape x back to points
        pts = x.reshape(-1, 3)
        # Compute distance matrix
        dist_matrix = compute_distance_matrix(pts)
        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(dist_matrix, np.inf)
        # Minimize negative of minimum distance (maximize minimum distance)
        min_dist = np.min(dist_matrix)
        # Maximize ratio by minimizing negative ratio
        return -min_dist
    
    def constraint_func(x):
        # Constraint: points must lie on unit sphere
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1)
        return norms - 1.0  # Should equal zero for all points
    
    # Flatten points for optimization
    x0 = points.flatten()
    
    # Optimization constraints
    cons = {'type': 'eq', 'fun': constraint_func}
    
    try:
        result = minimize(objective, x0, method='SLSQP', constraints=cons, 
                         options={'maxiter': max_iter, 'ftol': 1e-8})
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            return spherical_projection(optimized_points)
    except:
        pass
    
    return points

def compute_min_max_ratio(points):
    """Compute the minimum and maximum distances between all pairs of points"""
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    
    # Compute pairwise distances
    distances = compute_distance_matrix(points)
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    
    # Find min and max distances
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    
    # Avoid division by zero
    if max_distance == 0:
        ratio = 0.0
    else:
        ratio = min_distance / max_distance
    
    return min_distance, max_distance, ratio

def sphere_tiling_evolution():
    """Main evolutionary algorithm using spherical tiling approach"""
    # Start with icosahedron points
    points = icosahedron_points()
    
    # Refine to get closer to 14 points (we'll use 20 initially then optimize)
    points = subdivision_refinement(points, 1)
    
    # Trim to 14 points by selecting those that produce the best configuration
    # We'll just take first 14 points as starting configuration
    if len(points) >= 14:
        points = points[:14]
    else:
        # If we don't have enough, create additional points
        extra_points = 14 - len(points)
        additional = np.random.randn(extra_points, 3)
        norms = np.linalg.norm(additional, axis=1, keepdims=True)
        additional = additional / norms
        points = np.vstack([points, additional])
    
    # Ensure we have exactly 14 points
    if len(points) > 14:
        points = points[:14]
    elif len(points) < 14:
        # Add more points via random placement on sphere
        additional = np.random.randn(14 - len(points), 3)
        norms = np.linalg.norm(additional, axis=1, keepdims=True)
        additional = additional / norms
        points = np.vstack([points, additional])
    
    # Initial optimization
    points = local_optimization_step(points)
    
    # Iterative improvement
    best_ratio = 0
    best_points = points.copy()
    
    # Try multiple configurations with different refinements
    for iteration in range(500):
        # Apply small perturbations with different scales
        perturbation_magnitude = 0.01 * (1.0 - iteration/500.0)  # Decreasing over time
        
        # Create new candidate solution
        candidate_points = points.copy()
        
        # Apply random perturbations to a few points
        num_perturbed = max(1, int(14 * 0.2))  # Perturb about 20% of points
        indices_to_perturb = np.random.choice(14, num_perturbed, replace=False)
        
        for idx in indices_to_perturb:
            # Small random perturbation
            delta = np.random.normal(0, perturbation_magnitude, 3)
            candidate_points[idx] += delta
            
            # Project back to sphere
            norm = np.linalg.norm(candidate_points[idx])
            if norm > 0:
                candidate_points[idx] = candidate_points[idx] / norm
        
        # Local optimization on new configuration
        candidate_points = local_optimization_step(candidate_points)
        
        # Evaluate both configurations
        _, _, current_ratio = compute_min_max_ratio(points)
        _, _, candidate_ratio = compute_min_max_ratio(candidate_points)
        
        # Accept better solution
        if candidate_ratio > current_ratio:
            points = candidate_points
            if candidate_ratio > best_ratio:
                best_ratio = candidate_ratio
                best_points = points.copy()
        else:
            # Occasionally accept worse solutions for escape from local minima
            if np.random.rand() < 0.01:
                points = candidate_points
                
        # Occasionally reinitialize with better random start
        if iteration % 100 == 0 and iteration > 0:
            # Try a fresh initialization with better distribution
            fresh_points = icosahedron_points()
            fresh_points = subdivision_refinement(fresh_points, 1)
            if len(fresh_points) >= 14:
                fresh_points = fresh_points[:14]
            fresh_points = local_optimization_step(fresh_points)
            _, _, fresh_ratio = compute_min_max_ratio(fresh_points)
            if fresh_ratio > best_ratio:
                best_points = fresh_points.copy()
                best_ratio = fresh_ratio
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Run the sphere tiling evolution algorithm
    points = sphere_tiling_evolution()
    
    # Final normalization to ensure they're on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    points = points / norms
    
    return points

# EVOLVE-BLOCK-END