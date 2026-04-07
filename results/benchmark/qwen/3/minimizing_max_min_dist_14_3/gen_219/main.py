# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
from numba import jit

@jit(nopython=True)
def fast_pdist_matrix(points):
    """Fast computation of pairwise distances using Numba."""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def icosahedral_initialization(n_points):
    """Initialize points using icosahedral symmetry for better uniformity."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Vertices of icosahedron scaled to unit sphere
    vertices = []
    # Add vertices at (±1, 0, ±φ) and permutations
    for i in [1, -1]:
        for j in [0, 1, -1]:
            for k in [0, 1, -1]:
                if i*j*k != 0:
                    vertices.append([i, j*phi, k/phi])
                elif i*j != 0:
                    vertices.append([i, j, k*phi])
                elif i*k != 0:
                    vertices.append([i, j*phi, k])
                elif j*k != 0:
                    vertices.append([i*phi, j, k])
    
    # Normalize to unit sphere and take first n_points
    vertices = np.array(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    
    # Select first n_points vertices (or generate additional ones)
    if len(vertices) >= n_points:
        return vertices[:n_points]
    
    # For 14 points, we'll use a modified approach
    # Generate points on icosahedron faces
    points = []
    
    # Add vertices of icosahedron
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    ico_vertices = ico_vertices / norms
    
    # For 14 points, we can start with 12 vertices and add 2 more
    # Or use a carefully chosen subset of icosahedral points
    selected_vertices = ico_vertices[[0,2,4,6,8,10,1,3,5,7,9,11]]
    
    # Add additional points in strategic locations
    additional_points = []
    for i in range(n_points - len(selected_vertices)):
        # Add points along great circles
        angle = 2 * np.pi * i / (n_points - len(selected_vertices))
        point = [np.cos(angle), np.sin(angle), 0]
        additional_points.append(point)
    
    points = np.vstack([selected_vertices, additional_points])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Apply slight random perturbations to avoid perfect symmetry
    np.random.seed(42)
    perturbations = np.random.normal(0, 0.02, (len(points), 3))
    points += perturbations
    # Project back to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points

def spherical_map(points):
    """Map points from 3D space to unit sphere using normalization."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def spherical_voronoi_quality(sphere_points):
    """Calculate quality based on Voronoi cell areas on sphere."""
    if len(sphere_points) < 2:
        return 0

    # Create spherical Voronoi diagram
    try:
        sv = SphericalVoronoi(sphere_points)
        # Calculate total area of Voronoi cells
        cell_areas = sv.calculate_areas()
        # Quality is inversely related to variance of cell areas
        # More uniform areas indicate better distribution
        if len(cell_areas) > 0:
            mean_area = np.mean(cell_areas)
            if mean_area > 0:
                variance = np.var(cell_areas)
                # Return inverse variance (higher is better)
                return 1.0 / (1.0 + variance / mean_area**2)
    except Exception:
        pass
    return 0

def min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0

    # Calculate pairwise distances
    distances = pdist(points)

    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)

    # Avoid division by zero
    if d_max == 0:
        return 0

    return d_min / d_max

def voronoi_based_objective(x_flat):
    """Objective function based on Voronoi cell uniformity and min/max ratio."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Calculate min/max ratio
    ratio = min_max_ratio(points)
    
    # Calculate spherical Voronoi quality
    sphere_points = spherical_map(points)
    
    try:
        sv = SphericalVoronoi(sphere_points)
        cell_areas = sv.calculate_areas()
        if len(cell_areas) > 0:
            mean_area = np.mean(cell_areas)
            if mean_area > 0:
                variance = np.var(cell_areas)
                # Variance of cell areas as penalty (lower is better)
                voronoi_penalty = variance / mean_area**2
                # Combine with ratio: maximize ratio but minimize variance
                return -ratio + 0.5 * voronoi_penalty
    except Exception:
        pass
    
    # If Voronoi calculation fails, just return negative ratio
    return -ratio

def constrained_point_projection(points, bounds):
    """Project points to valid bounds."""
    points = np.clip(points, bounds[:, 0], bounds[:, 1])
    return points

def spherical_voronoi_optimization(initial_points, max_iterations=500):
    """Optimize points using spherical Voronoi structure analysis."""
    points = initial_points.copy()
    bounds = np.array([(0, 1) for _ in range(14 * 3)])
    
    # Convert to flattened for optimization interface
    x0 = points.flatten()
    
    # Precompute useful quantities
    current_ratio = min_max_ratio(points)
    
    for iteration in range(max_iterations):
        try:
            # Use scipy minimize with L-BFGS-B for local refinement
            result = minimize(
                voronoi_based_objective,
                x0,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(x0))],
                options={'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                new_points = result.x.reshape((14, 3))
                new_ratio = min_max_ratio(new_points)
                
                # Only accept improvement
                if new_ratio > current_ratio:
                    points = new_points.copy()
                    current_ratio = new_ratio
                    x0 = points.flatten()
                else:
                    # Try to escape local minimum with small perturbations
                    np.random.seed(iteration)
                    perturbations = np.random.normal(0, 0.001, (14, 3))
                    test_points = points + perturbations
                    test_points = spherical_map(test_points)
                    
                    test_ratio = min_max_ratio(test_points)
                    if test_ratio > current_ratio:
                        points = test_points.copy()
                        current_ratio = test_ratio
                        x0 = points.flatten()
                        
        except Exception as e:
            # If optimization fails, try perturbation
            np.random.seed(iteration)
            perturbations = np.random.normal(0, 0.005, (14, 3))
            test_points = points + perturbations
            test_points = spherical_map(test_points)
            
            test_ratio = min_max_ratio(test_points)
            if test_ratio > current_ratio:
                points = test_points.copy()
                current_ratio = test_ratio
                x0 = points.flatten()
    
    return points

def adaptive_voronoi_optimization(initial_points, max_time_seconds=360):
    """Main optimization loop using Voronoi-based approach."""
    # Start with icosahedral initialization for high-quality starting point
    points = icosahedral_initialization(14)
    
    # Apply spherical Voronoi optimization
    optimized_points = spherical_voronoi_optimization(points, max_iterations=100)
    
    # Final local refinement
    try:
        x0 = optimized_points.flatten()
        bounds = [(0, 1) for _ in range(14 * 3)]
        
        result = minimize(
            voronoi_based_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 500},
            tol=1e-12
        )
        
        if result.success:
            final_points = result.x.reshape((14, 3))
            final_ratio = min_max_ratio(final_points)
            
            # If we've improved, return the better result
            if final_ratio > min_max_ratio(optimized_points):
                optimized_points = final_points
    except Exception:
        pass
    
    return optimized_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Use the Voronoi-based optimization approach
    try:
        # Time-limited optimization with better initial configuration
        points = adaptive_voronoi_optimization(np.random.rand(14, 3), max_time_seconds=360)
        return points
    except Exception:
        # Fallback to simple Fibonacci approach
        points = icosahedral_initialization(14)
        return points

# EVOLVE-BLOCK-END