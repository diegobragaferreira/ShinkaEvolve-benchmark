# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from scipy.spatial import ConvexHull
from numba import jit
import warnings

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

def fibonacci_sphere(n):
    """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def spherical_map(points):
    """Map points from 3D space to unit sphere using normalization."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

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

def spherical_voronoi_objective(points):
    """Combined objective function using both min/max ratio and Voronoi uniformity."""
    if len(points) < 2:
        return 0
    
    # Calculate min/max ratio
    ratio = min_max_ratio(points)
    
    # Calculate spherical Voronoi quality (higher is better)
    sphere_points = spherical_map(points)
    voronoi_quality = spherical_voronoi_quality(sphere_points)
    
    # Weighted combination: prioritize ratio but consider uniformity
    return ratio + 0.15 * voronoi_quality

def constraint_penalty(points, penalty_weight=1e6):
    """Calculate penalty for constraint violations."""
    penalty = 0
    for i in range(len(points)):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2
    return penalty

def constrained_objective(x_flat, penalty_weight=1e6):
    """Objective function combining main goal with constraints."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))
    
    # Calculate main objective
    main_obj = spherical_voronoi_objective(points)
    
    # Add penalty for constraint violations
    penalty = constraint_penalty(points, penalty_weight)
    
    # Return negative since we minimize in scipy.optimize
    return -main_obj + penalty

def project_to_unit_cube(points):
    """Project points to unit cube [0,1]^3 while preserving relationships."""
    # Find bounding box
    mins = np.min(points, axis=0)
    maxes = np.max(points, axis=0)
    
    # Avoid division by zero
    ranges = maxes - mins
    ranges[ranges == 0] = 1
    
    # Normalize to [0,1] cube
    normalized = (points - mins) / ranges
    return normalized

def initialize_with_voronoi_optimization():
    """Initialize points using a Voronoi-based approach that directly seeks uniformity."""
    # Start with Fibonacci distribution for good initial spread
    initial_points = fibonacci_sphere(14)
    
    # Add some randomness to break symmetries
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, (14, 3))
    initial_points += noise
    
    # Project to unit sphere
    initial_points = spherical_map(initial_points)
    
    # Convert to unit cube [0,1]^3
    initial_points = (initial_points + 1) / 2
    
    # Apply iterative refinement to improve Voronoi uniformity
    best_points = initial_points.copy()
    best_score = spherical_voronoi_objective(best_points)
    
    # Iterative improvement loop
    for iteration in range(15):
        # Create a slightly perturbed version
        perturbed_points = best_points.copy()
        noise = np.random.normal(0, 0.01, (14, 3))
        perturbed_points += noise
        
        # Project back to unit cube
        perturbed_points = np.clip(perturbed_points, 0, 1)
        
        # Evaluate
        current_score = spherical_voronoi_objective(perturbed_points)
        if current_score > best_score:
            best_points = perturbed_points
            best_score = current_score
    
    return best_points

def adaptive_local_search(start_points, max_iterations=200):
    """Perform adaptive local search to improve point distribution."""
    current_points = start_points.copy()
    
    # Use a more aggressive optimization approach
    x0 = current_points.flatten()
    bounds = [(0, 1) for _ in range(14 * 3)]
    
    # Multi-stage refinement with varying tolerances
    refinement_stages = [
        {'method': 'L-BFGS-B', 'options': {'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 50}},
        {'method': 'L-BFGS-B', 'options': {'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 100}},
        {'method': 'L-BFGS-B', 'options': {'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 150}}
    ]
    
    best_points = current_points.copy()
    best_score = spherical_voronoi_objective(best_points)
    
    for stage in refinement_stages:
        try:
            result = minimize(
                constrained_objective,
                x0,
                method=stage['method'],
                bounds=bounds,
                options=stage['options'],
                tol=stage['options']['ftol']
            )
            
            if result.success:
                refined_points = result.x.reshape((14, 3))
                refined_score = spherical_voronoi_objective(refined_points)
                
                if refined_score > best_score:
                    best_points = refined_points
                    best_score = refined_score
                    x0 = refined_points.flatten()
                else:
                    # Early exit if improvement stops
                    break
            else:
                break
                
        except Exception:
            break
    
    return best_points

def spherical_voronoi_evolution():
    """Main evolutionary optimization process based on spherical Voronoi principles."""
    
    # Phase 1: Multiple initialization strategies to find promising starting points
    initial_strategies = [
        lambda: initialize_with_voronoi_optimization(),
        lambda: (fibonacci_sphere(14) + 1) / 2,
        lambda: np.random.rand(14, 3),
    ]
    
    best_points = None
    best_score = -np.inf
    
    for i, strategy in enumerate(initial_strategies):
        try:
            # Generate initial points
            initial_points = strategy()
            
            # Apply local optimization to the initial points
            optimized_points = adaptive_local_search(initial_points, max_iterations=100)
            
            # Evaluate the final result
            score = spherical_voronoi_objective(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
        except Exception as e:
            continue
    
    # Phase 2: Final refinement if needed
    if best_points is not None:
        final_refinement = adaptive_local_search(best_points, max_iterations=200)
        final_score = spherical_voronoi_objective(final_refinement)
        
        if final_score > best_score:
            best_points = final_refinement
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Use the specialized spherical Voronoi evolution approach
    best_points = spherical_voronoi_evolution()
    
    # Fallback to ensure we always return valid points
    if best_points is None:
        # Use enhanced Fibonacci initialization
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(14):
            y = 1 - (i / float(14 - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        best_points = np.array(points)
        
        # Improve distribution by applying multiple small perturbations
        np.random.seed(42)
        for _ in range(15):
            # Add small random perturbations
            perturbation = np.random.normal(0, 0.015, (14, 3))
            best_points += perturbation

            # Project back to sphere surface using normalization
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            best_points = best_points / norms

        # Normalize to unit sphere and scale to unit cube [0,1]^3
        best_points = (best_points + 1) / 2
    
    return best_points

# EVOLVE-BLOCK-END