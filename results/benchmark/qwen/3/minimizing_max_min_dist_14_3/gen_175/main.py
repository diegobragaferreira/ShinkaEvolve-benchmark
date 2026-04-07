# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
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

    # Calculate pairwise distances efficiently with numba
    distances = fast_pdist_matrix(points)
    
    # Get min and max distances (excluding diagonal zeros)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    masked_distances = distances[mask]
    
    if len(masked_distances) == 0:
        return 0
        
    d_min = np.min(masked_distances)
    d_max = np.max(masked_distances)

    # Avoid division by zero
    if d_max == 0:
        return 0

    return d_min / d_max

def adaptive_penalty_objective(x_flat, iteration=0, base_penalty=1e6):
    """Objective function with adaptive penalty for out-of-bounds points."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Apply penalty for constraint violations
    penalty = 0
    penalty_weight = base_penalty * (1 + iteration * 0.1)
    for i in range(n_points):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Return value to minimize (negative ratio + penalty)
    return -ratio + penalty

def calculate_distribution_score(points):
    """Calculate a score based on how uniformly distributed the points are."""
    if len(points) < 2:
        return 0
    
    # Compute distances and analyze spread
    distances = fast_pdist_matrix(points)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    masked_distances = distances[mask]
    
    if len(masked_distances) == 0:
        return 0
    
    # Check for clustering by examining variance in distances
    mean_dist = np.mean(masked_distances)
    var_dist = np.var(masked_distances)
    
    # Prefer more uniform distribution (lower variance)
    if mean_dist > 0:
        uniformity_score = 1.0 / (1.0 + var_dist / mean_dist**2)
    else:
        uniformity_score = 0.0
        
    return uniformity_score

def improved_objective_function(x_flat):
    """Enhanced objective function combining min/max ratio with distribution quality."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Calculate min/max ratio
    ratio = min_max_ratio(points)
    
    # Calculate distribution quality score
    distribution_score = calculate_distribution_score(points)
    
    # Combined objective: prioritize ratio but consider distribution
    combined = ratio + 0.15 * distribution_score

    # Return negative because we minimize in scipy.optimize
    return -combined

def create_initial_placement():
    """Create initial point placement using enhanced spherical code approach."""
    # Method: Generate points using Fibonacci-like distribution with multiple adjustment phases
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(14):
        y = 1 - (i / float(14 - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    initial_points = np.array(points)

    # Improve distribution with multiple refinement passes
    np.random.seed(42)
    
    # Phase 1: Larger perturbations to break symmetries
    perturbation = np.random.normal(0, 0.035, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Phase 2: Medium perturbations for better distribution
    perturbation = np.random.normal(0, 0.018, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Phase 3: Fine adjustments
    perturbation = np.random.normal(0, 0.008, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Phase 4: Final optimization step using iterative local adjustment
    for _ in range(5):
        # Simple iterative improvement
        for i in range(14):
            # Move point away from neighbors
            neighbor_sum = np.zeros(3)
            count = 0
            for j in range(14):
                if i != j:
                    diff = initial_points[i] - initial_points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq > 0:
                        neighbor_sum += diff / (dist_sq + 1e-8)
                        count += 1
            
            if count > 0:
                move_direction = -neighbor_sum / count
                # Limit movement magnitude
                move_magnitude = np.min([0.005, np.linalg.norm(move_direction)])
                if move_magnitude > 0:
                    move_direction = move_direction * move_magnitude / np.linalg.norm(move_direction)
                    initial_points[i] += move_direction
                    
        # Project back to sphere surface
        initial_points = spherical_map(initial_points)

    # Normalize to unit sphere and scale to unit cube [0,1]^3
    initial_points = (initial_points + 1) / 2

    return initial_points

def hybrid_optimization_strategy():
    """Perform hybrid optimization with multiple restarts and strategies."""
    best_ratio = 0
    best_points = None
    
    # Multiple initialization strategies
    initialization_strategies = [
        # Strategy 1: Enhanced Fibonacci sphere
        lambda: create_initial_placement(),
        
        # Strategy 2: Random initialization with better spread
        lambda: np.random.rand(14, 3),
        
        # Strategy 3: Fibonacci sphere with slight perturbation
        lambda: (fibonacci_sphere(14) + 1) / 2 + np.random.normal(0, 0.01, (14, 3)),
    ]

    # Try different initialization strategies with multiple restarts
    for restart in range(3):
        for i, init_func in enumerate(initialization_strategies):
            # Generate initial points
            initial_points = init_func()
            
            # Flatten initial points for optimization
            x0 = initial_points.flatten()
            
            # Define bounds for each coordinate (0 to 1)
            bounds = [(0, 1) for _ in range(14 * 3)]
            
            # Phase 1: Global optimization with differential evolution
            try:
                # Set optimization parameters
                popsize = 20 + restart * 5  # Increase population size with restarts
                maxiter = 120 + restart * 40  # More iterations with restarts
                
                result = differential_evolution(
                    adaptive_penalty_objective,
                    bounds,
                    seed=42 + restart * 10 + i,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-7,
                    mutation=(0.5, 1),
                    recombination=0.8,
                    disp=False
                )
                
                # Extract optimized points
                optimized_points = result.x.reshape((14, 3))
                
                # Calculate final ratio
                final_ratio = min_max_ratio(optimized_points)
                
                # Store best result
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()
                    
                # Early stopping if we get very close to target
                if best_ratio >= 0.4898 * 0.98:
                    break
                    
            except Exception as e:
                continue  # Skip this strategy if optimization fails

    return best_points, best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    best_points, best_ratio = hybrid_optimization_strategy()
    
    # If we didn't find any good solution, use fallback
    if best_points is None:
        # Use the enhanced initialization as fallback
        best_points = create_initial_placement()
        best_ratio = min_max_ratio(best_points)
    
    # Phase 2: Progressive local refinement with multiple tolerances  
    if best_points is not None and best_ratio > 0:
        try:
            # Flatten the best points
            x0_refine = best_points.flatten()
            
            # Multiple refinement stages with decreasing tolerances
            refinements = [
                {'tol': 1e-8, 'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 400},
                {'tol': 1e-9, 'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 400},
                {'tol': 1e-10, 'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 400}
            ]
            
            for i, options in enumerate(refinements):
                # Use the current best points as starting point
                result_refined = minimize(
                    improved_objective_function,
                    x0_refine,
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(14 * 3)],
                    options=options,
                    tol=options['tol']
                )
                
                refined_points = result_refined.x.reshape((14, 3))
                final_ratio = min_max_ratio(refined_points)
                
                # Update if improved
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = refined_points.copy()
                    
                # Early stopping if ratio is already good enough
                if best_ratio >= 0.4898 * 0.95:
                    break
                
                x0_refine = refined_points.flatten()
                
        except Exception as e:
            pass  # Keep original best points if refinement fails
            
    # Final validation and return
    if best_points is None:
        # Last resort fallback
        best_points = create_initial_placement()
        
    return best_points

# EVOLVE-BLOCK-END