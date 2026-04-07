# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')


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


def spherical_voronoi_quality(sphere_points):
    """Calculate quality based on Voronoi cell areas on sphere."""
    if len(sphere_points) < 2:
        return 0
    try:
        sv = SphericalVoronoi(sphere_points)
        cell_areas = sv.calculate_areas()
        if len(cell_areas) > 0:
            mean_area = np.mean(cell_areas)
            if mean_area > 0:
                variance = np.var(cell_areas)
                # Return inverse variance (higher is better) - more uniform distribution
                return 1.0 / (1.0 + variance / mean_area**2)
    except Exception:
        pass
    return 0


def min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    d_min = np.min(distances)
    d_max = np.max(distances)
    return d_min / d_max if d_max > 0 else 0


def normalize_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return points / norms


def project_to_cube(points):
    """Project points from sphere to unit cube [0,1]^3."""
    # Normalize to unit sphere first
    sphere_points = normalize_to_sphere(points)
    # Map to cube [0,1]^3
    return (sphere_points + 1) / 2


def adaptive_penalty_objective(x_flat, penalty_weight=1e6, iteration=0):
    """Objective function with adaptive penalty for out-of-bounds points."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Apply penalty for constraint violations
    penalty = 0
    for i in range(n_points):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2 * (1 + iteration * 0.1)
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2 * (1 + iteration * 0.1)

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Return value to minimize (negative ratio + penalty)
    return -ratio + penalty


def adaptive_differential_evolution(x0, bounds, seed, maxiter, popsize, tol, mutation, recombination):
    """
    Adaptive differential evolution that increases population size when convergence stalls
    """
    # Track convergence
    prev_best = np.inf
    convergence_stall_count = 0
    max_stall_count = 10

    # Initial run
    result = differential_evolution(
        adaptive_penalty_objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        disp=False
    )

    # Monitor for convergence
    current_best = -result.fun
    if abs(prev_best - current_best) < 1e-6:
        convergence_stall_count += 1
    else:
        convergence_stall_count = 0
    prev_best = current_best

    # If convergence stalled, try with larger population
    if convergence_stall_count >= max_stall_count:
        larger_popsize = min(50, popsize + 10)  # Increase population size
        try:
            result = differential_evolution(
                adaptive_penalty_objective,
                bounds,
                seed=seed,
                maxiter=maxiter,
                popsize=larger_popsize,
                tol=tol,
                mutation=mutation,
                recombination=recombination,
                disp=False
            )
        except:
            pass  # Fall back to previous result

    return result


def enhanced_fibonacci_with_voronoi(n):
    """Enhanced Fibonacci sphere with Voronoi quality consideration"""
    points = fibonacci_sphere(n)
    
    # Add perturbations to break symmetries and improve distribution
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, points.shape)
    points += noise
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Scale to make better use of volume
    points *= 0.9
    
    return points


def initialize_multiple_strategies():
    """Initialize points using multiple strategies and select the best"""
    # Strategy 1: Enhanced Fibonacci with Voronoi quality
    points1 = enhanced_fibonacci_with_voronoi(14)
    # Transform to [0,1]^3 cube
    points1_cube = (points1 + 1) / 2

    # Strategy 2: Random uniform initialization
    np.random.seed(42)
    points2 = np.random.rand(14, 3)

    # Strategy 3: Perturbed Fibonacci
    points3 = (fibonacci_sphere(14) + 1) / 2 + np.random.normal(0, 0.03, (14, 3))
    points3 = np.clip(points3, 0, 1)

    # Strategy 4: Two-layered approach
    points4 = np.vstack([
        np.random.rand(7, 3),
        np.random.rand(7, 3) + 0.5
    ])
    points4 = np.clip(points4, 0, 1)

    # Strategy 5: Spherical Voronoi approach
    points5 = enhanced_fibonacci_with_voronoi(14)
    points5 = (points5 + 1) / 2

    # Evaluate all initialization strategies
    strategies = [
        ("enhanced_fib", points1_cube),
        ("random", points2),
        ("perturbed_fib", points3),
        ("two_layer", points4),
        ("spherical_voronoi", points5)
    ]
    
    best_ratio = 0
    best_points = None
    
    for name, points in strategies:
        ratio = min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
            
    return best_points.flatten()


def improved_local_refinement(points, method='L-BFGS-B'):
    """Apply improved local optimization refinement using multiple techniques"""
    try:
        # Convert to flat array for optimization
        x0 = points.flatten()
        
        def obj_func(x):
            points_refined = x.reshape((14, 3))
            ratio = min_max_ratio(points_refined)
            return -ratio  # Return negative for minimization
            
        if method == 'L-BFGS-B':
            result = minimize(
                obj_func,
                x0,
                method=method,
                bounds=[(0, 1)] * 14 * 3,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000},
                tol=1e-12
            )
            
            refined_points = result.x.reshape((14, 3))
            return np.clip(refined_points, 0, 1)
            
    except Exception:
        return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    best_ratio = 0
    best_points = None

    # Multiple initialization strategies with better evaluation
    initialization_strategies = [
        # Strategy 1: Enhanced Fibonacci sphere scaled to unit cube
        lambda: (fibonacci_sphere(n) + 1) / 2,

        # Strategy 2: Latin Hypercube Sampling (simulated with random sampling)
        lambda: np.random.rand(n, 3),

        # Strategy 3: Random initialization
        lambda: np.random.rand(n, 3),

        # Strategy 4: Two-layered approach
        lambda: np.vstack([
            np.random.rand(n//2, 3),
            np.random.rand(n//2, 3) + 0.5
        ]),

        # Strategy 5: Enhanced Fibonacci with Voronoi quality
        lambda: enhanced_fibonacci_with_voronoi(n),
        
        # Strategy 6: Perturbed Fibonacci
        lambda: np.clip((fibonacci_sphere(n) + 1) / 2 + np.random.normal(0, 0.03, (n, 3)), 0, 1)
    ]

    # Try different initialization strategies with multiple restarts
    for restart in range(5):  # More restarts for better exploration
        for i, init_func in enumerate(initialization_strategies):
            try:
                # Generate initial points
                initial_points = init_func()

                # Flatten initial points for optimization
                x0 = initial_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(n * 3)]

                # Phase 1: Global optimization with differential evolution
                # Use adaptive parameters based on restart round
                base_popsize = 20 + restart * 5  # Increase population size with restarts
                maxiter = 100 + restart * 50  # More iterations with restarts

                # Adaptive population sizing based on convergence behavior
                result = adaptive_differential_evolution(
                    x0, bounds, seed=42 + restart * 10 + i,
                    maxiter=maxiter, popsize=base_popsize,
                    tol=1e-6, mutation=(0.5, 1.0), recombination=0.7
                )

                # Extract optimized points
                optimized_points = result.x.reshape((n, 3))
                optimized_points = np.clip(optimized_points, 0, 1)

                # Calculate final ratio
                final_ratio = min_max_ratio(optimized_points)

                # Store best result
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()

            except Exception as e:
                continue  # Skip this strategy if optimization fails

    # Phase 2: Local refinement with L-BFGS-B if we found a good candidate
    if best_points is not None and best_ratio > 0:
        try:
            # Second refinement stage with L-BFGS-B
            refined_points = improved_local_refinement(best_points, 'L-BFGS-B')
            final_ratio = min_max_ratio(refined_points)

            # Update if improved
            if final_ratio > best_ratio:
                best_points = refined_points

        except Exception as e:
            pass  # Keep original best points if refinement fails

    # Phase 3: Additional refinement using alternate method
    if best_points is not None:
        try:
            # Try another refinement with slightly different tolerance
            refined_points = improved_local_refinement(best_points, 'L-BFGS-B')
            final_ratio = min_max_ratio(refined_points)
            
            if final_ratio > best_ratio:
                best_points = refined_points

        except Exception as e:
            pass  # Keep original best points if refinement fails

    # Ensure we return valid points even if optimization failed
    if best_points is None:
        # Fallback to enhanced Fibonacci initialization
        best_points = (fibonacci_sphere(n) + 1) / 2

    return best_points


# EVOLVE-BLOCK-END