# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
from scipy.spatial.distance import pdist
import warnings

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

def adaptive_penalty_objective(x_flat, iteration=0, base_penalty=1e6):
    """Objective function with adaptive penalty for out-of-bounds points.

    Args:
        x_flat: Flattened array of point coordinates [x1, y1, z1, x2, y2, z2, ...]
        iteration: Current iteration number for adaptive scaling
        base_penalty: Base weight for constraint penalty

    Returns:
        Value to minimize (includes penalty for constraint violations)
    """
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Apply penalty for constraint violations
    penalty = 0
    penalty_weight = base_penalty * (1 + iteration * 0.2)
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

def objective_function(x_flat, use_spherical_quality=True):
    """Objective function to maximize the min/max distance ratio.

    Args:
        x_flat: Flattened array of point coordinates [x1, y1, z1, x2, y2, z2, ...]
        use_spherical_quality: Whether to include spherical voronoi quality term

    Returns:
        Negative of combined objective (since we minimize in scipy.optimize)
    """
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Normalize points to unit sphere
    sphere_points = spherical_map(points)

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Calculate spherical Voronoi quality
    voronoi_quality = spherical_voronoi_quality(sphere_points)

    # Combine objectives: prioritize min/max ratio but also consider geometric distribution
    combined = ratio + 0.1 * voronoi_quality

    # Return negative because we want to maximize (minimize negative)
    return -combined

def create_initial_placement():
    """Create initial point placement using an enhanced spherical code approach."""
    # Method: Generate points using Fibonacci-like distribution but with enhanced perturbation
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

    # Improve distribution with multiple phases of perturbations
    np.random.seed(42)

    # Phase 1: Larger perturbations to break symmetries
    perturbation = np.random.normal(0, 0.03, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Phase 2: Medium perturbations for better distribution
    perturbation = np.random.normal(0, 0.015, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Phase 3: Fine adjustments
    perturbation = np.random.normal(0, 0.005, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Normalize to unit sphere and scale to unit cube [0,1]^3
    initial_points = (initial_points + 1) / 2

    return initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    best_ratio = 0
    best_points = None

    # Multiple initialization strategies
    initialization_strategies = [
        # Strategy 1: Enhanced Fibonacci sphere scaled to unit cube
        lambda: create_initial_placement(),

        # Strategy 2: Random initialization with better spread
        lambda: np.random.rand(14, 3)
    ]

    # Try different initialization strategies
    for i, init_func in enumerate(initialization_strategies):
        # Generate initial points
        initial_points = init_func()

        # Flatten initial points for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(14 * 3)]

        # Phase 1: Global optimization with differential evolution
        try:
            # Use a larger population and more iterations for better exploration
            result = differential_evolution(
                adaptive_penalty_objective,
                bounds,
                seed=42 + i,
                maxiter=150,
                popsize=25,
                tol=1e-6,
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

        except Exception as e:
            continue  # Skip this strategy if optimization fails

    # If we didn't find any good solution, use fallback
    if best_points is None:
        # Use the enhanced initialization as fallback
        best_points = create_initial_placement()

    # Phase 2: Progressive local refinement with multiple tolerances
    if best_points is not None:
        try:
            # Flatten the best points
            x0_refine = best_points.flatten()

            # Multiple refinement stages with decreasing tolerances
            refinements = [
                {'tol': 1e-8, 'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 500},
                {'tol': 1e-9, 'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 500},
                {'tol': 1e-10, 'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 500}
            ]

            for i, options in enumerate(refinements):
                # Use the current best points as starting point
                result_refined = minimize(
                    adaptive_penalty_objective,
                    x0_refine,
                    method='L-BFGS-B',
                    bounds=bounds,
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