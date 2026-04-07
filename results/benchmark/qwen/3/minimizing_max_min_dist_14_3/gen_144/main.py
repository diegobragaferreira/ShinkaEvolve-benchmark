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
    """Create initial point placement using a multi-phase approach for better uniformity."""
    # Phase 1: Generate points using Fibonacci-like distribution on sphere
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

    # Phase 2: Apply multiple rounds of perturbations with decreasing magnitude to avoid local optima
    np.random.seed(42)

    # First round: larger perturbations to break initial symmetries
    perturbation = np.random.normal(0, 0.05, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Second round: smaller perturbations for fine-tuning
    perturbation = np.random.normal(0, 0.02, (14, 3))
    initial_points += perturbation
    initial_points = spherical_map(initial_points)

    # Third round: smallest perturbations for final adjustment
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

    # Create initial placement
    initial_points = create_initial_placement()

    # Flatten initial points for optimization
    x0 = initial_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(14 * 3)]

    # Phase 1: Global optimization with adaptive differential evolution
    try:
        # Use adaptive parameters for better convergence
        # Start with fewer iterations and gradually increase if needed
        maxiter = 100
        popsize = 20

        # Run differential evolution with adaptive parameters
        result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-7,
            mutation=(0.5, 1),
            recombination=0.8,
            disp=False
        )

        # Extract optimized points
        optimized_points = result.x.reshape((14, 3))

        # Calculate final ratio to check if we have a good solution
        final_ratio = min_max_ratio(optimized_points)

        # If we achieved good results, proceed to local refinement
        if final_ratio >= 0.45:  # Threshold to avoid unnecessary refinement
            # Phase 2: Local refinement with L-BFGS-B
            try:
                # Use the optimized points as starting point for refinement
                x0_refine = optimized_points.flatten()

                # Refinement with L-BFGS-B using very tight tolerances
                result_refined = minimize(
                    objective_function,
                    x0_refine,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-10, 'gtol': 1e-10},
                    tol=1e-10
                )

                refined_points = result_refined.x.reshape((14, 3))
                final_ratio = min_max_ratio(refined_points)

                # Return refined points
                return refined_points

            except Exception:
                # If refinement fails, return the DE result
                return optimized_points

        # If not good enough, return the DE result anyway as fallback
        return optimized_points

    except Exception:
        pass  # Fall through to fallback

    # Final fallback to initial placement
    return initial_points

# EVOLVE-BLOCK-END