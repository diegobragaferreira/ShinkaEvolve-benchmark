# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import SphericalVoronoi
import itertools


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


def latin_hypercube_sampling(n, d, seed=42):
    """Generate n points using Latin Hypercube Sampling in d dimensions."""
    np.random.seed(seed)
    samples = np.zeros((n, d))

    for i in range(d):
        # Generate random permutation for each dimension
        perm = np.random.permutation(n)
        samples[:, i] = perm

    # Normalize to [0, 1]
    samples = samples / (n - 1)

    return samples


def icosahedron_initialization(n, seed=42):
    """Initialize points using vertices of regular icosahedron with perturbations."""
    np.random.seed(seed)

    # Vertices of regular icosahedron (normalized to unit sphere)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]

    # Normalize vertices to unit sphere
    vertices = np.array(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms

    # For 14 points, we'll use 12 vertices of icosahedron plus 2 additional points
    # Place additional points at poles
    additional_points = np.array([[0, 0, 1], [0, 0, -1]])

    # Combine and add some randomness
    points_sphere = np.vstack([vertices[:12], additional_points])

    # Apply small random perturbations to improve distribution
    perturbation_magnitude = 0.03
    noise = np.random.normal(0, perturbation_magnitude, (14, 3))
    points_sphere += noise

    # Normalize again to maintain unit sphere
    norms = np.linalg.norm(points_sphere, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    points_sphere = points_sphere / norms

    # Project to unit cube [0,1]^3
    points_cube = (points_sphere + 1) / 2

    return points_cube


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


def adaptive_penalty_objective(x_flat, penalty_weight=1e6, iteration=0):
    """Objective function with adaptive penalty for out-of-bounds points.

    Args:
        x_flat: Flattened array of point coordinates [x1, y1, z1, x2, y2, z2, ...]
        penalty_weight: Base weight for constraint penalty
        iteration: Current iteration number for adaptive scaling

    Returns:
        Value to minimize (includes penalty for constraint violations)
    """
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


def adaptive_differential_evolution(objective_func, bounds, max_iter=200, popsize=None, seed=42):
    """Run differential evolution with adaptive population sizing."""
    if popsize is None:
        popsize = 15

    # Start with standard DE
    result = differential_evolution(
        objective_func,
        bounds,
        seed=seed,
        maxiter=max_iter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False
    )

    return result


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
        # Strategy 1: Fibonacci sphere scaled to unit cube
        lambda: (fibonacci_sphere(14) + 1) / 2,

        # Strategy 2: Latin Hypercube Sampling
        lambda: latin_hypercube_sampling(14, 3, seed=42),

        # Strategy 3: Random initialization
        lambda: np.random.rand(14, 3),

        # Strategy 4: Icosahedron-based initialization
        lambda: icosahedron_initialization(14, seed=42),

        # Strategy 5: Spherical Voronoi initialization
        lambda: spherical_voronoi_initialization(14, seed=42)
    ]

    # Track improvement for early stopping
    previous_best = -1
    improvement_streak = 0
    max_improvement_streak = 10

    # Try different combinations of strategies with multiple restarts
    for restart in range(3):  # 3 restart rounds
        for i, init_func in enumerate(initialization_strategies):
            # Generate initial points
            initial_points = init_func()

            # Flatten initial points for optimization
            x0 = initial_points.flatten()

            # Define bounds for each coordinate (0 to 1)
            bounds = [(0, 1) for _ in range(14 * 3)]

            # Phase 1: Global optimization with differential evolution
            try:
                # Use adaptive population size based on restart round
                popsize = 15 + restart * 5  # Increase population size with restarts

                result = adaptive_differential_evolution(
                    lambda x: adaptive_penalty_objective(x, iteration=restart),
                    bounds,
                    max_iter=100 + restart * 50,  # More iterations with restarts
                    popsize=popsize,
                    seed=42 + restart * 10 + i  # Different seed for each strategy
                )

                # Extract optimized points
                optimized_points = result.x.reshape((14, 3))

                # Calculate final ratio
                final_ratio = min_max_ratio(optimized_points)

                # Store best result
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()
                    improvement_streak = 0  # Reset streak
                else:
                    improvement_streak += 1

                # Early stopping if no improvement for too many attempts
                if improvement_streak >= max_improvement_streak:
                    break

            except Exception as e:
                continue  # Skip this strategy if optimization fails

    # Phase 2: Local refinement with L-BFGS-B if we found a good candidate
    if best_points is not None and best_ratio > 0:
        try:
            # Flatten the best points
            x0_refine = best_points.flatten()

            # Refinement with L-BFGS-B
            result_refined = minimize(
                lambda x: adaptive_penalty_objective(x, iteration=10),
                x0_refine,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9},
                tol=1e-9
            )

            refined_points = result_refined.x.reshape((14, 3))
            final_ratio = min_max_ratio(refined_points)

            # Update if improved
            if final_ratio > best_ratio:
                best_points = refined_points

        except Exception as e:
            pass  # Keep original best points if refinement fails

    # Ensure we return valid points even if optimization failed
    if best_points is None:
        # Fallback to Latin Hypercube Sampling
        best_points = latin_hypercube_sampling(14, 3, seed=42)

    return best_points


# EVOLVE-BLOCK-END