# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi


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


def penalty_objective(x_flat, penalty_weight=1e6):
    """Objective function with penalty for out-of-bounds points."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Apply penalty for constraint violations
    penalty = 0
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


def objective_function(x_flat, use_spherical_quality=True):
    """Objective function to maximize the min/max distance ratio."""
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
    """Create initial point placement using enhanced spherical code approach."""
    # Method: Generate points using Fibonacci-like distribution but with better uniformity
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

    # Improve distribution by applying multiple small perturbations
    np.random.seed(42)
    for _ in range(15):  # More perturbations for better uniformity
        # Add small random perturbations
        perturbation = np.random.normal(0, 0.015, (14, 3))
        initial_points += perturbation

        # Project back to sphere surface
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

        # Strategy 2: Fibonacci sphere scaled to unit cube
        lambda: (fibonacci_sphere(14) + 1) / 2,

        # Strategy 3: Latin Hypercube Sampling
        lambda: latin_hypercube_sampling(14, 3, seed=42),

        # Strategy 4: Random initialization
        lambda: np.random.rand(14, 3)
    ]

    # Track improvement for early stopping
    improvement_streak = 0
    max_improvement_streak = 5
    best_ratio_history = []
    min_improvement_threshold = 1e-8

    # Try different combinations of strategies with multiple restarts
    for restart in range(3):  # 3 restart rounds for better exploration
        for i, init_func in enumerate(initialization_strategies):
            # Generate initial points
            initial_points = init_func()

            # Flatten initial points for optimization
            x0 = initial_points.flatten()

            # Define bounds for each coordinate (0 to 1)
            bounds = [(0, 1) for _ in range(14 * 3)]

            # Phase 1: Global optimization with differential evolution
            try:
                # Use adaptive population size and mutation strategy based on restart round
                base_popsize = 15 + restart * 5  # Increase population size with restarts
                maxiter = 100 + restart * 50  # More iterations with restarts

                # Adaptive mutation strategy based on convergence
                mutation_rates = [(0.5, 1.0), (0.7, 1.0), (0.8, 1.0)]
                mutation_strategy = mutation_rates[min(restart, len(mutation_rates)-1)]

                # Try different recombination rates
                recombination_rates = [0.7, 0.8, 0.9]
                recombination_rate = recombination_rates[min(restart, len(recombination_rates)-1)]

                result = differential_evolution(
                    lambda x: adaptive_penalty_objective(x, iteration=restart),
                    bounds,
                    seed=42 + restart * 10 + i,
                    maxiter=maxiter,
                    popsize=base_popsize,
                    tol=1e-6,
                    mutation=mutation_strategy,
                    recombination=recombination_rate,
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
                    improvement_streak = 0  # Reset streak
                    best_ratio_history = []  # Reset history
                else:
                    improvement_streak += 1
                    # Check for convergence by monitoring recent improvements
                    best_ratio_history.append(final_ratio)
                    if len(best_ratio_history) > 10:
                        # Remove oldest entry
                        best_ratio_history.pop(0)
                        # Check if improvement is minimal
                        if len(best_ratio_history) >= 2:
                            recent_improvement = abs(best_ratio_history[-1] - best_ratio_history[0])
                            if recent_improvement < min_improvement_threshold:
                                improvement_streak += 1  # Count as stagnation

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

            # Refinement with L-BFGS-B - use adaptive tolerances
            ftol_vals = [1e-6, 1e-8, 1e-9]
            gtol_vals = [1e-6, 1e-8, 1e-9]

            # Try different tolerance settings to balance speed and accuracy
            for ftol, gtol in zip(ftol_vals, gtol_vals):
                result_refined = minimize(
                    lambda x: adaptive_penalty_objective(x, iteration=10),
                    x0_refine,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': ftol, 'gtol': gtol},
                    tol=ftol
                )

                refined_points = result_refined.x.reshape((14, 3))
                final_ratio = min_max_ratio(refined_points)

                # Update if improved
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = refined_points
                    break  # Stop once we achieve improvement

        except Exception as e:
            pass  # Keep original best points if refinement fails

    # Ensure we return valid points even if optimization failed
    if best_points is None:
        # Fallback to enhanced Fibonacci initialization
        best_points = create_initial_placement()

    return best_points

# EVOLVE-BLOCK-END