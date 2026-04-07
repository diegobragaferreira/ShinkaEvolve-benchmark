# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute pairwise distances
        distances = cdist(points, points)
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Minimize negative of minimum distance (maximize minimum distance)
        return -np.min(distances)

    def objective_ratio(x):
        # Reshape x back to points array
        points = x.reshape(-1, 3)
        # Compute pairwise distances
        distances = cdist(points, points)
        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        # Avoid division by zero
        if d_max == 0:
            return -np.inf
        # Return negative ratio to maximize it (since scipy minimizes)
        return -d_min / d_max

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_bounds(x):
        # Ensure points stay within [0,1]^3 bounds
        points = x.reshape(-1, 3)
        # Check that all coordinates are within [0,1]
        return np.concatenate([points.min(axis=0), 1 - points.max(axis=0)])

    def generate_fibonacci_points(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            theta = np.arccos(1 - 2*(i/(n-1)))
            phi = i * 2 * np.pi / golden_ratio
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
        return np.array(points)

    def normalize_points(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def generate_perturbed_fibonacci_points(n, perturbation_strength=0.05):
        """Generate fibonacci points with small random perturbations"""
        base_points = generate_fibonacci_points(n)
        perturbations = np.random.normal(0, perturbation_strength, (n, 3))
        perturbed_points = base_points + perturbations
        # Normalize back to unit sphere
        perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        return perturbed_points

    # Multiple starting configurations
    configs = []

    # Configuration 1: Fibonacci spiral on sphere
    configs.append(generate_fibonacci_points(14))

    # Configuration 2: Perturbed Fibonacci
    np.random.seed(100)
    configs.append(generate_perturbed_fibonacci_points(14, 0.02))

    # Configuration 3: Another Fibonacci variant
    np.random.seed(200)
    configs.append(generate_fibonacci_points(14))

    # Configuration 4: Random points on sphere
    np.random.seed(300)
    random_points = np.random.randn(14, 3)
    configs.append(normalize_points(random_points))

    # Configuration 5: Another perturbed version
    np.random.seed(400)
    configs.append(generate_perturbed_fibonacci_points(14, 0.03))

    # Configuration 6: Another random seed
    np.random.seed(500)
    random_points2 = np.random.randn(14, 3)
    configs.append(normalize_points(random_points2))

    # Main optimization loop with multiple restarts using differential evolution
    best_final_points = None
    best_ratio = -np.inf

    # First try differential evolution on each configuration
    for i, initial_config in enumerate(configs):
        try:
            # Use differential evolution for global search first
            n_points = 14
            n_vars = n_points * 3  # 14 points * 3 coordinates each

            # Bounds for each coordinate: [-1, 1] to allow for sphere constraint
            bounds = [(-1, 1) for _ in range(n_vars)]

            # Run differential evolution with the current initial config
            result = differential_evolution(
                objective_ratio,
                bounds,
                seed=42 + i,
                maxiter=500,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )

            # Extract optimized points and normalize
            optimized_points = result.x.reshape(-1, 3)
            optimized_points = normalize_points(optimized_points)

            # Evaluate this solution
            distances = cdist(optimized_points, optimized_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_final_points = optimized_points.copy()

        except Exception as e:
            continue

    # If no good solution from DE, try local optimization from best configurations
    if best_final_points is None:
        # Try optimizing from the best initial configurations using local method
        for i, initial_config in enumerate(configs[:3]):  # Try first 3 configs
            try:
                # Local optimization around initial point
                x0 = initial_config.flatten()
                cons = [
                    {'type': 'ineq', 'fun': constraint_sphere}
                ]

                result = minimize(objective, x0, method='SLSQP', constraints=cons,
                                options={'ftol': 1e-8, 'maxiter': 500})

                optimized_points = result.x.reshape(-1, 3)
                optimized_points = normalize_points(optimized_points)

                # Evaluate this solution
                distances = cdist(optimized_points, optimized_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_final_points = optimized_points.copy()

            except Exception as e:
                continue

    # Final refinement step with L-BFGS-B from the best found configuration
    if best_final_points is not None:
        try:
            x0 = best_final_points.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere}
            ]

            # Use L-BFGS-B for final polishing
            refined_result = minimize(objective_ratio, x0, method='L-BFGS-B', constraints=cons,
                                    options={'ftol': 1e-12, 'maxiter': 300})

            refined_points = refined_result.x.reshape(-1, 3)
            refined_points = normalize_points(refined_points)

            # Re-evaluate final solution
            distances = cdist(refined_points, refined_points)
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist > 0:
                refined_ratio = min_dist / max_dist
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_final_points = refined_points.copy()
        except Exception as e:
            pass

    # If still no solution, return the last attempt or fallback to Fibonacci
    if best_final_points is None:
        return generate_fibonacci_points(14)

    return best_final_points

# EVOLVE-BLOCK-END