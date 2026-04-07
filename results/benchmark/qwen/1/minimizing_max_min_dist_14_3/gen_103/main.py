# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import warnings
warnings.filterwarnings('ignore')

# Import Sobol sequence generator
try:
    from sobol_seq import i4_sobol_generate
except ImportError:
    # Fallback to custom Sobol implementation if needed
    def i4_sobol_generate(dim_num, n, skip=1):
        # Simple pseudo-Sobol sequence generator
        import random
        random.seed(42)
        result = np.zeros((n, dim_num))
        for i in range(n):
            for j in range(dim_num):
                result[i, j] = random.random()
        return result

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

    def constraint_sphere(x):
        # Ensure points stay within unit sphere
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1 - norms  # Should be >= 0

    def constraint_max_distance(x):
        # Ensure maximum distance doesn't exceed some reasonable bound
        points = x.reshape(-1, 3)
        distances = cdist(points, points)
        np.fill_diagonal(distances, 0)
        max_dist = np.max(distances)
        return 2 - max_dist  # Should be >= 0 (allowing up to diameter 2)

    def normalize_points(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def generate_sobol_points(n):
        """Generate points using Sobol sequence on sphere"""
        # Generate Sobol points in [0,1]^3
        sobol_points = i4_sobol_generate(3, n)

        # Transform to sphere using spherical coordinates
        points = []
        for i in range(n):
            # Map [0,1]^3 to spherical coordinates
            u, v, w = sobol_points[i]
            theta = np.arccos(1 - 2*u)  # Polar angle
            phi = 2 * np.pi * v         # Azimuthal angle
            r = w ** (1/3)              # Radius (cube root for uniform volume distribution)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            points.append([x, y, z])

        return np.array(points)

    def generate_sobol_on_sphere(n):
        """Generate points using Sobol sequence directly on unit sphere"""
        # Generate Sobol points in [0,1]^3
        sobol_points = i4_sobol_generate(3, n)

        # Map to unit sphere using normal distribution approach (better than spherical coordinates)
        points = []
        for i in range(n):
            # Each component from Sobol sequence transformed to standard normal
            # But we'll use a simpler approach with direct spherical mapping
            u, v, w = sobol_points[i]
            theta = np.arccos(1 - 2*u)  # Polar angle from 0 to pi
            phi = 2 * np.pi * v         # Azimuthal angle from 0 to 2pi
            r = 1.0                     # On unit sphere

            # Convert spherical to Cartesian
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            points.append([x, y, z])

        return np.array(points)

    def generate_perturbed_sobol_points(n, perturbation_strength=0.05):
        """Generate Sobol points with small random perturbations"""
        base_points = generate_sobol_on_sphere(n)
        perturbations = np.random.normal(0, perturbation_strength, (n, 3))
        perturbed_points = base_points + perturbations
        # Normalize back to unit sphere
        perturbed_points = perturbed_points / np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        return perturbed_points

    # Multiple starting configurations - now with Sobol sequences
    configs = []

    # Configuration 1: Sobol sequence on sphere
    configs.append(generate_sobol_on_sphere(14))

    # Configuration 2: Random points on sphere
    np.random.seed(42)
    random_points = np.random.randn(14, 3)
    random_points = normalize_points(random_points)
    configs.append(random_points)

    # Configuration 3: Perturbed Sobol
    np.random.seed(100)
    configs.append(generate_perturbed_sobol_points(14, 0.02))

    # Configuration 4: Another random seed
    np.random.seed(200)
    random_points2 = np.random.randn(14, 3)
    random_points2 = normalize_points(random_points2)
    configs.append(random_points2)

    # Configuration 5: Sobol points again
    configs.append(generate_sobol_on_sphere(14))

    # Configuration 6: Perturbed Sobol with larger perturbation
    np.random.seed(300)
    configs.append(generate_perturbed_sobol_points(14, 0.05))

    # Configuration 7: Another deterministic seed
    np.random.seed(123)
    configs.append(normalize_points(np.random.randn(14, 3)))

    # Configuration 8: Yet another perturbed version
    np.random.seed(456)
    configs.append(generate_perturbed_sobol_points(14, 0.03))

    # Main optimization loop with multiple restarts
    best_final_points = None
    best_ratio = 0

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
                objective,
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
                    {'type': 'ineq', 'fun': constraint_sphere},
                    {'type': 'ineq', 'fun': constraint_max_distance}
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

    # Final refinement step with SLSQP from the best found configuration
    if best_final_points is not None:
        try:
            x0 = best_final_points.flatten()
            cons = [
                {'type': 'ineq', 'fun': constraint_sphere},
                {'type': 'ineq', 'fun': constraint_max_distance}
            ]

            # Use L-BFGS-B for final polishing
            refined_result = minimize(objective, x0, method='L-BFGS-B', constraints=cons,
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

    # If still no solution, return the last attempt or fallback to Sobol
    if best_final_points is None:
        return generate_sobol_on_sphere(14)

    return best_final_points

# EVOLVE-BLOCK-END