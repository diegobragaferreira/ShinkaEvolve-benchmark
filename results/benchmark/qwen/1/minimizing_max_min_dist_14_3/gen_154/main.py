# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import qmc
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def distance_ratio(points_flat):
        """Calculate the ratio of minimum to maximum distance"""
        points = points_flat.reshape(-1, 3)
        distances = squareform(pdist(points))
        # Set diagonal to large value so it doesn't affect min/max
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist

    def objective_function(points_flat):
        """Minimize negative of distance ratio (since we want to maximize)"""
        return -distance_ratio(points_flat)

    def sobol_sphere_initialization(n):
        """
        Generate initial points using Sobol sequence projected onto sphere.
        This provides superior space filling compared to standard methods.
        """
        # Generate Sobol sequence in [0,1]^3
        sampler = qmc.Sobol(d=3, seed=42)
        points = sampler.random(n=n)

        # Convert to unit sphere using spherical coordinates
        # Map [0,1] to [0,pi] for theta and [0,2pi] for phi
        theta = points[:, 0] * np.pi  # 0 to pi
        phi = points[:, 1] * 2 * np.pi  # 0 to 2pi
        r = points[:, 2]  # Radius component

        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Combine into points array
        points_cartesian = np.column_stack([x, y, z])

        # Normalize to unit sphere
        norms = np.linalg.norm(points_cartesian, axis=1)
        normalized_points = points_cartesian / norms[:, np.newaxis]

        return normalized_points

    def gaussian_perturbation_initialization(n, base_points, sigma=0.015):
        """
        Apply Gaussian perturbations to break symmetry and improve optimization chances.
        """
        # Create perturbed version of base points
        perturbed_points = base_points.copy()

        # Apply Gaussian noise with controlled magnitude
        np.random.seed(42)
        noise = np.random.normal(0, sigma, perturbed_points.shape)
        perturbed_points += noise

        # Normalize to unit sphere again
        norms = np.linalg.norm(perturbed_points, axis=1)
        perturbed_points = perturbed_points / norms[:, np.newaxis]

        return perturbed_points

    def adaptive_optimization(x0, maxiter=1000):
        """
        Adaptive optimization with progressive tightening and better convergence control
        """
        # Define constraints for normalization (points should be on unit sphere)
        constraints = []

        def constraint_func(x):
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0  # Should be near 0 for unit sphere

        # Add constraint for each point to lie on unit sphere
        for i in range(14):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]})

        # Phase 1: Coarse optimization (fast, loose constraints)
        bounds = [(-1.2, 1.2)] * len(x0)

        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-4},
            tol=1e-4
        )

        # Phase 2: Medium optimization (moderate constraints)
        x1 = result.x
        bounds = [(-1.1, 1.1)] * len(x0)

        result = minimize(
            objective_function,
            x1,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )

        # Phase 3: Fine optimization (tight constraints)
        x2 = result.x
        bounds = [(-1.05, 1.05)] * len(x0)

        result = minimize(
            objective_function,
            x2,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )

        return result.x

    best_ratio = -np.inf
    best_points = None

    # Multi-start optimization with enhanced initialization strategies
    for restart in range(12):  # Increased restarts for better exploration
        np.random.seed(42 + restart)

        # Strategy 1: Sobol-based initialization
        if restart % 3 == 0:
            initial_points = sobol_sphere_initialization(14)
        # Strategy 2: Gaussian perturbed Sobol initialization
        elif restart % 3 == 1:
            base_points = sobol_sphere_initialization(14)
            initial_points = gaussian_perturbation_initialization(14, base_points, sigma=0.015)
        # Strategy 3: Another Gaussian perturbed version with different sigma
        else:
            base_points = sobol_sphere_initialization(14)
            initial_points = gaussian_perturbation_initialization(14, base_points, sigma=0.02)

        # Flatten for optimization
        x0 = initial_points.flatten()

        # Optimize with adaptive approach
        try:
            optimized_points = adaptive_optimization(x0, maxiter=1000)

            # Calculate final ratio
            ratio = distance_ratio(optimized_points)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

        except Exception as e:
            continue

    # If no good solution found, fallback to Sobol initialization
    if best_points is None:
        initial_points = sobol_sphere_initialization(14)
        best_points = initial_points.flatten()

    # Convert back to 14x3 array
    final_points = best_points.reshape(14, 3)

    return final_points

# EVOLVE-BLOCK-END