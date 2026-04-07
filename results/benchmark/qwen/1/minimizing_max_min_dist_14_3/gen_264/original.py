# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import qmc
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def fibonacci_spiral_on_sphere(n_points: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral with golden angle."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def sobol_initialization(n: int, seed: int = 42) -> np.ndarray:
        """Generate points using Sobol sequence for better space-filling properties"""
        try:
            # Create Sobol sequence sampler
            sampler = qmc.Sobol(d=3, seed=seed)
            # Generate points
            points = sampler.random(n)
            # Scale to [-1, 1]^3
            points = points * 2 - 1
            return points
        except:
            # Fallback to random initialization if Sobol fails
            return np.random.uniform(-1, 1, (n, 3))

    def objective_energy(x):
        """Objective function that minimizes potential energy (inverse distance)"""
        points = x.reshape(-1, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)
        # Compute distance matrix
        dist_matrix = cdist(points, points)
        # Avoid division by zero
        np.fill_diagonal(dist_matrix, 1e-10)
        # Energy is sum of inverse distances
        return np.sum(1.0 / dist_matrix)

    def objective_ratio(x):
        """Objective function that maximizes min/max distance ratio"""
        points = x.reshape(-1, 3)
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)

        # Compute distance matrix
        dist_matrix = cdist(points, points)
        np.fill_diagonal(dist_matrix, np.inf)

        min_dist = np.min(dist_matrix)
        max_dist = np.max(dist_matrix)

        if max_dist == 0:
            return -1.0

        # We want to maximize min_dist / max_dist, so minimize -min_dist / max_dist
        return -min_dist / max_dist

    def sphere_constraint(x):
        """Constraint function ensuring all points lie on unit sphere."""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return difference from unit radius (should be close to 0)
        return norms - 1.0

    # Multi-start approach with diverse initializations
    best_ratio = -np.inf
    best_points = None

    # Strategy 1: Standard Fibonacci spiral
    fib_points = fibonacci_spiral_on_sphere(14)

    # Strategy 2: Sobol sequence initialization
    sobol_points = sobol_initialization(14, seed=42)

    # Strategy 3: Perturbed Fibonacci points
    np.random.seed(42)
    perturbed_fib = fib_points + np.random.normal(0, 0.03, fib_points.shape)

    # Strategy 4: Perturbed Sobol points
    perturbed_sobol = sobol_points + np.random.normal(0, 0.03, sobol_points.shape)

    initializations = [
        fib_points,
        sobol_points,
        perturbed_fib,
        perturbed_sobol
    ]

    # Try each initialization with optimization
    for i, initial_points in enumerate(initializations):
        try:
            # Normalize to unit sphere
            norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
            initial_points = initial_points / np.where(norms > 0, norms, 1)

            # First, optimize using energy minimization to get a good starting configuration
            x0 = initial_points.flatten()

            # Use L-BFGS-B for energy minimization first
            result_energy = minimize(
                objective_energy,
                x0,
                method='L-BFGS-B',
                options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 500}
            )

            if result_energy.success:
                points_energy = result_energy.x.reshape(-1, 3)
                # Re-normalize
                norms = np.linalg.norm(points_energy, axis=1, keepdims=True)
                points_energy = points_energy / np.where(norms > 0, norms, 1)
            else:
                points_energy = initial_points

            # Then optimize using the actual ratio objective with SLSQP
            # Use the energy-optimized points as the starting point
            x0_final = points_energy.flatten()

            # Define constraint: points must be on unit sphere
            cons = {'type': 'eq', 'fun': sphere_constraint}

            result = minimize(
                objective_ratio,
                x0_final,
                method='SLSQP',
                constraints=cons,
                options={'ftol': 1e-12, 'maxiter': 1000}
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                # Final normalization to ensure unit sphere
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                optimized_points = optimized_points / np.where(norms > 0, norms, 1)

                # Calculate the ratio for this run
                dist_matrix = cdist(optimized_points, optimized_points)
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)
                max_dist = np.max(dist_matrix)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            else:
                # Even if SLSQP fails, use the energy-optimized points
                dist_matrix = cdist(points_energy, points_energy)
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)
                max_dist = np.max(dist_matrix)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points_energy.copy()

        except Exception as e:
            warnings.warn(f"Initialization {i} failed: {str(e)}")
            continue

    # Fallback to standard Fibonacci spiral if everything else fails
    if best_points is None:
        initial_points = fibonacci_spiral_on_sphere(14)
        norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
        initial_points = initial_points / np.where(norms > 0, norms, 1)
        return initial_points

    return best_points

# EVOLVE-BLOCK-END