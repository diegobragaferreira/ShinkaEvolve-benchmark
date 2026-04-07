# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)

        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        if len(distances) == 0 or np.allclose(distances[distances != np.inf], 0):
            return -1.0  # Worst possible case

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0 or np.isinf(min_dist):
            return -1.0

        return -min_dist / max_dist

    def constrained_objective(x):
        # Boundary clamping in objective function for better numerical stability
        points = np.clip(x.reshape(-1, 2), 0, 1).flatten()
        return objective(points)

    def initialize_hexagonal_grid(n_points=16):
        """Initialize using a hexagonal pattern"""
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points.append([x, y])

        points = np.array(points[:n_points])

        # Normalize and scale to fit nicely in [0,1] x [0,1]
        if len(points) > 0:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])

            if x_range > 0:
                points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
            if y_range > 0:
                points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range

            # Scale to fit nicely in unit square with padding
            points[:, 0] *= 0.9
            points[:, 1] *= 0.9
            points[:, 0] += 0.05
            points[:, 1] += 0.05

        return points

    def initialize_fibonacci_spiral(n_points=16):
        """Initialize using fibonacci spiral pattern"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            theta = 2 * np.pi * i / phi
            r = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0.5
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])

        return np.array(points)

    def initialize_spiral_pattern(n_points=16):
        """Initialize using spiral pattern"""
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            radius = 0.4 * (i / n_points)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)

    def generate_multiple_initializations():
        """Generate multiple diverse initial configurations"""
        initial_configs = []

        # Hexagonal grid
        initial_configs.append(initialize_hexagonal_grid(16))

        # Fibonacci spiral
        initial_configs.append(initialize_fibonacci_spiral(16))

        # Spiral pattern
        initial_configs.append(initialize_spiral_pattern(16))

        # Random initialization
        initial_configs.append(np.random.rand(16, 2))

        # Perturbed hexagonal grid
        hex_grid = initialize_hexagonal_grid(16)
        perturbed = hex_grid + np.random.normal(0, 0.01, hex_grid.shape)
        initial_configs.append(np.clip(perturbed, 0, 1))

        return initial_configs

    # Set up optimization parameters
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

    # Generate multiple initial configurations
    initial_configs = generate_multiple_initializations()

    best_points = None
    best_ratio = -np.inf
    start_time = time.time()
    timeout = 170  # Leave 10 seconds for final processing

    # Multi-start optimization with diverse strategies
    for i, initial_config in enumerate(initial_configs):
        if time.time() - start_time > timeout:
            break

        # Add small random perturbation to break symmetry
        perturbed_config = initial_config + np.random.normal(0, 0.005, initial_config.shape)
        perturbed_config = np.clip(perturbed_config, 0, 1)

        x0 = perturbed_config.flatten()

        # Phase 1: Differential Evolution for global exploration
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    constrained_objective,
                    bounds,
                    seed=42 + i,
                    maxiter=50,
                    popsize=10,
                    tol=1e-6
                )

            if de_result.success:
                x0 = de_result.x
        except Exception:
            pass

        # Phase 2: Local optimization with L-BFGS-B
        try:
            local_result = minimize(
                constrained_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            if local_result.success:
                optimized_points = local_result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)

                # Calculate the actual ratio
                ratio = -constrained_objective(local_result.x)  # Convert back to positive ratio

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

        except Exception:
            pass

        # Phase 3: Additional local refinement with different method as fallback
        if best_points is not None and time.time() - start_time < timeout:
            try:
                # Try Nelder-Mead as alternative local optimizer
                nm_result = minimize(
                    constrained_objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'adaptive': True}
                )

                if nm_result.success:
                    refined_points = nm_result.x.reshape(-1, 2)
                    refined_points = np.clip(refined_points, 0, 1)
                    ratio = -constrained_objective(nm_result.x)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()

            except Exception:
                pass

    # Final fallback to best configuration found so far
    if best_points is None:
        # Use the best of our initial configurations
        best_points = initial_configs[0]
        for config in initial_configs[1:]:
            try:
                ratio = -objective(config.flatten())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = config.copy()
            except Exception:
                continue

    # Final refinement with high iteration count if we have a reasonable solution
    if best_points is not None and time.time() - start_time < timeout:
        try:
            final_result = minimize(
                constrained_objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )

            if final_result.success:
                final_points = final_result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                final_ratio = -constrained_objective(final_result.x)

                if final_ratio > best_ratio:
                    best_points = final_points

        except Exception:
            pass

    # Ensure we return a valid solution
    if best_points is None:
        best_points = initialize_hexagonal_grid(16)

    return best_points


# EVOLVE-BLOCK-END