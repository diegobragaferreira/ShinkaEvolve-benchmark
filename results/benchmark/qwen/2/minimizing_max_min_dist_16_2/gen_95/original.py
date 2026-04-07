# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape flat array back to points
        pts = x.reshape(16, 2)

        # Calculate all pairwise distances efficiently using scipy
        distances = pdist(pts)

        # Handle case where there are no distances (shouldn't happen)
        if len(distances) == 0:
            return 0

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize the ratio
        if max_dist <= 1e-12:  # Avoid division by zero
            return 0
        return -min_dist / max_dist

    def generate_initial_configurations():
        """Generate multiple high-quality initial configurations"""
        configs = []

        # Configuration 1: Hexagonal grid (improved version)
        hex_points = np.zeros((16, 2))
        idx = 0
        for i in range(4):
            for j in range(4):
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset) / 3.0
                y = i / 3.0
                # Add controlled perturbation
                x += np.random.normal(0, 0.015)
                y += np.random.normal(0, 0.015)
                # Ensure within bounds
                x = np.clip(x, 0.001, 0.999)
                y = np.clip(y, 0.001, 0.999)
                hex_points[idx] = [x, y]
                idx += 1
        configs.append(hex_points)

        # Configuration 2: Concentric rings (better coverage)
        ring_points = np.zeros((16, 2))
        angles = np.linspace(0, 2*np.pi, 17)[:-1]
        # Two rings: inner radius 0.3, outer radius 0.7
        radii = np.concatenate([np.full(8, 0.3), np.full(8, 0.7)])
        for i in range(16):
            angle = angles[i]
            radius = radii[i]
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            ring_points[i] = [x, y]
        configs.append(ring_points)

        # Configuration 3: Fibonacci spiral-like arrangement
        fib_points = np.zeros((16, 2))
        for i in range(16):
            # Use a variant of fibonacci spiral but mapped to square
            angle = i * 2.4  # Changed from golden ratio to encourage better spacing
            radius = np.sqrt(i/15.0) if i > 0 else 0
            x = 0.5 + radius * np.cos(angle) * 0.45
            y = 0.5 + radius * np.sin(angle) * 0.45
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            fib_points[i] = [x, y]
        configs.append(fib_points)

        # Configuration 4: Perturbed regular grid
        grid_points = np.zeros((16, 2))
        for i in range(16):
            row = i // 4
            col = i % 4
            x = col / 3.0 + np.random.normal(0, 0.02)
            y = row / 3.0 + np.random.normal(0, 0.02)
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            grid_points[i] = [x, y]
        configs.append(grid_points)

        return configs

    # Generate multiple initial configurations
    initial_configs = generate_initial_configurations()

    best_ratio = -np.inf
    best_points = None

    # Try optimization from different starting points
    for i, initial_config in enumerate(initial_configs):
        # Define bounds (points must be in [0.001, 0.999] x [0.001, 0.999] to avoid edge issues)
        bounds = [(0.001, 0.999) for _ in range(32)]

        # Stage 1: Global optimization with differential evolution (exploration)
        try:
            # Use differential evolution for better global search
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42 + i,
                tol=1e-8,
                mutation=(0.5, 1),
                recombination=0.7
            )

            # Stage 2: Local refinement with L-BFGS-B (exploitation)
            refined_result = minimize(
                objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if refined_result.success:
                optimized_points = refined_result.x.reshape(16, 2)

                # Calculate the actual ratio for this solution
                distances = pdist(optimized_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()

        except Exception as e:
            continue  # Skip this configuration if optimization fails

    # If no good solution was found, return the best of our initial configurations
    if best_points is None:
        # Evaluate all initial configurations to select the best one
        max_ratio = -np.inf
        for config in initial_configs:
            distances = pdist(config)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_points = config.copy()

    # Ensure we have a valid result
    if best_points is None:
        # Fallback to a simple 4x4 grid with small random perturbations
        base_grid = np.array([[i, j] for i in range(4) for j in range(4)])
        best_points = base_grid.astype(float) / 3.0
        np.random.seed(42)
        best_points += np.random.uniform(-0.02, 0.02, best_points.shape)
        best_points = np.clip(best_points, 0.001, 0.999)

    return best_points


# EVOLVE-BLOCK-END