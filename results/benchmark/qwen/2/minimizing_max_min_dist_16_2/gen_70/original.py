# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative ratio (since we want to maximize)
        return -d_min / d_max

    def constraint(x):
        # Ensure points stay within [0,1] x [0,1]
        points = x.reshape(-1, 2)
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # x coordinates <= 1
            points[:, 1],           # y coordinates >= 0
            1 - points[:, 1]        # y coordinates <= 1
        ])

    def compute_ratio(points):
        """Compute the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def create_initial_configurations():
        """Create multiple high-quality initial configurations."""
        configs = []

        # Configuration 1: Structured 4x4 grid with perturbations
        np.random.seed(42)
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
        noise = np.random.normal(0, 0.05, (16, 2))
        config1 = np.clip(grid_points + noise, 0, 1)
        configs.append(config1)

        # Configuration 2: Random with clustering avoidance
        np.random.seed(123)
        config2 = np.random.uniform(0.05, 0.95, (16, 2))
        # Add some structure to avoid very tight clusters
        for i in range(0, 16, 4):  # Group every 4 points
            group_center = np.mean(config2[i:i+4], axis=0)
            config2[i:i+4] += np.random.normal(0, 0.03, (4, 2))
            config2[i:i+4] = np.clip(config2[i:i+4], 0, 1)
        configs.append(config2)

        # Configuration 3: Fibonacci spiral-like arrangement
        np.random.seed(456)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.sqrt(np.linspace(0.05, 0.45, 16))  # Square root for uniform distribution
        fib_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        fib_points = np.clip((fib_points + 1) / 2, 0, 1)  # Normalize to [0,1]
        configs.append(fib_points)

        # Configuration 4: Hexagonal grid approximation
        np.random.seed(789)
        hex_x = np.array([0.15, 0.45, 0.75, 0.3, 0.6, 0.15, 0.45, 0.75, 0.225, 0.525, 0.825, 0.375, 0.675, 0.225, 0.525, 0.825])
        hex_y = np.array([0.15, 0.15, 0.15, 0.45, 0.45, 0.75, 0.75, 0.75, 0.3, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.9])
        hex_points = np.column_stack([hex_x, hex_y])
        noise = np.random.normal(0, 0.03, (16, 2))
        config4 = np.clip(hex_points + noise, 0, 1)
        configs.append(config4)

        return configs

    def optimize_with_refinement(x0):
        """Perform sequential optimization with refinement stages."""
        # Stage 1: Fast optimization with L-BFGS-B
        bounds = [(0, 1) for _ in range(32)]
        result1 = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6}
        )

        if not result1.success:
            return None

        # Stage 2: Precise optimization with SLSQP
        result2 = minimize(
            objective,
            result1.x,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint},
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
        )

        if result2.success:
            return result2.x
        return None

    # Multi-start optimization with improved initializations
    best_ratio = -np.inf
    best_points = None

    # Generate multiple initial configurations
    initial_configs = create_initial_configurations()

    # Run optimizations with different initial configurations
    for i, initial_points in enumerate(initial_configs):
        try:
            # Optimize using our refined two-stage approach
            optimized_x = optimize_with_refinement(initial_points.flatten())

            if optimized_x is not None:
                optimized_points = optimized_x.reshape(-1, 2)
                final_ratio = compute_ratio(optimized_points)

                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = optimized_points.copy()

        except Exception as e:
            continue

    # If no successful optimization, return the best initial configuration
    if best_points is None:
        best_points = initial_configs[0] if initial_configs else np.random.uniform(0, 1, (16, 2))

    return best_points

# EVOLVE-BLOCK-END