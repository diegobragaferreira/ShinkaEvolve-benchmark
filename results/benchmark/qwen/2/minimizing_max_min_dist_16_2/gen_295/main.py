# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def golden_spiral_2d(n_points):
        """Generate points on a 2D golden spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        return np.array(points)

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Compute pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -min_dist / max_dist

    def objective_with_regularization(x):
        """Objective with regularization to avoid numerical issues"""
        points = x.reshape(-1, 2)
        distances = pdist(points)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-12
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio

    def constraint_bounds(x):
        """Constraint function for bounds checking"""
        points = x.reshape(-1, 2)
        # Check bounds: each coordinate should be in [0,1]
        violations = []
        for coord in [points[:, 0], points[:, 1]]:
            violations.extend(np.maximum(0 - coord, 0))   # lower bound
            violations.extend(np.maximum(coord - 1, 0))   # upper bound
        return np.array(violations)

    def create_dispersed_corner_pattern():
        """Create a specialized corner-based pattern that maximizes initial spread"""
        # Place 4 corner points
        corners = [[0, 0], [1, 0], [0, 1], [1, 1]]
        # Add 4 edge midpoints
        edges = [[0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]]
        # Add 4 interior points in a cross pattern
        cross = [[0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5]]
        # Add 4 more points in a diamond pattern
        diamond = [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]]

        points = corners + edges + cross + diamond
        return np.array(points)

    def create_hexagonal_grid():
        """Create a hexagonal grid pattern with some perturbation"""
        # Create a 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0
                y = i / 3.0
                # Offset every other row
                if i % 2 == 1:
                    x += 1.0 / (3 * 2)
                points.append([x, y])

        # Convert to numpy array and add slight perturbations
        points = np.array(points[:16], dtype=np.float64)
        # Add small random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.01, (16, 2))
        # Clip to valid range
        points = np.clip(points, 0, 1)
        return points

    def create_clustered_initial():
        """Create an initial configuration with points clustered in specific regions"""
        np.random.seed(42)
        # Create 4 clusters of 4 points each
        clusters = []
        cluster_centers = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        for center in cluster_centers:
            for _ in range(4):
                point = [center[0] + np.random.normal(0, 0.05), center[1] + np.random.normal(0, 0.05)]
                clusters.append(point)
        return np.array(clusters)

    def create_adaptive_perturbed_grid():
        """Create a grid with adaptive perturbations based on distance analysis"""
        # Start with regular 4x4 grid
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]

        # Apply perturbations with special emphasis on boundary points
        np.random.seed(42)
        for i in range(16):
            row, col = i // 4, i % 4
            # Emphasize perturbations for corner and edge points to encourage spreading
            if row in [0, 3] or col in [0, 3]:
                std_factor = 1.5
            else:
                std_factor = 1.0
            grid_points[i] += np.random.normal(0, 0.02 * std_factor, 2)

        grid_points = np.clip(grid_points, 0, 1)
        return grid_points

    def optimize_with_restarts():
        """Run optimization with multiple enhanced initial configurations"""
        best_ratio = -np.inf
        best_points = None
        best_result = None

        # Try multiple different initial configurations
        initial_configs = []

        # 1. Golden spiral pattern
        spiral_points = golden_spiral_2d(16)
        # Scale and center the spiral
        spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
            np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
        spiral_points = spiral_points * 0.8 + 0.1  # Scale to [0.1, 0.9]
        initial_configs.append(spiral_points.copy())

        # 2. Dispersed corner pattern
        initial_configs.append(create_dispersed_corner_pattern())

        # 3. Hexagonal grid pattern
        initial_configs.append(create_hexagonal_grid())

        # 4. Clustered initial configuration
        initial_configs.append(create_clustered_initial())

        # 5. Adaptive perturbed grid
        initial_configs.append(create_adaptive_perturbed_grid())

        # 6. Perturbed grid
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]
        grid_points += np.random.normal(0, 0.05, (16, 2))
        grid_points = np.clip(grid_points, 0, 1)
        initial_configs.append(grid_points)

        # 7. Random uniform points
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        initial_configs.append(random_points)

        # Run optimization for each configuration
        for i, init_points in enumerate(initial_configs):
            try:
                # Flatten for optimization
                x0 = init_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]

                # Optimize
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-9, 'eps': 1e-6},
                    # Add a callback to monitor progress
                    callback=None
                )

                if result.success:
                    # Extract final points
                    final_points = result.x.reshape(-1, 2)

                    # Compute actual ratio
                    distances = pdist(final_points)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)

                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            best_result = result

            except Exception as e:
                warnings.warn(f"Optimization failed for initial config {i}: {e}")
                continue

        return best_points if best_points is not None else initial_configs[0]

    # Try optimized approach first
    try:
        final_points = optimize_with_restarts()
    except:
        # Fallback to simple approach if something fails
        np.random.seed(42)
        final_points = np.random.rand(16, 2)

    return final_points


# EVOLVE-BLOCK-END