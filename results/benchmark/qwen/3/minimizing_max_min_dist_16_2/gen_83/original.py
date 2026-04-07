# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        distances = pdist(points)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return np.min(distances) / max_dist

    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        return -compute_min_max_ratio(points)

    def create_hexagonal_grid():
        """Create initial configuration using a hexagonal grid pattern"""
        # Create a 4x4 hexagonal grid
        points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = 1.0 / (rows - 1)

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x
                y = i * spacing_y
                # Add slight offset to create hexagonal pattern
                if i % 2 == 1:
                    x += spacing_x * 0.5
                points.append([x, y])

        # Convert to numpy array and normalize
        points = np.array(points[:16])  # Take first 16 points
        return points

    def create_alternative_configurations():
        """Generate multiple alternative initial configurations"""
        configs = []

        # Configuration 1: Hexagonal grid
        configs.append(create_hexagonal_grid())

        # Configuration 2: Random but constrained
        np.random.seed(42)
        configs.append(np.random.rand(16, 2))

        # Configuration 3: Grid with perturbations
        grid_points = create_hexagonal_grid()
        np.random.seed(43)
        perturbations = np.random.normal(0, 0.05, (16, 2))
        configs.append(np.clip(grid_points + perturbations, 0, 1))

        # Configuration 4: Simple 4x4 uniform grid
        uniform_grid = []
        for i in range(4):
            for j in range(4):
                uniform_grid.append([i/3, j/3])
        configs.append(np.array(uniform_grid[:16]))

        return configs

    # Generate multiple initial configurations
    initial_configs = create_alternative_configurations()

    best_ratio = -np.inf
    best_points = None

    # Try each initial configuration with optimization
    for i, initial_points in enumerate(initial_configs):
        # Clip initial points to valid bounds
        initial_points = np.clip(initial_points, 0, 1)

        # Flatten for optimization
        x0 = initial_points.flatten()

        # Set up bounds
        bounds = [(0, 1) for _ in range(32)]

        # First optimization pass with SLSQP
        try:
            result1 = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-6}
            )

            # Extract optimized points
            optimized_points = result1.x.reshape(-1, 2)

            # Refinement with L-BFGS-B
            result2 = minimize(
                objective_function,
                optimized_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8}
            )

            final_points = result2.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)

            # Evaluate the result
            current_ratio = compute_min_max_ratio(final_points)

            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = final_points.copy()

        except Exception as e:
            print(f"Error in optimization {i}: {e}")
            continue

    # If no optimization succeeded, return the last attempt
    if best_points is None:
        # Fallback to simple random configuration with good seed
        np.random.seed(42)
        best_points = np.random.rand(16, 2)

    return best_points

# EVOLVE-BLOCK-END