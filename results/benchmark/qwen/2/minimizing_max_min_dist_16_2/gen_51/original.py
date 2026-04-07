# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0

        # Calculate pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0

        return dmin / dmax

    def objective_function(points):
        """Objective function to maximize (negative because we minimize in scipy)."""
        return -calculate_min_max_ratio(points)

    def create_hexagonal_initialization():
        """Create a hexagonal-like arrangement of points."""
        points = np.zeros((16, 2))

        # Create a roughly hexagonal arrangement
        rows = 4
        cols = 4
        spacing = 1.0 / (rows + 1)

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < 16:
                    # Offset every other row
                    x = (j + 0.5 * (i % 2)) * spacing
                    y = i * spacing
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_grid_initialization():
        """Create a regular grid initialization."""
        points = np.zeros((16, 2))
        idx = 0

        # Create 4x4 grid
        for i in range(4):
            for j in range(4):
                if idx < 16:
                    x = j / 3.0 if j > 0 else 0.0
                    y = i / 3.0 if i > 0 else 0.0
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_perturbed_initialization(base_points, perturbation_magnitude=0.02):
        """Create a perturbed version of base initialization."""
        perturbed = base_points.copy()
        # Add random perturbation
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        # Ensure points stay within bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def optimize_with_local_refinement(initial_points, max_iter=500):
        """Perform local optimization refinement on initial configuration."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]

        # Optimize using L-BFGS-B method
        result = minimize(
            lambda flat_points: objective_function(flat_points.reshape(-1, 2)),
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=None
        )

        # Extract optimized points
        optimized_points = result.x.reshape(-1, 2)

        # Ensure all points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)

        return optimized_points

    def multi_start_optimization():
        """Perform multi-start optimization with different initial configurations."""
        best_ratio = -np.inf
        best_points = None

        # Try different initialization strategies
        initial_configs = []

        # Hexagonal initialization
        hex_initial = create_hexagonal_initialization()
        initial_configs.append(create_perturbed_initialization(hex_initial, 0.02))

        # Grid initialization
        grid_initial = create_grid_initialization()
        initial_configs.append(create_perturbed_initialization(grid_initial, 0.02))

        # Random initialization with hex pattern influence
        np.random.seed(42)
        random_initial = np.random.rand(16, 2)
        initial_configs.append(random_initial)

        # Try each initialization
        for i, initial_config in enumerate(initial_configs):
            try:
                # First perform global optimization with DE
                bounds = [(0, 1)] * 32

                # Use differential evolution for global search
                de_result = differential_evolution(
                    lambda x: objective_function(x.reshape(-1, 2)),
                    bounds,
                    seed=42 + i,
                    maxiter=200,
                    popsize=15,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    tol=1e-6,
                    disp=False
                )

                # Refine with local optimization
                refined_points = optimize_with_local_refinement(de_result.x.reshape(-1, 2), 300)
                ratio = calculate_min_max_ratio(refined_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points

            except Exception as e:
                # If optimization fails, continue with next configuration
                continue

        # If no better solution found, return the best local optimization
        if best_points is None:
            # Fallback to single local optimization with hexagonal initialization
            hex_initial = create_hexagonal_initialization()
            perturbed_hex = create_perturbed_initialization(hex_initial, 0.01)
            best_points = optimize_with_local_refinement(perturbed_hex, 500)

        return best_points

    # Main optimization routine
    np.random.seed(42)

    # Use multi-start optimization approach
    final_points = multi_start_optimization()

    return final_points


# EVOLVE-BLOCK-END