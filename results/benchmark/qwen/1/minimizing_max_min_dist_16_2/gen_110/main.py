# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n_points = 16
    dimension = 2
    bounds = (0, 1)

    # Set seed for reproducibility
    np.random.seed(42)

    def calculate_distance_matrix(points):
        """Calculate pairwise distance matrix with numerical stability"""
        # Use squareform for better numerical stability
        distances = squareform(pdist(points))
        # Ensure diagonal is infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        return distances

    def fitness_function(points_flat):
        """Calculate fitness as min/max distance ratio"""
        # Reshape flat array back to points
        points = points_flat.reshape(-1, dimension)

        # Ensure points are within bounds with small epsilon to avoid edge cases
        epsilon = 1e-8
        points = np.clip(points, bounds[0] + epsilon, bounds[1] - epsilon)

        # Calculate distance matrix
        try:
            dist_matrix = calculate_distance_matrix(points)

            # Calculate min and max distances (excluding infinity from diagonal)
            min_dist = np.min(dist_matrix)
            max_dist = np.max(dist_matrix)

            # Avoid division by zero or extremely small values
            if max_dist < 1e-12:
                return 0

            # Return ratio (we want to maximize this)
            ratio = min_dist / max_dist
            return ratio

        except Exception:
            return 0

    def generate_multiple_initializations():
        """Generate multiple initial point configurations and select the best"""
        best_points = None
        best_ratio = 0

        # Strategy 1: Hexagonal grid pattern
        points_hex = np.zeros((n_points, dimension))
        rows = 4
        cols = 4
        spacing = 0.25
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < n_points:
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * np.sqrt(3) / 2
                    points_hex[idx] = [x, y]
                    idx += 1
        # Normalize to [0,1] range
        points_hex[:, 0] = (points_hex[:, 0] - points_hex[:, 0].min()) / (points_hex[:, 0].max() - points_hex[:, 0].min()) * 0.8 + 0.1
        points_hex[:, 1] = (points_hex[:, 1] - points_hex[:, 1].min()) / (points_hex[:, 1].max() - points_hex[:, 1].min()) * 0.8 + 0.1
        # Add noise
        points_hex += np.random.normal(0, 0.01, points_hex.shape)
        points_hex = np.clip(points_hex, 0, 1)

        # Strategy 2: Golden spiral
        indices = np.arange(n_points)
        golden_angle = 2.399963229728653  # ~2π/(φ^2) where φ is golden ratio
        angles = golden_angle * indices
        radii = np.log(indices + 1) / np.log(n_points)  # Better distribution
        points_golden = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        points_golden = np.clip(points_golden, 0, 1)

        # Strategy 3: Perturbed square grid
        points_grid = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0 + np.random.normal(0, 0.015)
                y = (j + 0.5) / 4.0 + np.random.normal(0, 0.015)
                points_grid.append([x, y])
        points_grid = np.array(points_grid)
        points_grid = np.clip(points_grid, 0, 1)

        # Strategy 4: Random with boundary awareness
        points_random = np.random.rand(n_points, dimension)
        points_random = np.clip(points_random, 0.05, 0.95)

        # Test all strategies and return the best
        strategies = [points_hex, points_golden, points_grid, points_random]
        for points_strategy in strategies:
            ratio = fitness_function(points_strategy.flatten())
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points_strategy.copy()

        return best_points if best_points is not None else points_random

    # Multi-start optimization with restart capability
    best_final_points = None
    best_final_ratio = 0
    max_restarts = 3

    for restart in range(max_restarts):
        # Generate initial points using multiple strategies
        if restart == 0:
            # First restart: use enhanced initialization
            points = generate_multiple_initializations()
        else:
            # Subsequent restarts: use random initialization with boundary awareness
            points = np.random.rand(n_points, dimension)
            points = np.clip(points, 0.05, 0.95)

        # Method 1: Try differential evolution for global search
        try:
            # Flatten bounds for scipy
            flat_bounds = [(bounds[0], bounds[1]) for _ in range(n_points * dimension)]

            # Use differential evolution for global search with improved parameters
            de_result = differential_evolution(
                fitness_function,
                flat_bounds,
                maxiter=200,      # Increased iterations for better search
                popsize=25,       # Larger population for better diversity
                tol=1e-8,         # Tighter tolerance
                mutation=(0.7, 1.2),  # Wider mutation range
                recombination=0.8,    # Higher recombination rate
                seed=42 + restart,  # Different seed for each restart
                disp=False
            )

            # Local refinement with L-BFGS-B using stricter tolerances
            bounds_list = [(bounds[0], bounds[1]) for _ in range(n_points * dimension)]
            lbfgs_result = minimize(
                fitness_function,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            # Reshape result
            final_points = lbfgs_result.x.reshape(-1, 2)
            final_ratio = fitness_function(final_points.flatten())

            # Keep the best solution found so far
            if final_ratio > best_final_ratio:
                best_final_ratio = final_ratio
                best_final_points = final_points.copy()

        except Exception as e:
            # Fallback to current points if optimization fails
            print(f"Optimization failed on restart {restart}: {e}")
            continue

    # If no successful optimization, return the best initialization
    if best_final_points is None:
        points = generate_multiple_initializations()
        # Final verification of bounds
        points = np.clip(points, bounds[0], bounds[1])
        return points

    # Ensure final points are within bounds
    best_final_points = np.clip(best_final_points, bounds[0], bounds[1])

    return best_final_points


# EVOLVE-BLOCK-END