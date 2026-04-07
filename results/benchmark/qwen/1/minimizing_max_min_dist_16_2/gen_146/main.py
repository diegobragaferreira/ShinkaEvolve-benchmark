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

        # Strategy 1: Improved hexagonal grid pattern with better spacing
        points_hex = np.zeros((n_points, dimension))
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Hexagon height
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < n_points:
                    x = col * spacing_x + (row % 2) * spacing_x / 2
                    y = row * spacing_y
                    points_hex[idx] = [x, y]
                    idx += 1
        # Add small random perturbation to break symmetries
        points_hex += np.random.normal(0, 0.005, points_hex.shape)
        points_hex = np.clip(points_hex, 0, 1)

        # Strategy 2: Golden spiral with better distribution
        indices = np.arange(n_points)
        golden_angle = 2.399963229728653  # ~2π/(φ^2) where φ is golden ratio
        angles = golden_angle * indices
        # Use logarithmic spiral with better scaling
        radii = np.log(indices + 1) / np.log(n_points + 1)
        points_golden = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        points_golden = np.clip(points_golden, 0, 1)

        # Strategy 3: Modified grid with non-uniform spacing
        points_grid = []
        # Create a 4x4 grid but with staggered positions for better distribution
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Add jitter to break symmetry
                x += np.random.normal(0, 0.01) * (1 - abs(i - 1.5)/1.5)
                y += np.random.normal(0, 0.01) * (1 - abs(j - 1.5)/1.5)
                points_grid.append([x, y])
        points_grid = np.array(points_grid)
        points_grid = np.clip(points_grid, 0, 1)

        # Strategy 4: Random with boundary awareness and clustering avoidance
        points_random = np.random.rand(n_points, dimension)
        points_random = np.clip(points_random, 0.05, 0.95)

        # Strategy 5: Central cluster with peripheral points for better spread
        points_cluster = np.zeros((n_points, dimension))
        # Place first 4 points in a small cluster at center
        points_cluster[:4] = np.random.rand(4, 2) * 0.2 + 0.4
        # Place remaining points around the edges
        for i in range(4, 16):
            # Distribute along edges with some randomness
            side = i % 4
            t = (i // 4) / 3.0  # Parameter along edge
            if side == 0:  # Top edge
                points_cluster[i] = [t, 0]
            elif side == 1:  # Right edge
                points_cluster[i] = [1, t]
            elif side == 2:  # Bottom edge
                points_cluster[i] = [1-t, 1]
            else:  # Left edge
                points_cluster[i] = [0, 1-t]
            # Add some noise
            points_cluster[i] += np.random.normal(0, 0.02, 2)
        points_cluster = np.clip(points_cluster, 0, 1)

        # Test all strategies and return the best
        strategies = [points_hex, points_golden, points_grid, points_random, points_cluster]
        for points_strategy in strategies:
            ratio = fitness_function(points_strategy.flatten())
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points_strategy.copy()

        return best_points if best_points is not None else points_random

    # Multi-start optimization with restart capability
    best_final_points = None
    best_final_ratio = 0
    max_restarts = 5  # Increase restarts for better exploration

    # Keep track of previous good solutions for seeding
    previous_solutions = []

    for restart in range(max_restarts):
        # Generate initial points using multiple strategies
        if restart == 0:
            # First restart: use enhanced initialization
            points = generate_multiple_initializations()
        elif restart == 1:
            # Second restart: use the best previous solution if available
            if previous_solutions:
                points = previous_solutions[-1]
            else:
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
                maxiter=300,      # Increased iterations for better search
                popsize=30,       # Larger population for better diversity
                tol=1e-8,         # Tighter tolerance
                mutation=(0.7, 1.2),  # Wider mutation range
                recombination=0.8,    # Higher recombination rate
                seed=42 + restart,  # Different seed for each restart
                disp=False
            )

            # Local refinement with multiple strategies to improve results
            bounds_list = [(bounds[0], bounds[1]) for _ in range(n_points * dimension)]

            # Strategy 1: L-BFGS-B with strict tolerances
            lbfgs_result = minimize(
                fitness_function,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            # Strategy 2: Try SLSQP as backup if L-BFGS-B fails
            if not lbfgs_result.success:
                slsqp_result = minimize(
                    fitness_function,
                    de_result.x,
                    method='SLSQP',
                    bounds=bounds_list,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if slsqp_result.success:
                    final_points = slsqp_result.x.reshape(-1, 2)
                else:
                    final_points = de_result.x.reshape(-1, 2)
            else:
                final_points = lbfgs_result.x.reshape(-1, 2)

            final_ratio = fitness_function(final_points.flatten())

            # Keep the best solution found so far
            if final_ratio > best_final_ratio:
                best_final_ratio = final_ratio
                best_final_points = final_points.copy()
                previous_solutions.append(final_points.copy())

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