# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0

        # Compute pairwise distances with enhanced numerical stability
        distance_matrix = squareform(pdist(points))

        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distance_matrix, np.inf)

        # Get all finite distances (excluding NaN and inf values)
        finite_distances = distance_matrix[np.isfinite(distance_matrix)]

        if len(finite_distances) == 0:
            return 0

        # Get min and max distances
        dmin = np.min(finite_distances)
        dmax = np.max(finite_distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax

    def generate_hexagonal_grid():
        """Generate a hexagonal grid arrangement."""
        points = []
        rows = 4
        cols = 4

        # Hexagonal packing parameters
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Height of equilateral triangle

        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                points.append([x, y])

        return np.array(points)

    def generate_initial_strategies():
        """Generate multiple initial point configurations."""
        strategies = {}

        # Strategy 1: Hexagonal grid (primary)
        strategies['hex'] = generate_hexagonal_grid()

        # Strategy 2: Perturbed hexagonal grid
        np.random.seed(42)
        perturbed_hex = strategies['hex'] + np.random.normal(0, 0.02, strategies['hex'].shape)
        strategies['hex_perturbed'] = np.clip(perturbed_hex, 0, 1)

        # Strategy 3: Regular grid with jitter
        regular_grid = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                regular_grid.append([x, y])
        regular_grid = np.array(regular_grid)
        jittered_grid = regular_grid + np.random.normal(0, 0.01, regular_grid.shape)
        strategies['grid_jittered'] = np.clip(jittered_grid, 0, 1)

        # Strategy 4: Golden spiral (better radial distribution)
        indices = np.arange(16)
        golden_angle = 2.399963229728653
        angles = golden_angle * indices
        # Use logarithmic distribution for better point spreading
        radii = np.log(indices + 1) / np.log(16)
        golden_spiral = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])
        strategies['spiral'] = np.clip(golden_spiral, 0, 1)

        # Strategy 5: Random points with edge avoidance
        np.random.seed(123)
        random_points = np.random.rand(16, 2)
        strategies['random'] = np.clip(random_points, 0.05, 0.95)

        return strategies

    def evaluate_all_strategies(strategies):
        """Evaluate all initial strategies and return the best one."""
        best_strategy = None
        best_ratio = 0

        for name, points in strategies.items():
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_strategy = points.copy()

        return best_strategy, best_ratio

    def robust_optimization(initial_points, max_evaluations=1000):
        """Perform robust optimization with multiple strategies."""
        # Flatten for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]

        try:
            # First aggressive differential evolution with good parameters
            de_result = differential_evolution(
                lambda x: -compute_min_max_ratio(x.reshape(-1, 2)),
                bounds,
                maxiter=max_evaluations // 10,
                popsize=20,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-8,
                strategy='best1bin'
            )

            # Try local refinement with L-BFGS-B
            if de_result.success:
                refined_result = minimize(
                    lambda x: -compute_min_max_ratio(x.reshape(-1, 2)),
                    de_result.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )

                if refined_result.success:
                    final_points = refined_result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    return final_points

            # If local refinement fails, return DE result
            final_points = de_result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            return final_points

        except Exception:
            # If all else fails, return the initial points
            return initial_points

    # Generate initial strategies
    strategies = generate_initial_strategies()

    # Find the best initial configuration
    best_initial, initial_ratio = evaluate_all_strategies(strategies)

    # Initialize best results
    best_points = best_initial.copy()
    best_ratio = initial_ratio

    # Multi-start optimization with different initial variations
    for restart in range(5):
        # Generate new variation of the initial points
        np.random.seed(restart + 1000)
        perturbed_points = best_initial.copy()
        noise_level = 0.03 + restart * 0.005  # Gradually increasing noise
        perturbed_points += np.random.normal(0, noise_level, best_initial.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)

        # Optimize this variant
        optimized_points = robust_optimization(perturbed_points, max_evaluations=500)
        optimized_ratio = compute_min_max_ratio(optimized_points)

        if optimized_ratio > best_ratio:
            best_ratio = optimized_ratio
            best_points = optimized_points.copy()

    # Final optimization on the best configuration found
    final_points = robust_optimization(best_points, max_evaluations=300)
    final_ratio = compute_min_max_ratio(final_points)

    # One final refinement attempt
    np.random.seed(9999)
    last_attempt = final_points + np.random.normal(0, 0.01, final_points.shape)
    last_attempt = np.clip(last_attempt, 0, 1)
    refined_final = robust_optimization(last_attempt, max_evaluations=200)
    refined_ratio = compute_min_max_ratio(refined_final)

    if refined_ratio > final_ratio:
        return refined_final
    else:
        return final_points

# EVOLVE-BLOCK-END