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

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0

        # Compute pairwise distances
        distances = pdist(points)

        # Get min and max distances
        dmin = np.min(distances)
        dmax = np.max(distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax

    def compute_distance_matrix(points):
        """Compute full pairwise distance matrix for additional analysis."""
        return squareform(pdist(points))

    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)

        # Ensure points are within bounds [0,1]
        points = np.clip(points, 0, 1)

        # Compute ratio
        ratio = compute_min_max_ratio(points)

        # Return negative because we want to maximize
        return -ratio

    def generate_initial_points():
        """Generate good initial point configurations using multiple strategies."""
        # Strategy 1: Golden spiral pattern (good for uniform distribution)
        indices = np.arange(16)
        golden_angle = 2.399963229728653  # ~2π/(φ^2) where φ is golden ratio
        angles = golden_angle * indices
        radii = np.sqrt(indices / 15)  # Normalize to [0,1]
        golden_spiral = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])

        # Strategy 2: Hexagonal grid with perturbation
        hex_points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Hexagon height
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                # Add small perturbation to break symmetry
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                hex_points.append([x, y])
        hex_points = np.array(hex_points)

        # Strategy 3: Square grid with jitter
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0 + np.random.normal(0, 0.01)
                y = (j + 0.5) / 4.0 + np.random.normal(0, 0.01)
                grid_points.append([x, y])
        grid_points = np.array(grid_points)

        # Strategy 4: Random points with some clustering avoidance
        np.random.seed(42)
        random_points = np.random.rand(16, 2)

        # Return the best of these initializations based on their ratio
        candidates = [golden_spiral, hex_points, grid_points, random_points]
        best_candidate = None
        best_ratio = 0

        for candidate in candidates:
            # Clip to bounds
            candidate = np.clip(candidate, 0, 1)
            ratio = compute_min_max_ratio(candidate)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate.copy()

        return best_candidate

    def adaptive_local_refinement(initial_points, max_iterations=1000):
        """Perform adaptive local refinement with monitoring."""
        # Flatten for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]

        # First, use differential evolution for global search
        try:
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=max_iterations // 10,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-6
            )

            # Then refine with local optimization
            refined_result = minimize(
                objective_function,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9},
                tol=1e-9
            )

            # Final validation and clipping
            final_points = refined_result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)

            return final_points

        except Exception as e:
            # Fallback to original points if optimization fails
            return initial_points

    # Generate good initial points
    initial_points = generate_initial_points()

    # Perform adaptive refinement with early stopping
    final_points = adaptive_local_refinement(initial_points, max_iterations=500)

    # Ensure we return the best of our attempts
    final_ratio = compute_min_max_ratio(final_points)

    # Try a few more random restarts for better results
    for restart in range(3):
        # Generate new random starting point with better spread
        np.random.seed(restart + 100)
        perturbed_points = initial_points.copy()
        # Apply larger perturbations to explore more area
        perturbed_points += np.random.normal(0, 0.05, initial_points.shape)
        perturbed_points = np.clip(perturbed_points, 0, 1)

        # Refine this alternative
        alt_points = adaptive_local_refinement(perturbed_points, max_iterations=300)
        alt_ratio = compute_min_max_ratio(alt_points)

        if alt_ratio > final_ratio:
            final_ratio = alt_ratio
            final_points = alt_points

    return final_points

# EVOLVE-BLOCK-END