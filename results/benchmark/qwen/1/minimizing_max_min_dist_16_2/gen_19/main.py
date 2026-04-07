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

        # Compute pairwise distances with enhanced numerical stability
        distances = pdist(points)

        # Filter out any potentially problematic distances
        distances = distances[distances > 1e-12]

        if len(distances) == 0:
            return 0

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
        # Strategy 1: Improved golden spiral pattern (better distribution)
        indices = np.arange(16)
        golden_angle = 2.399963229728653  # ~2π/(φ^2) where φ is golden ratio
        angles = golden_angle * indices
        # Use logarithmic spiral for better spread
        radii = np.log(indices + 1) / np.log(16)  # Better distribution than sqrt
        golden_spiral = np.column_stack([
            0.5 + 0.45 * radii * np.cos(angles),
            0.5 + 0.45 * radii * np.sin(angles)
        ])

        # Strategy 2: Hexagonal grid with perturbation (better coverage)
        hex_points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = np.sqrt(3) / 2 / (rows - 1)  # Hexagon height
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                # Add larger perturbation to break symmetry and encourage better spread
                x += np.random.normal(0, 0.02)
                y += np.random.normal(0, 0.02)
                hex_points.append([x, y])
        hex_points = np.array(hex_points)

        # Strategy 3: Square grid with improved jitter
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0 + np.random.normal(0, 0.015)
                y = (j + 0.5) / 4.0 + np.random.normal(0, 0.015)
                grid_points.append([x, y])
        grid_points = np.array(grid_points)

        # Strategy 4: Random points with clustering avoidance and boundary awareness
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        # Push points away from boundaries slightly to avoid edge effects
        random_points = np.clip(random_points, 0.05, 0.95)

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
        """Perform adaptive local refinement with monitoring and convergence detection."""
        # Flatten for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]

        try:
            # Adaptive DE params based on problem size
            de_params = {
                'maxiter': max_iterations // 10,
                'popsize': 20,
                'mutation': (0.5, 1),
                'recombination': 0.7,
                'seed': 42,
                'disp': False,
                'tol': 1e-8
            }

            # First, use differential evolution for global search
            de_result = differential_evolution(
                objective_function,
                bounds,
                **de_params
            )

            # Check if DE was successful
            if not hasattr(de_result, 'success') or not de_result.success:
                # If DE failed, try a more conservative approach
                de_result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=max_iterations // 20,
                    popsize=10,
                    mutation=(0.3, 0.7),
                    recombination=0.5,
                    seed=42,
                    disp=False,
                    tol=1e-6
                )

            # Then refine with local optimization using L-BFGS-B
            refined_result = minimize(
                objective_function,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )

            # Final validation and clipping
            final_points = refined_result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)

            # Additional check to see if we actually improved
            if not refined_result.success:
                # If local refinement failed, return the better of DE result or original
                de_points = de_result.x.reshape(-1, 2)
                de_points = np.clip(de_points, 0, 1)
                de_ratio = compute_min_max_ratio(de_points)
                orig_ratio = compute_min_max_ratio(initial_points)

                if de_ratio > orig_ratio:
                    return de_points
                else:
                    return initial_points

            return final_points

        except Exception as e:
            # Fallback to original points if optimization fails
            return initial_points

    def run_optimization_with_restart():
        """Run the main optimization with restart capability."""
        best_points = None
        best_ratio = 0

        # Multiple restarts with different strategies
        for restart in range(5):
            # Generate different initial configurations based on restart number
            np.random.seed(restart + 42)

            # Select different initialization strategy per restart
            if restart == 0:
                # Golden spiral
                indices = np.arange(16)
                golden_angle = 2.399963229728653
                angles = golden_angle * indices
                radii = np.log(indices + 1) / np.log(16)
                initial_points = np.column_stack([
                    0.5 + 0.45 * radii * np.cos(angles),
                    0.5 + 0.45 * radii * np.sin(angles)
                ])
            elif restart == 1:
                # Hexagonal grid
                hex_points = []
                rows = 4
                cols = 4
                spacing_x = 1.0 / (cols - 1)
                spacing_y = np.sqrt(3) / 2 / (rows - 1)
                for i in range(rows):
                    for j in range(cols):
                        x = j * spacing_x + (i % 2) * spacing_x / 2
                        y = i * spacing_y
                        x += np.random.normal(0, 0.02)
                        y += np.random.normal(0, 0.02)
                        hex_points.append([x, y])
                initial_points = np.array(hex_points)
            else:
                # Random + perturbation
                initial_points = np.random.rand(16, 2)
                # Make it more evenly distributed
                initial_points = np.clip(initial_points, 0.05, 0.95)

            # Apply bounds
            initial_points = np.clip(initial_points, 0, 1)

            # Run refinement
            refined_points = adaptive_local_refinement(initial_points, max_iterations=1000)
            refined_ratio = compute_min_max_ratio(refined_points)

            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()

        # If we didn't find anything, return the best among initial attempts
        if best_points is None:
            initial_points = generate_initial_points()
            return initial_points

        return best_points

    # Main execution
    return run_optimization_with_restart()

# EVOLVE-BLOCK-END