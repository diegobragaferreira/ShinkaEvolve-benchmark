# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    np.random.seed(42)

    def compute_ratio(points):
        """Compute min/max distance ratio for given point configuration."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Handle case where all points might be coincident
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def initialize_better_distribution():
        """Create initial point distribution using better geometric principles."""
        # Use Fibonacci-like distribution for more uniform coverage
        points = np.zeros((16, 2))

        # Golden angle for spiral-like distribution
        golden_angle = np.pi * (3 - np.sqrt(5))

        # Distribute points along a spiral pattern
        for i in range(16):
            # Radial position (avoid center)
            radius = 0.4 * np.sqrt(i / 15.0) + 0.1

            # Angular position with golden angle
            angle = i * golden_angle

            # Convert to Cartesian coordinates
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)

            # Add slight random perturbation to break symmetry
            x += np.random.normal(0, 0.01) * 0.1
            y += np.random.normal(0, 0.01) * 0.1

            points[i] = [x, y]

        # Ensure all points remain within bounds
        points = np.clip(points, 0.01, 0.99)

        return points

    def initialize_hexagonal_distribution():
        """Create initial point distribution using hexagonal grid pattern."""
        points = np.zeros((16, 2))

        # Create hexagonal grid pattern with 4x4 structure
        rows, cols = 4, 4
        spacing_x = 0.25
        spacing_y = 0.25 * math.sqrt(3) / 2

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < 16:
                    # Offset every other row for hexagonal packing
                    x = j * spacing_x + (i % 2) * spacing_x * 0.5
                    y = i * spacing_y

                    # Add small random perturbation to avoid perfect grid
                    x += np.random.normal(0, 0.01) * 0.5
                    y += np.random.normal(0, 0.01) * 0.5

                    points[idx] = [x, y]
                    idx += 1

        # Normalize to [0.1, 0.9] range
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

        # Clamp to bounds
        points = np.clip(points, 0.01, 0.99)

        return points

    def hybrid_optimization(initial_points):
        """Use hybrid optimization combining global and local search."""
        # Try multiple initial configurations
        initial_configs = [
            initialize_better_distribution(),
            initialize_hexagonal_distribution(),
            np.random.uniform(0.1, 0.9, (16, 2))
        ]

        best_points = None
        best_ratio = -np.inf

        for initial_config in initial_configs:
            try:
                # Phase 1: Global optimization with differential evolution
                def objective(x):
                    points = x.reshape(-1, 2)
                    ratio = compute_ratio(points)
                    return -ratio  # Negative because we want to maximize

                bounds = [(0, 1) for _ in range(32)]

                # Run differential evolution
                de_result = differential_evolution(
                    objective,
                    bounds,
                    seed=42,
                    maxiter=200,
                    popsize=15,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    disp=False
                )

                if de_result.success:
                    # Phase 2: Local refinement with L-BFGS-B
                    refined_x = de_result.x
                    refined_bounds = [(0, 1) for _ in range(32)]

                    local_result = minimize(
                        objective,
                        refined_x,
                        method='L-BFGS-B',
                        bounds=refined_bounds,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                        tol=1e-12
                    )

                    if local_result.success:
                        final_points = local_result.x.reshape(-1, 2)
                        final_ratio = compute_ratio(final_points)

                        if final_ratio > best_ratio:
                            best_ratio = final_ratio
                            best_points = final_points.copy()
                else:
                    # Fallback to DE result if it didn't succeed
                    final_points = de_result.x.reshape(-1, 2)
                    final_ratio = compute_ratio(final_points)

                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = final_points.copy()

            except Exception:
                continue

        # If no successful optimization, return the best initial configuration
        if best_points is None:
            # Use the best performing initial configuration
            ratios = [compute_ratio(config) for config in initial_configs]
            best_idx = np.argmax(ratios)
            best_points = initial_configs[best_idx]

        return best_points

    # Main optimization process
    # Step 1: Initialize with better distribution
    initial_points = initialize_better_distribution()

    # Step 2: Hybrid optimization (DE + L-BFGS-B)
    optimized_points = hybrid_optimization(initial_points)

    # Step 3: Final polishing with local search
    def final_local_refinement(points, iterations=200):
        """Final local refinement to squeeze out remaining improvement."""
        current_points = points.copy()

        for _ in range(iterations):
            current_ratio = compute_ratio(current_points)

            # Gradient estimation via finite differences
            eps = 1e-5
            best_points = current_points.copy()
            best_ratio = current_ratio

            # Try moving each point in small increments
            for i in range(len(current_points)):
                for dim in range(2):
                    # Try positive and negative step
                    for step_sign in [-1, 1]:
                        test_points = current_points.copy()
                        test_points[i, dim] += step_sign * eps

                        # Clamp to bounds with epsilon padding
                        test_points[i, 0] = np.clip(test_points[i, 0], 0.001, 0.999)
                        test_points[i, 1] = np.clip(test_points[i, 1], 0.001, 0.999)

                        test_ratio = compute_ratio(test_points)
                        if test_ratio > best_ratio:
                            best_ratio = test_ratio
                            best_points = test_points.copy()

            # If we found an improvement, use it
            if best_ratio > current_ratio:
                current_points = best_points
            else:
                # Reduce step size and try again
                eps *= 0.5

                # Early stopping if changes become negligible
                if eps < 1e-10:
                    break

        return current_points

    # Apply final refinement
    final_points = final_local_refinement(optimized_points, 100)

    return final_points

# EVOLVE-BLOCK-END