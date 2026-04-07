# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Implements a grid-adaptive optimization approach that balances exploration and exploitation.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_ratio(points):
        """Compute the minimum to maximum distance ratio for given points."""
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return 0.0, min_dist, max_dist

        ratio = min_dist / max_dist
        return ratio, min_dist, max_dist

    def objective(x):
        """Objective function to minimize (negative ratio)."""
        points = x.reshape(-1, 2)
        ratio, _, _ = compute_ratio(points)
        return -ratio

    def initialize_grid_points():
        """Initialize points using a structured 4x4 grid with adaptive perturbations."""
        np.random.seed(42)
        points = []
        rows, cols = 4, 4

        # Generate points on a grid with position-based perturbation strategy
        for i in range(rows):
            for j in range(cols):
                # Base grid positions
                x = j * (1.0 / (cols - 1)) if cols > 1 else 0.5
                y = i * (1.0 / (rows - 1)) if rows > 1 else 0.5

                # Position-based perturbation strength
                if (i == 0 or i == rows-1) and (j == 0 or j == cols-1):
                    # Corner points - smallest perturbation
                    perturbation = 0.015
                elif i == 0 or i == rows-1 or j == 0 or j == cols-1:
                    # Edge points - medium perturbation
                    perturbation = 0.025
                else:
                    # Interior points - larger perturbation
                    perturbation = 0.035

                # Apply perturbation
                x += np.random.normal(0, perturbation)
                y += np.random.normal(0, perturbation)
                points.append([x, y])

        # Ensure all points are within bounds [0,1]
        points = np.clip(points, 0, 1)
        return np.array(points[:16])

    def adaptive_optimization(initial_points, max_time=170):
        """Adaptive optimization that adjusts strategy based on convergence behavior."""
        start_time = time.time()
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio, _, _ = compute_ratio(best_points)

        # Track convergence history
        convergence_history = []
        stagnation_count = 0
        max_stagnation = 20

        # Parameters for adaptive strategy
        exploration_phase = True
        current_step_size = 0.02
        fine_tune_threshold = 0.25

        # Main optimization loop
        for iteration in range(2000):  # Limited iterations to respect time budget
            if time.time() - start_time > max_time - 1:
                break

            # Store previous state for convergence check
            prev_points = current_points.copy()
            prev_ratio = best_ratio

            # Choose optimization strategy based on current phase
            if exploration_phase:
                # Use a more aggressive approach for exploration
                try:
                    # Coarse optimization step
                    bounds = [(0.001, 0.999) for _ in range(32)]
                    result = minimize(
                        objective,
                        current_points.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 20, 'ftol': 1e-6, 'gtol': 1e-4}
                    )

                    if result.success:
                        current_points = result.x.reshape(-1, 2)

                        # Check if we made progress
                        current_ratio, _, _ = compute_ratio(current_points)
                        if current_ratio > best_ratio:
                            best_ratio = current_ratio
                            best_points = current_points.copy()
                            stagnation_count = 0  # Reset stagnation counter
                        else:
                            stagnation_count += 1

                except:
                    # Fallback behavior
                    pass

                # Switch to fine-tuning phase if appropriate
                if best_ratio > fine_tune_threshold:
                    exploration_phase = False
                    current_step_size = 0.005

            else:
                # Fine-tuning phase - more careful optimization
                try:
                    bounds = [(0.001, 0.999) for _ in range(32)]
                    result = minimize(
                        objective,
                        current_points.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}
                    )

                    if result.success:
                        current_points = result.x.reshape(-1, 2)

                        # Check if we made progress
                        current_ratio, _, _ = compute_ratio(current_points)
                        if current_ratio > best_ratio:
                            best_ratio = current_ratio
                            best_points = current_points.copy()
                            stagnation_count = 0
                        else:
                            stagnation_count += 1

                except:
                    # Fallback behavior
                    pass

            # Adapt strategy if stagnating
            if stagnation_count > max_stagnation:
                # Inject more randomness to escape local optima
                current_points = best_points.copy()
                noise_magnitude = 0.01 + (stagnation_count - max_stagnation) * 0.001
                current_points += np.random.normal(0, noise_magnitude, current_points.shape)
                current_points = np.clip(current_points, 0.001, 0.999)
                stagnation_count = 0  # Reset
                exploration_phase = True  # Reset to exploration mode

        return best_points

    # Initialize with structured grid
    initial_points = initialize_grid_points()

    # Perform adaptive optimization
    optimized_points = adaptive_optimization(initial_points)

    return optimized_points

# EVOLVE-BLOCK-END