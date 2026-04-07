# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import itertools
from typing import Tuple

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance"""
        distances = pdist(points)
        if len(distances) == 0 or np.max(distances) <= 0:
            return 0.0
        return np.min(distances) / np.max(distances)

    def objective_function(points_flat):
        """Objective function to minimize (negative ratio)"""
        points = points_flat.reshape(-1, 2)
        return -calculate_min_max_ratio(points)

    def generate_diverse_initial_configs():
        """Generate multiple diverse initial configurations"""
        configs = []
        import math

        # 1. Hexagonal grid pattern
        np.random.seed(42)
        hex_points = []
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(hex_points) >= 16:
                    break
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3)/2
                hex_points.append([x, y])

        # Normalize and scale
        hex_points = np.array(hex_points[:16])
        x_min, y_min = np.min(hex_points, axis=0)
        x_max, y_max = np.max(hex_points, axis=0)
        if x_max > x_min and y_max > y_min:
            hex_points[:, 0] = (hex_points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            hex_points[:, 1] = (hex_points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        configs.append(hex_points.copy())

        # 2. Spiral pattern
        spiral_points = []
        angle_step = 2 * np.pi / 10
        radius_step = 1.0 / 10
        for i in range(16):
            if i == 0:
                spiral_points.append([0.5, 0.5])
            else:
                angle = i * angle_step
                radius = min(0.45, i * radius_step)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                spiral_points.append([x, y])
        configs.append(np.array(spiral_points[:16]))

        # 3. Random uniform distribution
        configs.append(np.random.rand(16, 2))

        # 4. Grid pattern
        grid_points = []
        for i in range(4):
            for j in range(4):
                if len(grid_points) >= 16:
                    break
                grid_points.append([i * 0.25 + 0.125, j * 0.25 + 0.125])
        configs.append(np.array(grid_points[:16]))

        # 5. Spherical projection pattern (fibonacci sphere)
        sph_points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        for i in range(16):
            y = 1 - (i / float(16 - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            sph_points.append([x, y])

        # Normalize to unit square
        sph_points = np.array(sph_points)
        x_min, y_min = np.min(sph_points, axis=0)
        x_max, y_max = np.max(sph_points, axis=0)
        if x_max > x_min and y_max > y_min:
            sph_points[:, 0] = (sph_points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
            sph_points[:, 1] = (sph_points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        configs.append(sph_points.copy())

        return configs

    def global_optimization(initial_points, max_iter=1000):
        """Use differential evolution for global optimization"""
        bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=max_iter,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception as e:
            print(f"Differential evolution error: {e}")

        # Fallback to local optimization if global fails
        return local_refinement(initial_points)

    def local_refinement(initial_points, max_iter=1000):
        """Apply local refinement to improve configuration"""
        best_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(best_points)

        # Multiple local optimization attempts
        for attempt in range(5):
            # Add some noise to diversify the starting points
            noise = np.random.normal(0, 0.01, best_points.shape)
            perturbed = best_points + noise
            perturbed = np.clip(perturbed, 0, 1)

            try:
                result = minimize(
                    objective_function,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(len(perturbed.flatten()))],
                    options={'maxiter': max_iter // 5},
                    tol=1e-6
                )

                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = calculate_min_max_ratio(optimized_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            except Exception as e:
                print(f"Local optimization error: {e}")
                continue

        return best_points, best_ratio

    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_configs()

    best_ratio = -np.inf
    best_points = None

    # Try each initial configuration with global optimization
    for i, initial_config in enumerate(initial_configs):
        try:
            # Try global optimization first
            global_points, global_ratio = global_optimization(initial_config, max_iter=500)

            # If global optimization didn't work well, try local refinement
            if global_ratio < 0.1:  # Threshold for poor results
                local_points, local_ratio = local_refinement(initial_config, max_iter=1000)
                final_points, final_ratio = (local_points, local_ratio)
            else:
                final_points, final_ratio = (global_points, global_ratio)

            # Final local refinement on the best found solution
            final_points, final_ratio = local_refinement(final_points, max_iter=500)

            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()

        except Exception as e:
            print(f"Error with initial config {i}: {e}")
            continue

    # If nothing worked, return the best from local refinement alone
    if best_points is None:
        initial_config = np.random.rand(16, 2)
        best_points, _ = local_refinement(initial_config, max_iter=1000)

    return best_points

# EVOLVE-BLOCK-END