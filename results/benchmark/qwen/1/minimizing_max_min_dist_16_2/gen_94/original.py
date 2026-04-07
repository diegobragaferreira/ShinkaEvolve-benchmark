# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from typing import Tuple
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance with improved numerical stability"""
        distances = pdist(points)
        if len(distances) == 0 or np.max(distances) <= 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 0:
            return 0.0
        return d_min / d_max

    def objective_function(points_flat):
        """Objective function to minimize (negative ratio)"""
        points = points_flat.reshape(-1, 2)
        return -calculate_min_max_ratio(points)

    def generate_diverse_initial_configs():
        """Generate multiple diverse initial configurations"""
        configs = []

        # 1. Hexagonal grid pattern with better spacing
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

        # Normalize and scale to fit nicely in [0,1] x [0,1]
        hex_points = np.array(hex_points[:16])
        x_min, y_min = np.min(hex_points, axis=0)
        x_max, y_max = np.max(hex_points, axis=0)
        if x_max > x_min and y_max > y_min:
            scale_factor = min(0.9, 0.9 / max(x_max - x_min, y_max - y_min))
            hex_points[:, 0] = (hex_points[:, 0] - x_min) * scale_factor + 0.05
            hex_points[:, 1] = (hex_points[:, 1] - y_min) * scale_factor + 0.05
        configs.append(hex_points.copy())

        # 2. Spiral pattern with better distribution
        spiral_points = []
        for i in range(16):
            if i == 0:
                spiral_points.append([0.5, 0.5])
            else:
                angle = i * 2.5  # Increased angular spacing for better spread
                radius = min(0.45, i * 0.08)  # Smaller radial increment
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                spiral_points.append([x, y])
        configs.append(np.array(spiral_points[:16]))

        # 3. Random uniform distribution
        configs.append(np.random.rand(16, 2))

        # 4. Grid pattern with slight jitter
        grid_points = []
        for i in range(4):
            for j in range(4):
                if len(grid_points) >= 16:
                    break
                base_x = i * 0.33 + 0.165
                base_y = j * 0.33 + 0.165
                # Add jitter to avoid perfect grid artifacts
                jitter_x = np.random.normal(0, 0.02)
                jitter_y = np.random.normal(0, 0.02)
                grid_points.append([base_x + jitter_x, base_y + jitter_y])
        configs.append(np.array(grid_points[:16]).clip(0, 1))

        # 5. Better hexagonal pattern
        hex_points2 = []
        for i in range(4):
            for j in range(4):
                if len(hex_points2) >= 16:
                    break
                x = j * 0.25 + 0.125 + (i % 2) * 0.125
                y = i * 0.25 + 0.125
                hex_points2.append([x, y])
        configs.append(np.array(hex_points2[:16]).clip(0, 1))

        return configs

    def adaptive_global_optimization(initial_points, max_iter=1000, target_ratio=None):
        """Use differential evolution with adaptive parameters based on performance"""
        bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

        # Adjust population size based on expected performance
        popsize = 15
        if target_ratio and target_ratio > 0.2:
            popsize = 10  # Smaller population for faster convergence
        elif target_ratio and target_ratio < 0.1:
            popsize = 20  # Larger population for better exploration

        try:
            start_time = time.time()
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=max_iter,
                popsize=popsize,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                callback=lambda x, convergence: None
            )
            elapsed_time = time.time() - start_time

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio, elapsed_time
        except Exception as e:
            print(f"Differential evolution error: {e}")

        # Fallback to local optimization if global fails
        return local_refinement(initial_points, max_iter=max_iter//2)

    def adaptive_local_refinement(initial_points, max_iter=1000, target_ratio=None):
        """Apply adaptive local refinement to improve configuration"""
        best_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(best_points)

        # Use adaptive iteration counts based on performance
        if target_ratio:
            if target_ratio > 0.25:
                refine_iters = max_iter // 4
            elif target_ratio > 0.15:
                refine_iters = max_iter // 2
            else:
                refine_iters = max_iter
        else:
            refine_iters = max_iter

        # Multiple local optimization attempts with varying strategies
        for attempt in range(3):
            # Strategy 1: Direct optimization
            try:
                result = minimize(
                    objective_function,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(len(best_points.flatten()))],
                    options={'maxiter': refine_iters // 3},
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
                pass

            # Strategy 2: Add noise and retry with different local optimization
            try:
                noise = np.random.normal(0, 0.02, best_points.shape) if attempt > 0 else 0
                perturbed = best_points + noise
                perturbed = np.clip(perturbed, 0, 1)

                result = minimize(
                    objective_function,
                    perturbed.flatten(),
                    method='TNC',
                    bounds=[(0, 1) for _ in range(len(perturbed.flatten()))],
                    options={'maxiter': refine_iters // 3},
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
                pass

            # Strategy 3: Slightly perturb and continue refining
            if attempt < 2:
                noise = np.random.normal(0, 0.005, best_points.shape)
                best_points += noise
                best_points = np.clip(best_points, 0, 1)

        return best_points, best_ratio

    # Generate diverse initial configurations
    initial_configs = generate_diverse_initial_configs()

    best_ratio = -np.inf
    best_points = None

    # Try each initial configuration with adaptive optimization
    for i, initial_config in enumerate(initial_configs):
        try:
            # Apply progressive optimization strategy
            # Phase 1: Global optimization with reduced iterations for quick exploration
            global_points, global_ratio, _ = adaptive_global_optimization(
                initial_config, max_iter=300, target_ratio=None
            )

            # Phase 2: Adaptive local refinement
            refined_points, refined_ratio = adaptive_local_refinement(
                global_points, max_iter=500, target_ratio=global_ratio
            )

            # Phase 3: Final local refinement if significantly better
            final_points, final_ratio = adaptive_local_refinement(
                refined_points, max_iter=300, target_ratio=refined_ratio
            )

            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()

        except Exception as e:
            print(f"Error with initial config {i}: {e}")
            continue

    # If nothing worked, return the best from local refinement alone with enhanced parameters
    if best_points is None:
        initial_config = np.random.rand(16, 2)
        best_points, _ = adaptive_local_refinement(initial_config, max_iter=1000)

    return best_points

# EVOLVE-BLOCK-END