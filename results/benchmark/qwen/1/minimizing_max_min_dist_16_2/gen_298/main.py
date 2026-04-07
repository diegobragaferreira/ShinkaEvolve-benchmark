# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
from typing import Tuple, List
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0 or np.max(distances) <= 0:
            return 0.0
        return np.min(distances) / np.max(distances)

    def objective_function(points_flat):
        """Objective function to minimize (negative ratio)"""
        points = points_flat.reshape(-1, 2)
        return -calculate_min_max_ratio(points)

    def generate_hexagonal_grid():
        """Generate initial points using hexagonal grid pattern"""
        # Create a hexagonal lattice pattern
        n_points = 16
        rows = 4
        cols = 4

        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = j + 0.5 * (i % 2)
                y = i * np.sqrt(3)/2
                points.append([x, y])

        # Normalize to unit square [0,1] x [0,1]
        points = np.array(points[:n_points])

        # Scale and shift to fit in [0,1] x [0,1]
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        if x_max > x_min and y_max > y_min:
            points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
            points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

        # Further adjust to make it more uniform
        points[:, 0] *= 0.9
        points[:, 1] *= 0.9
        points[:, 0] += 0.05
        points[:, 1] += 0.05

        return points

    def generate_spiral_pattern():
        """Generate points in a spiral pattern"""
        points = []
        angle_step = 2 * np.pi / 10
        radius_step = 1.0 / 10

        for i in range(16):
            if i == 0:
                points.append([0.5, 0.5])
            else:
                angle = i * angle_step
                radius = min(0.45, i * radius_step)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])

        return np.array(points)

    def generate_random_points():
        """Generate random points"""
        return np.random.rand(16, 2)

    def generate_grid_points():
        """Generate grid pattern points"""
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) >= 16:
                    break
                points.append([i * 0.25 + 0.125, j * 0.25 + 0.125])
        return np.array(points)

    def initialize_population() -> List[np.ndarray]:
        """Generate multiple diverse initial configurations"""
        configs = []

        # Different initialization strategies - focused on proven effective patterns
        configs.append(generate_hexagonal_grid())
        configs.append(generate_spiral_pattern())
        configs.append(generate_random_points())
        configs.append(generate_grid_points())

        # Add some noise to diversify but keep it minimal
        for i in range(len(configs)):
            # Use decreasing noise levels for diversity
            noise_level = 0.02 / (i + 1)
            noise = np.random.normal(0, noise_level, configs[i].shape)
            configs[i] = np.clip(configs[i] + noise, 0, 1)

        return configs

    def adaptive_local_optimization(initial_points, max_iter=500, method='L-BFGS-B'):
        """Apply adaptive local optimization with multiple methods and early stopping"""
        best_points = initial_points.copy()
        best_ratio = calculate_min_max_ratio(best_points)

        # Try different optimization methods with varying parameters
        if method == 'L-BFGS-B':
            try:
                result = minimize(
                    objective_function,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(len(best_points.flatten()))],
                    options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
                    tol=1e-8
                )

                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = calculate_min_max_ratio(optimized_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()

            except Exception:
                pass

        elif method == 'Nelder-Mead':
            try:
                result = minimize(
                    objective_function,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'adaptive': True, 'fatol': 1e-8, 'xatol': 1e-8}
                )

                if result.success:
                    optimized_points = result.x.reshape(-1, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = calculate_min_max_ratio(optimized_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()

            except Exception:
                pass

        return best_points, best_ratio

    def multi_stage_optimization(initial_points):
        """Apply multi-stage optimization for better results with adaptive parameters"""
        current_points = initial_points.copy()
        current_ratio = calculate_min_max_ratio(current_points)

        # Stage 1: Coarse optimization with fewer iterations but higher tolerance
        coarse_points, coarse_ratio = adaptive_local_optimization(current_points, max_iter=150, method='L-BFGS-B')
        if coarse_ratio > current_ratio:
            current_points = coarse_points
            current_ratio = coarse_ratio

        # Stage 2: Medium optimization (more iterations with tighter tolerances)
        medium_points, medium_ratio = adaptive_local_optimization(current_points, max_iter=250, method='L-BFGS-B')
        if medium_ratio > current_ratio:
            current_points = medium_points
            current_ratio = medium_ratio

        # Stage 3: Fine optimization (most iterations with very tight tolerances)
        fine_points, fine_ratio = adaptive_local_optimization(current_points, max_iter=350, method='L-BFGS-B')
        if fine_ratio > current_ratio:
            current_points = fine_points
            current_ratio = fine_ratio

        # Try hybrid approach with Nelder-Mead for additional improvement
        hybrid_points, hybrid_ratio = adaptive_local_optimization(current_points, max_iter=100, method='Nelder-Mead')
        if hybrid_ratio > current_ratio:
            current_points = hybrid_points
            current_ratio = hybrid_ratio

        return current_points, current_ratio

    def global_search_optimization(initial_points):
        """Use global optimization to improve initial points"""
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            return -calculate_min_max_ratio(points)

        # Use differential evolution with aggressive parameters
        bounds = [(0, 1) for _ in range(len(initial_points.flatten()))]
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=150,  # Reduced iterations for faster processing
                popsize=20,   # Increased population size for better exploration
                mutation=(0.8, 1),  # More aggressive mutation
                recombination=0.8,  # Higher recombination rate
                seed=42,
                disp=False,
                tol=1e-8,  # Tighter tolerance
                strategy='best1bin'
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = calculate_min_max_ratio(optimized_points)
                return optimized_points, ratio
        except Exception:
            pass
            
        return initial_points, calculate_min_max_ratio(initial_points)

    # Generate diverse initial configurations
    initial_configs = initialize_population()

    best_ratio = -np.inf
    best_points = None

    # Try each initial configuration with optimization
    for i, initial_config in enumerate(initial_configs):
        try:
            # Apply global search first to expand search space
            global_points, global_ratio = global_search_optimization(initial_config)
            
            # Apply multi-stage optimization for fine-tuning
            optimized_points, optimized_ratio = multi_stage_optimization(global_points)

            if optimized_ratio > best_ratio:
                best_ratio = optimized_ratio
                best_points = optimized_points.copy()

        except Exception as e:
            continue

    # If nothing worked, return a default configuration
    if best_points is None:
        best_points = generate_hexagonal_grid()

    return best_points

# EVOLVE-BLOCK-END