# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def golden_spiral_2d(n_points):
        """Generate points on a 2D golden spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        return np.array(points)

    def compute_min_max_ratio(points):
        """Compute the minimum to maximum distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

        # Set diagonal to infinity to exclude self-distances
        if len(distances) == 0:
            return 0.0

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def objective(x):
        """Objective function to maximize the min/max distance ratio."""
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Compute pairwise distances
        distances = pdist(points)

        # Avoid division by zero
        if len(distances) == 0:
            return -np.inf

        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio since we want to maximize
        if max_dist == 0:
            return -np.inf
        return -min_dist / max_dist

    def objective_with_regularization(x):
        """Objective with regularization to avoid numerical issues"""
        points = x.reshape(-1, 2)
        distances = pdist(points)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-12
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio

    def generate_initial_configurations():
        """Generate multiple initial configurations"""
        configs = []
        np.random.seed(42)

        # 1. Golden spiral pattern
        spiral_points = golden_spiral_2d(16)
        # Scale and center the spiral
        if np.max(spiral_points) > np.min(spiral_points):
            spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
                np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
        spiral_points = spiral_points * 0.8 + 0.1  # Scale to [0.1, 0.9]
        configs.append(spiral_points.copy())

        # 2. Perturbed grid
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]
        grid_points += np.random.normal(0, 0.05, (16, 2))
        grid_points = np.clip(grid_points, 0, 1)
        configs.append(grid_points)

        # 3. Random uniform points
        random_points = np.random.rand(16, 2)
        configs.append(random_points)

        return configs

    def optimize_with_restarts(initial_configs, max_time=170):
        """Run optimization with multiple restarts using different strategies"""
        best_ratio = -np.inf
        best_points = None
        
        start_time = time.time()
        
        # Try multiple different initial configurations
        for i, init_points in enumerate(initial_configs):
            try:
                # Flatten for optimization
                x0 = init_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]

                # First stage: Differential evolution for global search
                remaining_time = max_time - (time.time() - start_time)
                if remaining_time > 10:  # Only use DE if we have enough time
                    result_de = differential_evolution(
                        objective_with_regularization,
                        bounds,
                        seed=42,
                        maxiter=min(50, int(remaining_time/2)),
                        popsize=15,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        tol=1e-6,
                        timeout=remaining_time
                    )
                    
                    if result_de.success:
                        # Use differential evolution result as starting point for refinement
                        x0 = result_de.x
                        
                # Second stage: Local optimization with L-BFGS-B
                remaining_time = max_time - (time.time() - start_time)
                if remaining_time > 5:
                    result = minimize(
                        objective_with_regularization,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-9},
                        timeout=remaining_time
                    )
                    
                    if result.success:
                        # Extract final points
                        final_points = result.x.reshape(-1, 2)
                        
                        # Compute actual ratio
                        ratio = compute_min_max_ratio(final_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()

            except Exception as e:
                warnings.warn(f"Optimization failed for initial config {i}: {e}")
                continue

        return best_points if best_points is not None else initial_configs[0]

    # Generate initial configurations
    initial_configs = generate_initial_configurations()

    # Optimize with multiple restarts
    try:
        final_points = optimize_with_restarts(initial_configs, max_time=170)
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to the best initial configuration
        best_initial = max(initial_configs, key=lambda x: compute_min_max_ratio(x))
        final_points = best_initial

    # Ensure final points are within bounds
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END