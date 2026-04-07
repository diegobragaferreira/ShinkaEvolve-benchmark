# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
import warnings
from scipy.stats import qmc

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_min_max_ratio(points):
        """Compute the minimum to maximum distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances
        distances = pdist(points)

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

    def hexagonal_lattice_2d(n_points):
        """Generate points on a 2D hexagonal lattice"""
        # Calculate grid parameters
        rows = int(np.ceil(np.sqrt(n_points)))
        cols = int(np.ceil(n_points / rows))

        # Create hexagonal grid
        points = []
        spacing = 1.0 / max(rows, cols)

        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                # Offset every other row
                x_offset = (i % 2) * spacing / 2
                x = (j * spacing) + x_offset
                y = i * spacing
                points.append([x, y])

        return np.array(points[:n_points])

    def sobol_sequence_2d(n_points):
        """Generate points using Sobol sequence for better space-filling"""
        sampler = qmc.Sobol(d=2, seed=42)
        points = sampler.random(n_points)
        return points

    def generate_initial_configurations():
        """Generate multiple diverse initial configurations"""
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

        # 2. Hexagonal lattice pattern
        hex_points = hexagonal_lattice_2d(16)
        # Normalize to [0.1, 0.9] range
        if np.max(hex_points) > np.min(hex_points):
            hex_points = (hex_points - np.min(hex_points, axis=0)) / (
                np.max(hex_points, axis=0) - np.min(hex_points, axis=0) + 1e-12)
        hex_points = hex_points * 0.8 + 0.1
        configs.append(hex_points.copy())

        # 3. Perturbed grid (more sophisticated)
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]
        # Apply adaptive perturbation based on position
        for i in range(16):
            row, col = i // 4, i % 4
            # Larger perturbations for corners to encourage spreading
            if row in [0, 3] and col in [0, 3]:
                std = 0.03
            else:
                std = 0.015
            grid_points[i] += np.random.normal(0, std, 2)
        grid_points = np.clip(grid_points, 0, 1)
        configs.append(grid_points)

        # 4. Random uniform points
        random_points = np.random.rand(16, 2)
        configs.append(random_points)

        # 5. Sobol sequence points
        sobol_points = sobol_sequence_2d(16)
        configs.append(sobol_points)

        return configs

    def optimize_with_adaptive_restarts(initial_configs, max_time=170):
        """Run optimization with multiple restarts using adaptive strategies"""
        best_ratio = -np.inf
        best_points = None
        start_time = time.time()
        
        # Try multiple different initial configurations
        for i, init_points in enumerate(initial_configs):
            try:
                # Evaluate initial configuration quality
                initial_ratio = compute_min_max_ratio(init_points)
                
                # Flatten for optimization
                x0 = init_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]
                
                # Adaptive optimization based on initial quality
                remaining_time = max_time - (time.time() - start_time)
                
                if remaining_time <= 5:
                    continue
                    
                # Use different strategies based on initial quality
                if initial_ratio > 0.20:  # High quality starting point
                    # Focus on local refinement with high precision
                    result = minimize(
                        objective_with_regularization,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                elif initial_ratio > 0.15:  # Medium quality starting point  
                    # Use L-BFGS-B with moderate precision
                    result = minimize(
                        objective_with_regularization,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                else:  # Low quality starting point
                    # Use global search first, then local refinement
                    de_timeout = min(remaining_time * 0.6, 60)
                    if de_timeout > 10:
                        # Differential evolution for global search
                        result_de = differential_evolution(
                            objective_with_regularization,
                            bounds,
                            seed=42,
                            maxiter=int(de_timeout/2),
                            popsize=min(20, 16 * 2),
                            mutation=(0.5, 1),
                            recombination=0.7,
                            tol=1e-8
                        )
                        
                        if result_de.success:
                            # Use differential evolution result as starting point for refinement
                            x0 = result_de.x
                            
                    # Local optimization with L-BFGS-B
                    lbfgs_timeout = remaining_time - de_timeout if de_timeout > 10 else remaining_time
                    if lbfgs_timeout > 5:
                        result = minimize(
                            objective_with_regularization,
                            x0,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-10}
                        )
                    else:
                        # Skip local optimization if time is too limited
                        result = type('obj', (object,), {'x': x0, 'success': True})()
                        
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

        # If no successful optimization, return the best initial configuration
        if best_points is None:
            return initial_configs[np.argmax([compute_min_max_ratio(cfg) for cfg in initial_configs])]

        return best_points

    # Generate initial configurations
    initial_configs = generate_initial_configurations()

    # Optimize with adaptive restarts
    try:
        final_points = optimize_with_adaptive_restarts(initial_configs, max_time=170)
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to the best initial configuration
        best_initial = max(initial_configs, key=lambda x: compute_min_max_ratio(x))
        final_points = best_initial

    # Ensure final points are within bounds
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END