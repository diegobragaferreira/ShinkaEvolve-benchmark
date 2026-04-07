# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
import warnings
from scipy.stats import qmc
from numba import jit
import math

@jit(nopython=True)
def fast_pdist_squared(points):
    """Fast computation of squared pairwise distances using numba"""
    n = points.shape[0]
    distances_squared = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy
            distances_squared[i, j] = dist_sq
            distances_squared[j, i] = dist_sq
    return distances_squared

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

        # Use faster distance calculation
        distances_squared = fast_pdist_squared(points)
        distances = np.sqrt(distances_squared[np.triu_indices_from(distances_squared, k=1)])
        
        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def objective_with_regularization(x):
        """Objective with regularization to avoid numerical issues"""
        points = x.reshape(-1, 2)
        distances_squared = fast_pdist_squared(points)
        distances = np.sqrt(distances_squared[np.triu_indices_from(distances_squared, k=1)])

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-12
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio

    def create_dispersed_corner_pattern():
        """Create a specialized corner-based pattern that maximizes initial spread"""
        # Place 4 corner points
        corners = [[0, 0], [1, 0], [0, 1], [1, 1]]
        # Add 4 edge midpoints
        edges = [[0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]]
        # Add 4 interior points in a cross pattern
        cross = [[0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5]]
        # Add 4 more points in a diamond pattern
        diamond = [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]]
        
        points = corners + edges + cross + diamond
        return np.array(points)

    def create_hexagonal_grid():
        """Create a hexagonal grid pattern with some perturbation"""
        # Create a 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0
                y = i / 3.0
                # Offset every other row
                if i % 2 == 1:
                    x += 1.0 / (3 * 2)
                points.append([x, y])
        
        # Convert to numpy array and add slight perturbations
        points = np.array(points[:16], dtype=np.float64)
        # Add small random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.01, (16, 2))
        # Clip to valid range
        points = np.clip(points, 0, 1)
        return points

    def create_spherical_pattern():
        """Create a pattern approximating a spherical distribution"""
        # Golden spiral approach for 2D
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(16):
            angle = 2 * np.pi * i / phi
            # Use sine/cosine to distribute points more evenly
            radius = np.sqrt(i / 15.0) if i > 0 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
            
        # Normalize and scale
        points = np.array(points)
        if np.max(points) > np.min(points):
            points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0) + 1e-12)
        points = points * 0.8 + 0.1  # Scale to [0.1, 0.9]
        return points

    def create_distributed_initial():
        """Create a distributed initial configuration that promotes good spread"""
        # Start with corner pattern
        base = create_dispersed_corner_pattern()
        
        # Add perturbations from the center to encourage spreading
        center = np.mean(base, axis=0)
        for i in range(16):
            # Move points away from center slightly
            direction = base[i] - center
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                base[i] += direction * 0.02
                
        # Final clip to bounds
        base = np.clip(base, 0, 1)
        return base

    def generate_initial_configurations():
        """Generate multiple diverse initial configurations"""
        configs = []
        
        # 1. Dispersed corner pattern
        configs.append(create_dispersed_corner_pattern())
        
        # 2. Hexagonal grid
        configs.append(create_hexagonal_grid())
        
        # 3. Spherical pattern
        configs.append(create_spherical_pattern())
        
        # 4. Distributed initial
        configs.append(create_distributed_initial())
        
        # 5. Random uniform points
        np.random.seed(42)
        configs.append(np.random.rand(16, 2))
        
        # 6. Sobol sequence points
        sampler = qmc.Sobol(d=2, seed=42)
        sobol_points = sampler.random(16)
        configs.append(sobol_points)
        
        # 7. Stratified grid with larger perturbations
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]
        np.random.seed(42)
        for i in range(16):
            # Larger perturbations for better diversity
            grid_points[i] += np.random.normal(0, 0.03, 2)
        grid_points = np.clip(grid_points, 0, 1)
        configs.append(grid_points)
        
        # 8. Spiral with different parameters
        # Different golden ratio spiral
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            angle = 2 * np.pi * i / phi * 1.5  # Different multiplier
            radius = np.sqrt(i / 15.0) if i > 0 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
        points = np.array(points)
        if np.max(points) > np.min(points):
            points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0) + 1e-12)
        points = points * 0.8 + 0.1
        configs.append(points)

        return configs

    def optimize_with_hierarchical_strategy(initial_configs, max_time=170):
        """Run optimization with hierarchical strategy - coarse to fine approach"""
        best_ratio = -np.inf
        best_points = None
        start_time = time.time()
        
        # Sort initial configs by quality (descending)
        initial_ratios = [compute_min_max_ratio(cfg) for cfg in initial_configs]
        sorted_indices = np.argsort(initial_ratios)[::-1]
        sorted_configs = [initial_configs[i] for i in sorted_indices]
        
        # Process in order of quality
        for i, init_points in enumerate(sorted_configs):
            try:
                # Early termination check
                remaining_time = max_time - (time.time() - start_time)
                if remaining_time <= 15:
                    break
                    
                # Evaluate initial configuration quality
                initial_ratio = compute_min_max_ratio(init_points)
                
                # Flatten for optimization
                x0 = init_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]
                
                # Strategy selection based on quality and time
                if initial_ratio > 0.25:  # Very high quality starting point
                    # Use high precision L-BFGS-B optimization
                    result = minimize(
                        objective_with_regularization,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
                    )
                elif initial_ratio > 0.20:  # High quality starting point
                    # Use L-BFGS-B with moderate precision
                    result = minimize(
                        objective_with_regularization,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-13}
                    )
                elif initial_ratio > 0.15:  # Medium quality starting point  
                    # Use L-BFGS-B with less strict tolerances
                    result = minimize(
                        objective_with_regularization,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1200, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                else:  # Low quality starting point
                    # First, try global optimization to escape local minima
                    de_timeout = min(remaining_time * 0.4, 60)
                    if de_timeout > 20:
                        try:
                            result_de = differential_evolution(
                                objective_with_regularization,
                                bounds,
                                seed=42,
                                maxiter=int(de_timeout/2),
                                popsize=min(25, 16 * 2),
                                mutation=(0.5, 1),
                                recombination=0.7,
                                tol=1e-9
                            )
                            
                            if result_de.success:
                                # Use differential evolution result as starting point for refinement
                                x0 = result_de.x
                        except:
                            pass
                    
                    # Then local refinement
                    lbfgs_timeout = remaining_time - de_timeout if de_timeout > 20 else remaining_time
                    if lbfgs_timeout > 15:
                        try:
                            result = minimize(
                                objective_with_regularization,
                                x0,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
                            )
                        except:
                            # Fallback to basic optimization
                            result = type('obj', (object,), {'x': x0, 'success': True})()
                    else:
                        # Skip local optimization if time is too limited
                        result = type('obj', (object,), {'x': x0, 'success': True})()
                        
                # Handle the result
                if hasattr(result, 'success') and result.success:
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
            return sorted_configs[0]

        return best_points

    # Generate initial configurations
    initial_configs = generate_initial_configurations()

    # Optimize with hierarchical strategy
    try:
        final_points = optimize_with_hierarchical_strategy(initial_configs, max_time=165)
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to the best initial configuration
        best_initial = max(initial_configs, key=lambda x: compute_min_max_ratio(x))
        final_points = best_initial

    # Ensure final points are within bounds
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END