# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
from sklearn.metrics.pairwise import euclidean_distances
from typing import Tuple, Optional, List
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points_flat):
        """Compute the min/max distance ratio for given flattened point coordinates."""
        points = points_flat.reshape(-1, 2)
        
        # Compute pairwise distances efficiently
        distances = euclidean_distances(points)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Compute min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return ratio (avoid division by zero)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def energy_objective(points_flat):
        """Energy-based objective function that encourages uniform distribution."""
        points = points_flat.reshape(-1, 2)
        distances = euclidean_distances(points)
        np.fill_diagonal(distances, np.inf)
        
        # Energy is sum of inverse squared distances (repulsive force model)
        # This promotes points being spread out
        energy = 0.0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = distances[i, j]
                if dist > 0:
                    energy += 1.0 / (dist * dist)
        return energy
    
    def objective(points_flat):
        """Minimize negative of min/max ratio (equivalent to maximizing the ratio)."""
        return -compute_min_max_ratio(points_flat)
    
    def constraint_func(points_flat):
        """Ensure points stay within [0,1] x [0,1]."""
        points = points_flat.reshape(-1, 2)
        # Each point coordinate should be between 0 and 1
        return np.concatenate([
            points[:, 0],  # x coordinates
            points[:, 1],  # y coordinates
            1 - points[:, 0],  # 1 - x coordinates
            1 - points[:, 1]   # 1 - y coordinates
        ])
    
    def geometric_initialization():
        """Initialize points using multiple geometric strategies for better coverage."""
        # Strategy 1: Square grid with perturbation
        grid_x = np.linspace(0.1, 0.9, 4)
        grid_y = np.linspace(0.1, 0.9, 4)
        x_grid, y_grid = np.meshgrid(grid_x, grid_y)
        points = np.column_stack([x_grid.ravel(), y_grid.ravel()])[:16]
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        
        # Strategy 2: Hexagonal pattern
        hex_points = []
        for i in range(4):
            for j in range(4):
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125
                y = i * 0.25 + 0.125
                hex_points.append([x, y])
        
        hex_points = np.array(hex_points[:16])
        hex_points += np.random.normal(0, 0.015, hex_points.shape)
        hex_points = np.clip(hex_points, 0, 1)
        
        # Strategy 3: Random with better spread
        rand_points = np.random.rand(16, 2)
        # Apply some basic spacing to prevent clustering
        for i in range(16):
            center_vec = rand_points[i] - [0.5, 0.5]
            center_distance = np.linalg.norm(center_vec)
            if center_distance > 0:
                rand_points[i] += center_vec * 0.05 / (center_distance + 0.01)
        rand_points = np.clip(rand_points, 0, 1)
        
        # Return the one that gives better initial spread
        return points  # Default to grid-based
    
    def adaptive_constraint_handler(points_flat, penalty_weight=100.0):
        """Adaptive constraint handling that penalizes violations more heavily when they occur frequently."""
        points = points_flat.reshape(-1, 2)
        violations = []
        
        # Check x bounds
        x_violations = np.concatenate([points[:, 0], 1 - points[:, 0]])
        violations.extend(x_violations[x_violations < 0])
        
        # Check y bounds  
        y_violations = np.concatenate([points[:, 1], 1 - points[:, 1]])
        violations.extend(y_violations[y_violations < 0])
        
        # Return penalty based on violation magnitude
        if len(violations) > 0:
            return penalty_weight * np.sum(np.array(violations)**2)
        return 0.0
    
    def enhanced_objective(points_flat):
        """Enhanced objective combining distance ratio with constraint penalties."""
        main_obj = objective(points_flat)
        constraint_penalty = adaptive_constraint_handler(points_flat)
        return main_obj + constraint_penalty * 1e-6
    
    def evaluate_initial_quality(points):
        """Evaluate the quality of initial point configuration."""
        ratio = compute_min_max_ratio(points.flatten())
        energy = energy_objective(points.flatten())
        return ratio, energy
    
    def multi_level_optimization(initial_points, time_limit: float) -> Tuple[np.ndarray, float]:
        """Perform a multi-level optimization approach."""
        start_time = time.time()
        
        # Level 1: Fast coarse optimization (very quick, lower quality)
        try:
            initial_flat = initial_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            cons = {'type': 'ineq', 'fun': constraint_func}
            
            result_coarse = minimize(
                enhanced_objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-5, 'gtol': 1e-5}
            )
            
            # Level 2: Medium refinement (more thorough)
            if result_coarse.success and (time.time() - start_time) < time_limit * 0.6:
                result_medium = minimize(
                    enhanced_objective,
                    result_coarse.x,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                # Level 3: Fine-tuning if time permits
                if result_medium.success and (time.time() - start_time) < time_limit * 0.8:
                    # Try Nelder-Mead as a local refinement
                    result_fine = minimize(
                        objective,
                        result_medium.x,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'fatol': 1e-9, 'xatol': 1e-9}
                    )
                    
                    if result_fine.success:
                        optimized_points = result_fine.x.reshape(-1, 2)
                    else:
                        optimized_points = result_medium.x.reshape(-1, 2)
                else:
                    optimized_points = result_medium.x.reshape(-1, 2)
            else:
                optimized_points = result_coarse.x.reshape(-1, 2)
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            optimized_points = initial_points.copy()
            
        # Ensure all points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)
        
        return optimized_points, time.time() - start_time
    
    # Multi-stage optimization approach with adaptive strategy
    best_result = None
    best_ratio = -np.inf
    best_time = float('inf')
    
    # Stage 1: Different initialization strategies
    strategies = []
    
    # Strategy 1: Determinant-maximizing initialization
    np.random.seed(42)
    init1 = geometric_initialization()
    strategies.append(("det_init", init1))
    
    # Strategy 2: Random initialization with better bounds
    init2 = np.random.uniform(0.05, 0.95, (16, 2))
    strategies.append(("rand_init", init2))
    
    # Strategy 3: Hexagonal + Perturbation
    np.random.seed(123)
    hex_grid_x = np.linspace(0.1, 0.9, 4)
    hex_grid_y = np.linspace(0.1, 0.9, 4)
    hex_x_grid, hex_y_grid = np.meshgrid(hex_grid_x, hex_grid_y)
    init3 = np.column_stack([hex_x_grid.ravel(), hex_y_grid.ravel()])[:16]
    init3 += np.random.normal(0, 0.015, init3.shape)
    init3 = np.clip(init3, 0, 1)
    strategies.append(("hex_init", init3))
    
    # Strategy 4: Regular grid with more randomness
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.1, 0.9, 4)
    x_grid, y_grid = np.meshgrid(grid_x, grid_y)
    init4 = np.column_stack([x_grid.ravel(), y_grid.ravel()])[:16]
    init4 += np.random.normal(0, 0.025, init4.shape)
    init4 = np.clip(init4, 0, 1)
    strategies.append(("grid_init", init4))
    
    # Run optimizations with different strategies
    for strategy_name, initial_points in strategies:
        try:
            # Early assessment of initial quality
            initial_ratio, initial_energy = evaluate_initial_quality(initial_points)
            
            # Skip poor quality initializations that are unlikely to improve
            if initial_ratio < 0.05:  # Threshold to skip obviously bad starts
                continue
                
            # Run multi-level optimization
            optimized_points, eval_time = multi_level_optimization(initial_points, 180.0)
            
            # Compute final metrics
            final_ratio = compute_min_max_ratio(optimized_points.flatten())
            benchmark_ratio = final_ratio / 0.2786  # AlphaEvolve benchmark
            
            print(f"{strategy_name} - Initial ratio: {initial_ratio:.6f}, Final ratio: {final_ratio:.6f}, Benchmark ratio: {benchmark_ratio:.6f}")
            
            # Keep track of best result
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_result = optimized_points.copy()
                best_time = eval_time
                
        except Exception as e:
            print(f"{strategy_name} failed with error: {e}")
            continue
    
    # Fallback to deterministic initialization if nothing worked
    if best_result is None:
        print("All optimization attempts failed, returning deterministic configuration")
        fallback_points = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.25, 0.75], [0.75, 0.75],
            [0.1, 0.1], [0.9, 0.1],
            [0.1, 0.9], [0.9, 0.9],
            [0.3, 0.5], [0.7, 0.5],
            [0.5, 0.3], [0.5, 0.7],
            [0.4, 0.4], [0.6, 0.6],
            [0.4, 0.6], [0.6, 0.4]
        ])
        return fallback_points
    
    print(f"\nBest final min/max ratio: {best_ratio:.6f}")
    print(f"Best benchmark ratio: {best_ratio / 0.2786:.6f}")
    print(f"Best evaluation time: {best_time:.6f}s")
    
    return best_result


# EVOLVE-BLOCK-END