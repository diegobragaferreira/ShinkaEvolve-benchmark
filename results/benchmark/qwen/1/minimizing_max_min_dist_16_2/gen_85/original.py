# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances using squareform for numerical stability
        distances = pdist(points)
        
        # Handle edge cases
        if len(distances) == 0 or np.allclose(distances, 0):
            return float('inf')  # Worst possible case

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return float('inf')

        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        return -min_dist / max_dist

    def constraint_func(x):
        # Ensure points are within [0+eps,1-eps] x [0+eps,1-eps] for numerical stability
        eps = 1e-8
        points = x.reshape(-1, 2)
        constraints = []

        # x coordinates in [eps, 1-eps]
        constraints.append(points[:, 0].min() - eps)  # x_min >= eps
        constraints.append(1 - eps - points[:, 0].max())  # x_max <= 1-eps

        # y coordinates in [eps, 1-eps]
        constraints.append(points[:, 1].min() - eps)  # y_min >= eps
        constraints.append(1 - eps - points[:, 1].max())  # y_max <= 1-eps

        return np.array(constraints)

    def bounded_objective(x):
        # Boundary checking with clamping to safe bounds
        eps = 1e-8
        points = np.clip(x.reshape(-1, 2), eps, 1-eps).flatten()
        return objective(points)

    def generate_initial_configurations():
        """Generate multiple diverse initial configurations"""
        configs = []
        
        # 1. Hexagonal grid pattern (better than simple random)
        np.random.seed(42)
        hex_points = []
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(hex_points) >= 16:
                    break
                # Hexagonal offset pattern
                x = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
                y = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
                if i % 2 == 1:  # Offset odd rows
                    x += 0.4 / (cols - 1) if cols > 1 else 0.2
                hex_points.append([x, y])
        
        # Normalize and add slight randomness
        hex_points = np.array(hex_points[:16])
        # Add jitter to break symmetry
        hex_points += np.random.normal(0, 0.01, hex_points.shape)
        hex_points = np.clip(hex_points, 0, 1)
        configs.append(hex_points.copy())
        
        # 2. Spiral pattern for better distribution
        spiral_points = []
        for i in range(16):
            if i == 0:
                spiral_points.append([0.5, 0.5])
            else:
                angle = i * 2.5
                radius = min(0.4, i * 0.05)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                spiral_points.append([x, y])
        configs.append(np.array(spiral_points[:16]).clip(0, 1))
        
        # 3. Random uniform distribution
        configs.append(np.random.rand(16, 2))
        
        # 4. Grid pattern with jitter
        grid_points = []
        for i in range(4):
            for j in range(4):
                if len(grid_points) >= 16:
                    break
                base_x = 0.1 + i * 0.225
                base_y = 0.1 + j * 0.225
                # Add jitter to avoid perfect grid artifacts
                jitter_x = np.random.normal(0, 0.015)
                jitter_y = np.random.normal(0, 0.015)
                grid_points.append([base_x + jitter_x, base_y + jitter_y])
        configs.append(np.array(grid_points[:16]).clip(0, 1))

        return configs

    # Generate multiple initial configurations
    initial_configs = generate_initial_configurations()
    
    best_ratio = float('inf')
    best_points = None
    
    # Try each initial configuration with hybrid optimization
    for i, initial_config in enumerate(initial_configs):
        try:
            # Flatten for optimization
            x0 = initial_config.flatten()
            
            # Define bounds for each coordinate (safe bounds to avoid numerical issues)
            bounds = [(1e-8, 1-1e-8) for _ in range(32)]
            
            # Phase 1: Global optimization with Differential Evolution
            de_result = differential_evolution(
                bounded_objective,
                bounds,
                seed=42+i,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            # If DE finds a good solution, refine it locally
            if de_result.success and -de_result.fun > 0.1:
                x0 = de_result.x
                
            # Phase 2: Local optimization with SLSQP for fine-tuning
            result = minimize(
                bounded_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9},
                callback=None
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                # Calculate actual ratio for this configuration
                distances = pdist(optimized_points)
                if len(distances) > 0 and np.max(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    ratio = min_dist / max_dist
                    
                    if ratio < best_ratio:  # We want to maximize ratio, so minimize negative ratio
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
        except Exception as e:
            continue
    
    # If we still don't have a good solution, use a fallback approach
    if best_points is None:
        # Use the most promising initial configuration with more aggressive local optimization
        fallback_config = np.random.rand(16, 2)
        x0 = fallback_config.flatten()
        bounds = [(1e-8, 1-1e-8) for _ in range(32)]
        
        # More aggressive local optimization
        result = minimize(
            bounded_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result.success:
            best_points = result.x.reshape(-1, 2)
    
    # Final safety check - ensure points are within bounds
    if best_points is not None:
        best_points = np.clip(best_points, 1e-8, 1-1e-8)
    else:
        # Last resort: return random configuration
        best_points = np.random.rand(16, 2)
    
    return best_points

# EVOLVE-BLOCK-END