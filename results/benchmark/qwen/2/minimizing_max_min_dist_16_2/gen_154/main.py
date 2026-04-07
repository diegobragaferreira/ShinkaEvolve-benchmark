# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
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
        
        # Calculate all pairwise distances efficiently
        distances = cdist(points, points, metric='euclidean')
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio to maximize the ratio (negative because we minimize)
        if max_dist <= 0:
            return 0
        return -min_dist / max_dist
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0

        # Efficiently compute all pairwise distances
        distances = cdist(points, points, metric='euclidean')

        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist <= 0:
            return 0

        return min_dist / max_dist
    
    # Multiple initialization strategies
    init_strategies = [
        # Strategy 1: Hexagonal-like grid with adaptive perturbations
        lambda: _hexagonal_grid_init(),
        
        # Strategy 2: Perturbed regular grid
        lambda: _perturbed_grid_init(),
        
        # Strategy 3: Spiral pattern for good spread
        lambda: _spiral_init()
    ]
    
    best_ratio = -np.inf
    best_points = None
    
    # Try each initialization strategy with multiple seeds
    for strategy_idx, init_func in enumerate(init_strategies):
        for seed_val in [42, 123, 456]:
            np.random.seed(seed_val)
            
            try:
                # Get initial points using the strategy
                points = init_func()
                
                # Apply local optimization with L-BFGS-B
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(objective, points.flatten(), method='L-BFGS-B', 
                                bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-12})
                
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                
                # Calculate final ratio
                ratio = calculate_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception:
                continue
    
    # If we didn't find a good solution, fallback to a refined grid approach
    if best_points is None:
        # Create structured initial configuration
        initial_points = np.zeros((16, 2))
        idx = 0
        for i in range(4):
            for j in range(4):
                # Standard grid positions with adaptive perturbations
                x = j / 3.0 + np.random.uniform(-0.01, 0.01)
                y = i / 3.0 + np.random.uniform(-0.01, 0.01)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                initial_points[idx] = [x, y]
                idx += 1
        
        # Apply optimization
        bounds = [(0, 1) for _ in range(32)]
        try:
            result = minimize(objective, initial_points.flatten(), method='L-BFGS-B', 
                            bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-12})
            best_points = result.x.reshape(-1, 2)
            best_points = np.clip(best_points, 0, 1)
        except:
            best_points = initial_points
    
    # Final refinement with differential evolution for global search
    try:
        bounds = [(0, 1) for _ in range(32)]
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=30,      # Reduced iterations for speed
            popsize=10,      # Smaller population
            seed=42,
            tol=1e-12,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        de_points = de_result.x.reshape(-1, 2)
        de_points = np.clip(de_points, 0, 1)
        de_ratio = calculate_min_max_ratio(de_points)
        current_ratio = calculate_min_max_ratio(best_points)
        
        if de_ratio > current_ratio:
            best_points = de_points
            
    except Exception:
        pass
    
    return best_points

def _hexagonal_grid_init():
    """Initialize with hexagonal packing pattern for better space utilization."""
    points = []
    rows = 4
    cols = 4
    
    for i in range(rows):
        for j in range(cols):
            # offset every other row for hexagonal packing
            x_offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + x_offset) / 3.0
            y = i / 3.0
            points.append([x, y])
    
    # Add small random perturbations to break symmetry
    points = np.array(points)
    np.random.seed(42)
    points += np.random.uniform(-0.015, 0.015, points.shape)
    points = np.clip(points, 0, 1)
    return points

def _perturbed_grid_init():
    """Initialize with regular grid plus adaptive perturbations."""
    grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
    points = grid_points.astype(float) / 3.0  # Normalize to [0,1] range
    
    # Add perturbations based on grid position
    np.random.seed(42)
    for i in range(16):
        # Vary perturbation strength by position
        row, col = i // 4, i % 4
        scale = 0.02 if (row + col) % 2 == 0 else 0.015
        points[i] += np.random.uniform(-scale, scale, 2)
    
    points = np.clip(points, 0, 1)
    return points

def _spiral_init():
    """Initialize with spiral pattern for good coverage."""
    points = []
    angles = np.linspace(0, 4 * 2 * np.pi, 16)
    radii = np.linspace(0.1, 0.45, 16)
    
    for angle, radius in zip(angles, radii):
        x = 0.5 + radius * np.cos(angle) * 0.4
        y = 0.5 + radius * np.sin(angle) * 0.4
        points.append([x, y])
    
    points = np.array(points)
    points = np.clip(points, 0, 1)
    return points

# EVOLVE-BLOCK-END