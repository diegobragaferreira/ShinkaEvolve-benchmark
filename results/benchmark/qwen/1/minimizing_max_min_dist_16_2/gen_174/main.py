# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_distance_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Handle case where all points might be coincident
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def initialize_hexagonal_grid():
        """Initialize points using a refined hexagonal grid pattern."""
        np.random.seed(42)
        n = 16
        points = np.zeros((n, 2))

        # Create hexagonal grid pattern with better distribution
        rows = 4
        cols = 4
        spacing_x = 0.25
        spacing_y = 0.25 * math.sqrt(3) / 2

        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < n:
                    # Offset every other row for hexagonal packing
                    x = col * spacing_x + (row % 2) * spacing_x * 0.5
                    y = row * spacing_y
                    points[idx] = [x, y]
                    idx += 1

        # Scale and shift points to fit within [0.1, 0.9] x [0.1, 0.9] with some randomness
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

        # Add small random perturbation
        points += np.random.normal(0, 0.01, points.shape)

        # Clamp to bounds with epsilon padding
        epsilon = 1e-8
        points = np.clip(points, epsilon, 1-epsilon)

        return points

    def coordinate_wise_refinement(points, max_iterations=100):
        """
        Perform coordinate-wise refinement of point positions to improve the ratio.
        Each coordinate is optimized individually while fixing others.
        """
        points = points.copy()
        n = len(points)
        
        # Precompute current distances to avoid recomputation
        current_ratio = compute_distance_ratio(points)
        
        for iteration in range(max_iterations):
            improved = False
            # Try to optimize each point coordinate-wise
            for i in range(n):
                best_point = points[i].copy()
                best_ratio = current_ratio
                
                # Try small perturbations in x and y directions
                for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                    for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                        if abs(dx) == 0 and abs(dy) == 0:
                            continue
                            
                        test_point = points[i].copy()
                        test_point[0] += dx
                        test_point[1] += dy
                        
                        # Enforce bounds with epsilon padding
                        epsilon = 1e-8
                        test_point[0] = np.clip(test_point[0], epsilon, 1-epsilon)
                        test_point[1] = np.clip(test_point[1], epsilon, 1-epsilon)
                        
                        # Temporarily update this point
                        old_point = points[i].copy()
                        points[i] = test_point
                        
                        try:
                            ratio = compute_distance_ratio(points)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_point = test_point.copy()
                                improved = True
                        except:
                            pass
                            
                        # Restore original point
                        points[i] = old_point
                
                # Update to best point found
                points[i] = best_point
                current_ratio = best_ratio
                
            # Early termination if no significant improvement
            if not improved and iteration > 20:
                break
                
        return points

    # Phase 1: Initialize with a good hexagonal grid pattern
    initial_points = initialize_hexagonal_grid()
    
    # Phase 2: Global optimization with differential evolution
    def objective_function(x):
        points = x.reshape(-1, 2)
        ratio = compute_distance_ratio(points)
        return -ratio  # Negative because scipy minimizes

    bounds = [(0, 1) for _ in range(32)]
    
    try:
        # Use differential evolution for global search
        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if de_result.success:
            global_optimized = de_result.x.reshape(-1, 2)
        else:
            global_optimized = initial_points.copy()
    except:
        global_optimized = initial_points.copy()

    # Phase 3: Local refinement with coordinate-wise optimization
    try:
        local_optimized = coordinate_wise_refinement(global_optimized, max_iterations=50)
        final_ratio = compute_distance_ratio(local_optimized)
    except:
        local_optimized = global_optimized.copy()
        final_ratio = compute_distance_ratio(local_optimized)
    
    # Phase 4: Final polishing with L-BFGS-B if needed
    try:
        # Flatten for scipy optimization
        x0 = local_optimized.flatten()
        bounds_scipy = [(0, 1) for _ in range(32)]
        
        # Optimize using L-BFGS-B solver
        result = minimize(
            fun=objective_function,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds_scipy,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            polished_points = result.x.reshape(-1, 2)
            polished_ratio = compute_distance_ratio(polished_points)
            
            # Use polished result if it's better
            if polished_ratio > final_ratio:
                return polished_points
    except:
        pass
    
    return local_optimized

# EVOLVE-BLOCK-END
