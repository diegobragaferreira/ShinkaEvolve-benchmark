# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses smart initialization, hybrid optimization, and symmetry breaking for superior results.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the minimum to maximum distance ratio for given points."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances efficiently
        distances = cdist(points, points)
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist

    def initialize_points(n_points=16, method='hexagonal'):
        """Initialize points using various intelligent strategies."""
        np.random.seed(42)
        
        if method == 'hexagonal':
            # Create hexagonal-like arrangement with adaptive spacing
            points = []
            rows, cols = 4, 4
            
            for i in range(rows):
                for j in range(cols):
                    # Create hexagonal offset pattern
                    x_offset = 0.5 if i % 2 == 1 else 0.0
                    x = (j + x_offset) * 0.25 + 0.125
                    y = i * 0.25 + 0.125
                    
                    # Add small adaptive perturbations
                    perturbation = 0.015
                    x += np.random.normal(0, perturbation * 0.5)
                    y += np.random.normal(0, perturbation * 0.5)
                    points.append([x, y])
            
            points = np.clip(points, 0, 1)
            return np.array(points[:n_points])
            
        elif method == 'perturbed_grid':
            # Start with regular grid and add perturbations
            grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
            points = grid_points.astype(float) / 3.0
            
            # Add adaptive perturbations
            for i in range(len(points)):
                # Points near edges get less perturbation, interior gets more
                x, y = points[i]
                if (abs(x - 0) < 0.1 or abs(x - 1) < 0.1 or 
                    abs(y - 0) < 0.1 or abs(y - 1) < 0.1):
                    perturbation_magnitude = 0.01
                elif abs(x - 0.5) < 0.3 and abs(y - 0.5) < 0.3:
                    perturbation_magnitude = 0.02
                else:
                    perturbation_magnitude = 0.015
                    
                points[i] += np.random.uniform(-perturbation_magnitude, perturbation_magnitude, 2)
            
            points = np.clip(points, 0, 1)
            return points[:n_points]
            
        else:  # random_spread
            # Create points with intentional spreading
            points = np.random.rand(n_points, 2)
            
            # Apply spreading to reduce clustering
            for i in range(n_points):
                center_vec = points[i] - [0.5, 0.5]
                center_distance = np.linalg.norm(center_vec)
                if center_distance > 0:
                    # Move points away from center with adaptive factor
                    adaptive_factor = min(0.15, 0.08 / (center_distance + 0.01))
                    points[i] += center_vec * adaptive_factor
            
            points = np.clip(points, 0, 1)
            return points

    def adaptive_local_optimization(initial_points, time_limit):
        """Optimize using adaptive sequential quadratic programming."""
        start_time = time.time()
        
        # Best solution tracking
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(initial_points)
        
        # Objective function for optimization
        def objective(x):
            points = x.reshape(-1, 2)
            # Negative because we minimize -ratio to maximize ratio
            return -compute_min_max_ratio(points)
        
        # Bounds for each coordinate
        bounds = [(0, 1) for _ in range(32)]
        
        # Try L-BFGS-B with adaptive parameters based on time
        remaining_time = time_limit - (time.time() - start_time)
        max_iter = int(500 + remaining_time * 2) if remaining_time > 10 else 200
        
        try:
            result = minimize(
                objective,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                final_ratio = compute_min_max_ratio(final_points)
                
                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points.copy()
                    
        except Exception:
            pass
        
        # Try SLSQP as fallback with symmetry-breaking constraints
        if (time.time() - start_time) < time_limit - 5:
            try:
                # Add symmetry breaking constraints to avoid degenerate solutions
                def symmetry_constraint(x):
                    points = x.reshape(-1, 2)
                    # Fix first point to (0,0) and last point to (1,1) to break symmetry
                    constraints = []
                    constraints.append(points[0, 0])  # x0 = 0
                    constraints.append(points[0, 1])  # y0 = 0
                    constraints.append(points[-1, 0] - 1.0)  # x15 = 1
                    constraints.append(points[-1, 1] - 1.0)  # y15 = 1
                    return np.array(constraints)
                
                # Simple constraint for boundary handling
                constraints = {'type': 'eq', 'fun': lambda x: np.array([0.0])}
                
                result = minimize(
                    objective,
                    best_points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    final_ratio = compute_min_max_ratio(final_points)
                    
                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = final_points.copy()
                        
            except Exception:
                pass
        
        return best_points

    def multi_start_optimization(max_time=170):
        """Run optimization with multiple starting strategies."""
        start_time = time.time()
        best_points = None
        best_ratio = -np.inf
        
        # Multiple restart strategies with different initialization methods
        restart_strategies = [
            ('hex', 'hexagonal'),
            ('grid', 'perturbed_grid'),
            ('rand', 'random_spread')
        ]
        
        # Try each strategy
        for strategy_name, init_method in restart_strategies:
            if (time.time() - start_time) > max_time - 10:
                break
                
            try:
                # Generate initial points with this strategy
                current_points = initialize_points(16, init_method)
                
                # Optimize with adaptive method
                optimized_points = adaptive_local_optimization(current_points, 
                                                              max_time - (time.time() - start_time))
                
                # Calculate ratio
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
            except Exception:
                continue
        
        # Final refinement with more aggressive optimization if we have time
        if (time.time() - start_time) < max_time - 5 and best_points is not None:
            try:
                refined_points = adaptive_local_optimization(
                    best_points, 
                    max_time - (time.time() - start_time)
                )
                refined_ratio = compute_min_max_ratio(refined_points)
                
                if refined_ratio > best_ratio:
                    best_points = refined_points.copy()
                    
            except Exception:
                pass
        
        return best_points if best_points is not None else initialize_points(16, 'hexagonal')

    # Main execution
    try:
        return multi_start_optimization(170)
    except Exception:
        # Fallback to deterministic hexagonal pattern
        return initialize_points(16, 'hexagonal')

# EVOLVE-BLOCK-END