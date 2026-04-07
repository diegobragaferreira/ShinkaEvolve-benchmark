# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)

        # Calculate pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing)
        if d_max <= 0:
            return -1.0  # Avoid division by zero or invalid distances
        return -d_min / d_max

    def constraint_bounds(x_flat):
        # Ensure all points are within [0,1] x [0,1] with epsilon padding
        points = x_flat.reshape(-1, 2)
        # Lower bounds (negative for inequality constraints) - points >= 1e-6
        lower = -points.flatten() + 1e-6
        # Upper bounds (positive for inequality constraints) - points <= 1-1e-6
        upper = points.flatten() - (1 - 1e-6)
        return np.concatenate([lower, upper])

    def adaptive_minimize(obj_func, x0, bounds, maxiter, ftol, gtol):
        """Minimize with adaptive stopping criteria"""
        previous_obj_val = float('inf')
        consecutive_no_improvement = 0
        max_no_improvement = 20

        # Use a callback to track progress
        def callback(xk):
            nonlocal previous_obj_val, consecutive_no_improvement
            obj_val = obj_func(xk)
            if abs(previous_obj_val - obj_val) < 1e-12:
                consecutive_no_improvement += 1
            else:
                consecutive_no_improvement = 0
            previous_obj_val = obj_val

        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
                callback=callback if consecutive_no_improvement < max_no_improvement else None
            )
            return result
        except Exception:
            return None

    # Phase 1: Multi-start initialization with different strategies
    np.random.seed(42)
    
    # Strategy 1: Hexagonal grid initialization
    points_hex = []
    rows, cols = 4, 4
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
            y = i / (rows - 1) if rows > 1 else 0.5
            # Add controlled random perturbation
            x += (np.random.rand() - 0.5) * 0.1
            y += (np.random.rand() - 0.5) * 0.1
            # Ensure within safe bounds
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)
            points_hex.append([x, y])
    initial_hex = np.array(points_hex[:16])
    
    # Strategy 2: Random initialization
    initial_random = np.random.rand(16, 2) * 0.8 + 0.1  # Keep away from edges
    
    # Phase 2: Optimization with multiple approaches
    best_solution = None
    best_ratio = float('inf')
    
    # Try both initializations with Differential Evolution
    for i, initial_points in enumerate([initial_hex, initial_random]):
        x0 = initial_points.flatten()
        bounds = [(1e-6, 1-1e-6) for _ in range(32)]
        
        try:
            # Global optimization with Differential Evolution
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=200,  # Increased iterations
                popsize=25,   # Larger population size
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            
            current_obj_value = -objective(de_result.x)
            if current_obj_value < best_ratio:
                best_ratio = current_obj_value
                best_solution = de_result.x
            
        except Exception as e:
            continue
    
    # Phase 3: Local refinement with adaptive L-BFGS-B
    if best_solution is not None:
        try:
            # Use the best solution from DE as starting point for L-BFGS-B
            bounds = [(1e-6, 1-1e-6) for _ in range(32)]
            result = adaptive_minimize(
                objective,
                best_solution,
                bounds,
                maxiter=1000,
                ftol=1e-10,
                gtol=1e-10
            )
            
            if result and result.success:
                final_obj_value = -objective(result.x)
                if final_obj_value < best_ratio:
                    best_ratio = final_obj_value
                    best_solution = result.x
                    
        except Exception:
            pass
    
    # Phase 4: Final fallback to initial solution if needed
    if best_solution is None:
        initial_points = initial_hex
        best_solution = initial_points.flatten()
    
    # Convert final solution to required format
    final_points = best_solution.reshape(-1, 2)
    
    # Ensure points respect bounds
    final_points = np.clip(final_points, 1e-6, 1 - 1e-6)
    
    return final_points


# EVOLVE-BLOCK-END