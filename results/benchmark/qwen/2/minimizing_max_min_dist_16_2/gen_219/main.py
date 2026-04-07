# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from numba import jit
import time

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

    def create_grid_initialization():
        """Create a structured 4x4 grid with adaptive perturbations to improve spacing"""
        # Start with regular 4x4 grid
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = i / 3.0  # Normalized to [0,1] range
                y = j / 3.0
                grid_points.append([x, y])
        
        points = np.array(grid_points)
        
        # Apply adaptive perturbations based on location
        # Emphasize corners and edges to encourage better spreading
        for i in range(16):
            row, col = i // 4, i % 4
            
            # More aggressive perturbations for corners to break symmetry
            if (row in [0, 3] and col in [0, 3]):
                std = 0.035
            elif (row in [0, 3] or col in [0, 3]):
                std = 0.02
            else:
                std = 0.01
                
            # Apply perturbation
            points[i, 0] += np.random.normal(0, std)
            points[i, 1] += np.random.normal(0, std)
        
        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)
        return points

    def distance_aware_local_search(points, max_iterations=50):
        """Perform a local search that specifically targets improving minimum distance"""
        current_points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try moving each point to improve the minimum distance
            for i in range(len(current_points)):
                original_point = current_points[i].copy()
                best_move = original_point.copy()
                best_improvement = 0.0
                
                # Test small movements in different directions
                movements = [
                    [-0.01, -0.01], [-0.01, 0], [-0.01, 0.01],
                    [0, -0.01], [0, 0.01],
                    [0.01, -0.01], [0.01, 0], [0.01, 0.01]
                ]
                
                # Test each movement
                for dx, dy in movements:
                    test_point = original_point.copy()
                    test_point[0] += dx
                    test_point[1] += dy
                    
                    # Clip to bounds
                    test_point[0] = np.clip(test_point[0], 0.001, 0.999)
                    test_point[1] = np.clip(test_point[1], 0.001, 0.999)
                    
                    # Temporarily update this point
                    temp_points = current_points.copy()
                    temp_points[i] = test_point
                    
                    # Check ratio improvement
                    ratio = compute_min_max_ratio(temp_points)
                    improvement = ratio - best_ratio
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = test_point.copy()
                
                # Apply the best move if it improves the solution
                if best_improvement > 0:
                    current_points[i] = best_move
                    improved = True
                    
                    # Update best solution if this is better
                    ratio = compute_min_max_ratio(current_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()
            
            # Early stopping if no improvement
            if not improved:
                break
                
        return best_points

    def progressive_optimization(initial_points, max_time=170):
        """Apply multi-stage optimization with increasing precision"""
        start_time = time.time()
        current_points = initial_points.copy()
        
        # Stage 1: Coarse optimization with relaxed tolerances
        if time.time() - start_time < max_time * 0.3:
            result = minimize(
                objective_with_regularization,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.success:
                current_points = result.x.reshape(-1, 2)
        
        # Stage 2: Refinement with medium precision
        if time.time() - start_time < max_time * 0.6:
            # Apply distance-aware local search
            current_points = distance_aware_local_search(current_points, max_iterations=30)
            
            # Fine optimization
            result = minimize(
                objective_with_regularization,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success:
                current_points = result.x.reshape(-1, 2)
        
        # Stage 3: Final high precision optimization
        if time.time() - start_time < max_time * 0.9:
            # Apply final local search
            current_points = distance_aware_local_search(current_points, max_iterations=50)
            
            # Very tight optimization
            result = minimize(
                objective_with_regularization,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success:
                current_points = result.x.reshape(-1, 2)
        
        return current_points

    # Generate initial configuration
    np.random.seed(42)
    initial_points = create_grid_initialization()
    
    # Apply progressive optimization
    final_points = progressive_optimization(initial_points, max_time=165)
    
    # Final local search for any remaining improvements
    final_points = distance_aware_local_search(final_points, max_iterations=100)
    
    # Ensure final points are within bounds
    final_points = np.clip(final_points, 0, 1)
    
    return final_points

# EVOLVE-BLOCK-END