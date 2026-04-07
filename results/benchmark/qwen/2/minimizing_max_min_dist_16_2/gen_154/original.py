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
    
    # Create a structured initial configuration based on 4x4 grid with strategic placement
    # This is designed to avoid degenerate cases and provide good starting balance
    initial_points = np.zeros((16, 2))
    
    # Create a structured 4x4 grid pattern with some strategic offset and perturbation
    idx = 0
    for i in range(4):
        for j in range(4):
            # Standard grid positions
            x = j / 3.0
            y = i / 3.0
            
            # Add small perturbations to break perfect symmetry and encourage better distribution
            # Using a deterministic yet varied pattern
            np.random.seed(i * 4 + j + 42)
            pert_x = np.random.uniform(-0.01, 0.01)
            pert_y = np.random.uniform(-0.01, 0.01)
            
            x += pert_x
            y += pert_y
            
            # Ensure points are within bounds
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            
            initial_points[idx] = [x, y]
            idx += 1
    
    # Apply multi-stage optimization to enhance the solution
    current_points = initial_points.copy()
    
    # Stage 1: Coarse grid refinement using L-BFGS-B
    bounds = [(0, 1) for _ in range(32)]
    
    try:
        # First optimization pass with higher tolerance to quickly improve the structure
        result1 = minimize(objective, current_points.flatten(), method='L-BFGS-B', 
                          bounds=bounds, options={'ftol': 1e-8, 'gtol': 1e-8})
        current_points = result1.x.reshape(-1, 2)
        current_points = np.clip(current_points, 0, 1)
    except:
        pass
    
    # Stage 2: Progressive fine-tuning with adaptive perturbation hill climbing
    def adaptive_hill_climbing(points, max_iterations=500):
        """Improve solution using adaptive hill climbing with decreasing step size"""
        current_ratio = calculate_min_max_ratio(points)
        step_size = 0.01
        stagnation_count = 0
        max_stagnation = 50
        
        for iteration in range(max_iterations):
            best_improvement = 0
            best_move = None
            best_new_points = None
            
            # Try moving each point
            for i in range(len(points)):
                temp_points = points.copy()
                
                # Try multiple random moves for this point
                for _ in range(5):  # Multiple attempts per point
                    move = np.random.uniform(-step_size, step_size, 2)
                    temp_points[i] = points[i] + move
                    
                    # Keep within bounds
                    temp_points[i] = np.clip(temp_points[i], 0, 1)
                    
                    new_ratio = calculate_min_max_ratio(temp_points)
                    
                    if new_ratio > current_ratio:
                        improvement = new_ratio - current_ratio
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_move = move
                            best_new_points = temp_points.copy()
            
            # If improvement found, accept it
            if best_improvement > 0:
                points = best_new_points
                current_ratio = calculate_min_max_ratio(points)
                stagnation_count = 0
            else:
                stagnation_count += 1
                # Reduce step size if no progress
                if stagnation_count > 10:
                    step_size *= 0.8
                    stagnation_count = 0
                    
            # Early stopping if improvement becomes negligible
            if best_improvement < 1e-12:
                break
                
        return points
    
    # Apply adaptive hill climbing
    current_points = adaptive_hill_climbing(current_points)
    
    # Stage 3: Differential Evolution to escape local optima and find better solutions
    try:
        de_bounds = [(0, 1) for _ in range(32)]
        de_result = differential_evolution(
            objective,
            de_bounds,
            maxiter=50,      # Reduced iterations for speed
            popsize=15,      # Smaller population
            seed=42,
            tol=1e-10,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        # Evaluate this result against our current best
        de_points = de_result.x.reshape(-1, 2)
        de_points = np.clip(de_points, 0, 1)
        de_ratio = calculate_min_max_ratio(de_points)
        current_ratio = calculate_min_max_ratio(current_points)
        
        if de_ratio > current_ratio:
            current_points = de_points
            
    except:
        pass
    
    # Final stage: Refine with L-BFGS-B one last time
    try:
        final_result = minimize(objective, current_points.flatten(), method='L-BFGS-B', 
                               bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-12})
        current_points = final_result.x.reshape(-1, 2)
        current_points = np.clip(current_points, 0, 1)
    except:
        pass
    
    return current_points

# EVOLVE-BLOCK-END