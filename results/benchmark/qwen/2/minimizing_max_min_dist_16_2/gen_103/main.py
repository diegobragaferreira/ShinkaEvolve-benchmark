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

    def generate_initial_configuration():
        """Generate a high-quality initial configuration using multiple strategies."""
        # Strategy 1: Grid with adaptive perturbations
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
        points = grid_points.astype(float) / 3.0
        
        # Adaptive perturbation based on expected distribution quality
        # Calculate current ratio to determine perturbation magnitude
        current_ratio = calculate_min_max_ratio(points)
        if current_ratio > 0.15:
            perturbation_magnitude = 0.015
        elif current_ratio > 0.08:
            perturbation_magnitude = 0.03
        else:
            perturbation_magnitude = 0.05
            
        np.random.seed(42)
        points += np.random.uniform(-perturbation_magnitude, perturbation_magnitude, points.shape)
        points = np.clip(points, 0, 1)
        
        # Strategy 2: Try hexagonal pattern as backup
        hex_points = np.zeros((16, 2))
        rows, cols = 4, 4
        for i in range(rows):
            for j in range(cols):
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125
                y = i * 0.25 + 0.125
                hex_points[i * cols + j] = [x, y]
        
        # Evaluate both and pick better starting point
        ratio1 = calculate_min_max_ratio(points)
        ratio2 = calculate_min_max_ratio(hex_points)
        
        return hex_points if ratio2 > ratio1 else points

    def adaptive_local_search(points, max_iter=500):
        """Enhanced local search with adaptive step sizes and early stopping."""
        current_ratio = calculate_min_max_ratio(points)
        step_size = 0.02
        stagnation_count = 0
        max_stagnation = 30
        
        for iteration in range(max_iter):
            improvement_found = False
            # Try moving each point
            for i in range(len(points)):
                temp_points = points.copy()
                move = np.random.uniform(-step_size, step_size, 2)
                temp_points[i] = points[i] + move
                temp_points[i] = np.clip(temp_points[i], 0, 1)
                
                new_ratio = calculate_min_max_ratio(temp_points)
                
                if new_ratio > current_ratio:
                    points = temp_points
                    current_ratio = new_ratio
                    improvement_found = True
                    stagnation_count = 0
                    break  # Move to next point after successful change
            
            if not improvement_found:
                stagnation_count += 1
                if stagnation_count > 10:
                    step_size *= 0.9  # Gradually decrease step size
                    stagnation_count = 0
                    
            # Early stopping for small improvements
            if not improvement_found and stagnation_count > max_stagnation:
                break
                
        return points

    # Generate initial configuration
    initial_points = generate_initial_configuration()
    
    # Multi-stage optimization approach
    current_points = initial_points.copy()
    
    # Stage 1: Global optimization with Differential Evolution
    bounds = [(0, 1) for _ in range(32)]
    
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=80,      # Reduced iterations for speed
            popsize=20,      # Larger population for better exploration
            seed=42,
            tol=1e-10,
            mutation=(0.5, 1),
            recombination=0.7
        )
        
        if hasattr(de_result, 'success') and de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            de_ratio = calculate_min_max_ratio(de_points)
            
            # Use DE result if it's better
            if de_ratio > calculate_min_max_ratio(current_points):
                current_points = de_points
                
    except Exception as e:
        pass
    
    # Stage 2: Progressive local refinement with adaptive tolerances
    # First coarse refinement
    try:
        result_coarse = minimize(
            objective,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 50}
        )
        if result_coarse.success:
            current_points = result_coarse.x.reshape(-1, 2)
            current_points = np.clip(current_points, 0, 1)
    except:
        pass

    # Stage 3: Adaptive local search with symmetry breaking
    current_points = adaptive_local_search(current_points)
    
    # Stage 4: Fine-tune with stricter local optimization
    try:
        result_fine = minimize(
            objective,
            current_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 100}
        )
        if result_fine.success:
            current_points = result_fine.x.reshape(-1, 2)
            current_points = np.clip(current_points, 0, 1)
    except:
        pass
    
    # Stage 5: Final adaptive refinement with symmetry breaking
    current_points = adaptive_local_search(current_points, max_iter=300)
    
    return current_points

# EVOLVE-BLOCK-END