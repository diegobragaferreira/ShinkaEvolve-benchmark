# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid structured-initialization with adaptive simulated annealing approach.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    
    # Create initial hexagonal pattern with better distribution
    def create_initial_hexagonal():
        # Arrange 16 points in a 4x4 grid with hexagonal spacing
        points = []
        rows = 4
        cols = 4
        spacing_x = 1.0
        spacing_y = np.sqrt(3) / 2

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points)

        # Normalize to [0,1] x [0,1]
        max_x = (cols - 1) + 0.5  # Account for offset in last row
        max_y = (rows - 1) * spacing_y

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y

        # Add small random noise to break symmetry
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        
        return points

    # Calculate min/max distance ratio efficiently
    def calculate_ratio(points):
        if len(points) < 2:
            return 0

        # Use more efficient distance calculation for better performance
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max <= 0:
            return 0

        return d_min / d_max

    # Advanced perturbation strategy with adaptive sizing
    def adaptive_perturbation(points, temperature):
        new_points = points.copy()
        
        # Determine whether to perturb single point or use coordinated moves
        if np.random.rand() < 0.7:
            # Single point perturbation
            idx = np.random.randint(len(points))
            # Adaptive step size based on temperature and local density
            step_size = temperature * 0.1
            
            # Reduce step size if point is in dense region
            distances = cdist([points[idx]], points)[0]
            distances = distances[distances > 0]  # Exclude self-distance
            if len(distances) > 0:
                min_dist = np.min(distances)
                if min_dist < 0.1:  # Point is in dense region
                    step_size *= 0.5
            
            new_points[idx, 0] += np.random.normal(0, step_size)
            new_points[idx, 1] += np.random.normal(0, step_size)
            
            # Boundary handling with reflection
            if new_points[idx, 0] < 0:
                new_points[idx, 0] = -new_points[idx, 0]
            elif new_points[idx, 0] > 1:
                new_points[idx, 0] = 2 - new_points[idx, 0]
                
            if new_points[idx, 1] < 0:
                new_points[idx, 1] = -new_points[idx, 1]
            elif new_points[idx, 1] > 1:
                new_points[idx, 1] = 2 - new_points[idx, 1]
                
        else:
            # Coordinated move for two nearby points
            # Find two nearby points to move together
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            
            # Find two points that are relatively close
            closest_pairs = np.unravel_index(np.argmin(distances), distances.shape)
            idx1, idx2 = closest_pairs
            
            # Coordinate the movement
            step_size = temperature * 0.08
            delta = np.random.normal(0, step_size, 2)
            
            # Apply same movement to both points
            new_points[idx1] += delta
            new_points[idx2] += delta
            
            # Boundary handling for both points
            for i in [idx1, idx2]:
                if new_points[i, 0] < 0:
                    new_points[i, 0] = -new_points[i, 0]
                elif new_points[i, 0] > 1:
                    new_points[i, 0] = 2 - new_points[i, 0]
                    
                if new_points[i, 1] < 0:
                    new_points[i, 1] = -new_points[i, 1]
                elif new_points[i, 1] > 1:
                    new_points[i, 1] = 2 - new_points[i, 1]
        
        return new_points

    # Optimized simulated annealing with adaptive cooling
    def optimized_annealing(initial_points, max_iter=10000):
        current_points = initial_points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # Better cooling schedule
        T = 0.2  # Higher initial temperature
        cooling_rate = 0.9995  # Moderate cooling
        min_temp = 1e-6
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Track recent improvements for dynamic cooling
        recent_improvements = []
        improvement_window = 50
        
        for iteration in range(max_iter):
            T *= cooling_rate
            
            if T < min_temp:
                break
                
            # Generate neighbor solution with adaptive perturbation
            new_points = adaptive_perturbation(current_points, T)
            
            # Evaluate new solution
            new_ratio = calculate_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = current_points.copy()
            
            # Track improvements for adaptive cooling
            if new_ratio > current_ratio:
                recent_improvements.append(new_ratio - current_ratio)
                if len(recent_improvements) > improvement_window * 2:
                    recent_improvements.pop(0)
            
            # Adaptive cooling based on performance
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-6:
                    T *= 0.98  # Cool faster if stagnating
                    
        return best_points, best_ratio

    # Main optimization loop with multiple restarts
    best_solution = None
    best_ratio = 0
    
    # Multiple restarts with different initializations
    for restart in range(5):
        # Create initial configuration with different random seed for variety
        np.random.seed(42 + restart)
        initial_points = create_initial_hexagonal()
        
        # Optimize with simulated annealing
        optimized_points, ratio = optimized_annealing(initial_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()
    
    # Final refinement with greedy local search
    if best_solution is not None:
        final_points = best_solution.copy()
        prev_ratio = best_ratio
        
        # Do a few rounds of greedy improvements
        for _ in range(500):
            # Try moving each point to see if we can improve
            for i in range(len(final_points)):
                original_point = final_points[i].copy()
                
                # Try small perturbations
                for _ in range(10):  # Try several candidates per point
                    candidate = original_point.copy()
                    candidate[0] += np.random.normal(0, 0.005)
                    candidate[1] += np.random.normal(0, 0.005)
                    
                    # Clip to boundary
                    candidate[0] = np.clip(candidate[0], 0, 1)
                    candidate[1] = np.clip(candidate[1], 0, 1)
                    
                    # Test if this improves the solution
                    test_points = final_points.copy()
                    test_points[i] = candidate
                    
                    new_ratio = calculate_ratio(test_points)
                    if new_ratio > prev_ratio:
                        final_points[i] = candidate
                        prev_ratio = new_ratio
                        
        return final_points
    
    # Fallback if nothing worked
    return create_initial_hexagonal()

# EVOLVE-BLOCK-END