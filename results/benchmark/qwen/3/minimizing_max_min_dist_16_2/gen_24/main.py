# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_distances(points):
        """Compute distance matrix and extract min/max distances"""
        if len(points) < 2:
            return 0, float('inf')
        dist_matrix = squareform(pdist(points))
        np.fill_diagonal(dist_matrix, float('inf'))
        dmin = np.min(dist_matrix)
        dmax = np.max(dist_matrix)
        return dmin, dmax
    
    def min_max_ratio(points):
        """Calculate the min/max distance ratio"""
        dmin, dmax = compute_distances(points)
        if dmax == 0:
            return 0
        return dmin / dmax
    
    def golden_spiral_initialization(n):
        """Initialize points using golden spiral distribution"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            theta = i * 2 * np.pi / phi
            r = np.sqrt(i + 1) / np.sqrt(n)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y])
        return np.array(points)
    
    def project_to_domain(points):
        """Project points to [0,1] x [0,1] domain"""
        # Scale to fit within [0,1] x [0,1]
        points = np.clip(points, 0, 1)
        return points
    
    def adaptive_cooling(initial_temp, iteration, max_iter):
        """Adaptive cooling schedule that responds to convergence"""
        # Start with a high temperature and cool faster initially
        # Then slow down as we converge
        temp = initial_temp * (0.99 ** iteration)
        
        # If we're converging slowly, cool slower
        if iteration > max_iter * 0.7:
            temp *= (0.999 ** (iteration - max_iter * 0.7))
            
        return max(temp, 1e-6)
    
    def neighbor_perturbation(points, sigma=0.01):
        """Apply neighborhood-based perturbations"""
        # Create a copy of points
        new_points = points.copy()
        
        # Select some points to perturb
        indices = np.random.choice(len(points), size=max(1, len(points)//4), replace=False)
        
        for idx in indices:
            # Add small random perturbation
            new_points[idx] += np.random.normal(0, sigma, 2)
        
        # Project back to valid domain
        new_points = project_to_domain(new_points)
        
        return new_points
    
    def local_refinement(points, max_iter=20):
        """Use gradient-based local refinement"""
        # Simple coordinate ascent approach
        current_points = points.copy()
        
        for _ in range(max_iter):
            best_points = current_points.copy()
            best_ratio = min_max_ratio(current_points)
            
            # Try small perturbations to each point
            for i in range(len(current_points)):
                original_pos = current_points[i].copy()
                
                # Try small movements in each direction
                for dx, dy in [(0.001, 0), (0, 0.001), (-0.001, 0), (0, -0.001)]:
                    test_points = current_points.copy()
                    test_points[i] = original_pos + np.array([dx, dy])
                    
                    # Ensure within bounds
                    test_points[i] = np.clip(test_points[i], 0, 1)
                    
                    ratio = min_max_ratio(test_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = test_points.copy()
                        
            # Update if we found improvement
            if best_ratio > min_max_ratio(current_points):
                current_points = best_points
            else:
                break  # No improvement, stop
                
        return current_points
    
    # Main optimization loop
    np.random.seed(42)
    
    # Initialize with golden spiral
    points = golden_spiral_initialization(16)
    
    # Normalize to [0,1] domain
    points = project_to_domain(points)
    
    # Optimization parameters
    max_iterations = 1000
    initial_temperature = 1.0
    best_points = points.copy()
    best_ratio = min_max_ratio(best_points)
    
    # Adaptive optimization loop
    start_time = time.time()
    
    for iteration in range(max_iterations):
        # Calculate current temperature
        temp = adaptive_cooling(initial_temperature, iteration, max_iterations)
        
        # Apply neighbor perturbation
        new_points = neighbor_perturbation(best_points, sigma=0.01)
        
        # Local refinement
        new_points = local_refinement(new_points, max_iter=10)
        
        # Evaluate new solution
        new_ratio = min_max_ratio(new_points)
        
        # Accept or reject based on temperature
        if new_ratio > best_ratio or np.random.random() < np.exp((new_ratio - best_ratio) / temp):
            best_points = new_points
            best_ratio = new_ratio
            
            # Early stopping condition
            if best_ratio > 0.5:  # Early exit if we get very good results
                break
    
    # Final local refinement
    final_points = local_refinement(best_points, max_iter=50)
    final_ratio = min_max_ratio(final_points)
    
    return final_points

# EVOLVE-BLOCK-END
