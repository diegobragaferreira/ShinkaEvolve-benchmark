# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from typing import Tuple, List
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a spherical simulated annealing approach with adaptive cooling and strategic perturbations.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Configuration parameters
    n_points = 14
    n_dimensions = 3
    max_iterations = 100000
    initial_temp = 1.0
    final_temp = 1e-6
    cooling_rate = 0.9995
    perturbation_magnitude = 0.1
    log_interval = 1000
    
    # Initialize points on unit sphere using Fibonacci spiral method
    def fibonacci_sphere(n: int) -> np.ndarray:
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    # Calculate minimum and maximum distances
    def calculate_distances(points: np.ndarray) -> Tuple[float, float]:
        if len(points) < 2:
            return 0.0, 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        return d_min, d_max
    
    # Calculate the ratio of min/max distances
    def calculate_ratio(points: np.ndarray) -> float:
        d_min, d_max = calculate_distances(points)
        if d_max <= 0:
            return 0.0
        return d_min / d_max
    
    # Project points onto unit sphere
    def project_to_sphere(points: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return points / norms
    
    # Generate a neighbor by perturbing a random point
    def generate_neighbor(current_points: np.ndarray, temp: float, 
                         perturbation_mag: float = None) -> np.ndarray:
        if perturbation_mag is None:
            perturbation_mag = temp * 0.1
            
        # Choose a random point to perturb
        idx = np.random.randint(0, len(current_points))
        neighbor_points = current_points.copy()
        
        # Add Gaussian noise to selected point
        noise = np.random.normal(0, perturbation_mag, n_dimensions)
        neighbor_points[idx] += noise
        
        # Project back to sphere
        neighbor_points = project_to_sphere(neighbor_points)
        
        return neighbor_points
    
    # Simulated Annealing main loop
    def simulated_annealing() -> Tuple[np.ndarray, float]:
        # Start with Fibonacci sphere initialization
        current_points = fibonacci_sphere(n_points)
        
        # Initial state
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Track recent improvements for adaptive cooling
        recent_improvements = []
        max_recent_improvements = 10
        
        # Temperature schedule
        temp = initial_temp
        
        iteration = 0
        start_time = time.time()
        
        while iteration < max_iterations and temp > final_temp:
            # Generate candidate solution
            candidate_points = generate_neighbor(current_points, temp)
            candidate_ratio = calculate_ratio(candidate_points)
            
            # Accept or reject based on Metropolis criterion
            delta_ratio = candidate_ratio - current_ratio
            
            # Avoid numerical issues with very small temperature
            if temp < 1e-12:
                accept_prob = 1.0 if delta_ratio > 0 else 0.0
            else:
                accept_prob = min(1.0, np.exp(delta_ratio / temp))
            
            if np.random.random() < accept_prob:
                current_points = candidate_points
                current_ratio = candidate_ratio
                
                # Update best solution
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
                    
                    # Record improvement
                    recent_improvements.append(iteration)
                    if len(recent_improvements) > max_recent_improvements:
                        recent_improvements.pop(0)
            
            # Adaptive cooling: decrease temperature more slowly if we're not improving
            if len(recent_improvements) > 0 and len(recent_improvements) < max_recent_improvements:
                temp *= cooling_rate
            else:
                # More aggressive cooling if we've been stagnating
                temp *= cooling_rate * 0.9
            
            iteration += 1
            
            # Log progress
            if iteration % log_interval == 0:
                elapsed_time = time.time() - start_time
                # print(f"Iteration {iteration}: Ratio = {current_ratio:.6f}, Best = {best_ratio:.6f}, Temp = {temp:.6f}, Time = {elapsed_time:.2f}s")
        
        return best_points, best_ratio
    
    # Run optimization
    try:
        optimized_points, best_ratio = simulated_annealing()
        
        # Final check of results
        final_min, final_max = calculate_distances(optimized_points)
        if final_max <= 0:
            # Fallback to Fibonacci initialization if optimization failed
            warnings.warn("Optimization failed, returning Fibonacci sphere initialization")
            optimized_points = fibonacci_sphere(n_points)
            
        return optimized_points
    
    except Exception as e:
        # Fallback to basic initialization if anything fails
        warnings.warn(f"Optimization failed with error: {str(e)}, returning Fibonacci sphere initialization")
        return fibonacci_sphere(n_points)


# EVOLVE-BLOCK-END
