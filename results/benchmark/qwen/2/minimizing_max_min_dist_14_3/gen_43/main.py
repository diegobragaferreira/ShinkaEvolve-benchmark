# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def fibonacci_sphere(n: int) -> np.ndarray:
    """
    Generate n points distributed as evenly as possible on a unit sphere
    using Fibonacci spiral method.
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def compute_min_max_ratio(points: np.ndarray) -> tuple:
    """
    Compute the minimum and maximum distances between all pairs of points,
    and return their ratio.
    """
    if len(points) < 2:
        return 0.0, 0.0
    
    # Compute pairwise distances
    distances = cdist(points, points)
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    
    # Find min and max distances
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    
    # Avoid division by zero
    if max_distance == 0:
        ratio = 0.0
    else:
        ratio = min_distance / max_distance
    
    return min_distance, max_distance, ratio

def perturb_points(points: np.ndarray, temperature: float, perturbation_scale: float = 0.01) -> np.ndarray:
    """
    Apply small random perturbations to points while keeping them on the unit sphere.
    """
    # Create a copy of the points
    new_points = points.copy()
    
    # Select a random point to perturb
    idx = np.random.randint(len(points))
    
    # Generate small random perturbation
    delta = np.random.normal(0, perturbation_scale * temperature, 3)
    
    # Add perturbation to selected point
    new_points[idx] += delta
    
    # Project back to unit sphere
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    new_points = new_points / norms
    
    return new_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Initialize points using Fibonacci sphere method for better distribution
    points = fibonacci_sphere(14)
    
    # Optimization parameters
    max_iterations = 100000
    initial_temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 0.0001
    
    # Track best solution
    best_points = points.copy()
    best_min_dist, best_max_dist, best_ratio = compute_min_max_ratio(points)
    
    # Current state
    current_points = points.copy()
    current_min_dist, current_max_dist, current_ratio = best_ratio, best_max_dist, best_ratio
    
    # Simulated Annealing
    temp = initial_temperature
    
    for iteration in range(max_iterations):
        # Perturb the current solution
        new_points = perturb_points(current_points, temp)
        
        # Compute new ratio
        new_min_dist, new_max_dist, new_ratio = compute_min_max_ratio(new_points)
        
        # Accept or reject the new solution using Metropolis criterion
        if new_ratio > current_ratio:
            # Always accept better solutions
            current_points = new_points
            current_ratio = new_ratio
            current_min_dist = new_min_dist
            current_max_dist = new_max_dist
            
            # Update best solution if this is better
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
                best_min_dist = new_min_dist
                best_max_dist = new_max_dist
        else:
            # Accept worse solutions with probability based on temperature
            acceptance_prob = np.exp((new_ratio - current_ratio) / temp)
            if np.random.rand() < acceptance_prob:
                current_points = new_points
                current_ratio = new_ratio
                current_min_dist = new_min_dist
                current_max_dist = new_max_dist
        
        # Cool down the temperature
        temp = max(temp * cooling_rate, min_temperature)
        
        # Early stopping if temperature is too low
        if temp < min_temperature:
            break
    
    return best_points

# EVOLVE-BLOCK-END