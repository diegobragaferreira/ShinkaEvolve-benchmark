# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def create_spherical_code_initialization(n_points: int = 14) -> np.ndarray:
    """
    Create initial point configuration based on spherical code principles.
    Uses optimized spherical code for 14 points.
    """
    # Known good configuration for 14 points on sphere from literature
    # These coordinates are normalized to unit sphere
    spherical_points = np.array([
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, -1.0000],
        [0.9343, 0.0000, 0.3564],
        [-0.9343, 0.0000, 0.3564],
        [0.0000, 0.9343, 0.3564],
        [0.0000, -0.9343, 0.3564],
        [0.0000, 0.9343, -0.3564],
        [0.0000, -0.9343, -0.3564],
        [0.9343, 0.0000, -0.3564],
        [-0.9343, 0.0000, -0.3564],
        [0.3564, 0.9343, 0.0000],
        [-0.3564, 0.9343, 0.0000],
        [0.3564, -0.9343, 0.0000],
        [-0.3564, -0.9343, 0.0000]
    ])
    
    # Normalize to unit sphere if needed
    norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
    spherical_points = spherical_points / np.where(norms == 0, 1, norms)
    
    # Add small perturbations to escape local optima
    np.random.seed(42)
    perturbation = np.random.normal(0, 0.01, spherical_points.shape)
    spherical_points = spherical_points + perturbation
    
    # Normalize again after perturbation
    norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
    spherical_points = spherical_points / np.where(norms == 0, 1, norms)
    
    return spherical_points

def geometric_relaxation_step(points: np.ndarray, iterations: int = 20) -> np.ndarray:
    """
    Apply geometric relaxation using force-based repulsion model.
    Each point repels others with inverse-square law, projected back to sphere.
    """
    points = points.copy()
    
    for _ in range(iterations):
        # Calculate pairwise distances
        n = len(points)
        forces = np.zeros_like(points)
        
        # Compute repulsive forces between all pairs
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                
                # Avoid singularity
                if dist_sq > 1e-10:
                    force_magnitude = 1.0 / dist_sq
                    forces[i] += force_magnitude * diff
                    forces[j] -= force_magnitude * diff
        
        # Apply forces and project back to unit cube
        points += 0.005 * forces  # Smaller step size for more stable convergence
        points = np.clip(points, 0, 1)
    
    return points

def initialize_points(n: int = 14, d: int = 3, num_starts: int = 6) -> np.ndarray:
    """
    Initialize points using multiple strategies for better starting configuration.
    
    Args:
        n: number of points
        d: dimensionality
        num_starts: number of different initializations to try

    Returns:
        Best initial point configuration
    """
    best_points = None
    best_ratio = -float('inf')
    
    for start_idx in range(num_starts):
        # Different initialization methods
        if start_idx == 0:
            # Fibonacci sphere initialization
            points = fibonacci_sphere(n)
            # Scale to unit cube [0,1]^3
            points = (points + 1) / 2  # map from [-1,1] to [0,1]
            
        elif start_idx == 1:
            # Spherical code initialization
            points = create_spherical_code_initialization(n)
            points = (points + 1) / 2  # map from [-1,1] to [0,1]
            
        elif start_idx == 2:
            # Random initialization with seed
            np.random.seed(42 + start_idx)
            points = np.random.rand(n, d)
            
        elif start_idx == 3:
            # Perturbed Fibonacci points
            points = fibonacci_sphere(n)
            points = (points + 1) / 2
            # Add small perturbation
            np.random.seed(42 + start_idx)
            points += np.random.normal(0, 0.005, points.shape)
            # Clip to valid range
            points = np.clip(points, 0, 1)
            
        elif start_idx == 4:
            # Perturbed spherical code points
            points = create_spherical_code_initialization(n)
            points = (points + 1) / 2
            # Add small perturbation
            np.random.seed(42 + start_idx)
            points += np.random.normal(0, 0.01, points.shape)
            # Clip to valid range
            points = np.clip(points, 0, 1)
            
        else:  # start_idx == 5
            # Geometric relaxation on random points
            np.random.seed(42 + start_idx)
            points = np.random.rand(n, d)
            points = geometric_relaxation_step(points, iterations=25)
            
        # Calculate initial ratio
        distances = pdist(points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
    
    # Fallback to random if nothing worked
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.rand(n, d)
    
    return best_points

def calculate_distance_metrics(points: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate minimum, maximum, and ratio of distances between all point pairs.

    Args:
        points: Array of shape (n, d)

    Returns:
        Tuple of (min_distance, max_distance, ratio)
    """
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0, 0.0, 0.0
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist <= 0:
        return 0.0, 0.0, 0.0
    
    ratio = min_dist / max_dist
    return min_dist, max_dist, ratio

def objective_function(points_flat: np.ndarray) -> float:
    """
    Objective function to maximize the min/max distance ratio.
    Returns negative ratio since optimizers minimize by default.
    Uses penalty-based constraint handling for better stability.

    Args:
        points_flat: Flattened array of point coordinates

    Returns:
        Negative min/max ratio (to be minimized)
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)

    # Penalty for out-of-bounds points
    penalty = 0.0
    for i in range(n):
        for j in range(3):  # x, y, z coordinates
            if points[i,j] < 0:
                penalty += 1000 * (0 - points[i,j])**2
            elif points[i,j] > 1:
                penalty += 1000 * (points[i,j] - 1)**2

    # Calculate distances
    distances = pdist(points)

    if len(distances) == 0:
        return float('inf') + penalty

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero
    if max_dist <= 0:
        return float('inf') + penalty

    # Return negative ratio plus penalty
    return -min_dist / max_dist + penalty

def adaptive_optimization(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Perform adaptive optimization with changing population sizes and strategies.
    Implements progressive optimization approach.
    
    Args:
        initial_points: Starting point configuration
        max_time: Maximum optimization time in seconds

    Returns:
        Optimized point configuration
    """
    start_time = time.time()
    
    # Flatten initial points for optimization
    initial_flat = initial_points.flatten()
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0.0, 1.0)] * len(initial_flat)
    
    # Progressive optimization phases
    # Phase 1: Global search with large population
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=200,
            popsize=25,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
    except Exception as e:
        # Fallback to initial points if first phase fails
        return initial_points

    # Phase 2: Refinement with smaller population
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=250,
            popsize=15,
            tol=1e-7,
            mutation=(0.7, 1.0),
            recombination=0.8,
            seed=43,
            disp=False
        )
        
    except Exception as e:
        pass

    # Phase 3: Local refinement using L-BFGS-B with tighter tolerances
    def local_objective(x_flat):
        points = x_flat.reshape(-1, 3)
        # Add penalty for out-of-bounds
        penalty = 0.0
        for i in range(14):
            for j in range(3):
                if points[i,j] < 0:
                    penalty += 1000 * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += 1000 * (points[i,j] - 1)**2
                    
        distances = pdist(points)
        if len(distances) == 0:
            return float('inf') + penalty
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 0:
            return float('inf') + penalty
        return -min_dist / max_dist + penalty
    
    try:
        x0 = result.x.reshape(-1, 3).flatten()
        local_result = minimize(
            local_objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        result = local_result
    except:
        pass

    # Reshape optimized result
    optimized_points = result.x.reshape(14, 3)
    
    # Ensure all points are within valid range (final safety check)
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Phase 1: Initialize points with multiple strategies
    initial_points = initialize_points(14, 3, num_starts=6)
    
    # Phase 2: Apply geometric relaxation for better distribution
    relaxed_points = geometric_relaxation_step(initial_points, iterations=30)
    
    # Phase 3: Optimize points with adaptive strategy
    optimized_points = adaptive_optimization(relaxed_points)
    
    # Phase 4: Final validation and adjustment
    final_points = optimized_points.copy()
    
    # Calculate final metrics
    min_dist, max_dist, ratio = calculate_distance_metrics(final_points)
    
    # If optimization didn't work well, try final global optimization
    if max_dist <= 0 or min_dist <= 0 or ratio < 0.15:
        # Try one more optimization pass with better parameters if needed
        try:
            bounds = [(0.0, 1.0)] * (14 * 3)
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=400,
                popsize=30,
                tol=1e-10,
                mutation=(0.8, 1.0),
                recombination=0.9,
                seed=42,
                disp=False
            )
            final_points = result.x.reshape(14, 3)
            final_points = np.clip(final_points, 0, 1)
        except:
            pass

    # Final validation
    _, _, final_ratio = calculate_distance_metrics(final_points)
    if final_ratio < 0.05:  # Very poor result, use another fallback
        np.random.seed(42)
        final_points = np.random.rand(14, 3)
    
    return final_points

# EVOLVE-BLOCK-END