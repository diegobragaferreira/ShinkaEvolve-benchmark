# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import time
import math

def compute_distance_matrix(points):
    """Compute pairwise distance matrix for given points."""
    return squareform(pdist(points))

def calculate_min_max_ratio(distance_matrix):
    """Calculate the ratio of minimum to maximum distances."""
    # Exclude diagonal (distance to self)
    off_diagonal = distance_matrix[distance_matrix > 0]
    if len(off_diagonal) == 0:
        return 0.0
    d_min = np.min(off_diagonal)
    d_max = np.max(off_diagonal)
    return d_min / d_max if d_max > 0 else 0.0

def initialize_points_hexagonal_perturbed():
    """Initialize points using a perturbed hexagonal grid."""
    points = []
    rows = 4
    cols = 4

    for i in range(rows):
        for j in range(cols):
            x = j + 0.5 * (i % 2)
            y = i * math.sqrt(3)/2
            
            # Add significant perturbation to break symmetry
            if i % 2 == 1 and cols > 1:
                x += 0.125
                
            points.append([x, y])

    # Normalize to [0,1] range
    points = np.array(points)
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
    
    # Add substantial noise to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.03, points.shape)
    points = np.clip(points, 0, 1)
    
    return points

def initialize_points_random_better():
    """Initialize points with better random distribution."""
    np.random.seed(42)
    points = np.random.uniform(0.05, 0.95, (16, 2))
    return points

def initialize_points_spiral():
    """Initialize points using spiral arrangement to ensure good coverage."""
    points = []
    n = 16
    
    # Create a spiral pattern with radial distribution
    for i in range(n):
        angle = 2 * math.pi * i / n * 3  # Three full rotations
        radius = 0.4 * (i / (n - 1))  # From center outward
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        points.append([x, y])
    
    points = np.array(points)
    # Add some noise to break symmetry
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, 0, 1)
    
    return points

def initialize_points_adaptive_grid():
    """Initialize with adaptive grid spacing for better distribution."""
    # Create grid with non-uniform spacing
    points = []
    for i in range(4):
        for j in range(4):
            # Vary spacing to avoid regular patterns
            x = j * 0.33 + 0.1 * np.sin(i * 0.5)
            y = i * 0.33 + 0.1 * np.cos(j * 0.5)
            points.append([x, y])
    
    points = np.array(points)
    
    # Normalize and add noise
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
    
    np.random.seed(42)
    points += np.random.normal(0, 0.02, points.shape)
    points = np.clip(points, 0, 1)
    
    return points

def initialize_points_symmetric_perturbed():
    """Initialize with symmetric base then perturb to break symmetry."""
    # Create a symmetric but not uniform grid
    points = []
    for i in range(4):
        for j in range(4):
            points.append([i * 0.33, j * 0.33])
    
    points = np.array(points)
    # Add perturbation to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.015, points.shape)
    points = np.clip(points, 0, 1)
    
    return points

def optimize_with_differential_evolution(points, max_iter=300):
    """Optimize using differential evolution with improved parameters."""
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        try:
            dist_matrix = compute_distance_matrix(points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            # Add penalty for boundary proximity
            penalty = 0
            boundary_threshold = 0.02
            boundary_penalty = 0
            for point in points:
                for coord in point:
                    if coord < boundary_threshold or coord > (1 - boundary_threshold):
                        boundary_penalty += 0.005
            
            return -(ratio - boundary_penalty)
        except Exception:
            return 1e6
    
    # Flatten initial points
    x0 = points.flatten()
    
    # Define bounds with slight margin to prevent boundary issues
    bounds = [(0.01, 0.99) for _ in range(len(x0))]
    
    # Use improved differential evolution parameters
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max_iter,
        popsize=30,  # Increase population size
        mutation=(0.8, 1),  # More aggressive mutation
        recombination=0.9,  # Higher recombination
        seed=42,
        disp=False,
        atol=1e-6,
        rtol=1e-6
    )
    
    optimized_points = result.x.reshape(-1, 2)
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

def optimize_with_lbfgsb(points, max_iter=300):
    """Optimize using L-BFGS-B for local refinement."""
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        try:
            dist_matrix = compute_distance_matrix(points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            # Add penalty for boundary proximity
            penalty = 0
            boundary_threshold = 0.02
            for point in points:
                for coord in point:
                    if coord < boundary_threshold or coord > (1 - boundary_threshold):
                        penalty += 0.003
            
            return -(ratio - penalty)
        except Exception:
            return 1e6
    
    # Flatten initial points
    x0 = points.flatten()
    
    # Define bounds with margin
    bounds = [(0.01, 0.99) for _ in range(len(x0))]
    
    # Use L-BFGS-B with more iterations
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5},
        callback=None
    )
    
    optimized_points = result.x.reshape(-1, 2)
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    # Multiple initialization strategies
    init_strategies = [
        initialize_points_hexagonal_perturbed,
        initialize_points_random_better,
        initialize_points_spiral,
        initialize_points_adaptive_grid,
        initialize_points_symmetric_perturbed
    ]
    
    best_points = None
    best_ratio = 0
    
    # Try each initialization strategy
    for i, init_func in enumerate(init_strategies):
        try:
            # Generate initial points
            initial_points = init_func()
            
            # First phase: Differential Evolution for global search
            de_points = optimize_with_differential_evolution(initial_points, max_iter=200)
            
            # Second phase: Local refinement with L-BFGS-B  
            refined_points = optimize_with_lbfgsb(de_points, max_iter=200)
            
            # Calculate final ratio
            dist_matrix = compute_distance_matrix(refined_points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            # Update best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
                
        except Exception as e:
            continue
    
    # Fallback to default approach if nothing worked well
    if best_points is None:
        initial_points = initialize_points_hexagonal_perturbed()
        de_points = optimize_with_differential_evolution(initial_points, max_iter=150)
        best_points = optimize_with_lbfgsb(de_points, max_iter=150)
    
    return best_points

# EVOLVE-BLOCK-END