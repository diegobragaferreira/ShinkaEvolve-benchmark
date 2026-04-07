# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi
import time
from itertools import combinations
import math

def create_spherical_voronoi_initialization(n_points: int = 14) -> np.ndarray:
    """
    Create initial point configuration based on spherical Voronoi tiling principles.
    This provides a more structured starting point than random initialization.
    """
    # Generate points on a sphere using fibonacci-like method
    points = []
    
    # Use a modified fibonacci approach for better distribution
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(n_points):
        # Distribute points more evenly on sphere
        y = 1 - (i / (n_points - 1)) * 2  # y from 1 to -1
        radius = np.sqrt(1 - y*y)
        
        theta = np.arctan2(y, radius) + (i * 2 * np.pi / n_points) 
        
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points

def calculate_min_max_ratio(points: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate minimum and maximum distances between all point pairs.
    Returns (min_distance, max_distance, ratio).
    """
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0, 0.0, 0.0
        
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist <= 0:
        return 0.0, 0.0, 0.0
    
    ratio = min_dist / max_dist
    return min_dist, max_dist, ratio

def spherical_constraint(points: np.ndarray) -> np.ndarray:
    """Keep points on the unit sphere by normalizing them."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def objective_function_for_minmax(points_flat: np.ndarray) -> float:
    """
    Objective function to maximize the min/max distance ratio.
    Returns negative ratio since optimizers minimize by default.
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)
    
    # Keep points on sphere
    points = spherical_constraint(points)
    
    # Ensure points are within reasonable bounds
    points = np.clip(points, -1.0, 1.0)
    
    # Calculate distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return float('inf')
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Avoid division by zero
    if max_dist <= 0:
        return float('inf')
    
    # Prefer configurations with higher ratios, so return negative
    return -min_dist / max_dist

def local_refinement(points: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Apply local refinement using gradient-based optimization.
    """
    def objective_local(x_flat):
        points_local = x_flat.reshape(-1, 3)
        # Keep on sphere
        points_local = spherical_constraint(points_local)
        distances = pdist(points_local)
        if len(distances) == 0:
            return float('inf')
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 0:
            return float('inf')
        return -min_dist / max_dist  # Minimize negative ratio
    
    # Flatten points for optimization
    x0 = points.flatten()
    
    # Use L-BFGS-B for local refinement
    try:
        result = minimize(objective_local, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8})
        refined_points = result.x.reshape(-1, 3)
        return spherical_constraint(refined_points)
    except:
        return points

def symmetric_rotation(points: np.ndarray, num_rotations: int = 8) -> np.ndarray:
    """
    Generate rotated versions of point set using quaternion-based rotations.
    """
    # Simple rotation around z-axis
    angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    rotated_sets = []
    
    for angle in angles:
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        rotated_points = points @ rotation_matrix.T
        rotated_sets.append(rotated_points)
    
    return np.vstack(rotated_sets)

def adaptive_simulated_annealing(initial_points: np.ndarray, max_time: float) -> np.ndarray:
    """
    Adaptive simulated annealing with cooling schedule to escape local minima.
    """
    current_points = initial_points.copy()
    current_score = -objective_function_for_minmax(current_points.flatten())
    
    # Parameters
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 1e-6
    max_iterations = int(max_time * 100)  # Rough estimate
    patience = 50
    
    best_points = current_points.copy()
    best_score = current_score
    no_improvement_count = 0
    
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > max_time:
            break
            
        # Generate neighbor solution - slightly perturb points
        neighbor_points = current_points + np.random.normal(0, 0.01, current_points.shape)
        
        # Keep on sphere
        neighbor_points = spherical_constraint(neighbor_points)
        
        # Evaluate neighbor
        neighbor_score = -objective_function_for_minmax(neighbor_points.flatten())
        
        # Accept or reject
        if neighbor_score > current_score or \
           np.random.random() < np.exp((neighbor_score - current_score) / temperature):
            current_points = neighbor_points
            current_score = neighbor_score
            
            # Update best if improved
            if neighbor_score > best_score:
                best_points = neighbor_points.copy()
                best_score = neighbor_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
            
        # Cooling schedule
        temperature *= cooling_rate
        
        # Early stopping if no improvement
        if no_improvement_count > patience:
            break
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining spherical Voronoi initialization with adaptive optimization.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Phase 1: Initial configuration using spherical Voronoi principles
    initial_points = create_spherical_voronoi_initialization(14)
    
    # Phase 2: Apply adaptive simulated annealing
    optimized_points = adaptive_simulated_annealing(initial_points, max_time=300.0)
    
    # Phase 3: Local refinement 
    optimized_points = local_refinement(optimized_points, max_iter=100)
    
    # Phase 4: Generate symmetric variations for exploration
    symmetric_candidates = symmetric_rotation(optimized_points, num_rotations=6)
    
    # Evaluate all candidates and select best
    best_points = optimized_points.copy()
    best_ratio = 0.0
    
    # Check current optimized version
    _, _, current_ratio = calculate_min_max_ratio(optimized_points)
    if current_ratio > best_ratio:
        best_ratio = current_ratio
        best_points = optimized_points.copy()
    
    # Check symmetric variants (if any are better)
    num_candidates = len(symmetric_candidates) // 14
    for i in range(num_candidates):
        candidate_points = symmetric_candidates[i*14:(i+1)*14]
        _, _, ratio = calculate_min_max_ratio(candidate_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = candidate_points.copy()
    
    # Phase 5: Final refinement with global optimization
    try:
        # Flatten for optimization
        x0 = best_points.flatten()
        
        # Global optimization with bounds
        bounds = [(-1.0, 1.0) for _ in range(14*3)]
        
        # Use differential evolution for final polishing
        result = differential_evolution(
            lambda x: objective_function_for_minmax(x),
            bounds,
            maxiter=50,
            popsize=15,
            seed=42,
            disp=False,
            tol=1e-6
        )
        
        final_points = result.x.reshape(14, 3)
        
        # Ensure they're on sphere
        final_points = spherical_constraint(final_points)
        
        # Final ratio check
        _, _, final_ratio = calculate_min_max_ratio(final_points)
        _, _, current_ratio = calculate_min_max_ratio(best_points)
        
        if final_ratio > current_ratio:
            best_points = final_points
            
    except:
        pass
    
    # Ensure final result is properly bounded and normalized
    best_points = np.clip(best_points, -1.0, 1.0)
    best_points = spherical_constraint(best_points)
    
    return best_points

# EVOLVE-BLOCK-END