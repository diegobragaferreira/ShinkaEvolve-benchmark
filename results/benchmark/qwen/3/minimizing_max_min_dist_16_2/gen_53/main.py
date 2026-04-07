# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
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

def initialize_points_voronoi_based():
    """Initialize points using Voronoi-based approach with constrained Delaunay triangulation."""
    # Generate initial points using a combination of regular grid and perturbation
    # Create a 4x4 grid and then apply Voronoi-inspired constraints
    points = []
    rows = 4
    cols = 4
    
    # Create regular grid with slight perturbations
    for i in range(rows):
        for j in range(cols):
            # Add some randomness to avoid perfect symmetry
            x = j + np.random.normal(0, 0.02)
            y = i + np.random.normal(0, 0.02)
            points.append([x, y])
    
    points = np.array(points)
    
    # Normalize to [0,1] range
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
        
    # Apply boundary constraints
    points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
    points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
    
    return points

def voronoi_volume_objective(points):
    """Objective function based on Voronoi cell volumes to encourage uniform distribution."""
    try:
        vor = Voronoi(points)
        volumes = []
        
        # Calculate Voronoi cell areas (approximate)
        for region in vor.regions:
            if not region or -1 in region:
                continue
            vertices = [vor.vertices[i] for i in region]
            if len(vertices) >= 3:
                # Simple polygon area calculation
                vertices = np.array(vertices)
                area = 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) - 
                                   np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
                volumes.append(area)
        
        if volumes:
            return -np.mean(volumes)  # Negative because we want to maximize volume
        else:
            return 0
    except:
        return 0

def optimize_with_voronoi_refinement(initial_points, max_iterations=300):
    """Refine points using Voronoi-based local optimization."""
    
    def objective_voronoi_ratio(x_flat):
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        
        try:
            dist_matrix = compute_distance_matrix(points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            # Add penalty for points too close to boundary
            penalty = 0
            if np.any(points < 0.02) or np.any(points > 0.98):
                penalty = -0.01
                
            return -(ratio + penalty)
        except Exception:
            return 1e6
    
    # First, try differential evolution for global search
    bounds = [(0.01, 0.99) for _ in range(len(initial_points.flatten()))]
    
    try:
        de_result = differential_evolution(
            objective_voronoi_ratio,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            disp=False
        )
        x0 = de_result.x
    except:
        x0 = initial_points.flatten()
    
    # Then refine with local optimization
    try:
        result = minimize(
            objective_voronoi_ratio,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations//2, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        optimized_points = result.x.reshape(-1, 2)
    except:
        optimized_points = initial_points
    
    # Final boundary check
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Multiple initialization strategies based on Voronoi principles
    initial_strategies = [
        lambda: initialize_points_voronoi_based(),
        lambda: np.random.uniform(0.05, 0.95, (16, 2)),
        lambda: np.random.uniform(0, 1, (16, 2))
    ]
    
    best_points = None
    best_ratio = 0
    
    for i, init_func in enumerate(initial_strategies):
        try:
            initial_points = init_func()
            
            # Apply Voronoi refinement
            final_points = optimize_with_voronoi_refinement(initial_points, max_iterations=300)
            
            # Calculate the resulting ratio
            dist_matrix = compute_distance_matrix(final_points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
                
        except Exception as e:
            continue
    
    # Fallback to simple random initialization if nothing works
    if best_points is None:
        initial_points = np.random.uniform(0.05, 0.95, (16, 2))
        best_points = optimize_with_voronoi_refinement(initial_points)
    
    return best_points

# EVOLVE-BLOCK-END
