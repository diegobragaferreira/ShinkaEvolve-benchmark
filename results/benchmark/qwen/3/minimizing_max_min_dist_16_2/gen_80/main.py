# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
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

def compute_voronoi_mean_area(points):
    """Calculate mean Voronoi cell area to assess uniformity."""
    try:
        vor = Voronoi(points)
        areas = []
        for i, region in enumerate(vor.regions):
            if not region or -1 in region:
                continue
            vertices = [vor.vertices[j] for j in region if j >= 0]
            if len(vertices) >= 3:
                # Compute polygon area using shoelace formula
                vertices = np.array(vertices)
                area = 0.5 * np.abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) - 
                                   np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
                areas.append(area)
        
        return np.mean(areas) if areas else 0.0
    except:
        return 0.0

def initialize_hexagonal_voronoi():
    """Initialize points using a sophisticated hexagonal lattice with Voronoi guidance."""
    # Create a 4x4 hexagonal grid with optimized spacing
    points = []
    rows = 4
    cols = 4
    
    # Hexagonal lattice parameters
    sqrt3 = math.sqrt(3)
    row_spacing = sqrt3 / 2
    col_spacing = 1.0
    
    # Create hexagonal pattern with enhanced spread
    for i in range(rows):
        for j in range(cols):
            # Offset every other row for true hexagonal arrangement  
            x = j * col_spacing + (i % 2) * 0.5 * col_spacing
            y = i * row_spacing
            
            # Normalize to [0,1] range, maintaining aspect ratio
            points.append([x, y])
    
    # Convert to numpy array
    points = np.array(points)
    
    # Normalize to fit well in unit square
    x_range = np.max(points[:, 0]) - np.min(points[:, 0])
    y_range = np.max(points[:, 1]) - np.min(points[:, 1])
    
    if x_range > 0:
        points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range
    if y_range > 0:
        points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range
    
    # Scale to fit nicely within [0.05, 0.95] to keep distance from boundaries
    points[:, 0] = 0.05 + 0.9 * points[:, 0]
    points[:, 1] = 0.05 + 0.9 * points[:, 1]
    
    # Apply controlled perturbation to break symmetry and avoid local minima
    np.random.seed(42)
    noise_magnitude = 0.015
    noise = np.random.normal(0, noise_magnitude, points.shape)
    
    # Apply different noise patterns to different rows for better symmetry breaking
    for i in range(rows):
        row_noise = noise[i*cols:(i+1)*cols]
        if i % 3 == 0:  # Every third row gets stronger noise
            row_noise *= 1.5
        elif i % 3 == 1:  # Every second row gets moderate noise
            row_noise *= 1.0
        else:  # Last row gets less noise
            row_noise *= 0.7
            
        noise[i*cols:(i+1)*cols] = row_noise
    
    points += noise
    points = np.clip(points, 0, 1)
    
    return points

def initialize_fallback():
    """Fallback initialization strategy."""
    np.random.seed(42)
    points = np.random.uniform(0.05, 0.95, (16, 2))
    return points

def voronoi_guided_optimization(initial_points, max_iter=300):
    """Optimize using Voronoi-based guided approach."""
    
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        
        try:
            # Compute distance matrix and ratio
            dist_matrix = compute_distance_matrix(points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            # Add Voronoi-based uniformity penalty
            voronoi_area = compute_voronoi_mean_area(points)
            # Penalize non-uniform distributions (smaller mean area means less uniform)
            uniformity_penalty = -0.5 * voronoi_area
            
            # Boundary penalty - points too close to edges are penalized
            boundary_penalty = 0
            margin = 0.02
            if np.any(points < margin) or np.any(points > 1 - margin):
                boundary_penalty = -0.1
                
            # Total penalty
            total_penalty = uniformity_penalty + boundary_penalty
            
            # Return negative since we're minimizing
            return -(ratio + total_penalty)
            
        except Exception:
            return 1e6  # Invalid configuration penalty
    
    # Flatten points for optimization
    x0 = initial_points.flatten()
    
    # Set bounds (slightly inside to prevent boundary issues)
    bounds = [(0.02, 0.98) for _ in range(len(x0))]
    
    # Initial optimization with L-BFGS-B
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5}
    )
    
    # Reshape and ensure bounds
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
    initial_strategies = [
        initialize_hexagonal_voronoi,
        initialize_fallback
    ]
    
    best_points = None
    best_ratio = 0
    
    for init_func in initial_strategies:
        try:
            initial_points = init_func()
            
            # Apply Voronoi-guided optimization
            optimized_points = voronoi_guided_optimization(initial_points, max_iter=200)
            
            # Calculate resulting ratio
            dist_matrix = compute_distance_matrix(optimized_points)
            ratio = calculate_min_max_ratio(dist_matrix)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # Fallback to simple optimization if needed
    if best_points is None:
        initial_points = initialize_fallback()
        best_points = voronoi_guided_optimization(initial_points, max_iter=200)
    
    return best_points

# EVOLVE-BLOCK-END