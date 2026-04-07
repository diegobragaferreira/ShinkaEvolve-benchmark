# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
import time
from scipy.spatial import SphericalVoronoi
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    # Generate superior initial configuration based on geometric principles
    points = generate_geometric_initialization(14)
    
    # Apply multi-scale optimization with geometric constraint enforcement
    best_points, best_ratio = optimize_sphere_tiling(points)
    
    return best_points

def generate_geometric_initialization(n):
    """Generate initial points using geometric constructions for better spread"""
    if n == 14:
        # Use construction based on icosahedral symmetry with additional points
        # Start with icosahedral vertices (12 points) plus 2 extra points
        points = np.zeros((n, 3))
        
        # Icosahedral vertices scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        # Standard icosahedral vertices
        vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        
        # Normalize to unit sphere
        for i, vertex in enumerate(vertices):
            norm = np.linalg.norm(vertex)
            points[i] = np.array(vertex) / norm
        
        # Add two more points near poles but distributed to avoid clustering
        points[12] = [0, 0, 0.8]  # Near north pole
        points[13] = [0, 0, -0.8]  # Near south pole
        
        # Add small random perturbations to break perfect symmetry
        points += np.random.normal(0, 0.01, points.shape)
        
        # Normalize again to ensure they're on unit sphere
        for i in range(len(points)):
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] = points[i] / norm
                
    else:
        # Fallback to Fibonacci sphere for other sizes
        points = fibonacci_sphere(n)
    
    return points

def fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral method"""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        
        theta = phi * i  # golden angle increment
        
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        points.append([x, y, z])
    
    return np.array(points)

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances"""
    if len(points) < 2:
        return 0.0
    
    distances = pdist(points)
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    if d_max == 0:
        return 0.0
    
    return d_min / d_max

def compute_energy_gradient(points):
    """Calculate gradient based on electrostatic repulsion model"""
    n = len(points)
    gradients = np.zeros_like(points)
    
    # For each point, calculate repulsive force from all other points
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                
                # Avoid singularities
                if dist_sq > 1e-10:
                    # Force is inversely proportional to square of distance (like Coulomb)
                    force_magnitude = 1.0 / (dist_sq * np.sqrt(dist_sq))
                    gradients[i] += force_magnitude * diff
    
    return gradients

def project_to_sphere(points):
    """Project points to unit sphere while preserving relative positions"""
    new_points = points.copy()
    for i in range(len(new_points)):
        norm = np.linalg.norm(new_points[i])
        if norm > 0:
            new_points[i] = new_points[i] / norm
    return new_points

def optimize_sphere_tiling(initial_points, max_time=360):
    """
    Multi-scale optimization using sphere tiling principle
    """
    points = initial_points.copy()
    current_ratio = compute_min_max_ratio(points)
    
    # Phase 1: Global optimization with energy-based approach
    temp = 1.0
    min_temp = 1e-8
    cooling_rate = 0.999
    max_iter_phase1 = 50000
    iter_count = 0
    
    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    
    start_time = time.time()
    
    # Phase 1: Energy minimization with repulsion forces
    while temp > min_temp and iter_count < max_iter_phase1 and time.time() - start_time < max_time:
        # Calculate repulsion forces
        gradients = compute_energy_gradient(points)
        
        # Apply forces with adaptive step size (larger when points are far apart)
        step_size = 0.001 * temp
        
        # Move points according to forces
        new_points = points.copy()
        for i in range(len(points)):
            # Apply force direction with magnitude
            new_points[i] -= step_size * gradients[i]
            
            # Project back to sphere
            norm = np.linalg.norm(new_points[i])
            if norm > 0:
                new_points[i] = new_points[i] / norm
        
        # Compute new ratio
        new_ratio = compute_min_max_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
            points = new_points
            current_ratio = new_ratio
            
            # Update best solution if improved
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
        
        # Cool down temperature
        temp *= cooling_rate
        iter_count += 1
    
    # Phase 2: Local optimization focusing on distance ratio improvements
    # Decrease temperature for more focused search
    temp = 0.1
    max_iter_phase2 = 50000
    
    while temp > min_temp and iter_count < max_iter_phase1 + max_iter_phase2 and time.time() - start_time < max_time:
        # Create new candidate by targeted perturbations
        new_points = points.copy()
        
        # Select point for perturbation based on how close it is to neighbors
        distances = pdist(points)
        distance_matrix = np.zeros((len(points), len(points)))
        distance_matrix.flat[::len(points)+1] = 0  # Zero diagonal
        
        # Fill matrix
        k = 0
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    distance_matrix[i, j] = distances[k]
                    k += 1
        
        # Find points with neighbors that are too close (potential bottlenecks)
        avg_distances = np.mean(distance_matrix, axis=1)
        min_distances = np.min(distance_matrix, axis=1)
        close_neighbor_ratio = min_distances / (avg_distances + 1e-10)
        
        # Prefer points that are surrounded by too-close neighbors
        weights = 1.0 - close_neighbor_ratio  # Higher weight for those with tight neighbors
        weights = np.clip(weights, 0, 1)
        
        # Choose point with probability proportional to how tight its neighborhood is
        probabilities = weights / np.sum(weights)
        point_to_move = np.random.choice(len(points), p=probabilities)
        
        # Apply physics-inspired perturbation based on local geometry
        # If point is too close to neighbors, repel it more strongly
        # If point is well-spaced, adjust gently
        
        # Get neighbors within some threshold
        neighbor_threshold = np.percentile(distances, 20)  # 20th percentile as threshold
        neighbors = np.where(distance_matrix[point_to_move] < neighbor_threshold)[0]
        
        if len(neighbors) > 0:
            # Point has close neighbors - apply strong repulsion
            repulsion_force = np.zeros(3)
            for neighbor_idx in neighbors:
                diff = points[point_to_move] - points[neighbor_idx]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    repulsion_force += diff / dist / (dist * dist + 1e-8)
            
            if np.linalg.norm(repulsion_force) > 0:
                repulsion_force = repulsion_force / np.linalg.norm(repulsion_force)
                magnitude = 0.005 * temp
                new_points[point_to_move] += repulsion_force * magnitude
            else:
                # Fallback
                new_points[point_to_move] += np.random.normal(0, 0.002, 3)
        else:
            # Well-spaced point - gentle adjustment
            new_points[point_to_move] += np.random.normal(0, 0.001, 3)
        
        # Ensure it stays on sphere
        norm = np.linalg.norm(new_points[point_to_move])
        if norm > 0:
            new_points[point_to_move] = new_points[point_to_move] / norm
        
        # Compute new ratio
        new_ratio = compute_min_max_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
            points = new_points
            current_ratio = new_ratio
            
            # Update best solution if improved
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
        
        # Cool down temperature
        temp *= cooling_rate
        iter_count += 1
    
    return best_points, best_ratio

# EVOLVE-BLOCK-END