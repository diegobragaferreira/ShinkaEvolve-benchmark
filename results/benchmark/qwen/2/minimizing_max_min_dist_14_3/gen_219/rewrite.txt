# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from scipy.spatial import SphericalVoronoi
from scipy.optimize import minimize
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    # Multi-stage optimization approach
    # Stage 1: Coarse optimization with 8 points to establish good starting configuration
    coarse_points = initialize_coarse_config(8)
    
    # Stage 2: Refine using energy-based optimization
    refined_points = optimize_energy_based(coarse_points, 1000)
    
    # Stage 3: Fine-tune with targeted geometric optimization
    final_points = optimize_geometric_targeted(refined_points, 2000)
    
    # Stage 4: Final refinement using constrained optimization
    best_points = finalize_with_constraints(final_points)
    
    return best_points

def initialize_coarse_config(n_points):
    """Initialize with a coarse configuration based on icosahedral symmetry"""
    # Start with icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        (-1, 0, phi), (1, 0, phi), (-1, 0, -phi), (1, 0, -phi),
        (0, phi, 1), (0, phi, -1), (0, -phi, 1), (0, -phi, -1),
        (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)
    ])
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices[0])
    
    # For 8 points, take first 8 vertices and add some perturbations
    points = vertices[:8].copy()
    
    # Add random perturbations to get better spread
    for i in range(len(points)):
        perturbation = np.random.normal(0, 0.05, 3)
        points[i] += perturbation
        # Project back to sphere
        norm = np.linalg.norm(points[i])
        if norm > 0:
            points[i] = points[i] / norm
    
    return points

def compute_energy_score(points):
    """Compute energy score based on inverse distance squared (repulsion energy)"""
    if len(points) < 2:
        return 0.0
    
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    
    # Energy is sum of inverse squared distances (repulsion energy)
    energy = np.sum(1.0 / (distances ** 2 + 1e-10))
    return energy

def optimize_energy_based(initial_points, max_iter):
    """Optimize using energy-based minimization with adaptive cooling"""
    points = initial_points.copy()
    best_points = points.copy()
    best_energy = compute_energy_score(points)
    
    # Adaptive cooling schedule
    temp = 1.0
    cooling_rate = 0.999
    min_temp = 1e-6
    
    for iteration in range(max_iter):
        # Create candidate by perturbing a random point
        new_points = points.copy()
        point_idx = np.random.randint(len(points))
        
        # Adaptive perturbation based on current configuration
        current_energy = compute_energy_score(points)
        perturbation_magnitude = min(0.1, max(0.001, temp * 0.02))
        
        # More aggressive perturbation when energy is high (bad configuration)
        if current_energy > 100:
            perturbation_magnitude *= 2.0
            
        perturbation = np.random.normal(0, perturbation_magnitude, 3)
        new_points[point_idx] += perturbation
        
        # Project back to sphere
        norm = np.linalg.norm(new_points[point_idx])
        if norm > 0:
            new_points[point_idx] = new_points[point_idx] / norm
        
        new_energy = compute_energy_score(new_points)
        
        # Accept with Metropolis criterion
        if new_energy < best_energy or np.random.rand() < np.exp(-(new_energy - best_energy) / temp):
            points = new_points
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
        
        # Cool temperature
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp
    
    return best_points

def optimize_geometric_targeted(initial_points, max_iter):
    """Targeted geometric optimization focused on improving min/max ratio"""
    points = initial_points.copy()
    best_points = points.copy()
    
    # Precompute some geometric properties
    prev_ratio = 0.0
    stall_count = 0
    
    for iteration in range(max_iter):
        # Analyze current configuration
        distances = pdist(points)
        if len(distances) == 0:
            continue
            
        current_ratio = np.min(distances) / (np.max(distances) + 1e-10)
        
        # Check for stagnation
        if abs(current_ratio - prev_ratio) < 1e-8:
            stall_count += 1
        else:
            stall_count = 0
            
        # Reset if stuck
        if stall_count > 500:
            points = initialize_coarse_config(len(points))
            stall_count = 0
        
        prev_ratio = current_ratio
        
        # Create candidate by targeting problematic points
        new_points = points.copy()
        
        # Focus on points with extreme distances
        distance_matrix = squareform(distances)
        mean_distances = np.mean(distance_matrix, axis=1)
        
        # Identify points that are too close or too far
        too_close_indices = np.where(mean_distances < np.percentile(mean_distances, 30))[0]
        too_far_indices = np.where(mean_distances > np.percentile(mean_distances, 70))[0]
        
        # Select which point to modify
        if len(too_close_indices) > 0 and np.random.rand() < 0.5:
            point_idx = np.random.choice(too_close_indices)
            # Repel from neighbors
            repulsion = np.zeros(3)
            for i in range(len(points)):
                if i != point_idx:
                    diff = points[point_idx] - points[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        repulsion += diff / (dist * dist + 1e-8)
            
            if np.linalg.norm(repulsion) > 0:
                repulsion = repulsion / np.linalg.norm(repulsion)
                new_points[point_idx] += repulsion * 0.02
        else:
            # Random point with small perturbation
            point_idx = np.random.randint(len(points))
            perturbation = np.random.normal(0, 0.01, 3)
            new_points[point_idx] += perturbation
        
        # Project back to sphere
        norm = np.linalg.norm(new_points[point_idx])
        if norm > 0:
            new_points[point_idx] = new_points[point_idx] / norm
        
        # Evaluate improvement
        new_distances = pdist(new_points)
        if len(new_distances) > 0:
            new_ratio = np.min(new_distances) / (np.max(new_distances) + 1e-10)
            
            # Accept if improvement
            if new_ratio > current_ratio:
                points = new_points
                best_points = new_points.copy()
    
    return best_points

def finalize_with_constraints(initial_points):
    """Final optimization using constrained optimization techniques"""
    points = initial_points.copy()
    
    # Function to minimize (negative of ratio to convert to maximization)
    def objective(x):
        current_points = x.reshape(-1, 3)
        distances = pdist(current_points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        # Return negative ratio to maximize
        return -min_dist / (max_dist + 1e-10)
    
    # Constraint function: points must lie on unit sphere
    def constraint_func(x):
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1.0 - norms  # Positive when norm <= 1
    
    # Initial flatten
    x0 = points.flatten()
    
    # Use L-BFGS-B for final refinement
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                         constraints={'type': 'ineq', 'fun': constraint_func},
                         options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8},
                         bounds=[(-1, 1)] * 42)
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            # Ensure normalization
            for i in range(len(final_points)):
                norm = np.linalg.norm(final_points[i])
                if norm > 0:
                    final_points[i] = final_points[i] / norm
            return final_points
    except:
        pass
    
    # Fallback to simple iterative improvement
    return points

# EVOLVE-BLOCK-END