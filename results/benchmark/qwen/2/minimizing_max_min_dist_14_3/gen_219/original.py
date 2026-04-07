# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial import SphericalVoronoi
import time
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    # Initialize points using Fibonacci sphere for good spread
    points = fibonacci_sphere(14)
    
    # Optimize using advanced spherical Voronoi-based approach
    best_points, best_ratio = optimize_with_spherical_voronoi(points)
    
    return best_points

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
        return 0.0, 0.0
    
    distances = pdist(points)
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    if d_max == 0:
        return 0.0, 0.0
    
    return d_min / d_max, d_min

def compute_spherical_voronoi_constraints(points):
    """Analyze current configuration using spherical Voronoi to identify constraints"""
    try:
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points)
        
        # Compute Voronoi cell areas and identify small cells (indicating dense regions)
        cell_areas = sv.volume
        mean_area = np.mean(cell_areas)
        
        # Find points in cells that are below 15% of average area (more sensitive threshold)
        dense_point_indices = np.where(cell_areas < 0.15 * mean_area)[0]
        
        # Also identify points that have many close neighbors using stricter threshold
        distances = pdist(points)
        distance_matrix = squareform(distances)
        # Use 5th percentile instead of 10th for more sensitive detection
        close_threshold = np.percentile(distances, 5)
        close_neighbor_counts = np.sum(distance_matrix < close_threshold, axis=1)
        
        # Combine criteria for constraint identification
        constraint_scores = np.zeros(len(points))
        for i in range(len(points)):
            # Score based on close neighbors (weighted more heavily)
            constraint_scores[i] = close_neighbor_counts[i] * 2.0
            # Bonus for being in a dense region (higher weight)
            if i in dense_point_indices:
                constraint_scores[i] += 15.0
                
        return constraint_scores
    except:
        # Fallback if Voronoi computation fails
        return np.ones(len(points)) * 1.0

def select_point_for_perturbation(points, constraint_scores=None):
    """Select point to perturb based on constraint scores or fallback to random"""
    if constraint_scores is not None and np.any(constraint_scores > 0):
        # Select point with highest constraint score (most problematic)
        # But with some randomness to maintain exploration
        # Use weighted probability based on constraint scores
        weights = constraint_scores + 1e-8  # Add small value to avoid zeros
        weights = weights / np.sum(weights)  # Normalize weights
        
        # 80% chance to select based on weights, 20% random
        if np.random.random() < 0.8:
            selected_idx = np.random.choice(len(points), p=weights)
        else:
            # Select randomly from top 3 constrained points
            top_indices = np.argsort(constraint_scores)[-3:]
            selected_idx = np.random.choice(top_indices)
    else:
        # Fallback to random selection
        selected_idx = np.random.randint(0, len(points))
    
    return selected_idx

def compute_targeted_perturbation(points, point_idx, constraint_scores):
    """Compute a targeted perturbation that should improve the configuration"""
    # Determine if we should expand or contract based on local geometry
    distances = cdist([points[point_idx]], points)[0]
    # Remove self-distance
    distances = distances[distances > 0]
    
    if len(distances) == 0:
        # No neighbors, random perturb
        perturbation = np.random.normal(0, 0.015, 3)
        return perturbation
    
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    
    # If point is too close to others, push away
    if min_distance < avg_distance * 0.4:
        # Move away from nearby points
        # Calculate repulsion vector from close neighbors with stronger weighting
        repulsion = np.zeros(3)
        # Consider neighbors within 0.6 of average distance
        for i in range(len(points)):
            if i != point_idx and distances[i-1] < avg_distance * 0.6:  
                diff = points[point_idx] - points[i]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    # Use inverse square for stronger repulsion at closer distances
                    repulsion += diff / (dist * dist + 1e-8) 
        
        if np.linalg.norm(repulsion) > 0:
            repulsion = repulsion / np.linalg.norm(repulsion)
            # Increase magnitude for stronger correction
            magnitude = 0.03
            return repulsion * magnitude
        else:
            # Fallback to random perturbation
            return np.random.normal(0, 0.015, 3)
    else:
        # If well-spaced, adjust based on proximity to center
        center_dist = np.linalg.norm(points[point_idx])
        # Pull toward optimal radial distribution (around 0.5)
        radial_adjust = 0.5 - center_dist
        radial_perturbation = np.array([0, 0, 0])
        if center_dist > 0:
            radial_perturbation = points[point_idx] * radial_adjust / center_dist
        
        # Add small random perturbation for exploration
        random_perturbation = np.random.normal(0, 0.008, 3)
        
        return radial_perturbation + random_perturbation

def optimize_with_spherical_voronoi(initial_points, max_time=360):
    """Optimize using spherical Voronoi analysis combined with adaptive simulated annealing"""
    points = initial_points.copy()
    current_ratio, min_dist = compute_min_max_ratio(points)
    
    # Parameters for adaptive simulated annealing
    temp = 1.0
    min_temp = 1e-12
    base_cooling_rate = 0.9995
    max_iter = 1000000
    
    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    best_min_dist = min_dist
    
    start_time = time.time()
    iter_count = 0
    
    # Track convergence for adaptive cooling
    recent_ratios = []
    max_recent = 50
    
    # Store recent improvements for adaptive cooling
    last_improvement_iter = 0
    
    while temp > min_temp and iter_count < max_iter and time.time() - start_time < max_time:
        # Compute constraint scores for current configuration
        constraint_scores = compute_spherical_voronoi_constraints(points)
        
        # Select point to perturb
        point_to_move = select_point_for_perturbation(points, constraint_scores)
        
        # Create new candidate point with targeted perturbation
        new_points = points.copy()
        perturbation = compute_targeted_perturbation(points, point_to_move, constraint_scores)
        new_points[point_to_move] += perturbation
        
        # Project back onto sphere
        norm = np.linalg.norm(new_points[point_to_move])
        if norm > 0:
            new_points[point_to_move] = new_points[point_to_move] / norm
        
        # Compute new ratio
        new_ratio, new_min_dist = compute_min_max_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
            points = new_points
            current_ratio = new_ratio
            min_dist = new_min_dist
            
            # Update best solution if improved
            if new_ratio > best_ratio:
                best_points = new_points.copy()
                best_ratio = new_ratio
                best_min_dist = new_min_dist
                last_improvement_iter = iter_count
                
        # Adaptive cooling based on recent performance
        recent_ratios.append(current_ratio)
        if len(recent_ratios) > max_recent:
            recent_ratios.pop(0)
            
        # Adjust cooling rate based on recent improvement and stagnation
        cooling_rate = base_cooling_rate
        if len(recent_ratios) >= 10:
            recent_improvement = recent_ratios[-1] - recent_ratios[0]
            if recent_improvement < 1e-8:
                # Very slow progress, cool faster
                cooling_rate = base_cooling_rate * 1.15
            elif recent_improvement > 1e-5:
                # Fast progress, cool slower
                cooling_rate = base_cooling_rate * 0.98
            elif iter_count - last_improvement_iter > 20000:
                # No improvement in a while, cool faster to escape local minima
                cooling_rate = base_cooling_rate * 1.1
        
        # Apply cooling
        temp *= cooling_rate
        iter_count += 1
        
        # Periodic refinement - every 1000 iterations with increasing frequency
        if iter_count % (max(1000, 5000 - iter_count // 100)) == 0 and iter_count > 0:
            # Do a more intensive local search for current best
            refined_points = points.copy()
            for _ in range(100):
                point_idx = np.random.randint(0, len(points))
                # Small perturbation to see if we can improve
                new_point = refined_points[point_idx] + np.random.normal(0, 0.0005, 3)
                norm = np.linalg.norm(new_point)
                if norm > 0:
                    new_point = new_point / norm
                refined_points[point_idx] = new_point
                
                new_ratio, _ = compute_min_max_ratio(refined_points)
                if new_ratio > current_ratio:
                    points = refined_points.copy()
                    current_ratio = new_ratio
            # Reset temp to allow for further exploration after refinement
            temp = min(temp * 1.2, 1.0)
    
    return best_points, best_ratio

# EVOLVE-BLOCK-END