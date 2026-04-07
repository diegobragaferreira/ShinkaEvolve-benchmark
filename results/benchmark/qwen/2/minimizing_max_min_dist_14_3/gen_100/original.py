# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.vq import kmeans2
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
    
    # Multi-phase initialization strategy
    initial_points = generate_multi_resolution_initialization()
    
    # Advanced optimization with progressive refinement
    best_points, best_ratio = optimize_with_progressive_refinement(initial_points)
    
    return best_points

def generate_multi_resolution_initialization():
    """Generate diverse initial configurations using multiple methods"""
    # Phase 1: Fibonacci sphere initialization
    fib_points = fibonacci_sphere(14)
    
    # Phase 2: Icosahedral-inspired arrangement
    ico_points = generate_icosahedral_points()
    
    # Phase 3: Hybrid combination
    # Mix the two approaches with different weights
    mixed_points = 0.6 * fib_points + 0.4 * ico_points
    # Normalize to unit sphere
    norms = np.linalg.norm(mixed_points, axis=1)
    mixed_points = mixed_points / norms[:, np.newaxis]
    
    # Add slight random perturbations to ensure diversity
    noise = np.random.normal(0, 0.01, (14, 3))
    mixed_points += noise
    
    # Normalize again
    norms = np.linalg.norm(mixed_points, axis=1)
    mixed_points = mixed_points / norms[:, np.newaxis]
    
    return mixed_points

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

def generate_icosahedral_points():
    """Generate points arranged like an icosahedron for better symmetry"""
    # Icosahedron vertices scaled to unit sphere
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    points = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    # Convert to numpy array and normalize
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    # We need 14 points, so add two more from edge midpoints
    # For simplicity, we'll take the first two points and create additional ones
    # by averaging pairs and normalizing
    additional_points = []
    for i in range(0, len(points), 2):
        if len(additional_points) < 2:
            mid = (points[i] + points[i+1]) / 2
            norm = np.linalg.norm(mid)
            if norm > 0:
                additional_points.append(mid / norm)
    
    # Combine all points
    all_points = np.vstack([points[:12], additional_points])
    
    # Ensure exactly 14 points by padding or truncating
    if len(all_points) < 14:
        # Fill with random points on sphere
        extra_points = fibonacci_sphere(14 - len(all_points))
        all_points = np.vstack([all_points, extra_points])
    elif len(all_points) > 14:
        all_points = all_points[:14]
        
    return all_points

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

def compute_clustering_based_perturbation(points):
    """Compute optimal perturbations based on k-means clustering analysis"""
    if len(points) < 4:
        return np.random.randint(0, len(points))
    
    try:
        # Use k-means clustering to find dense regions
        # Try different numbers of clusters to find the most informative partition
        best_clusters = 4
        best_inertia = float('inf')
        
        # Test various k values
        for k in [2, 3, 4, 5]:
            try:
                centroids, labels = kmeans2(points, k, minit='random')
                # Compute inertia (sum of squared distances to centroids)
                inertia = 0
                for i in range(k):
                    cluster_points = points[labels == i]
                    if len(cluster_points) > 0:
                        inertia += np.sum(np.square(cluster_points - centroids[i]))
                
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_clusters = k
            except:
                continue
        
        # Run final clustering with best k
        centroids, labels = kmeans2(points, best_clusters, minit='random')
        
        # Identify dense clusters (clusters with high point density)
        cluster_sizes = np.bincount(labels)
        density_threshold = np.percentile(cluster_sizes, 20)  # 20th percentile
        dense_cluster_indices = np.where(cluster_sizes >= density_threshold)[0]
        
        # Identify points in dense clusters
        dense_point_indices = []
        for cluster_idx in dense_cluster_indices:
            cluster_points = np.where(labels == cluster_idx)[0]
            dense_point_indices.extend(cluster_points.tolist())
        
        # Create a density-based scoring system
        density_scores = np.zeros(len(points))
        
        # Score points based on how they relate to dense clusters
        for i in range(len(points)):
            # How many neighbors are in dense clusters?
            distances = cdist([points[i]], points)[0]
            distances = distances[distances > 0]  # Remove self-distance
            
            if len(distances) > 0:
                # Find nearest neighbors
                nearest_indices = np.argsort(distances)[:min(5, len(distances))]
                # Count how many of these are in dense clusters
                dense_neighbors = sum(1 for idx in nearest_indices if idx in dense_point_indices)
                
                # Score based on neighbor density
                density_scores[i] = dense_neighbors / max(1, len(nearest_indices))
                
                # Bonus for being in a dense cluster
                if labels[i] in dense_cluster_indices:
                    density_scores[i] += 1.0
        
        # Choose point with highest density score (but with some randomness)
        if np.max(density_scores) > 0:
            # Prefer higher-scoring points but with 30% randomness
            if np.random.random() < 0.7:
                chosen_idx = np.argmax(density_scores)
            else:
                # Select randomly from top 3 high-scoring points  
                top_indices = np.argsort(density_scores)[-3:]
                chosen_idx = np.random.choice(top_indices)
        else:
            # If no strong constraints, choose randomly
            chosen_idx = np.random.randint(0, len(points))

        return chosen_idx

    except Exception as e:
        # Fallback to random selection if clustering fails
        return np.random.randint(0, len(points))

def compute_targeted_directional_perturbation(points, point_idx, prev_ratio):
    """Compute a directional perturbation based on geometric analysis"""
    try:
        # Get distances to all other points
        distances = cdist([points[point_idx]], points)[0]
        distances = distances[distances > 0]  # Remove self-distance
        
        if len(distances) == 0:
            # No neighbors, random perturbation
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            magnitude = 0.02
            return direction * magnitude
            
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Analyze the point's local geometry
        # If point is too close to others, push it away
        if min_distance < avg_distance * 0.4:
            # Compute repulsion force from close neighbors
            repulsion = np.zeros(3)
            for i in range(len(points)):
                if i != point_idx:
                    diff = points[point_idx] - points[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0 and dist < avg_distance * 0.7:
                        # Inverse distance weighted repulsion
                        repulsion += diff / dist * (1.0 / dist**2)
            
            # If there's repulsion, normalize and apply
            if np.linalg.norm(repulsion) > 0:
                repulsion = repulsion / np.linalg.norm(repulsion)
                # Magnitude depends on how close it is
                magnitude = 0.05 * (1.0 - min_distance / avg_distance)
                return repulsion * magnitude
            else:
                # Fallback to random perturbation
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                return direction * 0.02
                
        elif max_distance > avg_distance * 1.5:
            # If point is far from others, maybe pull it closer to balance
            # Compute attraction toward average position
            attraction = np.mean(points, axis=0) - points[point_idx]
            attraction_norm = np.linalg.norm(attraction)
            if attraction_norm > 0:
                attraction = attraction / attraction_norm
                # Magnitude inversely related to distance from center
                center_distance = np.linalg.norm(points[point_idx])
                magnitude = 0.01 * (1.0 - center_distance * 0.5)
                return -attraction * magnitude
            else:
                # Random perturbation
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                return direction * 0.01
                
        else:
            # Moderate distances - small adjustment
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            # Magnitude based on how balanced the distances are
            balance_score = abs(min_distance - avg_distance) / avg_distance + \
                           abs(max_distance - avg_distance) / avg_distance
            magnitude = 0.01 * (1.0 - balance_score * 0.5)
            return direction * magnitude
            
    except Exception:
        # Fallback to simple random perturbation
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        return direction * 0.01

def optimize_with_progressive_refinement(initial_points, max_time=360):
    """Optimize using progressive refinement approach"""
    points = initial_points.copy()
    current_ratio, min_dist = compute_min_max_ratio(points)
    
    # Progressive phases: coarse, medium, fine
    phase = 0
    max_iter_per_phase = 150000
    total_iterations = 0
    
    # Parameters for simulated annealing
    temp = 1.0
    min_temp = 1e-12
    base_cooling_rate = 0.9995
    max_total_iter = 500000
    
    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    best_min_dist = min_dist
    
    start_time = time.time()
    iter_count = 0
    
    # Track recent ratios for adaptive cooling
    recent_ratios = []
    max_recent = 30
    
    # Progressive refinement phases
    phases = [
        {"temp": 1.0, "cooling_rate": 0.9995, "perturb_factor": 1.0, "description": "coarse"},
        {"temp": 0.7, "cooling_rate": 0.9998, "perturb_factor": 0.8, "description": "medium"},
        {"temp": 0.3, "cooling_rate": 0.9999, "perturb_factor": 0.5, "description": "fine"}
    ]
    
    while temp > min_temp and total_iterations < max_total_iter and time.time() - start_time < max_time:
        # Determine current phase based on iteration count
        if total_iterations < max_iter_per_phase:
            phase = 0
        elif total_iterations < 2 * max_iter_per_phase:
            phase = 1
        else:
            phase = 2
            
        current_phase = phases[phase]
        
        # Select point to perturb
        point_to_move = compute_clustering_based_perturbation(points)
        
        # Create new candidate point
        new_points = points.copy()
        
        # Apply targeted directional perturbation
        perturbation = compute_targeted_directional_perturbation(
            points, point_to_move, current_ratio)
        
        new_points[point_to_move] += perturbation * current_phase["perturb_factor"]
        
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
                
        # Adaptive cooling based on recent performance
        recent_ratios.append(current_ratio)
        if len(recent_ratios) > max_recent:
            recent_ratios.pop(0)
            
        # Adjust cooling rate based on performance trends
        cooling_rate = current_phase["cooling_rate"]
        if len(recent_ratios) >= 10:
            recent_improvement = recent_ratios[-1] - recent_ratios[0]
            if recent_improvement < 1e-7:
                # Very slow progress, cool faster
                cooling_rate = current_phase["cooling_rate"] * 1.1
            elif recent_improvement > 1e-4 and phase < 2:
                # Fast progress, cool slower (but not in final phase)
                cooling_rate = current_phase["cooling_rate"] * 0.99
                
        # Apply temperature cooling
        temp *= cooling_rate
        iter_count += 1
        total_iterations += 1
        
        # Periodically restart if no significant improvement
        if iter_count % 2000 == 0 and iter_count > 0:
            # Check if recent ratios have stagnated
            if len(recent_ratios) >= 10:
                improvement = recent_ratios[-1] - recent_ratios[-5]
                if improvement < 1e-6:
                    # Restart with better configuration
                    points = generate_multi_resolution_initialization()
                    current_ratio, min_dist = compute_min_max_ratio(points)
                    temp = min(temp * 1.5, 1.0)  # Increase temperature
                    
    return best_points, best_ratio

# EVOLVE-BLOCK-END