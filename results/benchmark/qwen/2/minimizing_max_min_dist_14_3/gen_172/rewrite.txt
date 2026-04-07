# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
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
    
    # Multi-stage optimization approach
    # Stage 1: Initial coarse positioning using icosahedral symmetry
    points = _generate_icosahedral_seed(14)
    
    # Stage 2: Voronoi-guided refinement with adaptive cooling
    points = _voronoi_guided_optimization(points)
    
    # Stage 3: Final refinement with constraint-aware optimization
    points = _constraint_aware_refinement(points)
    
    return points

def _generate_icosahedral_seed(n):
    """Generate initial seed points using icosahedral symmetry for better structure"""
    # Create basic icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    vertices = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    # Normalize to unit sphere
    vertices = np.array(vertices)
    norms = np.linalg.norm(vertices, axis=1)
    vertices = vertices / norms[:, np.newaxis]
    
    # Use vertices as initial points and add extra points
    if n <= 12:
        return vertices[:n]
    else:
        # Add more points by perturbing existing ones
        points = vertices.copy()
        extra_needed = n - 12
        for i in range(extra_needed):
            # Add points near existing ones with small perturbations
            base_idx = i % 12
            perturbation = np.random.normal(0, 0.05, 3)
            new_point = points[base_idx] + perturbation
            # Project back to sphere
            norm = np.linalg.norm(new_point)
            if norm > 0:
                new_point = new_point / norm
            points = np.vstack([points, new_point])
        return points

def _voronoi_guided_optimization(initial_points, max_time=300):
    """Optimize using Voronoi-based constraint analysis with adaptive cooling"""
    points = initial_points.copy()
    
    # Parameters
    temp = 0.1
    min_temp = 1e-10
    cooling_rate = 0.999
    max_iter = 50000
    start_time = time.time()
    
    # Track best solution
    best_points = points.copy()
    best_ratio = _compute_min_max_ratio(points)
    
    # Track convergence
    recent_ratios = []
    max_recent = 100
    
    iter_count = 0
    while temp > min_temp and iter_count < max_iter and time.time() - start_time < max_time:
        # Analyze current configuration using spherical Voronoi
        voronoi_constraints = _analyze_voronoi_constraints(points)
        
        # Select point based on Voronoi analysis
        point_idx = _select_point_by_voronoi(voronoi_constraints)
        
        # Compute targeted perturbation
        perturbation = _compute_targeted_perturbation(points, point_idx, voronoi_constraints, temp)
        
        # Apply perturbation
        new_points = points.copy()
        new_points[point_idx] += perturbation
        
        # Project back to unit sphere
        norm = np.linalg.norm(new_points[point_idx])
        if norm > 0:
            new_points[point_idx] = new_points[point_idx] / norm
        
        # Evaluate new configuration
        new_ratio = _compute_min_max_ratio(new_points)
        
        # Accept or reject based on Metropolis criterion
        if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) / temp):
            points = new_points
            if new_ratio > best_ratio:
                best_points = points.copy()
                best_ratio = new_ratio
        
        # Adaptive cooling based on convergence
        recent_ratios.append(new_ratio)
        if len(recent_ratios) > max_recent:
            recent_ratios.pop(0)
        
        # Adjust cooling rate based on recent improvements
        if len(recent_ratios) >= 10:
            recent_improvement = recent_ratios[-1] - recent_ratios[0]
            if recent_improvement < 1e-8:
                cooling_rate = min(0.9999, cooling_rate * 1.05)  # Speed up cooling
            elif recent_improvement > 1e-4:
                cooling_rate = max(0.999, cooling_rate * 0.98)  # Slow down cooling
        
        temp *= cooling_rate
        iter_count += 1
        
        # Restart if stagnation detected
        if iter_count % 5000 == 0 and len(recent_ratios) >= 10:
            recent_change = np.std(recent_ratios[-10:])
            if recent_change < 1e-6:
                # Restart with better configuration
                points = best_points.copy()
                temp = min(temp * 1.5, 0.5)
                recent_ratios = []
    
    return best_points

def _analyze_voronoi_constraints(points):
    """Analyze current configuration using spherical Voronoi to identify constraints"""
    try:
        # Create spherical Voronoi diagram
        sv = SphericalVoronoi(points)
        
        # Compute Voronoi cell areas
        cell_areas = sv.volume
        mean_area = np.mean(cell_areas)
        
        # Identify points in cells that are below 30% of average area (dense regions)
        dense_point_indices = np.where(cell_areas < 0.3 * mean_area)[0]
        
        # Also identify points that have many close neighbors
        distances = pdist(points)
        distance_matrix = squareform(distances)
        close_neighbor_counts = np.sum(distance_matrix < np.percentile(distances, 15), axis=1)
        
        # Combine criteria for constraint identification
        constraint_scores = np.zeros(len(points))
        
        # Score based on close neighbors (scaled)
        constraint_scores += close_neighbor_counts * 0.8
        
        # Bonus for being in a dense region
        for idx in dense_point_indices:
            constraint_scores[idx] += 3.0
            
        return constraint_scores
    except:
        # Fallback if Voronoi computation fails
        return np.ones(len(points)) * 0.5

def _select_point_by_voronoi(constraint_scores):
    """Select point to perturb based on constraint scores"""
    # Prefer higher-scoring points but with some randomness
    if np.max(constraint_scores) > 0:
        # 75% chance to select the most constrained point
        if np.random.random() < 0.75:
            selected_idx = np.argmax(constraint_scores)
        else:
            # 25% chance to select randomly from top 4 constrained
            top_indices = np.argsort(constraint_scores)[-4:]
            selected_idx = np.random.choice(top_indices)
    else:
        # If no strong constraints, choose randomly
        selected_idx = np.random.randint(0, len(constraint_scores))
    
    return selected_idx

def _compute_targeted_perturbation(points, point_idx, constraint_scores, temp):
    """Compute a targeted perturbation based on Voronoi analysis and local geometry"""
    try:
        # Get all distances for this point
        distances = pdist(points)
        distance_matrix = squareform(distances)
        
        # Extract distances to neighbors (excluding self)
        neighbor_distances = distance_matrix[point_idx]
        neighbor_distances = neighbor_distances[neighbor_distances > 0]
        
        if len(neighbor_distances) == 0:
            # Fallback to random perturbation
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            return direction * 0.02 * temp
        
        avg_distance = np.mean(neighbor_distances)
        min_distance = np.min(neighbor_distances)
        max_distance = np.max(neighbor_distances)
        
        # Determine if this point is well-positioned
        constraint_weight = constraint_scores[point_idx]
        is_dense_region = constraint_weight > 2.0
        
        # Compute perturbation based on geometric analysis
        if is_dense_region or min_distance < avg_distance * 0.5:
            # Point is in dense region or too close to neighbors - repel it
            repulsion = np.zeros(3)
            for i in range(len(points)):
                if i != point_idx:
                    diff = points[point_idx] - points[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0 and dist < avg_distance * 0.8:
                        # Weighted repulsion based on inverse distance squared
                        repulsion += diff / (dist * dist + 1e-8)
            
            if np.linalg.norm(repulsion) > 0:
                repulsion = repulsion / np.linalg.norm(repulsion)
                # Magnitude based on how much it violates spacing requirements
                violation_factor = min_distance / avg_distance
                magnitude = 0.05 * temp * (1.0 + 2.0 * (1.0 - violation_factor))
                return repulsion * magnitude
            else:
                # Fallback to random perturbation
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                return direction * 0.02 * temp
                
        elif max_distance > avg_distance * 1.3:
            # Point is too far from others - attract it
            attraction = np.mean(points, axis=0) - points[point_idx]
            attraction_norm = np.linalg.norm(attraction)
            if attraction_norm > 0:
                attraction = attraction / attraction_norm
                magnitude = 0.03 * temp
                return -attraction * magnitude
            else:
                # Fallback to random perturbation
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                return direction * 0.015 * temp
        else:
            # Point has reasonable spacing - small adjustment
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            # Reduce magnitude for stable configuration
            magnitude = 0.01 * temp
            return direction * magnitude
            
    except Exception:
        # Fallback to simple random perturbation
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        return direction * 0.015 * temp

def _compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances"""
    if len(points) < 2:
        return 0.0
    
    distances = pdist(points)
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    if d_max == 0:
        return 0.0
    
    return d_min / d_max

def _constraint_aware_refinement(initial_points, max_iter=5000):
    """Final refinement step focused on constraint satisfaction"""
    points = initial_points.copy()
    current_ratio = _compute_min_max_ratio(points)
    
    # More aggressive refinement with smaller temperature
    temp = 0.02
    min_temp = 1e-12
    cooling_rate = 0.9995
    
    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    
    iter_count = 0
    while temp > min_temp and iter_count < max_iter:
        # Analyze constraints
        constraint_scores = _analyze_voronoi_constraints(points)
        
        # Select point to perturb
        point_idx = _select_point_by_voronoi(constraint_scores)
        
        # Compute precise perturbation
        perturbation = _compute_targeted_perturbation(points, point_idx, constraint_scores, temp)
        
        # Apply perturbation
        new_points = points.copy()
        new_points[point_idx] += perturbation
        
        # Project back to sphere
        norm = np.linalg.norm(new_points[point_idx])
        if norm > 0:
            new_points[point_idx] = new_points[point_idx] / norm
        
        # Evaluate
        new_ratio = _compute_min_max_ratio(new_points)
        
        # Accept or reject
        if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
            points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_points = points.copy()
                best_ratio = new_ratio
        
        temp *= cooling_rate
        iter_count += 1
    
    return best_points

# EVOLVE-BLOCK-END