# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
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

    # Initialize points using optimized Fibonacci sphere method
    points = fibonacci_sphere(14)

    # Optimize using enhanced spherical simulated annealing
    best_points, _ = optimize_with_spherical_annealing(points)

    # Project to unit cube [0,1]^3
    points_in_cube = project_to_unit_cube(best_points)
    
    return points_in_cube

def fibonacci_sphere(n):
    """Generate points on sphere using Fibonacci spiral method with improved numerical stability"""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(max(0, 1 - y * y))  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum pairwise distances efficiently"""
    if len(points) < 2:
        return 0.0, 0.0

    distances = pdist(points)
    if len(distances) == 0:
        return 0.0, 0.0

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

        # Find points in cells that are below 20% of average area (dense regions)
        dense_point_indices = np.where(cell_areas < 0.2 * mean_area)[0]

        # Identify points with many close neighbors using distance percentiles
        distances = pdist(points)
        if len(distances) == 0:
            return np.ones(len(points)) * 0.5, np.zeros(len(points))

        distance_matrix = pdist(points)
        percentile_10 = np.percentile(distances, 10)
        percentile_25 = np.percentile(distances, 25)

        # Count neighbors within different thresholds
        close_neighbors_10 = np.sum(distance_matrix < percentile_10, axis=0)
        close_neighbors_25 = np.sum(distance_matrix < percentile_25, axis=0)

        # Combine criteria for constraint identification
        constraint_scores = np.zeros(len(points))
        for i in range(len(points)):
            # Score based on number of very close neighbors (more critical)
            constraint_scores[i] = close_neighbors_10[i] * 2.0
            # Add score for moderately close neighbors
            constraint_scores[i] += close_neighbors_25[i] * 0.5
            # Bonus for being in a dense region
            if i in dense_point_indices:
                constraint_scores[i] += 15.0

        return constraint_scores, close_neighbors_10

    except Exception:
        # Fallback if Voronoi computation fails
        return np.ones(len(points)) * 0.5, np.zeros(len(points))

def select_point_for_perturbation(points, constraint_scores):
    """Select point to perturb based on constraint scores with randomness"""
    # Select point with highest constraint score (most problematic)
    # But with some randomness to maintain exploration
    if np.random.random() < 0.7:
        # 70% chance to select the most constrained point
        selected_idx = np.argmax(constraint_scores)
    else:
        # 30% chance to select randomly from top 3 constrained
        top_indices = np.argsort(constraint_scores)[-3:]
        selected_idx = np.random.choice(top_indices)
    
    return selected_idx

def compute_targeted_perturbation(points, point_idx, constraint_scores, distances_from_others):
    """Compute a targeted perturbation that should improve the configuration"""
    # Determine local geometry characteristics
    distances = distances_from_others[point_idx]
    # Remove self-distance
    distances = distances[distances > 0]

    if len(distances) == 0:
        # No neighbors, random perturb
        return np.random.normal(0, 0.01, 3)

    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    # Calculate constraint severity for this point
    constraint_severity = constraint_scores[point_idx]

    # If point is too close to others, push away strongly
    if min_distance < avg_distance * 0.4:
        # Move away from nearby points with strong repulsion
        repulsion = np.zeros(3)
        total_weight = 0.0

        for i in range(len(points)):
            if i != point_idx:
                diff = points[point_idx] - points[i]
                dist = np.linalg.norm(diff)
                if dist > 0 and dist < avg_distance * 0.8:  # Close enough to matter
                    # Use inverse square law for stronger repulsion at shorter distances
                    weight = 1.0 / (dist * dist + 1e-8)
                    repulsion += diff / dist * weight
                    total_weight += weight

        if total_weight > 0:
            repulsion = repulsion / total_weight
            repulsion = repulsion / np.linalg.norm(repulsion)

            # Magnitude based on how constrained the point is and how close it is
            proximity_factor = min_distance / avg_distance
            magnitude = 0.03 * (1.0 + constraint_severity * 0.1) * (1.0 + proximity_factor * 2.0)
            return repulsion * magnitude
        else:
            # Fallback to random perturbation
            return np.random.normal(0, 0.015, 3)
    elif min_distance > avg_distance * 0.8 and max_distance > avg_distance * 1.5:
        # Point is relatively well-spaced but some are very far - try to pull them together
        attraction = np.zeros(3)
        total_weight = 0.0

        # Attract to points that are moderately close
        for i in range(len(points)):
            if i != point_idx and distances[i-1] < avg_distance * 1.2:
                diff = points[i] - points[point_idx]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    weight = 1.0 / (dist * dist + 1e-8)
                    attraction += diff / dist * weight
                    total_weight += weight

        if total_weight > 0:
            attraction = attraction / total_weight
            attraction = attraction / np.linalg.norm(attraction)
            magnitude = 0.01 * (1.0 + constraint_severity * 0.05)
            return attraction * magnitude
        else:
            # Fallback to small random perturbation
            return np.random.normal(0, 0.005, 3)
    else:
        # Medium spacing situation - small adjustment
        # Add a component that pushes toward the center for points near poles
        center_dist = np.linalg.norm(points[point_idx])
        center_push = np.zeros(3)
        if center_dist < 0.2 or center_dist > 0.8:
            # Push toward middle of sphere
            center_push = -points[point_idx]
            norm = np.linalg.norm(center_push)
            if norm > 0:
                center_push = center_push / norm

        # Normal random component
        random_perturb = np.random.normal(0, 0.008, 3)

        # Blend the two perturbations
        return random_perturb * 0.7 + center_push * 0.3

def project_to_unit_cube(points):
    """Project points to unit cube [0,1]^3"""
    # Find min/max along each axis
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)

    # Handle case where there's no variation
    ranges = max_coords - min_coords
    if np.any(ranges == 0):
        # If any dimension has no variation, return points centered at 0.5
        return np.full_like(points, 0.5)

    # Scale to [0,1] range
    normalized = (points - min_coords) / ranges

    # Ensure they're clipped to [0,1]
    return np.clip(normalized, 0, 1)

def optimize_with_spherical_annealing(initial_points, max_time=360):
    """Optimize using enhanced spherical simulated annealing with adaptive cooling"""
    points = initial_points.copy()
    current_ratio, min_dist = compute_min_max_ratio(points)

    # Parameters for adaptive simulated annealing
    temp = 1.0
    min_temp = 1e-10
    base_cooling_rate = 0.9995
    max_iter = 500000

    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    best_min_dist = min_dist

    start_time = time.time()
    iter_count = 0

    # Track convergence for adaptive cooling
    recent_ratios = []
    max_recent = 50

    # Track improvement steps for early stopping
    last_improvement = 0
    patience = 500  # How many iterations without improvement before cooling faster

    # Precompute distance matrix for efficiency
    distances_matrix = cdist(points, points)

    while temp > min_temp and iter_count < max_iter and time.time() - start_time < max_time:
        # Recompute constraint scores periodically
        if iter_count % 100 == 0:
            constraint_scores, _ = compute_spherical_voronoi_constraints(points)
        else:
            # Approximate for speed - just compute for one point each iteration
            constraint_scores, _ = compute_spherical_voronoi_constraints(points)

        # Select point to perturb
        point_to_move = select_point_for_perturbation(points, constraint_scores)

        # Create new candidate point with targeted perturbation
        new_points = points.copy()
        perturbation = compute_targeted_perturbation(points, point_to_move, constraint_scores, distances_matrix)
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

        # Adaptive cooling based on recent performance
        recent_ratios.append(current_ratio)
        if len(recent_ratios) > max_recent:
            recent_ratios.pop(0)

        # Adjust cooling rate based on recent improvement and early stopping
        if len(recent_ratios) >= 10:
            recent_improvement = recent_ratios[-1] - recent_ratios[0]
            if recent_improvement < 1e-8:
                # Slow progress, cool faster
                cooling_rate = base_cooling_rate * 1.15
            elif recent_improvement > 1e-4:
                # Fast progress, cool slower
                cooling_rate = base_cooling_rate * 0.98
            else:
                cooling_rate = base_cooling_rate
        else:
            cooling_rate = base_cooling_rate

        # Apply cooling
        temp *= cooling_rate
        iter_count += 1

        # Early stopping if no significant improvement
        if iter_count - last_improvement > patience and temp < 1e-6:
            # Reduce temperature more aggressively
            temp *= 0.95

        # Periodic refinement - every 1000 iterations
        if iter_count % 1000 == 0 and iter_count > 0:
            # Do a more intensive local search for current best
            refined_points = points.copy()
            for _ in range(50):
                point_idx = np.random.randint(0, len(points))
                # Small perturbation to see if we can improve
                new_point = refined_points[point_idx] + np.random.normal(0, 0.001, 3)
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

        # Update precomputed distance matrix for efficiency
        distances_matrix = cdist(points, points)

    return best_points, best_ratio

# EVOLVE-BLOCK-END