# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, SphericalVoronoi
from scipy.spatial.distance import pdist, cdist, squareform
import time
import math
from scipy.optimize import differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    # Multi-start optimization with different seeds for better exploration
    best_points = None
    best_ratio = 0.0
    
    # Try multiple starting configurations
    for seed in [42, 123, 456, 789, 999]:
        np.random.seed(seed)
        points = fibonacci_sphere(14)
        optimized_points, optimized_ratio = optimize_with_voronoi_hybrid(points)
        
        if optimized_ratio > best_ratio:
            best_ratio = optimized_ratio
            best_points = optimized_points.copy()
    
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

def compute_voronoi_based_perturbation(points, threshold_ratio=0.3):
    """Compute optimal perturbations based on enhanced Voronoi analysis"""
    if len(points) < 4:  # Need at least 4 points for meaningful Voronoi
        return np.random.randint(0, len(points))

    try:
        # Create Spherical Voronoi diagram for better sphere-specific analysis
        sv = SphericalVoronoi(points)

        # Analyze Voronoi cell areas to identify dense regions
        cell_areas = sv.volume
        mean_area = np.mean(cell_areas)

        # Identify points in cells that are below 20% of average area (dense regions)
        dense_point_indices = np.where(cell_areas < 0.2 * mean_area)[0]

        # Also identify points that are too close to their neighbors
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        avg_dist = np.mean(distances)
        threshold = threshold_ratio * avg_dist

        # Create a scoring system for how much each point needs adjustment
        adjustment_scores = np.zeros(len(points))

        # Score based on cell density
        for i in range(len(points)):
            if i in dense_point_indices:
                adjustment_scores[i] += 2.0  # High priority for dense regions

        # Score based on neighbor distances
        distance_matrix = squareform(distances)
        for i in range(len(points)):
            # Count neighbors that are too close
            close_neighbors = np.where(distance_matrix[i] < threshold)[0]
            close_neighbors = close_neighbors[close_neighbors != i]  # Exclude self
            adjustment_scores[i] += len(close_neighbors) * 0.5

        # Choose point with highest adjustment score (but with some randomness)
        if np.max(adjustment_scores) > 0:
            # Prefer higher-scoring points but with 30% randomness
            if np.random.random() < 0.7:
                chosen_idx = np.argmax(adjustment_scores)
            else:
                # Select randomly from top 3 high-scoring points
                top_indices = np.argsort(adjustment_scores)[-3:]
                chosen_idx = np.random.choice(top_indices)
        else:
            # If no strong constraints, choose randomly
            chosen_idx = np.random.randint(0, len(points))

        return chosen_idx

    except Exception as e:
        # Fallback to random selection if Voronoi fails
        return np.random.randint(0, len(points))

def optimize_with_voronoi_hybrid(initial_points, max_time=360):
    """Optimize using hybrid approach combining Voronoi-based refinement with SA"""
    points = initial_points.copy()
    current_ratio, min_dist = compute_min_max_ratio(points)

    # Parameters for simulated annealing
    temp = 1.0
    min_temp = 1e-10
    base_cooling_rate = 0.9995
    max_iter = 1000000

    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    best_min_dist = min_dist

    start_time = time.time()
    iter_count = 0

    # Track recent ratios for adaptive cooling
    recent_ratios = []
    max_recent = 50

    # Main optimization loop
    while temp > min_temp and iter_count < max_iter and time.time() - start_time < max_time:
        # Alternate between different strategies based on iteration count
        if iter_count % 10 == 0:
            # Every 10 iterations, use Voronoi-based selection
            point_to_move = compute_voronoi_based_perturbation(points)
        else:
            # Otherwise, use random selection
            point_to_move = np.random.randint(0, len(points))

        # Create new candidate point
        new_points = points.copy()

        # Apply more sophisticated perturbation based on local Voronoi analysis
        try:
            # Analyze local geometry around the selected point
            distances = cdist([points[point_to_move]], points)[0]
            distances = distances[distances > 0]  # Remove self-distance

            if len(distances) > 0:
                avg_distance = np.mean(distances)
                min_distance = np.min(distances)
                max_distance = np.max(distances)

                # Determine perturbation type based on local density
                if min_distance < avg_distance * 0.4:
                    # Point is too close to neighbors - repel it strongly
                    repulsion = np.zeros(3)
                    for i in range(len(points)):
                        if i != point_to_move and distances[i-1] < avg_distance * 0.6:
                            diff = points[point_to_move] - points[i]
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                repulsion += diff / dist * (1.0/(dist * dist + 1e-8))

                    if np.linalg.norm(repulsion) > 0:
                        repulsion = repulsion / np.linalg.norm(repulsion)
                        # Enhanced scaling based on how close we are to ideal spacing
                        proximity_factor = min_distance / avg_distance
                        magnitude = 0.04 * temp * (1.0 + proximity_factor * 2.0)
                        new_points[point_to_move] += repulsion * magnitude
                    else:
                        # Fallback to random perturbation
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        magnitude = 0.025 * temp
                        new_points[point_to_move] += direction * magnitude
                else:
                    # Point is relatively well-spaced - make small adjustment
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    # Magnitude inversely proportional to distance from center
                    center_dist = np.linalg.norm(points[point_to_move])
                    # Scale magnitude based on how far we are from the ideal range
                    distance_factor = 1.0 - min(abs(center_dist - 0.5), 0.5)
                    magnitude = 0.01 * temp * distance_factor
                    new_points[point_to_move] += direction * magnitude

            else:
                # Fallback for edge cases
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                magnitude = 0.025 * temp
                new_points[point_to_move] += direction * magnitude

        except Exception:
            # Fallback to simple random perturbation
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            magnitude = 0.025 * temp
            new_points[point_to_move] += direction * magnitude

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

        # Adjust cooling rate based on recent improvement
        cooling_rate = base_cooling_rate
        if len(recent_ratios) >= 10:
            recent_improvement = recent_ratios[-1] - recent_ratios[0]
            if recent_improvement < 1e-8 and temp > 1e-6:
                # Very slow progress, cool faster
                cooling_rate = base_cooling_rate * 1.15
            elif recent_improvement > 1e-4 and temp > 1e-5:
                # Fast progress, cool slower
                cooling_rate = base_cooling_rate * 0.98

        # Cool down temperature
        temp *= cooling_rate
        iter_count += 1

        # Occasionally do a full re-initialization if stuck
        if iter_count % 5000 == 0 and temp > 0.1:
            # Reinitialize with better sphere packing
            points = fibonacci_sphere(14)
            current_ratio, min_dist = compute_min_max_ratio(points)
            temp = min(temp * 1.2, 1.0)  # Increase temperature slightly

    return best_points, best_ratio

# EVOLVE-BLOCK-END