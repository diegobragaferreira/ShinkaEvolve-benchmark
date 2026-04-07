# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, SphericalVoronoi
from scipy.spatial.distance import pdist, cdist
import time
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)

    # Use a sophisticated initialization based on the golden spiral on sphere
    points = fibonacci_sphere(14)

    # Apply hybrid optimization: Voronoi-based refinement with simulated annealing
    best_points, best_ratio = optimize_with_voronoi_hybrid(points)

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
    """Compute optimal perturbations based on Voronoi analysis"""
    if len(points) < 4:  # Need at least 4 points for meaningful Voronoi
        return None

    try:
        # Create Voronoi diagram
        vor = Voronoi(points)

        # Find points that are too close (based on distance threshold)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        avg_dist = np.mean(distances)

        # Identify poorly spaced points (those with neighbors below threshold)
        threshold = threshold_ratio * avg_dist

        # Create a list of points to consider for adjustment
        perturb_indices = []
        for i in range(len(points)):
            # Check distances to neighbors in Voronoi diagram
            neighbors = []
            for j in range(len(vor.ridge_points)):
                if vor.ridge_points[j][0] == i or vor.ridge_points[j][1] == i:
                    if vor.ridge_points[j][0] == i:
                        neighbors.append(vor.ridge_points[j][1])
                    else:
                        neighbors.append(vor.ridge_points[j][0])

            # If any neighbor is too close, mark this point for adjustment
            point_distances = cdist([points[i]], points)[0]
            close_neighbors = np.where(point_distances < threshold)[0]
            if len(close_neighbors) > 0 and i in close_neighbors:
                close_neighbors = close_neighbors[close_neighbors != i]

            if len(close_neighbors) > 0:
                perturb_indices.append(i)

        if len(perturb_indices) == 0:
            # If no specific points are identified, pick one at random
            return np.random.randint(0, len(points))
        else:
            # Choose a point that's most constrained
            chosen_idx = np.random.choice(perturb_indices)
            return chosen_idx

    except:
        # Fallback to random selection if Voronoi fails
        return np.random.randint(0, len(points))

def optimize_with_voronoi_hybrid(initial_points, max_time=360):
    """Optimize using hybrid approach combining Voronoi-based refinement with SA"""
    points = initial_points.copy()
    current_ratio, min_dist = compute_min_max_ratio(points)

    # Parameters for simulated annealing
    temp = 1.0
    min_temp = 1e-8
    cooling_rate = 0.999
    max_iter = 500000

    # Track best solution
    best_points = points.copy()
    best_ratio = current_ratio
    best_min_dist = min_dist

    start_time = time.time()
    iter_count = 0

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

        # Apply more targeted perturbation - either expand or contract based on local geometry
        if np.random.random() < 0.5:
            # Expansion type perturbation (move away from cluster)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            magnitude = 0.03 * temp  # Scale by temperature
            new_points[point_to_move] += direction * magnitude
        else:
            # Contraction type perturbation (move towards center of mass)
            # Calculate centroid of neighbors
            centroid = np.mean(points, axis=0)
            direction = points[point_to_move] - centroid
            if np.linalg.norm(direction) > 0:
                direction /= np.linalg.norm(direction)
                magnitude = 0.02 * temp
                new_points[point_to_move] -= direction * magnitude
            else:
                # If points are identical, use random move
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                magnitude = 0.02 * temp
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

        # Cool down temperature
        temp *= cooling_rate
        iter_count += 1

        # Occasionally do a full re-initialization if stuck
        if iter_count % 5000 == 0 and temp > 0.1:
            # Reinitialize with better sphere packing
            points = fibonacci_sphere(14)
            current_ratio, min_dist = compute_min_max_ratio(points)
            temp = min(temp * 1.3, 1.0)  # Increase temperature slightly

    return best_points, best_ratio

# EVOLVE-BLOCK-END