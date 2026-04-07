# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    np.random.seed(42)

    # Enhanced hexagonal initialization with better spacing and symmetry breaking
    def initialize_enhanced_hexagonal():
        # Create a more optimal hexagonal arrangement
        # For 16 points, we can arrange them in 4 rows with staggered positions
        points = []

        # Use sqrt(3)/2 spacing for true hexagonal packing
        row_spacing = np.sqrt(3) / 2
        col_spacing = 1.0

        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x = j * col_spacing + (i % 2) * col_spacing * 0.5
                y = i * row_spacing
                points.append([x, y])

        points = np.array(points)

        # Normalize to fit in [0,1] x [0,1]
        max_x = (cols - 1) + 0.5  # Account for offset in last row
        max_y = (rows - 1) * row_spacing

        points[:, 0] = points[:, 0] / max_x
        points[:, 1] = points[:, 1] / max_y

        # Add strategic perturbations to break symmetry
        # Use different noise levels for different points
        noise = np.random.normal(0, 0.015, points.shape)

        # Apply more aggressive perturbation to corner points to break symmetry
        corner_indices = [0, 3, 12, 15]  # Four corners of 4x4 grid
        noise[corner_indices] *= 2.0

        points += noise
        points = np.clip(points, 0, 1)

        return points

    # Compute min/max distance ratio with efficient distance matrix
    def compute_ratio(points):
        if len(points) < 2:
            return 0

        # Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(points)

        # Find minimum distance (excluding self-distance)
        # Query for second closest point since first is self
        distances, indices = tree.query(points, k=2)
        d_min = np.min(distances[:, 1])  # Second closest (not self)

        # Find maximum distance
        # For large point sets, find max distance more efficiently
        if len(points) <= 100:
            # Direct computation for small sets
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)
            d_max = np.min(distances[distances != np.inf]) if np.any(distances != np.inf) else 0
        else:
            # For larger sets, use pairwise maximum
            distances = cdist(points, points)
            np.fill_diagonal(distances, 0)
            d_max = np.max(distances)

        if d_max == 0:
            return 0

        return d_min / d_max

    # Sphere packing objective function with geometric constraints
    def sphere_packing_objective(points):
        """Objective based on maximizing minimum distance (equivalent to maximum sphere radius)"""
        if len(points) < 2:
            return 0

        # Use cKDTree for efficient computation
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=2)
        min_distance = np.min(distances[:, 1])

        # Penalize if any points are too close to boundaries
        boundary_penalty = 0
        boundary_threshold = 0.02
        for point in points:
            if (point[0] < boundary_threshold or point[0] > 1-boundary_threshold or
                point[1] < boundary_threshold or point[1] > 1-boundary_threshold):
                boundary_penalty += 1000 * boundary_threshold

        return min_distance - boundary_penalty

    # Generate neighbor solution with adaptive moves and better boundary handling
    def generate_neighbor_adaptive(points, step_size=0.02):
        new_points = points.copy()

        # Choose move type: single point, pair move, or cluster move
        move_type = np.random.choice(['single', 'pair', 'cluster'], p=[0.6, 0.3, 0.1])

        if move_type == 'single':
            # Single point move
            idx = np.random.randint(len(points))
            movement = np.random.normal(0, step_size, 2)
            new_points[idx] += movement

        elif move_type == 'pair':
            # Move two nearby points together
            # Find two nearby points
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)

            # Get indices of two closest points
            min_indices = np.unravel_index(np.argmin(distances), distances.shape)
            idx1, idx2 = min_indices

            # Move them in coordinated way
            movement = np.random.normal(0, step_size * 0.8, 2)
            new_points[idx1] += movement
            new_points[idx2] += movement

        else:  # cluster
            # Move a small cluster of points together
            num_cluster = min(3, len(points))
            cluster_indices = np.random.choice(len(points), num_cluster, replace=False)

            # Move them towards their centroid
            centroid = np.mean(new_points[cluster_indices], axis=0)
            movement = np.random.normal(0, step_size * 0.5, 2)

            for idx in cluster_indices:
                new_points[idx] += movement

        # Apply boundary constraints with better handling
        # For better boundary handling, use clipping instead of reflection
        new_points = np.clip(new_points, 0, 1)

        return new_points

    # Advanced adaptive optimization with cooling schedule
    def adaptive_optimization(initial_points):
        points = initial_points.copy()
        current_ratio = compute_ratio(points)
        best_points = points.copy()
        best_ratio = current_ratio

        # Adaptive cooling schedule
        temp = 0.1  # Start with higher temperature
        cooling_rate = 0.999  # Slightly faster cooling
        min_temp = 1e-6

        # Track improvement for adaptive cooling
        improvement_window = 100
        recent_improvements = []

        max_iterations = 3000
        for iteration in range(max_iterations):
            # Adjust temperature based on recent performance
            if len(recent_improvements) >= improvement_window:
                avg_improvement = np.mean(recent_improvements[-improvement_window:])
                if avg_improvement < 1e-6:
                    temp *= 0.95  # Cool faster if stagnating
                elif avg_improvement > 1e-4:
                    temp *= 1.01  # Warm up if making good progress

            temp = max(temp * cooling_rate, min_temp)

            if temp < min_temp:
                break

            # Generate neighbor solution
            new_points = generate_neighbor_adaptive(points, step_size=temp)
            new_ratio = compute_ratio(new_points)

            # Track recent improvements
            if new_ratio > current_ratio:
                recent_improvements.append(new_ratio - current_ratio)
                if len(recent_improvements) > improvement_window * 2:
                    recent_improvements.pop(0)

            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                points = new_points.copy()
                current_ratio = new_ratio

                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = points.copy()

        return best_points, best_ratio

    # Initialize with enhanced hexagonal approach
    points = initialize_enhanced_hexagonal()

    # Try multiple restarts to escape local optima
    best_solution = points.copy()
    best_ratio = compute_ratio(points)

    # Multiple restarts with different random seeds
    for restart in range(5):
        # Different initialization for each restart
        points = initialize_enhanced_hexagonal()

        # Apply adaptive optimization
        optimized_points, ratio = adaptive_optimization(points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Return the best solution found
    return best_solution

# EVOLVE-BLOCK-END