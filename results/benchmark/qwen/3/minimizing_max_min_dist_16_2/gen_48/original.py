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

    # Sphere packing-inspired initialization with geometric constraints
    def initialize_sphere_packing():
        # Start with a regular hexagonal grid
        points = []
        rows = 4
        cols = 4
        spacing = 0.25

        for i in range(rows):
            for j in range(cols):
                x = j * spacing + (i % 2) * spacing * 0.5
                y = i * spacing
                points.append([x, y])

        points = np.array(points)

        # Add slight perturbations to break symmetry and improve optimization
        noise = np.random.normal(0, 0.01, points.shape)
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

    # Generate neighbor solution using sphere packing constraints
    def generate_neighbor(points, step_size=0.02):
        new_points = points.copy()

        # Select multiple points to move for better exploration
        num_moves = min(3, len(points))
        move_indices = np.random.choice(len(points), num_moves, replace=False)

        for idx in move_indices:
            # Move point with adaptive step size
            movement = np.random.normal(0, step_size, 2)
            new_points[idx] += movement

        # Apply boundary constraints with reflection
        for i in range(len(new_points)):
            for j in range(2):
                if new_points[i, j] < 0:
                    new_points[i, j] = -new_points[i, j]  # Reflect
                elif new_points[i, j] > 1:
                    new_points[i, j] = 2 - new_points[i, j]  # Reflect

        return new_points

    # Advanced multi-scale optimization
    def multi_scale_optimization(initial_points):
        best_points = initial_points.copy()
        best_ratio = compute_ratio(best_points)

        # Scale 1: Coarse-grained optimization with large steps
        points = initial_points.copy()
        for iter_coarse in range(500):
            # Large step size for broad search
            new_points = generate_neighbor(points, step_size=0.05)
            new_ratio = compute_ratio(new_points)

            # Accept with Metropolis criterion
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 100):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        # Scale 2: Medium-grained optimization
        for iter_medium in range(1000):
            # Medium step size for local refinement
            new_points = generate_neighbor(points, step_size=0.01)
            new_ratio = compute_ratio(new_points)

            # Accept with Metropolis criterion
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 500):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        # Scale 3: Fine-grained optimization with small steps
        for iter_fine in range(1500):
            # Small step size for fine adjustment
            new_points = generate_neighbor(points, step_size=0.002)
            new_ratio = compute_ratio(new_points)

            # Accept with Metropolis criterion (higher acceptance probability)
            if new_ratio > best_ratio or np.random.rand() < np.exp((new_ratio - best_ratio) * 1000):
                points = new_points.copy()
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_points = points.copy()

        return best_points

    # Initialize with sphere packing approach
    points = initialize_sphere_packing()

    # Try multiple restarts to escape local optima
    best_solution = points.copy()
    best_ratio = compute_ratio(points)

    # Multiple restarts with different random seeds
    for restart in range(5):
        # Different initialization for each restart
        points = initialize_sphere_packing()

        # Apply multi-scale optimization
        optimized_points = multi_scale_optimization(points)
        ratio = compute_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()

    # Final local refinement using gradient approximation
    final_points = best_solution.copy()

    # Simple gradient-based refinement
    for refinement_iter in range(500):
        current_ratio = compute_ratio(final_points)

        # Estimate gradient using finite differences
        gradient = np.zeros_like(final_points)
        eps = 1e-5

        for i in range(len(final_points)):
            for j in range(2):
                # Perturb point coordinate
                points_plus = final_points.copy()
                points_plus[i, j] += eps
                points_plus = np.clip(points_plus, 0, 1)

                points_minus = final_points.copy()
                points_minus[i, j] -= eps
                points_minus = np.clip(points_minus, 0, 1)

                ratio_plus = compute_ratio(points_plus)
                ratio_minus = compute_ratio(points_minus)

                gradient[i, j] = (ratio_plus - ratio_minus) / (2 * eps)

        # Update with gradient ascent
        step_size = 0.005
        final_points = final_points + step_size * gradient

        # Ensure bounds
        final_points = np.clip(final_points, 0, 1)

        # Early stopping condition
        if np.all(np.abs(gradient) < 1e-6):
            break

    return final_points

# EVOLVE-BLOCK-END