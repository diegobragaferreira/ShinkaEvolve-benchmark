# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    # Set seed for reproducibility
    np.random.seed(42)

    # Phase 1: Initialize points using hexagonal grid pattern for better starting configuration
    n = 16
    points = np.zeros((n, 2))

    # Create hexagonal grid pattern
    rows = 4
    cols = 4
    spacing = 0.25

    row_offset = 0.0
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < n:
                x = col * spacing + (row % 2) * spacing * 0.5
                y = row * spacing * math.sqrt(3) / 2
                points[idx] = [x, y]
                idx += 1

    # Adjust points to fit within [0,1]x[0,1] and add some randomness
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

    # Phase 2: Simulated Annealing optimization
    # Energy function that directly targets min/max distance ratio
    def compute_energy_and_ratio(points):
        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            ratio = 0
        else:
            ratio = min_dist / max_dist

        # Energy is negative ratio (we want to maximize ratio, so minimize negative ratio)
        # Also add penalty for points outside bounds
        penalty = 0
        for pt in points:
            if pt[0] < 0 or pt[0] > 1 or pt[1] < 0 or pt[1] > 1:
                penalty += 1000

        return -ratio + penalty, ratio

    # Simulated Annealing parameters
    temp = 1.0
    min_temp = 1e-6
    cooling_rate = 0.9995
    max_iter = 50000

    current_energy, current_ratio = compute_energy_and_ratio(points)
    best_points = points.copy()
    best_ratio = current_ratio

    # Track convergence
    last_improvement = 0
    patience = 1000

    # Main optimization loop
    for iteration in range(max_iter):
        # Generate neighbor solution (random perturbation)
        neighbor_points = points.copy()
        # Pick a random point to move
        move_idx = np.random.randint(0, n)
        # Small random displacement
        displacement = np.random.normal(0, 0.001, 2)
        neighbor_points[move_idx] += displacement

        # Apply boundary constraints
        neighbor_points[move_idx, 0] = np.clip(neighbor_points[move_idx, 0], 0, 1)
        neighbor_points[move_idx, 1] = np.clip(neighbor_points[move_idx, 1], 0, 1)

        # Compute energy of neighbor
        neighbor_energy, neighbor_ratio = compute_energy_and_ratio(neighbor_points)

        # Accept or reject move
        if neighbor_energy < current_energy:
            # Always accept better solutions
            points = neighbor_points
            current_energy = neighbor_energy
            current_ratio = neighbor_ratio
        else:
            # Accept worse solutions with probability based on temperature
            delta = neighbor_energy - current_energy
            if np.random.rand() < np.exp(-delta / temp):
                points = neighbor_points
                current_energy = neighbor_energy
                current_ratio = neighbor_ratio

        # Update best solution
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
            last_improvement = iteration

        # Adaptive cooling
        temp *= cooling_rate

        # Early stopping based on patience
        if iteration - last_improvement > patience:
            break

        # Cool down more aggressively if not improving
        if iteration % 1000 == 0 and temp > min_temp:
            temp *= 0.95

    return best_points

# EVOLVE-BLOCK-END