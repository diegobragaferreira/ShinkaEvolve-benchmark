# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
from typing import Tuple, List
import math

def compute_min_max_ratio(points):
    """Compute the minimum to maximum distance ratio for given points."""
    if len(points) < 2:
        return 0.0

    # Compute pairwise distances
    distances = cdist(points, points)

    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)

    # Find min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero
    if max_dist == 0:
        return 0.0

    return min_dist / max_dist

def create_geometric_patterns(n_points: int = 16) -> List[np.ndarray]:
    """Create multiple distinct geometric patterns as initialization strategies."""
    patterns = []

    # Pattern 1: Square Grid
    grid_points = []
    side = int(math.ceil(math.sqrt(n_points)))
    for i in range(side):
        for j in range(side):
            if len(grid_points) < n_points:
                x = j / (side - 1) if side > 1 else 0.5
                y = i / (side - 1) if side > 1 else 0.5
                grid_points.append([x, y])
    patterns.append(np.array(grid_points[:n_points]))

    # Pattern 2: Triangular Lattice (Hexagonal arrangement)
    tri_points = []
    rows = 4
    cols = 4
    for i in range(rows):
        for j in range(cols):
            if len(tri_points) < n_points:
                x = (j + (i % 2) * 0.5) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                tri_points.append([x, y])
    patterns.append(np.array(tri_points[:n_points]))

    # Pattern 3: Circular Distribution
    circ_points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        radius = 0.4 * (1.0 + 0.3 * np.sin(2 * angle))  # Varying radius
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        circ_points.append([x, y])
    patterns.append(np.array(circ_points))

    # Pattern 4: Spiral Pattern
    spiral_points = []
    for i in range(n_points):
        t = i * 0.5
        r = 0.4 * (i / (n_points - 1)) if n_points > 1 else 0.0
        x = 0.5 + r * np.cos(t)
        y = 0.5 + r * np.sin(t)
        spiral_points.append([x, y])
    patterns.append(np.array(spiral_points))

    # Pattern 5: Random with spacing constraint
    np.random.seed(42)
    rand_points = []
    for _ in range(n_points):
        x = np.random.random()
        y = np.random.random()
        rand_points.append([x, y])
    patterns.append(np.array(rand_points))

    return patterns

def adaptive_local_search(initial_points: np.ndarray, max_time: float) -> np.ndarray:
    """Apply adaptive local search with multiple refinement phases."""
    start_time = time.time()

    # Phase 1: Coarse optimization with relaxed tolerances
    points = initial_points.copy()
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(points)

    # Define objective function
    def objective(x):
        points = x.reshape(-1, 2)
        return -compute_min_max_ratio(points)  # Negative for maximization

    # Bounds
    bounds = [(0, 1) for _ in range(32)]

    # Try L-BFGS-B for coarse optimization
    if (time.time() - start_time) < max_time - 20:
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
            )

            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)

                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
        except:
            pass

    # Phase 2: Fine optimization with stricter tolerances
    if (time.time() - start_time) < max_time - 10:
        try:
            # Start from best current solution
            points = best_points.copy()

            # Try SLSQP with tight tolerances
            result = minimize(
                objective,
                points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)

                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
        except:
            pass

    # Phase 3: Final refinement with different method
    if (time.time() - start_time) < max_time - 5:
        try:
            # Try Nelder-Mead for additional refinement
            points = best_points.copy()
            result = minimize(
                objective,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'fatol': 1e-12, 'xatol': 1e-12}
            )

            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                refined_ratio = compute_min_max_ratio(refined_points)

                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points.copy()
        except:
            pass

    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Track time to respect 180 second limit
    start_time = time.time()
    time_limit = 180.0

    # Create multiple geometric patterns
    patterns = create_geometric_patterns(16)

    # Evaluate each pattern and select the best
    best_pattern_ratio = -np.inf
    best_pattern_points = None

    for i, pattern in enumerate(patterns):
        if (time.time() - start_time) > time_limit * 0.95:
            break

        try:
            # Apply local search to each pattern
            optimized_points = adaptive_local_search(pattern, time_limit - (time.time() - start_time))

            # Calculate ratio
            ratio = compute_min_max_ratio(optimized_points)

            if ratio > best_pattern_ratio:
                best_pattern_ratio = ratio
                best_pattern_points = optimized_points.copy()

        except Exception as e:
            continue

    # If we didn't get a good solution from patterns, fall back to a clean initialization
    if best_pattern_points is None or best_pattern_ratio < 0.1:
        # Create a simple but diverse grid with randomness
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + 0.125 + np.random.normal(0, 0.01)
                y = i * 0.25 + 0.125 + np.random.normal(0, 0.01)
                points.append([x, y])

        points = np.clip(points, 0, 1)
        points = points[:16]

        # Refine with local search
        best_pattern_points = adaptive_local_search(points, time_limit - (time.time() - start_time))

    return best_pattern_points

# EVOLVE-BLOCK-END