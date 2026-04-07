# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

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

def initialize_points(n_points=16, method='adaptive_grid'):
    """Initialize points using a structured adaptive grid approach."""
    np.random.seed(42)

    if method == 'adaptive_grid':
        # Create a 4x4 grid with adaptive perturbations
        points = []
        rows, cols = 4, 4

        # Generate base grid points
        for i in range(rows):
            for j in range(cols):
                # Base grid positions
                x = j * (1.0 / (cols - 1)) if cols > 1 else 0.5
                y = i * (1.0 / (rows - 1)) if rows > 1 else 0.5

                # Adaptive perturbation - vary based on position
                if (i == 0 or i == rows-1) and (j == 0 or j == cols-1):
                    # Corner points - smallest perturbation
                    perturbation = 0.005
                elif i == 0 or i == rows-1 or j == 0 or j == cols-1:
                    # Edge points - medium perturbation
                    perturbation = 0.01
                else:
                    # Interior points - larger perturbation
                    perturbation = 0.02

                # Apply random perturbation
                x += np.random.normal(0, perturbation)
                y += np.random.normal(0, perturbation)
                points.append([x, y])

        # Ensure points are within bounds [0,1]
        points = np.clip(points, 0, 1)
        return np.array(points[:n_points])

    elif method == 'hexagonal':
        # Create hexagonal-like arrangement
        points = []
        rows, cols = 4, 4

        for i in range(rows):
            for j in range(cols):
                x = j * 0.3 + (i % 2) * 0.15
                y = i * 0.3
                x += np.random.normal(0, 0.015)
                y += np.random.normal(0, 0.015)
                points.append([x, y])

        points = np.clip(points, 0, 1)
        return np.array(points[:n_points])

    else:
        # Default random initialization
        return np.random.rand(n_points, 2)

def enhanced_adaptive_optimization(initial_points, max_time=150):
    """Enhanced optimization with adaptive perturbations and iterative refinement."""
    start_time = time.time()

    # Best solution tracking
    current_points = initial_points.copy()
    best_points = initial_points.copy()
    best_ratio = compute_min_max_ratio(initial_points)

    # Main optimization loop with adaptive perturbations
    max_iterations = 3000
    iteration = 0

    while iteration < max_iterations and (time.time() - start_time) < max_time - 5:
        # Calculate current ratio for adaptive perturbation sizing
        current_ratio = compute_min_max_ratio(current_points)

        # Compute adaptive perturbation magnitude
        if current_ratio < 0.1:
            perturbation_magnitude = 0.02  # Large perturbation for poor solutions
        elif current_ratio < 0.2:
            perturbation_magnitude = 0.01  # Medium perturbation
        else:
            perturbation_magnitude = 0.005  # Small perturbation for good solutions

        # Create candidate solution by perturbing one point at a time
        candidate_points = current_points.copy()

        # Select random point to perturb
        point_idx = np.random.randint(0, len(candidate_points))

        # Apply adaptive perturbation
        candidate_points[point_idx, 0] += np.random.normal(0, perturbation_magnitude)
        candidate_points[point_idx, 1] += np.random.normal(0, perturbation_magnitude)

        # Enforce bounds
        candidate_points[point_idx, 0] = np.clip(candidate_points[point_idx, 0], 0, 1)
        candidate_points[point_idx, 1] = np.clip(candidate_points[point_idx, 1], 0, 1)

        # Evaluate candidate
        candidate_ratio = compute_min_max_ratio(candidate_points)

        # Accept improvement or use probabilistic acceptance
        if candidate_ratio > best_ratio:
            current_points = candidate_points.copy()
            best_points = candidate_points.copy()
            best_ratio = candidate_ratio
        elif np.random.random() < 0.1:  # 10% chance of accepting worse solution
            current_points = candidate_points.copy()

        iteration += 1

        # Early stopping if we achieve good results
        if best_ratio > 0.3:
            break

        # Periodic refinement with local optimization
        if iteration % 100 == 0:
            try:
                # Local optimization with L-BFGS-B
                def objective(x):
                    points = x.reshape(-1, 2)
                    return -compute_min_max_ratio(points)

                bounds = [(0, 1) for _ in range(32)]

                result = minimize(
                    objective,
                    current_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10},
                    timeout=max_time - (time.time() - start_time)
                )

                if result.success:
                    ref_points = result.x.reshape(-1, 2)
                    ref_points = np.clip(ref_points, 0, 1)
                    ref_ratio = compute_min_max_ratio(ref_points)

                    if ref_ratio > best_ratio:
                        current_points = ref_points.copy()
                        best_points = ref_points.copy()
                        best_ratio = ref_ratio

            except:
                pass

    return best_points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Initialize with adaptive grid
    initial_points = initialize_points(n_points=16, method='adaptive_grid')

    # Optimize using adaptive SQP approach
    optimized_points = adaptive_sqp_optimization(initial_points, max_time=150)

    return optimized_points

# EVOLVE-BLOCK-END