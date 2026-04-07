# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import math
import time

def _initialize_hexagonal_grid(n_points: int = 16) -> np.ndarray:
    """Initialize points using a hexagonal grid pattern."""
    np.random.seed(42)

    points = np.zeros((n_points, 2))
    rows = 4
    cols = 4
    spacing = 0.25

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx < n_points:
                # Offset every other row for hexagonal packing
                x = col * spacing + (row % 2) * spacing * 0.5
                y = row * spacing * math.sqrt(3) / 2
                points[idx] = [x, y]
                idx += 1

    # Normalize to [0.1, 0.9] range
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

    # Add small random perturbation to avoid degenerate cases
    points += np.random.normal(0, 0.01, points.shape)

    return points


def _initialize_spiral_pattern(n_points: int = 16) -> np.ndarray:
    """Initialize points using a spiral pattern."""
    points = np.zeros((n_points, 2))

    # Create spiral pattern
    angles = np.linspace(0, 4*np.pi, n_points)
    radii = np.linspace(0.1, 0.4, n_points)

    for i in range(n_points):
        points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
        points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

    return points


def _initialize_random(n_points: int = 16) -> np.ndarray:
    """Initialize points using random distribution."""
    return np.random.uniform(0.1, 0.9, (n_points, 2))


def _initialize_fibonacci_sphere(n_points: int = 16) -> np.ndarray:
    """Initialize points using Fibonacci sphere distribution for good spreading."""
    points = np.zeros((n_points, 2))
    phi = math.pi * (3 - math.sqrt(5))  # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        # Map to 2D plane
        points[i] = [0.5 + x * 0.4, 0.5 + z * 0.4]

    return points


def _initialize_grid_pattern(n_points: int = 16) -> np.ndarray:
    """Initialize points using a regular grid pattern."""
    points = np.zeros((n_points, 2))

    # Create 4x4 grid
    grid_x = np.linspace(0.1, 0.9, 4)
    grid_y = np.linspace(0.1, 0.9, 4)

    X, Y = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Take first n_points points and add some randomness
    points = grid_points[:n_points]
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, 0, 1)

    return points


def _compute_distance_ratio(points: np.ndarray) -> float:
    """Compute the ratio of minimum to maximum distance between all point pairs."""
    if len(points) < 2:
        return 0.0

    # Use squareform for numerical stability
    distances = squareform(pdist(points))

    # Mask diagonal elements (distance to self is 0) with large value instead of inf
    np.fill_diagonal(distances, 1e10)

    # Get min and max distances
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Handle case where all points might be coincident
    if max_dist == 0 or min_dist == 1e10:
        return 0.0

    return min_dist / max_dist


def _objective_function(points: np.ndarray) -> float:
    """Objective function to maximize (negative because scipy minimizes)."""
    # Reshape points array to (n, 2) format if needed
    if points.ndim == 1:
        points = points.reshape(-1, 2)

    # Compute negative of distance ratio (since we want to maximize)
    return -_compute_distance_ratio(points)


def _coordinate_wise_refinement(points, max_iterations=50):
    """
    Perform coordinate-wise refinement of point positions to improve the ratio.
    Each coordinate is optimized individually while fixing others.
    """
    points = points.copy()
    n = len(points)
    
    # Precompute current distances to avoid recomputation
    current_ratio = _compute_distance_ratio(points)
    
    for iteration in range(max_iterations):
        improved = False
        # Try to optimize each point coordinate-wise
        for i in range(n):
            best_point = points[i].copy()
            best_ratio = current_ratio
            
            # Try small perturbations in x and y directions
            for dx in [-0.005, -0.002, 0, 0.002, 0.005]:
                for dy in [-0.005, -0.002, 0, 0.002, 0.005]:
                    if abs(dx) == 0 and abs(dy) == 0:
                        continue
                        
                    test_point = points[i].copy()
                    test_point[0] += dx
                    test_point[1] += dy
                    
                    # Enforce bounds with epsilon padding
                    epsilon = 1e-8
                    test_point[0] = np.clip(test_point[0], epsilon, 1-epsilon)
                    test_point[1] = np.clip(test_point[1], epsilon, 1-epsilon)
                    
                    # Temporarily update this point
                    old_point = points[i].copy()
                    points[i] = test_point
                    
                    try:
                        ratio = _compute_distance_ratio(points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_point = test_point.copy()
                            improved = True
                    except:
                        pass
                        
                    # Restore original point
                    points[i] = old_point
            
            # Update to best point found
            points[i] = best_point
            current_ratio = best_ratio
            
        # Early termination if no significant improvement
        if not improved and iteration > 20:
            break
            
    return points


def _optimize_with_de_and_local_refinement(initial_points: np.ndarray) -> np.ndarray:
    """Run optimization with Differential Evolution followed by local refinement."""
    n = initial_points.shape[0]
    d = initial_points.shape[1]

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    # Define bounds for each coordinate [0, 1]
    bounds = [(0, 1) for _ in range(n * d)]

    # Stage 1: Global optimization with Differential Evolution
    try:
        de_result = differential_evolution(
            _objective_function,
            bounds,
            maxiter=100,  # Reduced for speed
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            atol=1e-12,
            rtol=1e-12
        )

        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
        else:
            de_points = initial_points.copy()
    except Exception:
        de_points = initial_points.copy()

    # Stage 2: Coordinate-wise refinement for better local search
    try:
        refined_points = _coordinate_wise_refinement(de_points, max_iterations=30)
    except Exception:
        refined_points = de_points.copy()

    # Stage 3: Local refinement with L-BFGS-B
    try:
        lbfgs_result = minimize(
            _objective_function,
            refined_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )

        if lbfgs_result.success:
            final_points = lbfgs_result.x.reshape(-1, 2)
        else:
            final_points = refined_points.copy()
    except Exception:
        final_points = refined_points.copy()

    # Stage 4: Additional SLSQP refinement as fallback
    try:
        slsqp_result = minimize(
            _objective_function,
            final_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )

        if slsqp_result.success:
            final_points = slsqp_result.x.reshape(-1, 2)
    except Exception:
        pass

    return final_points


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2

    # Try multiple initialization strategies with enhanced diversity
    initializations = [
        _initialize_hexagonal_grid(n),
        _initialize_spiral_pattern(n),
        _initialize_random(n),
        _initialize_fibonacci_sphere(n),
        _initialize_grid_pattern(n)
    ]

    best_points = None
    best_ratio = -np.inf

    # Run optimization from each initialization
    for i, init_points in enumerate(initializations):
        # Apply enhanced optimization with DE + local refinement
        optimized_points = _optimize_with_de_and_local_refinement(init_points)
        ratio = _compute_distance_ratio(optimized_points)

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    # If no successful optimization, return the best initialization
    if best_points is None:
        best_points = _initialize_hexagonal_grid(n)

    return best_points

# EVOLVE-BLOCK-END