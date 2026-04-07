# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective_function(points_flat):
        # Reshape flat array back to 16x2 points
        points = points_flat.reshape(-1, 2)

        # Ensure points are within unit square
        points = np.clip(points, 0, 1)

        # Compute pairwise distances
        distances = squareform(pdist(points))

        # Set diagonal to large value so it doesn't affect min
        np.fill_diagonal(distances, np.inf)

        # Get minimum and maximum distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio since we want to maximize
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return -np.inf

    # Create initial guess using an enhanced structured approach with symmetry breaking
    # Start with a hexagonal grid pattern and add adaptive noise
    np.random.seed(42)

    # Generate a hexagonal lattice pattern with better distribution
    rows = 4
    cols = 4
    points = []

    # Hexagonal grid with slight perturbation
    for i in range(rows):
        for j in range(cols):
            x = j * 0.25 + (i % 2) * 0.125
            y = i * 0.25
            points.append([x, y])

    # Convert to numpy array
    initial_points = np.array(points)

    # Apply symmetry breaking by fixing corner points explicitly
    # This helps avoid degenerate solutions and ensures good spatial distribution
    initial_points[0] = [0.0, 0.0]      # bottom-left corner
    initial_points[3] = [1.0, 0.0]      # bottom-right corner
    initial_points[12] = [0.0, 1.0]     # top-left corner
    initial_points[15] = [1.0, 1.0]     # top-right corner

    # Compute initial distance statistics to inform perturbation scaling
    test_points = initial_points.copy()
    distances = squareform(pdist(test_points))
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    initial_ratio = min_dist / max_dist if max_dist > 0 else 0.0

    # Apply adaptive perturbation scaling based on current configuration quality
    # If the initial configuration is poor, allow larger perturbations
    # If it's already good, use smaller perturbations to avoid disrupting progress
    base_perturbation = 0.05
    perturbation_scale = max(0.1, 1.0 - initial_ratio)  # Scale from 0.1 to 1.0
    actual_perturbation = base_perturbation * perturbation_scale

    # Add random perturbations to non-corner points
    for i in range(16):
        if i not in [0, 3, 12, 15]:  # Skip corner points that are fixed
            initial_points[i, 0] += (np.random.random() - 0.5) * actual_perturbation
            initial_points[i, 1] += (np.random.random() - 0.5) * actual_perturbation

    # Clip to unit square to ensure all points are within bounds
    initial_points = np.clip(initial_points, 0, 1)

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    # Use hybrid optimization approach: global search followed by local refinement
    from scipy.optimize import differential_evolution

    # First, try global optimization with differential evolution for better exploration
    bounds = [(0, 1) for _ in range(32)]

    try:
        # Run differential evolution with multiple restarts
        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )

        # If DE finds a better solution, use it as starting point for local refinement
        de_ratio = -objective_function(de_result.x)
        initial_ratio = -objective_function(initial_flat)

        if de_ratio > initial_ratio:
            # Use DE result as starting point for local optimization
            start_points = de_result.x
        else:
            # Fall back to initial configuration
            start_points = initial_flat

    except:
        # If DE fails, fall back to initial configuration
        start_points = initial_flat

    # Then use local optimization to refine the solution
    # We'll use L-BFGS-B which handles bounds well
    result = minimize(
        objective_function,
        start_points,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
    )

    # Extract the optimized solution
    optimized_points = result.x.reshape(-1, 2)
    # Ensure they're still in bounds
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points


# EVOLVE-BLOCK-END