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

    def create_enhanced_initialization():
        """Create an enhanced initial configuration with better symmetry breaking"""
        np.random.seed(42)

        # Start with a structured 4x4 grid pattern
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points.append([x, y])

        points = np.array(points)

        # Add corner anchors to break symmetry explicitly
        # These fixed points will help avoid degenerate solutions
        anchor_points = [
            [0.0, 0.0],    # bottom-left
            [1.0, 0.0],    # bottom-right
            [0.0, 1.0],    # top-left
            [1.0, 1.0]     # top-right
        ]

        # Replace the four corner points with anchors
        points[0] = anchor_points[0]      # bottom-left
        points[3] = anchor_points[1]      # bottom-right
        points[12] = anchor_points[2]     # top-left
        points[15] = anchor_points[3]     # top-right

        # Apply adaptive perturbations based on distance analysis
        # First compute current state
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        current_ratio = min_dist / max_dist if max_dist > 0 else 0.0

        # Adaptive perturbation based on current distribution quality
        base_perturbation = 0.03  # Reduced base perturbation
        perturbation_scale = max(0.1, 1.0 - current_ratio)  # Scale from 0.1 to 1.0
        actual_perturbation = base_perturbation * perturbation_scale

        # Apply perturbations to non-anchor points
        for i in range(16):
            if i not in [0, 3, 12, 15]:  # Skip anchor points
                points[i, 0] += (np.random.random() - 0.5) * actual_perturbation
                points[i, 1] += (np.random.random() - 0.5) * actual_perturbation

        # Clip to unit square
        points = np.clip(points, 0, 1)
        return points

    # Create multiple diverse initial configurations
    initial_configurations = []

    # Create several different starting points
    for i in range(3):
        np.random.seed(42 + i)  # Different seeds for variety
        initial_config = create_enhanced_initialization()
        initial_configurations.append(initial_config)

    # Find the best initial configuration
    best_initial_points = None
    best_initial_ratio = -np.inf

    for initial_config in initial_configurations:
        # Evaluate initial configuration
        distances = squareform(pdist(initial_config))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        initial_ratio = min_dist / max_dist if max_dist > 0 else 0.0

        if initial_ratio > best_initial_ratio:
            best_initial_ratio = initial_ratio
            best_initial_points = initial_config.copy()

    # Now run optimization on the best initial configuration
    initial_flat = best_initial_points.flatten()

    # Use local optimization to refine the initial configuration
    # We'll use L-BFGS-B which handles bounds well
    result = minimize(
        objective_function,
        initial_flat,
        method='L-BFGS-B',
        bounds=[(0, 1) for _ in range(32)],
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
    )

    # Extract the optimized solution
    optimized_points = result.x.reshape(-1, 2)
    # Ensure they're still in bounds
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points


# EVOLVE-BLOCK-END