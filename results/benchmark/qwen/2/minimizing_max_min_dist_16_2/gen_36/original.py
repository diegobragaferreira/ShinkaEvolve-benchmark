# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero and penalize out-of-bounds points
        if d_max <= 1e-12:
            return -np.inf

        ratio = d_min / d_max

        # Add penalty for points outside [0,1] bounds
        penalty = 0
        for i in range(16):
            if points[i, 0] < 0 or points[i, 0] > 1 or points[i, 1] < 0 or points[i, 1] > 1:
                penalty += 1000000  # Larger penalty for constraint violations

        # Return negative ratio to minimize (since we want to maximize the ratio)
        return -(ratio - penalty / 10000000)

    # Create bounds for each coordinate (0 to 1 for both x and y)
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

    # Initialize with a better starting configuration based on known optimal arrangements
    np.random.seed(42)

    # Create a more sophisticated initial configuration - use a known good starting point
    # Based on the 16-point optimal configuration for sphere packing in 2D
    # This is a variation of the 4x4 grid with offsets but optimized for uniformity

    # Try to place points in a way that mimics the optimal 16-point distribution
    # We'll use a combination of regular grid with strategic perturbations
    initial_points = np.zeros((16, 2))

    # Generate a good initial configuration - 4x4 grid with staggered rows
    idx = 0
    for i in range(4):
        for j in range(4):
            # Create staggered grid pattern
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + offset) / 3.0
            y = i / 3.0

            # Apply perturbations to break symmetry and create better spacing
            # Use slightly larger perturbations than before
            x += np.random.normal(0, 0.03)  # Increased from 0.015
            y += np.random.normal(0, 0.03)

            # Ensure within bounds - clip values
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)

            initial_points[idx] = [x, y]
            idx += 1

    # Also try a second initialization strategy - points arranged in two concentric rings
    # This can help avoid local minima
    ring_points = np.zeros((16, 2))
    angles = np.linspace(0, 2*np.pi, 17)[:-1]  # 16 angles
    radii = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.9, 0.9, 0.9, 0.9]

    for i in range(16):
        angle = angles[i]
        radius = radii[i]
        x = 0.5 + radius * np.cos(angle) * 0.4
        y = 0.5 + radius * np.sin(angle) * 0.4
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        ring_points[i] = [x, y]

    # Evaluate both initial configurations and pick the better one
    def evaluate_config(config):
        flat_config = config.flatten()
        return -objective(flat_config)

    score1 = evaluate_config(initial_points)
    score2 = evaluate_config(ring_points)

    # Use the better initial configuration
    if score2 > score1:
        initial_points = ring_points

    # Flatten for optimization
    initial_flat = initial_points.flatten()

    try:
        # Stage 1: More thorough global search with differential evolution
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=100,  # Increased iterations
            popsize=20,   # Larger population for better exploration
            seed=42,
            tol=1e-8,     # Tighter tolerance
            mutation=(0.5, 1),
            recombination=0.7,
            callback=None
        )

        # Stage 2: Local refinement with L-BFGS-B using more iterations
        refined_result = minimize(
            objective,
            result_de.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12},  # More iterations and tighter tolerances
            callback=None
        )

        final_result = refined_result.x

    except Exception as e:
        # Fallback to initial configuration
        print(f"Optimization failed: {e}")
        final_result = initial_flat

    # Convert back to points array
    points = final_result.reshape(-1, 2)

    # Ensure all points are within bounds
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)

    return points

# EVOLVE-BLOCK-END