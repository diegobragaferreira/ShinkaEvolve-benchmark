# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
import itertools

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 2)

        # Compute pairwise distances efficiently using scipy
        distances = pdist(points)

        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        return -min_dist / max_dist

    def create_initial_grid():
        """Create initial 4x4 grid points"""
        grid_size = 4
        x_vals = np.linspace(0.05, 0.95, grid_size)
        y_vals = np.linspace(0.05, 0.95, grid_size)
        return np.array([[x, y] for x in x_vals for y in y_vals])

    def perturb_points(points, magnitude):
        """Add controlled random perturbations"""
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        return np.clip(perturbed, 0, 1)

    def evaluate_solution(points):
        """Evaluate the quality of a solution"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist

    # Multi-start approach with different initialization strategies
    best_ratio = 0
    best_points = None

    # Strategy 1: Grid with small perturbations
    grid_points = create_initial_grid()
    for i in range(5):  # Multiple random perturbations of grid
        perturbed = perturb_points(grid_points, 0.01)
        x0 = perturbed.flatten()
        bounds = [(0, 1) for _ in range(32)]

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except:
            continue

    # Strategy 2: Random initialization
    np.random.seed(42)
    for i in range(5):  # Multiple random starts
        random_points = np.random.rand(16, 2)
        x0 = random_points.flatten()
        bounds = [(0, 1) for _ in range(32)]

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except:
            continue

    # Strategy 3: Better structured approach - spiral-like pattern
    np.random.seed(42)
    for i in range(3):  # Try different structured patterns
        # Create a more strategic initial setup
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles) * 0.8
        y = 0.5 + radii * np.sin(angles) * 0.8
        spiral_points = np.column_stack([x, y])

        # Add small perturbations
        perturbed = perturb_points(spiral_points, 0.02)
        x0 = perturbed.flatten()
        bounds = [(0, 1) for _ in range(32)]

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except:
            continue

    # If no good solution was found, return the grid-based solution
    if best_points is None:
        grid_points = create_initial_grid()
        best_points = grid_points.copy()

    return best_points

# EVOLVE-BLOCK-END