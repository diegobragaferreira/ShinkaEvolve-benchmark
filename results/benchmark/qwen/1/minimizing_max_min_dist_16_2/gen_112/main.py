# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    # Set seed for reproducibility
    np.random.seed(42)

    def compute_ratio(points):
        """Compute min/max distance ratio for given point configuration."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Handle case where all points might be coincident
        if max_dist == 0:
            return 0.0

        return min_dist / max_dist

    def objective_function(points_vec):
        """Objective function for optimization (negative ratio to maximize)."""
        points = points_vec.reshape(-1, 2)
        ratio = compute_ratio(points)
        return -ratio  # Negative because scipy minimizes

    def constraint_function(points_vec):
        """Constraint function to keep points within bounds [0,1] x [0,1]."""
        points = points_vec.reshape(-1, 2)
        # Check if any point is outside [0,1] bounds
        violations = 0

        # Check x bounds
        violations += np.sum(points[:, 0] < 0)
        violations += np.sum(points[:, 0] > 1)

        # Check y bounds
        violations += np.sum(points[:, 1] < 0)
        violations += np.sum(points[:, 1] > 1)

        # Return negative of violations (positive if all constraints satisfied)
        return -violations

    def initialize_hexagonal_grid():
        """Initialize points using a better hexagonal grid pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Create hexagonal grid pattern with better distribution
        rows = 4
        cols = 4
        spacing = 0.25

        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < n:
                    # Offset every other row for hexagonal packing
                    x = col * spacing + (row % 2) * spacing * 0.5
                    y = row * spacing * math.sqrt(3) / 2
                    points[idx] = [x, y]
                    idx += 1

        # Adjust points to fit within [0.1,0.9]x[0.1,0.9] with some randomness
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

        # Add small random perturbation
        points += np.random.normal(0, 0.01, points.shape)

        # Clamp to bounds
        points = np.clip(points, 0.01, 0.99)

        return points

    def initialize_spiral_pattern():
        """Initialize points using a spiral pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Create spiral pattern
        angles = np.linspace(0, 4*np.pi, n)
        radii = np.linspace(0.1, 0.4, n)

        for i in range(n):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

        return points

    def initialize_random():
        """Initialize points using random distribution."""
        return np.random.uniform(0.1, 0.9, (16, 2))

    # Try multiple initialization strategies
    initial_configs = [
        initialize_hexagonal_grid(),
        initialize_spiral_pattern(),
        initialize_random()
    ]

    best_points = None
    best_ratio = -np.inf

    # Run optimization from each initialization
    for init_config in initial_configs:
        points = init_config.copy()

        # Stage 1: Global search with Differential Evolution
        bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=200,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )

        # Extract best solution from DE
        de_points = de_result.x.reshape(-1, 2)

        # Stage 2: Local refinement with SLSQP
        constraints = {'type': 'ineq', 'fun': constraint_function}
        bounds = [(0, 1) for _ in range(32)]

        slsqp_result = minimize(
            objective_function,
            de_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10},
            tol=1e-10
        )

        final_points = slsqp_result.x.reshape(-1, 2)

        # Evaluate final solution
        current_ratio = compute_ratio(final_points)

        # Update global best if this run was better
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = final_points.copy()

    return best_points

# EVOLVE-BLOCK-END