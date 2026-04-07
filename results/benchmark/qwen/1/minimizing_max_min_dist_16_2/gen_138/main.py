# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    # Set seed for reproducibility
    np.random.seed(42)

    def compute_distance_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
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

    def constraint_function(points_vec):
        """Constraint function to ensure points stay within bounds [0,1] x [0,1]."""
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
        """Initialize points using a hexagonal grid pattern."""
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

        # Add small random perturbation to avoid degenerate cases
        points += np.random.normal(0, 0.01, points.shape)

        # Clamp to bounds with epsilon padding
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

    def initialize_fibonacci_sphere():
        """Initialize points using Fibonacci sphere distribution for good spreading."""
        n = 16
        points = np.zeros((n, 2))
        phi = math.pi * (3 - math.sqrt(5))  # golden angle in radians

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            # Map to 2D plane
            points[i] = [0.5 + x * 0.4, 0.5 + z * 0.4]

        return points

    def initialize_random():
        """Initialize points using random distribution."""
        return np.random.uniform(0.1, 0.9, (16, 2))

    # Try multiple initialization strategies
    initial_configs = [
        initialize_hexagonal_grid(),
        initialize_spiral_pattern(),
        initialize_fibonacci_sphere(),
        initialize_random()
    ]

    best_points = None
    best_ratio = -np.inf

    # Run optimization from each initialization
    for init_config in initial_configs:
        points = init_config.copy()

        # Phase 1: Global search using differential evolution
        from scipy.optimize import differential_evolution

        def objective_function(points_vec):
            points = points_vec.reshape(-1, 2)
            ratio = compute_distance_ratio(points)
            return -ratio  # Negative because scipy minimizes

        bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each

        # Run differential evolution for global search
        de_result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,  # Reduced iterations for speed
            popsize=10,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )

        # Extract result from DE
        de_points = de_result.x.reshape(-1, 2)

        # Phase 2: Local refinement with L-BFGS-B
        try:
            # Flatten for scipy optimization
            x0 = de_points.flatten()

            # Define bounds for each coordinate [0, 1]
            bounds_scipy = [(0, 1) for _ in range(32)]

            # Optimize using L-BFGS-B solver which handles bounds well
            result = minimize(
                fun=objective_function,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds_scipy,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )

            # Extract optimized points
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                ratio = compute_distance_ratio(optimized_points)

                # Update global best if this run was better
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
            else:
                # Fallback to DE result if L-BFGS-B fails
                ratio = compute_distance_ratio(de_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = de_points.copy()

        except Exception:
            # Fallback to DE result if L-BFGS-B fails
            ratio = compute_distance_ratio(de_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = de_points.copy()

    # If no successful optimization, return the best initialization
    if best_points is None:
        best_points = initialize_hexagonal_grid()

    return best_points

# EVOLVE-BLOCK-END