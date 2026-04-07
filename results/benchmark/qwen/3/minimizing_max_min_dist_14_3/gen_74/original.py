# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)

    def initialize_spherical_points(n_points):
        """Initialize points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle

        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def initialize_cube_grid_points(n_points):
        """Initialize points in a 3D cube grid"""
        # Find appropriate grid size
        grid_size = int(np.ceil(n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:n_points])

    def evaluate_initialization(points):
        """Fast evaluation of initialization quality"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 1e-12:
            return d_min / d_max
        return 0

    # Try multiple initialization strategies
    strategies = []

    # Strategy 1: Spherical Fibonacci points
    fib_points = initialize_spherical_points(14)
    fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("fibonacci", fib_points))

    # Strategy 2: Cube grid points
    cube_points = initialize_cube_grid_points(14)
    strategies.append(("cube_grid", cube_points))

    # Strategy 3: Random points
    np.random.seed(42)
    random_points = np.random.rand(14, 3)
    strategies.append(("random", random_points))

    # Strategy 4: Perturbed spherical points
    np.random.seed(42)
    perturbed_points = fib_points + np.random.normal(0, 0.05, (14, 3))
    perturbed_points = np.clip(perturbed_points, 0, 1)
    strategies.append(("perturbed", perturbed_points))

    # Evaluate all strategies and select the best
    best_initialization = None
    best_ratio = -np.inf

    for name, points in strategies:
        ratio = evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()

    # Use the best initialization as starting point
    x0 = best_initialization.flatten()

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Run optimization with reasonable settings
    result = differential_evolution(
        objective,
        bounds,
        x0=x0,
        seed=42,
        maxiter=1000,
        popsize=20,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False
    )

    # Extract the best solution
    points = result.x.reshape(-1, 3)

    return points


# EVOLVE-BLOCK-END