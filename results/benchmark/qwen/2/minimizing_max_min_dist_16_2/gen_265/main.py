# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    best_ratio = -np.inf
    best_points = None

    # Multiple restart strategies
    def _grid_perturbed_init():
        """Initialize with grid points plus adaptive random perturbations."""
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
        points = grid_points.astype(float) / 3.0  # Normalize to [0,1] range

        # Adaptive perturbation based on initial quality
        np.random.seed(42)
        points += np.random.uniform(-0.02, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def _hexagonal_init():
        """Initialize points using hexagonal packing pattern."""
        points = []
        rows, cols = 4, 4
        for i in range(rows):
            for j in range(cols):
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])
        return np.array(points)

    def _random_spread_init():
        """Initialize with random points that are intentionally spread out."""
        np.random.seed(42)
        points = np.random.rand(16, 2)

        # Apply basic spacing to prevent clustering
        for i in range(16):
            center_vec = points[i] - [0.5, 0.5]
            center_distance = np.linalg.norm(center_vec)
            if center_distance > 0:
                points[i] += center_vec * 0.1 / center_distance

        points = np.clip(points, 0, 1)
        return points

    def compute_ratio(points):
        """Compute the min/max distance ratio for given points."""
        distances = cdist(points, points, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist <= 0:
            return 0
        return min_dist / max_dist

    restart_strategies = [
        _grid_perturbed_init,
        _hexagonal_init,
        _random_spread_init
    ]

    # Function to perform local optimization with refinement
    def local_optimization_with_refinement(initial_points):
        """Perform local optimization followed by refinement."""
        # Define objective function: negative ratio (we'll minimize this)
        def objective(x):
            # Reshape flat array back to points
            pts = x.reshape(n, d)

            # Calculate all pairwise distances efficiently using cdist
            distances = cdist(pts, pts, metric='euclidean')
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            # Return negative ratio to maximize the ratio
            if max_dist <= 0:
                return 0
            return -min_dist / max_dist

        # Define symmetry-breaking constraints
        def symmetry_constraint(x):
            pts = x.reshape(n, d)
            # Fix bottom-left point at origin (breaks translation symmetry)
            con1 = pts[0, 0]  # x-coordinate of first point should be 0
            con2 = pts[0, 1]  # y-coordinate of first point should be 0
            return np.array([con1, con2])

        # Define lexicographic ordering constraint to break permutation symmetry
        def ordering_constraint(x):
            pts = x.reshape(n, d)
            constraints = []
            for i in range(1, n):
                # Each point should have x-coordinate >= previous point's x-coordinate
                constraints.append(pts[i, 0] - pts[i-1, 0])
                # If x-coordinates are equal, y-coordinate should be >= previous point's y-coordinate
                if pts[i, 0] == pts[i-1, 0]:
                    constraints.append(pts[i, 1] - pts[i-1, 1])
            return np.array(constraints)

        # Define bounds (points must be in [0,1] x [0,1])
        bounds = [(0, 1) for _ in range(n * d)]

        # Optimize using L-BFGS-B algorithm
        try:
            result = minimize(objective, initial_points.flatten(), method='L-BFGS-B', bounds=bounds,
                            options={'ftol': 1e-12, 'gtol': 1e-12})

            if result.success:
                # Extract optimized points
                optimized_points = result.x.reshape(n, d)

                # Perform a second optimization with SLSQP for further refinement
                try:
                    result2 = minimize(objective, optimized_points.flatten(), method='SLSQP', bounds=bounds,
                                     options={'ftol': 1e-12, 'gtol': 1e-12})
                    if result2.success:
                        optimized_points = result2.x.reshape(n, d)
                except:
                    pass

                return optimized_points
        except:
            return None
        return None

    # Try each initialization strategy multiple times
    for strategy_idx, init_func in enumerate(restart_strategies):
        for restart in range(3):  # 3 restarts per strategy
            np.random.seed(strategy_idx * 1000 + restart)

            # Get initial points
            points = init_func()

            # Perform local optimization
            optimized_points = local_optimization_with_refinement(points)

            if optimized_points is not None:
                # Calculate ratio for this optimization run
                ratio = compute_ratio(optimized_points)

                # Keep track of best solution
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()

    # Add evolutionary algorithm restarts for better global search
    try:
        # Define the objective function for differential evolution with constraints
        def de_objective(x):
            # Reshape x into points array
            points = x.reshape(-1, 2)

            # Calculate pairwise distances
            distances = cdist(points, points, metric='euclidean')

            # Calculate min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)

            # Avoid division by zero
            if d_max == 0:
                return -np.inf

            # Return negative ratio (since we want to maximize)
            return -d_min / d_max

        # Define constraints for differential evolution
        def de_constraints(x):
            points = x.reshape(-1, 2)
            # Fix bottom-left point at origin to break translation symmetry
            con1 = points[0, 0]  # x-coordinate of first point should be 0
            con2 = points[0, 1]  # y-coordinate of first point should be 0
            return np.array([con1, con2])

        # Run differential evolution for global search
        bounds = [(0, 1) for _ in range(n * d)]
        de_result = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )

        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            # Refine the DE result
            refined_de_points = local_optimization_with_refinement(de_points)
            if refined_de_points is not None:
                ratio = compute_ratio(refined_de_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_de_points.copy()

    except Exception as e:
        # Continue with existing best solution if DE fails
        pass

    # If no successful optimization was found, return the best we have
    if best_points is None:
        # Fallback to default initialization and optimization
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
        points = grid_points.astype(float) / 3.0
        np.random.seed(42)
        points += np.random.uniform(-0.02, 0.02, points.shape)
        points = np.clip(points, 0, 1)

        def objective(x):
            pts = x.reshape(n, d)
            distances = cdist(pts, pts, metric='euclidean')
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist <= 0:
                return 0
            return -min_dist / max_dist

        bounds = [(0, 1) for _ in range(n * d)]
        try:
            result = minimize(objective, points.flatten(), method='L-BFGS-B', bounds=bounds,
                            options={'ftol': 1e-12, 'gtol': 1e-12})
            if result.success:
                best_points = result.x.reshape(n, d)
        except:
            best_points = points

    return best_points


# EVOLVE-BLOCK-END