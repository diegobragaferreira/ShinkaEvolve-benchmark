# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
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

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0

        # Efficiently compute all pairwise distances
        distances = cdist(points, points, metric='euclidean')

        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist <= 0:
            return 0

        return min_dist / max_dist

    # Multiple restart strategies with adaptive initialization
    def _hexagonal_packing_init():
        """Initialize points using hexagonal packing pattern for better spacing."""
        points = []

        for i in range(4):
            for j in range(4):
                # offset every other row for hexagonal packing
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) * 0.25 + 0.125  # Scale and shift to [0.125, 0.875]
                y = i * 0.25 + 0.125
                points.append([x, y])

        return np.array(points)

    def _adaptive_grid_init():
        """Initialize with grid points plus adaptive random perturbations."""
        # Start with a regular grid
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)])
        points = grid_points.astype(float) / 3.0  # Normalize to [0,1] range

        # Calculate initial ratio to determine perturbation scale
        initial_ratio = calculate_min_max_ratio(points)

        # Adaptive perturbation scaling based on configuration quality
        if initial_ratio > 0.1:  # If already reasonably balanced
            perturbation_magnitude = 0.015
        elif initial_ratio > 0.05:  # Moderately unbalanced
            perturbation_magnitude = 0.03
        else:  # Very unbalanced, allow larger perturbations
            perturbation_magnitude = 0.06

        # Add random perturbations
        np.random.seed(42)
        points += np.random.uniform(-perturbation_magnitude, perturbation_magnitude, points.shape)

        # Ensure points stay within bounds
        points = np.clip(points, 0, 1)

        return points

    def _improved_random_init():
        """Initialize with better spread random points."""
        np.random.seed(42)
        points = np.random.rand(16, 2)

        # Apply basic spacing to prevent extreme clustering
        for i in range(16):
            # Move points away from edges slightly
            edge_distance = np.minimum(points[i], 1 - points[i])
            if np.min(edge_distance) < 0.1:
                points[i] = np.clip(points[i] + np.random.normal(0, 0.02, 2), 0, 1)

        return points

    def _differential_evolution_restart():
        """Use differential evolution to find a better starting point."""
        def objective(x):
            points = x.reshape(-1, 2)
            distances = cdist(points, points, metric='euclidean')
            np.fill_diagonal(distances, np.inf)

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist <= 0:
                return 0
            return -min_dist / max_dist

        bounds = [(0, 1) for _ in range(n * d)]

        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-10,
                mutation=(0.7, 1),
                recombination=0.9
            )

            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass

        return None

    def _golden_spiral_init():
        """Initialize points using a golden spiral pattern for better dispersion."""
        # Golden angle in radians
        golden_angle = 2.399963229728653  # 2π(1 - 1/φ) where φ is the golden ratio

        points = []
        for i in range(n):
            radius = np.sqrt(i / (n - 1)) if n > 1 else 0
            angle = i * golden_angle

            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4

            points.append([x, y])

        return np.array(points)

    # Try each initialization strategy multiple times
    restart_strategies = [
        _hexagonal_packing_init,
        _adaptive_grid_init,
        _improved_random_init,
        _golden_spiral_init
    ]

    # Add differential evolution restart as a fourth strategy
    evolutions = []
    for i in range(3):  # Try DE restart 3 times with different seeds
        try:
            np.random.seed(1000 + i)
            de_points = _differential_evolution_restart()
            if de_points is not None:
                evolutions.append(de_points)
        except:
            continue

    # Combine all initialization approaches
    all_initializations = []
    for strategy in restart_strategies:
        for restart in range(2):  # 2 restarts per strategy
            np.random.seed(100 + restart)  # Different seeds for diversity
            points = strategy()
            all_initializations.append(points)

    # Add DE results
    for de_points in evolutions:
        all_initializations.append(de_points)

    # Process all initial configurations with optimized pipeline
    max_time = 170  # Leave some buffer for final steps
    start_time = time.time()

    for i, initial_points in enumerate(all_initializations):
        if time.time() - start_time > max_time:
            break

        try:
            # Stage 1: Coarse optimization with L-BFGS-B
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

            # Add symmetry-breaking constraint to avoid degenerate configurations
            def symmetry_breaking_constraint(x):
                points = x.reshape(n, d)
                # Fix bottom-left point to break symmetry
                return np.array([points[0, 0] - 0.0, points[0, 1] - 0.0])  # Fix point 0 at (0,0)

            # Initial coarse optimization
            result1 = minimize(
                objective,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-8, 'gtol': 1e-8, 'maxiter': 200}
            )

            if not result1.success:
                continue

            optimized_points = result1.x.reshape(n, d)
            optimized_points = np.clip(optimized_points, 0, 1)

            # Stage 2: Local refinement with adaptive hill climbing
            def local_refinement(points, max_iter=100):
                current_points = points.copy()
                current_ratio = calculate_min_max_ratio(current_points)

                # Adaptive step sizes based on current performance
                for iteration in range(max_iter):
                    if time.time() - start_time > max_time:
                        break

                    best_improvement = 0
                    best_new_points = current_points.copy()

                    # Dynamic step size adjustment
                    if iteration < 30:
                        step_size = 0.02  # Larger steps initially
                    elif iteration < 60:
                        step_size = 0.01  # Medium steps
                    else:
                        step_size = 0.005  # Fine-tune steps

                    # Perturb each point multiple times
                    for i in range(n):
                        temp_points = current_points.copy()

                        # Try multiple random moves for this point
                        for _ in range(3):
                            move = np.random.uniform(-step_size, step_size, 2)
                            temp_points[i] = current_points[i] + move

                            # Keep within bounds
                            temp_points[i] = np.clip(temp_points[i], 0, 1)

                            new_ratio = calculate_min_max_ratio(temp_points)

                            if new_ratio > current_ratio:
                                improvement = new_ratio - current_ratio
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_new_points = temp_points.copy()

                    # Accept improvement if found
                    if best_improvement > 1e-12:
                        current_points = best_new_points
                        current_ratio = calculate_min_max_ratio(current_points)
                    else:
                        # Reduce step size if no progress
                        step_size *= 0.95
                        if step_size < 1e-6:
                            break

                return current_points

            refined_points = local_refinement(optimized_points)
            ratio = calculate_min_max_ratio(refined_points)

            # Keep track of best solution
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()

        except Exception:
            continue

    # If no successful optimization was found, return the best we have
    if best_points is None:
        # Fallback to hexagonal grid with symmetry breaking
        points = _hexagonal_packing_init()
        # Apply small perturbations to break any remaining symmetries
        np.random.seed(42)
        points += np.random.uniform(-0.01, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        # Manually enforce symmetry breaking constraint
        points[0] = [0.0, 0.0]  # Fix bottom-left point
        best_points = points

    return best_points

# EVOLVE-BLOCK-END