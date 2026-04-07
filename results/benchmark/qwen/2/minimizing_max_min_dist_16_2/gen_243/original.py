# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time

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

        # Avoid division by zero - return a very negative value for invalid cases
        if max_dist == 0:
            return -1e10

        return -min_dist / max_dist

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

    def create_spiral_pattern():
        """Create a spiral-like initial pattern"""
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles) * 0.8
        y = 0.5 + radii * np.sin(angles) * 0.8
        spiral_points = np.column_stack([x, y])
        return perturb_points(spiral_points, 0.02)

    def create_hexagonal_pattern():
        """Create a hexagonal pattern"""
        np.random.seed(42)
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                # Add small random perturbation
                x += (np.random.random() - 0.5) * 0.05
                y += (np.random.random() - 0.5) * 0.05
                points.append([x, y])
        return np.array(points)

    def create_voronoi_pattern():
        """Create a pattern based on Voronoi-like distribution"""
        np.random.seed(42)
        # Generate points clustered around a few centers
        centers = np.random.rand(4, 2)
        points = []
        for center in centers:
            for _ in range(4):
                point = center + np.random.normal(0, 0.08, 2)
                points.append(point)
        return np.clip(np.array(points), 0, 1)

    def create_refined_initial():
        """Create an initial configuration using differential evolution for better starting point"""
        # Use a simpler objective for finding a good initial configuration
        def simple_objective(x):
            points = x.reshape(-1, 2)
            distances = pdist(points)
            if len(distances) == 0:
                return 0
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 0
            # Prefer configurations where min_dist is not too small
            return -min_dist / max_dist

        # Bounds for differential evolution
        bounds = [(0, 1) for _ in range(32)]

        # Run differential evolution to get a good starting point
        try:
            de_result = differential_evolution(
                simple_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42
            )

            if de_result.success:
                return de_result.x.reshape(-1, 2)
        except:
            pass

        # Fallback to grid if DE fails
        return create_initial_grid()

    def multi_local_optimization(initial_points, max_evals=2000):
        """Apply multiple local optimization methods to refine solution"""
        best_points = initial_points.copy()
        best_ratio = evaluate_solution(best_points)

        # Try multiple optimization methods
        methods = ['L-BFGS-B', 'SLSQP']

        for method in methods:
            try:
                x0 = best_points.flatten()
                bounds = [(0, 1) for _ in range(32)]

                # Use different tolerances for different methods
                if method == 'L-BFGS-B':
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                else:
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                    )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = evaluate_solution(final_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

            except Exception as e:
                continue

        return best_points, best_ratio

    def evolutionary_restart_strategy(max_evals=1000):
        """Implement evolutionary algorithm restarts to provide better starting points"""
        # Use differential evolution with multiple random seeds for global search
        best_evolutionary_solution = None
        best_evolutionary_ratio = -np.inf

        # Try multiple DE runs with different seeds
        for seed_val in [42, 123, 456, 789]:
            try:
                # Simple objective for evolutionary search
                def de_objective(x):
                    points = x.reshape(-1, 2)
                    distances = pdist(points)
                    if len(distances) == 0:
                        return 0
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist == 0:
                        return 0
                    # Prefer configurations with larger minimum distances
                    return -min_dist / max_dist

                bounds = [(0, 1) for _ in range(32)]

                # Run differential evolution with specific seed
                de_result = differential_evolution(
                    de_objective,
                    bounds,
                    maxiter=30,   # Fewer iterations for speed
                    popsize=20,   # Larger population for better exploration
                    tol=1e-6,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=seed_val,
                    workers=1
                )

                if de_result.success:
                    solution_points = de_result.x.reshape(-1, 2)
                    ratio = evaluate_solution(solution_points)
                    if ratio > best_evolutionary_ratio:
                        best_evolutionary_ratio = ratio
                        best_evolutionary_solution = solution_points.copy()

            except Exception as e:
                continue

        return best_evolutionary_solution, best_evolutionary_ratio

    # Enhanced multi-start optimization with intelligent strategy selection
    best_ratio = 0
    best_points = None

    # Strategy 1: Evolutionary restarts to diversify starting points
    try:
        np.random.seed(42)
        evolutionary_points, evolutionary_ratio = evolutionary_restart_strategy()
        if evolutionary_points is not None:
            refined_points, ratio = multi_local_optimization(evolutionary_points, 1000)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
    except Exception as e:
        pass

    # Strategy 2: Differential evolution for global search
    try:
        np.random.seed(42)
        refined_initial = create_refined_initial()
        refined_points, ratio = multi_local_optimization(refined_initial, 1000)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined_points.copy()
    except Exception as e:
        pass

    # Strategy 3: Multiple grid-based patterns with varying perturbations
    try:
        grid_points = create_initial_grid()
        perturbation_levels = [0.005, 0.01, 0.02, 0.05]
        for mag in perturbation_levels:
            for _ in range(2):  # Try 2 variations per perturbation level
                perturbed = perturb_points(grid_points, mag)
                refined_points, ratio = multi_local_optimization(perturbed, 500)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
    except Exception as e:
        pass

    # Strategy 4: Spiral pattern
    try:
        spiral_points = create_spiral_pattern()
        refined_points, ratio = multi_local_optimization(spiral_points, 500)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined_points.copy()
    except Exception as e:
        pass

    # Strategy 5: Hexagonal pattern
    try:
        hex_points = create_hexagonal_pattern()
        refined_points, ratio = multi_local_optimization(hex_points, 500)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined_points.copy()
    except Exception as e:
        pass

    # Strategy 6: Voronoi-like pattern
    try:
        voronoi_points = create_voronoi_pattern()
        refined_points, ratio = multi_local_optimization(voronoi_points, 500)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined_points.copy()
    except Exception as e:
        pass

    # Strategy 7: Random initialization with refinement
    try:
        for i in range(5):  # More random starts
            np.random.seed(42 + i * 10)  # Different seed sequence
            random_points = np.random.rand(16, 2)
            refined_points, ratio = multi_local_optimization(random_points, 500)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
    except Exception as e:
        pass

    # Final refinement with high-precision optimization
    if best_points is not None and best_ratio < 0.3:  # Only if we haven't found a very good solution yet
        try:
            # Do one final high-precision optimization
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception as e:
            pass

    # If no good solution was found, return a default configuration
    if best_points is None:
        # Fallback to refined initial configuration
        best_points = create_refined_initial()

    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)

    return best_points

# EVOLVE-BLOCK-END