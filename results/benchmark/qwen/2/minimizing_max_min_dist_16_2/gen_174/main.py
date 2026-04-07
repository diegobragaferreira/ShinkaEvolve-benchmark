# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
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

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

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

    def perturb_points_adaptive(points, current_ratio):
        """Add adaptive random perturbations based on current solution quality"""
        # Scale perturbation magnitude inversely with the current ratio
        # If ratio is low (bad solution), use larger perturbations
        # If ratio is high (good solution), use smaller perturbations
        base_magnitude = 0.05
        # The adjustment factor ensures we don't perturb too aggressively when solution is already good
        adjustment_factor = max(0.1, 1.0 - current_ratio * 2.0)  # Decrease perturbation as ratio increases
        magnitude = base_magnitude * adjustment_factor
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        return np.clip(perturbed, 0, 1)

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

    def create_spiral_pattern():
        """Create a spiral-like initial pattern"""
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles) * 0.8
        y = 0.5 + radii * np.sin(angles) * 0.8
        spiral_points = np.column_stack([x, y])
        return perturb_points(spiral_points, 0.02)

    def create_random_initial():
        """Create random initial points"""
        np.random.seed(42)
        return np.random.rand(16, 2)

    # Multi-start optimization with multiple strategies
    best_ratio = 0
    best_points = None
    strategies_tried = 0

    # Strategy 1: Regular grid with adaptive perturbations
    try:
        grid_points = create_initial_grid()
        # Use adaptive perturbations that scale with solution quality
        # Start with a baseline configuration to estimate initial ratio
        x0_baseline = grid_points.flatten()
        baseline_result = minimize(
            objective,
            x0_baseline,
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(32)],
            options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        baseline_ratio = evaluate_solution(baseline_result.x.reshape(-1, 2)) if baseline_result.success else 0

        # Try multiple adaptive perturbations to avoid local minima
        for i in range(5):  # More attempts with different adaptive scales
            # Use adaptive perturbation based on current solution quality
            perturbed = perturb_points_adaptive(grid_points, baseline_ratio)
            x0 = perturbed.flatten()
            bounds = [(0, 1) for _ in range(32)]

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
            strategies_tried += 1
    except Exception as e:
        pass

    # Strategy 2: Hexagonal pattern
    try:
        hex_points = create_hexagonal_pattern()
        x0 = hex_points.flatten()
        bounds = [(0, 1) for _ in range(32)]

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if result.success:
            final_points = result.x.reshape(-1, 2)
            ratio = evaluate_solution(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
        strategies_tried += 1
    except Exception as e:
        pass

    # Strategy 3: Spiral pattern
    try:
        spiral_points = create_spiral_pattern()
        x0 = spiral_points.flatten()
        bounds = [(0, 1) for _ in range(32)]

        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        if result.success:
            final_points = result.x.reshape(-1, 2)
            ratio = evaluate_solution(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
        strategies_tried += 1
    except Exception as e:
        pass

    # Strategy 4: Random initialization with adaptive perturbations
    try:
        for i in range(3):  # Multiple random starts
            np.random.seed(42 + i)  # Different seed for each attempt
            random_points = np.random.rand(16, 2)
            # Apply adaptive perturbation to random points
            adaptive_random_points = perturb_points_adaptive(random_points, 0.0)  # No prior ratio info
            x0 = adaptive_random_points.flatten()
            bounds = [(0, 1) for _ in range(32)]

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
            strategies_tried += 1
    except Exception as e:
        pass

    # Strategy 5: Try SLSQP as a fallback with better tolerance settings
    if best_points is None or strategies_tried < 2:
        try:
            grid_points = create_initial_grid()
            x0 = grid_points.flatten()
            bounds = [(0, 1) for _ in range(32)]

            result = minimize(
                objective,
                x0,
                method='SLSQP',
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
            pass

    # If no good solution was found, return a default configuration
    if best_points is None:
        # Fallback to simple grid initialization
        grid_points = create_initial_grid()
        best_points = grid_points.copy()

    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)

    return best_points

# EVOLVE-BLOCK-END