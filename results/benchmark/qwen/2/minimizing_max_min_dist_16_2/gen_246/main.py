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
        points = np.array([[x, y] for x in x_vals for y in y_vals])

        # Add symmetry breaking constraints by fixing corner points
        # Bottom-left corner
        points[0] = [0.0, 0.0]
        # Bottom-right corner
        points[3] = [1.0, 0.0]
        # Top-left corner
        points[12] = [0.0, 1.0]
        # Top-right corner
        points[15] = [1.0, 1.0]

        return points

    def perturb_points(points, magnitude):
        """Add controlled random perturbations"""
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        return np.clip(perturbed, 0, 1)

    def compute_current_ratio(points):
        """Compute current min/max distance ratio for adaptive scaling"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return min_dist / max_dist

    def adaptive_perturb_points(points, base_magnitude=0.02):
        """Add random perturbations with adaptive magnitude based on current configuration quality"""
        # Calculate current ratio to determine perturbation size
        current_ratio = compute_current_ratio(points)

        # Also analyze the variance in distances to understand distribution quality
        distances = pdist(points)
        if len(distances) > 0:
            distance_variance = np.var(distances)
            # If distances are very uniform (low variance), we're likely in a good configuration
            # If distances vary greatly (high variance), we might need more exploration
            variance_factor = 1.0 + 0.5 * np.tanh(distance_variance - 0.2)
        else:
            variance_factor = 1.0

        # Scale perturbation based on multiple factors
        # If ratio is very low (bad configuration), use larger perturbations
        # If ratio is high (good configuration), use smaller perturbations
        if current_ratio < 0.1:
            # Very poor configuration - use larger perturbations for exploration
            magnitude = base_magnitude * 2.0 * variance_factor
        elif current_ratio < 0.2:
            # Poor configuration - use medium perturbations
            magnitude = base_magnitude * 1.5 * variance_factor
        else:
            # Reasonable configuration - use small perturbations for fine-tuning
            magnitude = base_magnitude * 0.5 * variance_factor

        return perturb_points(points, magnitude)

    def create_spiral_pattern():
        """Create a spiral-like initial pattern"""
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles) * 0.8
        y = 0.5 + radii * np.sin(angles) * 0.8
        spiral_points = np.column_stack([x, y])

        # Apply symmetry breaking by fixing corners
        spiral_points[0] = [0.0, 0.0]      # bottom-left corner
        spiral_points[3] = [1.0, 0.0]      # bottom-right corner
        spiral_points[12] = [0.0, 1.0]     # top-left corner
        spiral_points[15] = [1.0, 1.0]     # top-right corner

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

    def evolutionary_multistart():
        """Use evolutionary algorithm for global search to find multiple good starting points"""
        # Define objective for differential evolution (minimize negative ratio)
        def de_objective(x):
            points = x.reshape(-1, 2)
            distances = pdist(points)
            if len(distances) == 0:
                return 0
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return -1e10
            return -min_dist / max_dist

        # Bounds for differential evolution
        bounds = [(0, 1) for _ in range(32)]

        # Run differential evolution to get multiple good starting points
        try:
            # Use a more exploratory approach with higher population size
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=30,      # Reduced iterations to save time
                popsize=20,      # Larger population for better exploration
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                workers=1
            )

            if de_result.success:
                # Get the best solution from differential evolution
                best_de_points = de_result.x.reshape(-1, 2)
                return [best_de_points]
        except Exception as e:
            pass

        return []

    def local_refinement(points_list):
        """Refine a list of point configurations using local optimization"""
        refined_results = []
        for i, points in enumerate(points_list):
            try:
                # Apply local optimization to each starting point
                x0 = points.flatten()
                bounds = [(0, 1) for _ in range(32)]

                # Use multiple local optimizers for robustness
                for method in ['L-BFGS-B', 'SLSQP']:
                    result = minimize(
                        objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                    )

                    if result.success:
                        refined_points = result.x.reshape(-1, 2)
                        refined_results.append(refined_points)
                        break  # Break on first success
            except Exception as e:
                continue
        return refined_results

    # Multi-start optimization with multiple strategies
    best_ratio = 0
    best_points = None
    strategies_tried = 0

    # Strategy 1: Evolutionary algorithm for global search
    try:
        evolutions = evolutionary_multistart()
        if evolutions:
            # Refine the results from evolution
            refined_evolutions = local_refinement(evolutions)
            for refined_points in refined_evolutions:
                ratio = evaluate_solution(refined_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
    except Exception as e:
        pass

    # Strategy 2: Regular grid with adaptive perturbations
    try:
        grid_points = create_initial_grid()
        for i in range(3):  # Try multiple perturbations of grid
            # Use adaptive perturbation sizing
            perturbed = adaptive_perturb_points(grid_points, 0.02)
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

    # Strategy 3: Spiral pattern with adaptive perturbations
    try:
        spiral_points = create_spiral_pattern()
        # Apply adaptive perturbation to spiral pattern
        perturbed = adaptive_perturb_points(spiral_points, 0.03)
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

    # Strategy 4: Hexagonal pattern with adaptive perturbations
    try:
        hex_points = create_hexagonal_pattern()
        # Apply adaptive perturbation to hexagonal pattern
        perturbed = adaptive_perturb_points(hex_points, 0.02)
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

    # Strategy 5: Random initialization with multiple tries and adaptive perturbations
    try:
        for i in range(3):  # Multiple random starts
            np.random.seed(42 + i)  # Different seed for each attempt
            random_points = np.random.rand(16, 2)
            # Apply adaptive perturbation to random initialization
            perturbed = adaptive_perturb_points(random_points, 0.03)
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

    # Strategy 6: Try SLSQP as a fallback with better tolerance settings
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