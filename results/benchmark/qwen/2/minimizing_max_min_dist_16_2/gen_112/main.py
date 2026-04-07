# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0

        # Calculate pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0

        return dmin / dmax

    def objective_function(points):
        """Objective function to maximize (negative because we minimize in scipy)."""
        return -calculate_min_max_ratio(points)

    def create_better_hexagonal_initialization():
        """Create a more sophisticated hexagonal-like arrangement of points."""
        points = np.zeros((16, 2))

        # Create a more regular hexagonal arrangement with better spacing
        # Using a hexagonal grid pattern with 4 rows and 4 columns with offset
        row_positions = [0, 1, 2, 3]
        col_positions = [0, 1, 2, 3]
        spacing_x = 1.0 / 4.0
        spacing_y = spacing_x * np.sqrt(3) / 2.0

        idx = 0
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                if idx < 16:
                    # Offset every other row for proper hexagonal packing
                    x = (col + 0.5 * (row % 2)) * spacing_x
                    y = row * spacing_y
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_concentric_ring_initialization():
        """Create a concentric ring-like arrangement."""
        points = np.zeros((16, 2))

        # Place points in concentric rings
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.8, 4)  # Four layers
        layer_points = [4, 4, 4, 4]  # 4 points per layer

        idx = 0
        for i, radius in enumerate(radii):
            num_points_in_layer = layer_points[i]
            layer_angles = np.linspace(0, 2*np.pi, num_points_in_layer, endpoint=False)
            for angle in layer_angles:
                if idx < 16:
                    x = 0.5 + radius * np.cos(angle)
                    y = 0.5 + radius * np.sin(angle)
                    points[idx] = [x, y]
                    idx += 1

        return points

    def create_fibonacci_sphere_like_initialization():
        """Create a Fibonacci-like arrangement for better point distribution."""
        points = np.zeros((16, 2))

        # Use Fibonacci-inspired pattern in 2D
        golden_ratio = (1 + np.sqrt(5)) / 2.0
        for i in range(16):
            theta = 2 * np.pi * i / golden_ratio
            r = np.sqrt(i / 15.0)  # Normalize to [0,1]
            x = 0.5 + r * np.cos(theta) * 0.8
            y = 0.5 + r * np.sin(theta) * 0.8
            points[i] = [x, y]

        return points

    def create_perturbed_initialization(base_points, perturbation_magnitude=0.015, adaptive_scale=1.0):
        """Create a perturbed version of base initialization with adaptive scaling."""
        perturbed = base_points.copy()
        # Add random perturbation scaled by adaptive factor
        actual_perturbation = perturbation_magnitude * adaptive_scale
        perturbed += np.random.normal(0, actual_perturbation, base_points.shape)
        # Ensure points stay within bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def create_adaptive_perturbed_initialization(base_points, initial_ratio, iteration=0):
        """Create an adaptive perturbed initialization based on current optimization state."""
        # Base perturbation magnitude
        base_perturbation = 0.015

        # Adaptive scaling based on initial quality and optimization iteration
        if initial_ratio < 0.1:
            # Poor initial configuration - use larger perturbations to explore more
            perturbation_magnitude = base_perturbation * (1.0 + (0.1 - initial_ratio) * 5.0)
        elif initial_ratio > 0.25:
            # Good initial configuration - use smaller perturbations to refine
            perturbation_magnitude = base_perturbation * max(0.1, 1.0 - (initial_ratio - 0.25) * 2.0)
        else:
            # Medium quality - use moderate perturbations
            perturbation_magnitude = base_perturbation

        # Additional adjustment based on iteration (decrease over time)
        if iteration > 0:
            perturbation_magnitude *= max(0.1, 1.0 - iteration * 0.02)

        perturbed = base_points.copy()
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def optimize_with_local_refinement(initial_points, max_iter=500):
        """Perform local optimization refinement on initial configuration."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]

        # Optimize using L-BFGS-B method with stricter tolerances
        result = minimize(
            lambda flat_points: objective_function(flat_points.reshape(-1, 2)),
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10},
            callback=None
        )

        # Extract optimized points
        optimized_points = result.x.reshape(-1, 2)

        # Ensure all points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)

        return optimized_points

    def multi_start_optimization():
        """Perform multi-start optimization with different initial configurations and strategies."""
        best_ratio = -np.inf
        best_points = None

        # Try different initialization strategies with improved diversity
        initial_configs = []

        # Strategy 1: Better hexagonal initialization
        hex_initial = create_better_hexagonal_initialization()
        initial_configs.append(('hex', hex_initial))

        # Strategy 2: Concentric ring initialization
        ring_initial = create_concentric_ring_initialization()
        initial_configs.append(('ring', ring_initial))

        # Strategy 3: Fibonacci-like arrangement
        fib_initial = create_fibonacci_sphere_like_initialization()
        initial_configs.append(('fibonacci', fib_initial))

        # Strategy 4: Modified grid initialization
        grid_initial = np.zeros((16, 2))
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx < 16:
                    x = j / 3.0 if j > 0 else 0.0
                    y = i / 3.0 if i > 0 else 0.0
                    # Add slight perturbation to break symmetry
                    x = min(1.0, max(0.0, x + np.random.normal(0, 0.02)))
                    y = min(1.0, max(0.0, y + np.random.normal(0, 0.02)))
                    grid_initial[idx] = [x, y]
                    idx += 1
        initial_configs.append(('grid', grid_initial))

        # Strategy 5: Pure random with better seed control
        np.random.seed(42)
        random_initial = np.random.rand(16, 2)
        initial_configs.append(('random', random_initial))

        # Try each initialization with different optimization approaches
        for i, (config_type, initial_config) in enumerate(initial_configs):
            try:
                # Evaluate initial configuration quality
                initial_ratio = calculate_min_max_ratio(initial_config)

                # Choose optimization approach based on initial quality
                if initial_ratio < 0.1:
                    # Very poor initial configuration - aggressive optimization
                    print(f"Config {config_type}: Poor initial quality ({initial_ratio:.4f}), using aggressive approach")
                    # Use stronger global optimization and multiple refinement passes
                    perturbed_config = create_adaptive_perturbed_initialization(initial_config, initial_ratio, 0)

                    # First, use differential evolution with more iterations
                    bounds = [(0, 1)] * 32
                    de_result = differential_evolution(
                        lambda x: objective_function(x.reshape(-1, 2)),
                        bounds,
                        seed=42 + i,
                        maxiter=500,  # More iterations for poor starts
                        popsize=30,    # Larger population for better exploration
                        mutation=(0.5, 1),
                        recombination=0.7,
                        tol=1e-8,
                        disp=False
                    )

                    # Multiple rounds of local refinement with decreasing perturbations
                    current_points = de_result.x.reshape(-1, 2)
                    best_refined_points = current_points
                    best_refined_ratio = calculate_min_max_ratio(current_points)

                    for round_num in range(3):
                        # Decrease perturbation strength for later rounds
                        perturbation_factor = 0.8 ** round_num
                        refined_points = optimize_with_local_refinement(current_points, 300)
                        refined_ratio = calculate_min_max_ratio(refined_points)

                        if refined_ratio > best_refined_ratio:
                            best_refined_ratio = refined_ratio
                            best_refined_points = refined_points

                        # Use even smaller perturbations for next round
                        current_points = best_refined_points
                        current_points = create_adaptive_perturbed_initialization(
                            current_points, refined_ratio, round_num + 1)

                    final_ratio = best_refined_ratio
                    final_points = best_refined_points

                elif initial_ratio > 0.25:
                    # Good initial configuration - focused refinement
                    print(f"Config {config_type}: Good initial quality ({initial_ratio:.4f}), using focused approach")
                    # Use less aggressive but more precise optimization
                    perturbed_config = create_adaptive_perturbed_initialization(initial_config, initial_ratio, 0)

                    # Use DE with fewer iterations since quality is already good
                    bounds = [(0, 1)] * 32
                    de_result = differential_evolution(
                        lambda x: objective_function(x.reshape(-1, 2)),
                        bounds,
                        seed=42 + i,
                        maxiter=200,
                        popsize=20,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        tol=1e-8,
                        disp=False
                    )

                    # Single round of local refinement
                    final_points = optimize_with_local_refinement(de_result.x.reshape(-1, 2), 500)
                    final_ratio = calculate_min_max_ratio(final_points)

                else:
                    # Medium quality - balanced approach
                    print(f"Config {config_type}: Medium initial quality ({initial_ratio:.4f}), using balanced approach")
                    # Standard approach
                    perturbed_config = create_adaptive_perturbed_initialization(initial_config, initial_ratio, 0)

                    bounds = [(0, 1)] * 32
                    de_result = differential_evolution(
                        lambda x: objective_function(x.reshape(-1, 2)),
                        bounds,
                        seed=42 + i,
                        maxiter=300,
                        popsize=25,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        tol=1e-8,
                        disp=False
                    )

                    # One round of local refinement
                    final_points = optimize_with_local_refinement(de_result.x.reshape(-1, 2), 400)
                    final_ratio = calculate_min_max_ratio(final_points)

                if final_ratio > best_ratio:
                    best_ratio = final_ratio
                    best_points = final_points

            except Exception as e:
                # If optimization fails, continue with next configuration
                print(f"Config {config_type} failed with error: {e}")
                continue

        # If no better solution found, return the best local optimization
        if best_points is None:
            # Fallback to single local optimization with the best hexagonal initialization
            hex_initial = create_better_hexagonal_initialization()
            initial_ratio = calculate_min_max_ratio(hex_initial)
            perturbed_hex = create_adaptive_perturbed_initialization(hex_initial, initial_ratio, 0)
            best_points = optimize_with_local_refinement(perturbed_hex, 600)

        return best_points

    # Main optimization routine
    np.random.seed(42)

    # Use multi-start optimization approach
    final_points = multi_start_optimization()

    return final_points


# EVOLVE-BLOCK-END