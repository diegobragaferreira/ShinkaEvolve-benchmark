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

    def adaptive_perturbation_scaling(current_ratio, initial_ratio, iteration=0, max_iterations=100):
        """Adaptively scale perturbation based on optimization progress."""
        # Start with larger perturbations and decrease over time
        # If we're far from good solution (low initial ratio), use larger perturbations
        # If we're approaching good solution (high current ratio), use smaller perturbations
        progress = min(iteration / max_iterations, 1.0)

        # Dynamic scaling factor that decreases over iterations
        base_factor = 1.0 - 0.8 * progress

        # Adjust based on how close we are to our current best solution
        if current_ratio < 0.1:  # Very poor solution
            scale_factor = 1.5 * base_factor
        elif current_ratio < 0.2:  # Poor solution
            scale_factor = 1.2 * base_factor
        elif current_ratio < 0.3:  # Moderate solution
            scale_factor = 0.8 * base_factor
        else:  # Good solution
            scale_factor = 0.5 * base_factor

        return max(0.1, scale_factor)

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
        """Perform multi-start optimization with different initial configurations."""
        best_ratio = -np.inf
        best_points = None

        # Try different initialization strategies with improved diversity
        initial_configs = []

        # Strategy 1: Better hexagonal initialization
        hex_initial = create_better_hexagonal_initialization()
        initial_configs.append(create_perturbed_initialization(hex_initial, 0.015))

        # Strategy 2: Concentric ring initialization
        ring_initial = create_concentric_ring_initialization()
        initial_configs.append(create_perturbed_initialization(ring_initial, 0.02))

        # Strategy 3: Fibonacci-like arrangement
        fib_initial = create_fibonacci_sphere_like_initialization()
        initial_configs.append(create_perturbed_initialization(fib_initial, 0.01))

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
        initial_configs.append(grid_initial)

        # Strategy 5: Pure random with better seed control
        np.random.seed(42)
        random_initial = np.random.rand(16, 2)
        initial_configs.append(random_initial)

        # Strategy 6: Square grid with custom spacing
        square_grid = np.zeros((16, 2))
        points_per_row = 4
        spacing = 1.0 / (points_per_row - 1)
        idx = 0
        for i in range(points_per_row):
            for j in range(points_per_row):
                if idx < 16:
                    square_grid[idx] = [j * spacing, i * spacing]
                    idx += 1
        # Add perturbation
        square_grid = create_perturbed_initialization(square_grid, 0.02)
        initial_configs.append(square_grid)

        # Strategy 7: Spider web pattern
        spider_web = np.zeros((16, 2))
        center = [0.5, 0.5]
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.8, 16)
        for i in range(16):
            spider_web[i] = [center[0] + radii[i]*np.cos(angles[i]),
                           center[1] + radii[i]*np.sin(angles[i])]
        # Clip to bounds
        spider_web = np.clip(spider_web, 0, 1)
        initial_configs.append(spider_web)

        # Try each initialization with hybrid optimization
        for i, initial_config in enumerate(initial_configs):
            try:
                # Compute adaptive scaling factor based on initial quality
                initial_ratio = calculate_min_max_ratio(initial_config)
                # Scale perturbation inversely proportional to current ratio
                # When ratio is low (bad), use larger perturbations; when ratio is high (good), use smaller perturbations
                adaptive_factor = max(0.5, 1.0 - initial_ratio * 2.0)  # Between 0.5 and 1.0

                # First perform global optimization with DE
                bounds = [(0, 1)] * 32

                # Use differential evolution for global search with better parameters
                de_result = differential_evolution(
                    lambda x: objective_function(x.reshape(-1, 2)),
                    bounds,
                    seed=42 + i,
                    maxiter=300,  # Reduced iterations for efficiency
                    popsize=25,    # Increased population size for better exploration
                    mutation=(0.5, 1),
                    recombination=0.7,
                    tol=1e-8,      # Tighter tolerance
                    disp=False
                )

                # Refine with local optimization using adaptive perturbation
                refined_points = optimize_with_local_refinement(de_result.x.reshape(-1, 2), 400)
                ratio = calculate_min_max_ratio(refined_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points

            except Exception as e:
                # If optimization fails, continue with next configuration
                continue

        # If no better solution found, return the best local optimization
        if best_points is None:
            # Fallback to single local optimization with the best hexagonal initialization
            hex_initial = create_better_hexagonal_initialization()
            # Use adaptive scaling for the fallback
            initial_ratio = calculate_min_max_ratio(hex_initial)
            adaptive_factor = max(0.5, 1.0 - initial_ratio * 2.0)
            perturbed_hex = create_perturbed_initialization(hex_initial, 0.008, adaptive_factor)
            best_points = optimize_with_local_refinement(perturbed_hex, 600)

        return best_points

    # Main optimization routine
    np.random.seed(42)

    # Use multi-start optimization approach
    final_points = multi_start_optimization()

    return final_points


# EVOLVE-BLOCK-END