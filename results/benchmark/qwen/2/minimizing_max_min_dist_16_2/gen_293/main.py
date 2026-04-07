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

    def create_asymmetric_initial():
        """Create an asymmetric initial configuration to break symmetry"""
        # Start with a structured pattern but break symmetry explicitly
        np.random.seed(42)

        # Create a 4x4 grid with some deliberate asymmetry
        points = np.zeros((16, 2))
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.05  # Slight offset for odd rows
                y = i * 0.25 + (j % 2) * 0.03  # Slight offset for even columns
                points[i*4 + j] = [x, y]

        # Add small random perturbation
        points += np.random.normal(0, 0.02, points.shape)

        # Ensure points are within bounds
        points = np.clip(points, 0, 1)

        # Explicitly fix corner points to break rotational symmetry
        points[0] = [0.0, 0.0]      # bottom-left
        points[3] = [1.0, 0.0]      # bottom-right
        points[12] = [0.0, 1.0]     # top-left
        points[15] = [1.0, 1.0]     # top-right

        return points

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

        # Fallback to asymmetric initial if DE fails
        return create_asymmetric_initial()

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

    def hierarchical_optimization(max_time=170):
        """Hierarchical optimization approach: coarse to fine search"""
        start_time = time.time()
        best_ratio = -np.inf
        best_points = None

        # Phase 1: Coarse search using 2x2 grid to identify promising regions
        coarse_configs = []

        # Create 2x2 grid points to sample different regions
        coarse_x = np.linspace(0.1, 0.9, 2)
        coarse_y = np.linspace(0.1, 0.9, 2)
        coarse_x_grid, coarse_y_grid = np.meshgrid(coarse_x, coarse_y)

        # Generate coarse grid points to explore different areas
        for i in range(2):
            for j in range(2):
                # Create a dense set of points around each coarse grid point
                coarse_center = np.array([coarse_x_grid[i,j], coarse_y_grid[i,j]])
                # Generate 4 points around each coarse center
                for dx in [-0.15, -0.05, 0.05, 0.15]:
                    for dy in [-0.15, -0.05, 0.05, 0.15]:
                        point = coarse_center + np.array([dx, dy])
                        if 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                            coarse_configs.append(point)

        # Create configurations with points spread out across the coarse grids
        coarse_point_sets = []
        for coarse_point in coarse_configs:
            # Create a configuration around the coarse point
            config = np.array([coarse_point + np.random.normal(0, 0.05, 2) for _ in range(16)])
            config = np.clip(config, 0, 1)
            coarse_point_sets.append(config)

        # Evaluate coarse configurations with basic optimization
        coarse_results = []
        for i, coarse_cfg in enumerate(coarse_point_sets):
            try:
                # Quick quality check with minimal optimization
                ratio = evaluate_solution(coarse_cfg)
                coarse_results.append((ratio, coarse_cfg))
            except Exception:
                continue

        # Sort by quality and take top candidates
        coarse_results.sort(key=lambda x: x[0], reverse=True)
        top_coarse_configs = [cfg for _, cfg in coarse_results[:min(4, len(coarse_results))]]

        # Phase 2: Full optimization on promising configurations
        # Combine top coarse configs with original ones for more diverse search
        all_configs_for_fine = top_coarse_configs + [create_asymmetric_initial(), create_refined_initial()]

        # Sort fine configs by initial quality
        fine_config_ratios = [evaluate_solution(cfg) for cfg in all_configs_for_fine]
        sorted_indices = np.argsort(fine_config_ratios)[::-1]
        sorted_fine_configs = [all_configs_for_fine[i] for i in sorted_indices]

        # Now run optimization on top configurations
        for i, init_points in enumerate(sorted_fine_configs):
            remaining_time = max_time - (time.time() - start_time)
            if remaining_time <= 5:
                break

            try:
                # Evaluate initial configuration quality
                initial_ratio = evaluate_solution(init_points)

                # Flatten for optimization
                x0 = init_points.flatten()

                # Define bounds for each coordinate (0 to 1)
                bounds = [(0, 1) for _ in range(32)]

                # Use different strategies based on initial quality
                if initial_ratio > 0.25:  # Very high quality starting point
                    # Focus on local refinement with high precision
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15}
                    )
                elif initial_ratio > 0.20:  # High quality starting point
                    # Use L-BFGS-B with moderate precision
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1200, 'ftol': 1e-13, 'gtol': 1e-13}
                    )
                elif initial_ratio > 0.15:  # Medium quality starting point
                    # Use L-BFGS-B with less strict tolerances
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                else:  # Low quality starting point
                    # Use L-BFGS-B with reasonable tolerances
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 800, 'ftol': 1e-9, 'gtol': 1e-9}
                    )

                if result.success:
                    # Extract final points
                    final_points = result.x.reshape(-1, 2)

                    # Compute actual ratio
                    ratio = evaluate_solution(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

            except Exception:
                continue

        # If no successful optimization, return the best initial configuration
        if best_points is None:
            return sorted_fine_configs[0] if sorted_fine_configs else create_asymmetric_initial()

        return best_points

    # Use hierarchical optimization approach
    try:
        final_points = hierarchical_optimization(max_time=170)
    except Exception:
        # Fallback to original approach if hierarchical fails
        best_ratio = 0
        best_points = None

        # Strategy 1: Asymmetric initial configuration
        try:
            asymmetric_points = create_asymmetric_initial()
            refined_points, ratio = multi_local_optimization(asymmetric_points, 1000)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except Exception:
            pass

        # Strategy 2: Differential evolution for global search
        try:
            np.random.seed(42)
            refined_initial = create_refined_initial()
            refined_points, ratio = multi_local_optimization(refined_initial, 1000)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except Exception:
            pass

        # Strategy 3: Random initialization with refinement
        try:
            for i in range(3):  # Less random starts to save time
                np.random.seed(42 + i * 10)
                random_points = np.random.rand(16, 2)
                refined_points, ratio = multi_local_optimization(random_points, 500)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = refined_points.copy()
        except Exception:
            pass

        # If no good solution was found, return a default configuration
        if best_points is None:
            # Fallback to asymmetric initial configuration
            best_points = create_asymmetric_initial()

        final_points = best_points

    # Ensure final points are within bounds
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END