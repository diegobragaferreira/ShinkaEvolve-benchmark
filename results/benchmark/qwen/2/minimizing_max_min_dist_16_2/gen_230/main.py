# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
from scipy.stats import qmc

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective_with_regularization(x):
        """Objective with regularization to avoid numerical issues"""
        points = x.reshape(-1, 2)
        distances = pdist(points)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-15
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio

    def create_dispersed_corner_pattern():
        """Create a specialized corner-based pattern that maximizes initial spread"""
        # Place 4 corner points with strategic offsets
        corners = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
        # Add 4 edge midpoints with offsets
        edges = [[0.5, 0.05], [0.5, 0.95], [0.05, 0.5], [0.95, 0.5]]
        # Add 4 interior points in a cross pattern with strategic positioning
        cross = [[0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5]]
        # Add 4 more points in a diamond pattern with strategic distances
        diamond = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]

        points = corners + edges + cross + diamond
        return np.array(points)

    def create_hexagonal_grid():
        """Create a hexagonal grid pattern with optimized perturbation"""
        # Create a 4x4 grid with hexagonal offset
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0
                y = i / 3.0
                # Offset every other row for hexagonal packing
                if i % 2 == 1:
                    x += 1.0 / (3 * 2)
                points.append([x, y])

        # Convert to numpy array and add slight perturbations
        points = np.array(points[:16], dtype=np.float64)
        # Add small random perturbations with seed for reproducibility
        np.random.seed(42)
        points += np.random.normal(0, 0.015, (16, 2))
        # Clip to valid range
        points = np.clip(points, 0, 1)
        return points

    def create_golden_spiral():
        """Create golden spiral with better normalization"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(16):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / 15) if 15 > 0 else 0  # Normalize to [0,1]
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        points = np.array(points)
        # Normalize to [0.1, 0.9] range to avoid boundary issues
        if np.max(points) > np.min(points):
            points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0) + 1e-12)
        points = points * 0.8 + 0.1
        return points

    def create_adaptive_perturbed_grid():
        """Create a grid with adaptive perturbations based on distance analysis"""
        # Start with regular 4x4 grid
        grid_points = np.array([[i/3, j/3] for i in range(4) for j in range(4)])[:16]

        # Analyze the current configuration to determine perturbation strategy
        distances = pdist(grid_points)
        current_min = np.min(distances)
        current_max = np.max(distances)
        current_ratio = current_min / current_max if current_max > 0 else 0

        # Adjust perturbation strength based on current quality with more nuanced scaling
        if current_ratio > 0.20:  # Already relatively well spread
            perturbation_std = 0.01
        elif current_ratio > 0.15:  # Moderate spread
            perturbation_std = 0.02
        else:  # Poorly spread
            perturbation_std = 0.03

        # Apply perturbations with careful attention to boundary points
        for i in range(16):
            row, col = i // 4, i % 4
            # Emphasize perturbations for corner and edge points to encourage spreading
            if row in [0, 3] or col in [0, 3]:
                std_factor = 1.5
            else:
                std_factor = 1.0
            grid_points[i] += np.random.normal(0, perturbation_std * std_factor, 2)

        grid_points = np.clip(grid_points, 0, 1)
        return grid_points

    def create_sobol_points():
        """Create points using Sobol sequence for better space-filling"""
        sampler = qmc.Sobol(d=2, seed=42)
        points = sampler.random(16)
        return points

    def local_improvement_heuristic(points, max_iterations=20):
        """Apply a simple local improvement heuristic to enhance point distributions"""
        improved_points = points.copy()

        # Perform several rounds of local perturbations with adaptive step sizes
        for iteration in range(max_iterations):
            # Track if we made any improvements
            improved = False
            
            # Try improving each point individually
            for i in range(len(points)):
                original_point = improved_points[i].copy()
                best_point = original_point.copy()
                best_ratio = -objective_with_regularization(improved_points.flatten())

                # Try more systematic perturbations around current point
                perturbations = [
                    [-0.005, -0.005], [-0.005, 0], [-0.005, 0.005],
                    [0, -0.005], [0, 0.005],
                    [0.005, -0.005], [0.005, 0], [0.005, 0.005],
                    [-0.01, -0.01], [-0.01, 0], [-0.01, 0.01],
                    [0, -0.01], [0, 0.01],
                    [0.01, -0.01], [0.01, 0], [0.01, 0.01]
                ]
                
                for dx, dy in perturbations:
                    test_point = original_point + np.array([dx, dy])
                    test_point = np.clip(test_point, 0, 1)

                    # Temporarily replace point
                    temp_points = improved_points.copy()
                    temp_points[i] = test_point

                    # Check if this improves the ratio
                    test_ratio = -objective_with_regularization(temp_points.flatten())
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_point = test_point.copy()

                # Update point if improvement found
                if not np.array_equal(best_point, original_point):
                    improved_points[i] = best_point
                    improved = True

            # Early stopping if no improvements made
            if not improved:
                break

        return improved_points

    def evolutionary_restart_strategy(initial_points, max_evaluations=3000):
        """Enhanced evolutionary restart strategy combining DE and local refinement"""
        best_points = initial_points.copy()
        best_ratio = -objective_with_regularization(initial_points.flatten())

        # Run multiple differential evolution processes with different strategies
        de_strategies = [
            {'popsize': 30, 'mutation': (0.5, 1), 'recombination': 0.8, 'maxiter': 100},
            {'popsize': 25, 'mutation': (0.7, 1), 'recombination': 0.7, 'maxiter': 80},
            {'popsize': 20, 'mutation': (0.3, 1), 'recombination': 0.6, 'maxiter': 120},
        ]

        bounds = [(0, 1) for _ in range(32)]

        for i, strategy in enumerate(de_strategies):
            try:
                # Run differential evolution with current strategy
                de_result = differential_evolution(
                    objective_with_regularization,
                    bounds,
                    seed=42 + i,
                    maxiter=strategy['maxiter'],
                    popsize=strategy['popsize'],
                    mutation=strategy['mutation'],
                    recombination=strategy['recombination'],
                    tol=1e-10,
                    disp=False
                )

                if de_result.success:
                    # Refine with L-BFGS-B
                    refined_result = minimize(
                        objective_with_regularization,
                        de_result.x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
                    )

                    if refined_result.success:
                        refined_points = refined_result.x.reshape(-1, 2)
                        refined_ratio = -objective_with_regularization(refined_result.x)

                        if refined_ratio > best_ratio:
                            best_ratio = refined_ratio
                            best_points = refined_points.copy()

            except Exception as e:
                warnings.warn(f"DE strategy {i} failed: {e}")
                continue

        return best_points

    def optimize_with_adaptive_strategy(initial_points):
        """Run optimization with adaptive parameters based on initial configuration quality"""
        try:
            # Evaluate initial quality
            initial_ratio = -objective_with_regularization(initial_points.flatten())

            x0 = initial_points.flatten()
            bounds = [(0, 1) for _ in range(32)]

            # Select optimization strategy based on initial quality with refined thresholds
            if initial_ratio > 0.30:  # Very high quality starting point
                # Use high precision L-BFGS-B optimization
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2500, 'ftol': 1e-16, 'gtol': 1e-16}
                )
            elif initial_ratio > 0.25:  # High quality starting point
                # Use L-BFGS-B with moderate precision
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
                )
            elif initial_ratio > 0.20:  # Medium-high quality starting point
                # Use L-BFGS-B with less strict tolerances
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
            elif initial_ratio > 0.15:  # Medium quality starting point
                # Use L-BFGS-B with moderate precision
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1200, 'ftol': 1e-10, 'gtol': 1e-10}
                )
            else:  # Low quality starting point
                # Use evolutionary restart strategy for better global search
                evolved_points = evolutionary_restart_strategy(initial_points)
                # Apply L-BFGS-B refinement to the evolved solution
                try:
                    result = minimize(
                        objective_with_regularization,
                        evolved_points.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                except:
                    # Fallback to basic optimization if needed
                    result = type('obj', (object,), {'x': evolved_points.flatten(), 'success': True})()

            # Apply local improvement heuristic to final result
            if hasattr(result, 'success') and result.success:
                final_points = result.x.reshape(-1, 2)
                refined_points = local_improvement_heuristic(final_points, max_iterations=30)
                return refined_points
            else:
                # Even if optimization failed, apply local heuristic to initial points
                refined_points = local_improvement_heuristic(initial_points, max_iterations=30)
                return refined_points

        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            # Return initial points with local improvement
            refined_points = local_improvement_heuristic(initial_points, max_iterations=30)
            return refined_points

    def optimize_with_restarts():
        """Run optimization with multiple enhanced initial configurations"""
        best_ratio = -np.inf
        best_points = None

        # Try multiple different initial configurations with enhanced strategies
        initial_configs = []

        # 1. Dispersed corner pattern (enhanced strategic placement)
        initial_configs.append(create_dispersed_corner_pattern())

        # 2. Hexagonal grid with optimized perturbations
        initial_configs.append(create_hexagonal_grid())

        # 3. Golden spiral pattern (improved normalization)
        initial_configs.append(create_golden_spiral())

        # 4. Adaptive perturbed grid (smarter scaling)
        initial_configs.append(create_adaptive_perturbed_grid())

        # 5. Sobol sequence points for better space filling
        initial_configs.append(create_sobol_points())

        # 6. Random uniform points with fixed seed
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        initial_configs.append(random_points)

        # Run optimization for each configuration with adaptive strategy
        for i, init_points in enumerate(initial_configs):
            try:
                final_points = optimize_with_adaptive_strategy(init_points)

                # Compute actual ratio
                distances = pdist(final_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

            except Exception as e:
                warnings.warn(f"Optimization failed for initial config {i}: {e}")
                continue

        # If no successful optimization, return the best initial configuration
        if best_points is None:
            return initial_configs[0]

        return best_points

    # Try optimized approach first
    try:
        final_points = optimize_with_restarts()
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        # Fallback to simple approach if something fails
        np.random.seed(42)
        final_points = np.random.rand(16, 2)

    return final_points

# EVOLVE-BLOCK-END