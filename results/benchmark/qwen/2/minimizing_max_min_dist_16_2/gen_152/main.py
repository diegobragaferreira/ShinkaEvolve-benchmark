# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Compute pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -min_dist / max_dist

    def objective_with_regularization(x):
        """Objective with regularization to avoid numerical issues"""
        points = x.reshape(-1, 2)
        distances = pdist(points)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Add small epsilon to avoid division by zero
        eps = 1e-12
        if max_dist < eps:
            return -1.0  # Return worst possible value

        ratio = min_dist / (max_dist + eps)
        return -ratio

    def constraint_bounds(x):
        """Constraint function for bounds checking"""
        points = x.reshape(-1, 2)
        # Check bounds: each coordinate should be in [0,1]
        violations = []
        for coord in [points[:, 0], points[:, 1]]:
            violations.extend(np.maximum(0 - coord, 0))   # lower bound
            violations.extend(np.maximum(coord - 1, 0))   # upper bound
        return np.array(violations)

    def golden_spiral_2d(n_points):
        """Generate points on a 2D golden spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        for i in range(n_points):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n_points - 1)) if n_points > 1 else 0
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])

        return np.array(points)

    def create_dispersed_corner_pattern():
        """Create a specialized corner-based pattern that maximizes initial spread"""
        # Place 4 corner points
        corners = [[0, 0], [1, 0], [0, 1], [1, 1]]
        # Add 4 edge midpoints
        edges = [[0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]]
        # Add 4 interior points in a cross pattern
        cross = [[0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5]]
        # Add 4 more points in a diamond pattern
        diamond = [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]]

        points = corners + edges + cross + diamond
        return np.array(points)

    def create_hexagonal_grid():
        """Create a hexagonal grid pattern with some perturbation"""
        # Create a 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0
                y = i / 3.0
                # Offset every other row
                if i % 2 == 1:
                    x += 1.0 / (3 * 2)
                points.append([x, y])

        # Convert to numpy array and add slight perturbations
        points = np.array(points[:16], dtype=np.float64)
        # Add small random perturbations
        np.random.seed(42)
        points += np.random.normal(0, 0.01, (16, 2))
        # Clip to valid range
        points = np.clip(points, 0, 1)
        return points

    def create_adaptive_perturbed_grid():
        """Create a grid with adaptive perturbations based on distance analysis"""
        # Start with regular 4x4 grid
        grid_points = np.array([[i/4, j/4] for i in range(4) for j in range(4)])[:16]

        # Analyze the current configuration to determine perturbation strategy
        distances = pdist(grid_points)
        current_min = np.min(distances)
        current_max = np.max(distances)
        current_ratio = current_min / current_max if current_max > 0 else 0

        # Adjust perturbation strength based on current quality
        if current_ratio > 0.15:
            # Already relatively well spread, use small perturbations
            perturbation_std = 0.01
        elif current_ratio > 0.10:
            # Moderate spread, medium perturbations
            perturbation_std = 0.02
        else:
            # Poorly spread, use larger perturbations
            perturbation_std = 0.03

        # Apply perturbations with special handling for boundary points
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

    def local_improvement_heuristic(points):
        """Apply a simple local improvement heuristic to enhance point distributions"""
        # Try to improve the configuration by small local adjustments
        improved_points = points.copy()

        # Perform several rounds of local perturbations
        for iteration in range(10):
            # Try improving each point individually
            for i in range(len(points)):
                original_point = improved_points[i].copy()
                best_point = original_point.copy()
                best_ratio = objective_with_regularization(improved_points.flatten())

                # Try small perturbations around current point
                for _ in range(20):
                    # Generate small random perturbation
                    perturbation = np.random.normal(0, 0.002, 2)
                    test_point = original_point + perturbation
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
                improved_points[i] = best_point

        return improved_points

    def optimize_with_adaptive_strategy(initial_points):
        """Run optimization with adaptive parameters based on initial configuration quality"""
        try:
            # Evaluate initial quality
            initial_ratio = -objective_with_regularization(initial_points.flatten())

            x0 = initial_points.flatten()
            bounds = [(0, 1) for _ in range(32)]

            # Select optimization strategy based on initial quality
            if initial_ratio > 0.25:  # Very high quality starting point
                # Use high precision L-BFGS-B optimization
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
            elif initial_ratio > 0.20:  # High quality starting point
                # Use L-BFGS-B with moderate precision
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-13}
                )
            elif initial_ratio > 0.15:  # Medium quality starting point
                # Use L-BFGS-B with less strict tolerances
                result = minimize(
                    objective_with_regularization,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1200, 'ftol': 1e-10, 'gtol': 1e-10}
                )
            else:  # Low quality starting point
                # Use Differential Evolution first to escape local minima
                try:
                    result_de = differential_evolution(
                        objective_with_regularization,
                        bounds,
                        seed=42,
                        maxiter=100,
                        popsize=20,
                        mutation=(0.5, 1),
                        recombination=0.7,
                        tol=1e-9
                    )
                    if result_de.success:
                        x0 = result_de.x
                except:
                    pass

                # Then use L-BFGS-B for refinement
                try:
                    result = minimize(
                        objective_with_regularization,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
                    )
                except:
                    # Fallback to basic optimization if needed
                    result = type('obj', (object,), {'x': x0, 'success': True})()

            # Apply local improvement heuristic to final result
            if hasattr(result, 'success') and result.success:
                final_points = result.x.reshape(-1, 2)
                refined_points = local_improvement_heuristic(final_points)
                return refined_points
            else:
                # Even if optimization failed, apply local heuristic to initial points
                refined_points = local_improvement_heuristic(initial_points)
                return refined_points

        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            # Return initial points with local improvement
            refined_points = local_improvement_heuristic(initial_points)
            return refined_points

    def optimize_with_restarts():
        """Run optimization with multiple enhanced initial configurations"""
        best_ratio = -np.inf
        best_points = None

        # Try multiple different initial configurations with enhanced strategies
        initial_configs = []

        # 1. Dispersed corner pattern
        initial_configs.append(create_dispersed_corner_pattern())

        # 2. Hexagonal grid with perturbations
        initial_configs.append(create_hexagonal_grid())

        # 3. Golden spiral pattern
        spiral_points = golden_spiral_2d(16)
        spiral_points = (spiral_points - np.min(spiral_points, axis=0)) / (
            np.max(spiral_points, axis=0) - np.min(spiral_points, axis=0) + 1e-12)
        spiral_points = spiral_points * 0.8 + 0.1
        initial_configs.append(spiral_points.copy())

        # 4. Adaptive perturbed grid
        initial_configs.append(create_adaptive_perturbed_grid())

        # 5. Random uniform points with fixed seed
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