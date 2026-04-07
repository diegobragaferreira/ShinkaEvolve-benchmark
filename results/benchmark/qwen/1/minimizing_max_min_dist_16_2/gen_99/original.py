# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)

        # Calculate pairwise distances using squareform for stability
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing)
        if d_max <= 0:
            return -1.0  # Avoid division by zero or invalid distances
        return -d_min / d_max

    # Create multiple initial configurations to improve optimization chances
    def generate_initial_configurations():
        """Generate several different initial configurations"""
        configs = []
        np.random.seed(42)

        # Configuration 1: Golden ratio based arrangement
        # Arrange points in a grid-like pattern with golden ratio spacing
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        points = []
        for i in range(4):
            for j in range(4):
                # Use golden ratio spacing for better distribution
                x = (i * phi) % 1
                y = (j * phi) % 1
                points.append([x, y])
        configs.append(np.array(points[:16]))

        # Configuration 2: Grid with perturbations
        points = []
        rows, cols = 4, 4
        for i in range(rows):
            for j in range(cols):
                x = i / (rows - 1) if rows > 1 else 0.5
                y = j / (cols - 1) if cols > 1 else 0.5
                # Add controlled perturbation
                x += (np.random.rand() - 0.5) * 0.1
                y += (np.random.rand() - 0.5) * 0.1
                points.append([x, y])
        configs.append(np.array(points[:16]))

        # Configuration 3: Spiral arrangement
        points = []
        for i in range(16):
            angle = i * 2 * np.pi / 16
            radius = i / 16.0
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            points.append([x, y])
        configs.append(np.array(points))

        # Configuration 4: Random with boundary padding
        points = np.random.rand(16, 2) * 0.8 + 0.1  # Keep away from edges
        configs.append(points)

        return configs

    def adaptive_minimize(obj_func, x0, bounds, maxiter, ftol, gtol):
        """Minimize with adaptive stopping criteria"""
        previous_obj_val = float('inf')
        consecutive_no_improvement = 0
        max_no_improvement = 20

        # Use a callback to track progress
        def callback(xk):
            nonlocal previous_obj_val, consecutive_no_improvement
            obj_val = obj_func(xk)
            if abs(previous_obj_val - obj_val) < 1e-12:
                consecutive_no_improvement += 1
            else:
                consecutive_no_improvement = 0
            previous_obj_val = obj_val

        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
                callback=callback if consecutive_no_improvement < max_no_improvement else None
            )
            return result
        except Exception:
            return None

    # Generate multiple initial configurations
    initial_configs = generate_initial_configurations()

    best_solution = None
    best_ratio = float('inf')

    # Try each initial configuration with optimization
    for i, initial_points in enumerate(initial_configs):
        # Flatten the points for optimization
        x0 = initial_points.flatten()

        # Define bounds for each coordinate [0, 1] with small epsilon padding
        bounds = [(1e-6, 1-1e-6) for _ in range(32)]

        # Phase 1: Try L-BFGS-B optimization
        try:
            result = adaptive_minimize(
                objective,
                x0,
                bounds,
                maxiter=1000,
                ftol=1e-10,
                gtol=1e-10
            )

            if result and result.success:
                current_obj_value = -objective(result.x)  # Convert back to positive ratio
                if current_obj_value < best_ratio:
                    best_ratio = current_obj_value
                    best_solution = result.x.reshape(-1, 2)

        except Exception:
            pass

        # Phase 2: If L-BFGS failed, try Differential Evolution on this configuration
        if best_solution is None:
            try:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=200,  # Increased iterations
                    popsize=25,   # Larger population
                    mutation=(0.5, 1),
                    recombination=0.7,
                    seed=42,
                    disp=False
                )

                current_obj_value = -objective(de_result.x)
                if current_obj_value < best_ratio:
                    best_ratio = current_obj_value
                    best_solution = de_result.x.reshape(-1, 2)

            except Exception:
                pass

    # If no optimization worked, return the best initial configuration
    if best_solution is None:
        # Select the best initial configuration based on its objective value
        best_initial_idx = 0
        best_initial_ratio = float('inf')

        for i, initial_points in enumerate(initial_configs):
            # Test the initial configuration
            initial_flat = initial_points.flatten()
            initial_obj_value = -objective(initial_flat)

            if initial_obj_value < best_initial_ratio:
                best_initial_ratio = initial_obj_value
                best_initial_idx = i

        best_solution = initial_configs[best_initial_idx]

    # Ensure final solution respects bounds
    final_points = np.clip(best_solution, 1e-6, 1-1e-6)

    return final_points

# EVOLVE-BLOCK-END