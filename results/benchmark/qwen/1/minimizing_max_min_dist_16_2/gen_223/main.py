# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
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

        # Calculate pairwise distances using squareform for better numerical stability
        distances = pdist(points)

        # Proper handling of edge cases
        if len(distances) == 0 or np.allclose(distances, 0):
            # Return a large penalty value that's more gradual than huge values
            return 1e6  # Large penalty for degenerate configurations

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero - return large penalty if no distance
        if max_dist == 0:
            return 1e6

        # Return negative ratio (since we want to maximize ratio, we minimize its negative)
        # Add a small epsilon to min_dist to prevent extreme ratios when distances are very small
        epsilon = 1e-12
        return -(min_dist + epsilon) / (max_dist + epsilon)

    def constraint_func(x):
        # Ensure points are within [eps,1-eps] x [eps,1-eps] for numerical stability
        eps = 1e-8
        points = x.reshape(-1, 2)
        constraints = []

        # x coordinates in [eps, 1-eps] - with padding to prevent hitting boundaries exactly
        constraints.append(points[:, 0].min() - eps)  # x_min >= eps
        constraints.append(1 - eps - points[:, 0].max())  # x_max <= 1-eps

        # y coordinates in [eps, 1-eps] - with padding to prevent hitting boundaries exactly
        constraints.append(points[:, 1].min() - eps)  # y_min >= eps
        constraints.append(1 - eps - points[:, 1].max())  # y_max <= 1-eps

        return np.array(constraints)

    def bounded_objective(x):
        # Boundary checking with clamping to safe bounds
        eps = 1e-8
        points = np.clip(x.reshape(-1, 2), eps, 1-eps).flatten()
        return objective(points)

    def generate_hexagonal_initial_config():
        """Generate a more sophisticated hexagonal initial configuration"""
        np.random.seed(42)
        points = []

        # Create points in a true hexagonal lattice pattern
        # Based on the mathematical packing of circles in hexagonal pattern
        sqrt3 = np.sqrt(3)

        # Parameters for hexagonal packing
        # In a hexagonal lattice with point spacing s, the horizontal spacing is s
        # and vertical spacing is s * sqrt(3) / 2
        spacing = 0.8  # Total width/height available
        rows = 4
        cols = 4

        # Calculate actual spacing
        actual_spacing_x = spacing / (cols - 1) if cols > 1 else 0.5
        actual_spacing_y = spacing / (rows - 1) if rows > 1 else 0.5

        # Create hexagonal pattern with appropriate offsets
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Calculate position for this point
                x = 0.1 + j * actual_spacing_x
                y = 0.1 + i * actual_spacing_y

                # Apply hexagonal offset for odd rows
                if i % 2 == 1:
                    x += actual_spacing_x * 0.5

                points.append([x, y])

        # Convert to numpy array and add small random jitter to break symmetry
        points = np.array(points[:16])
        points += np.random.normal(0, 0.003, points.shape)  # Smaller jitter to maintain structure

        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)

        return points

    def generate_spiral_initial_config():
        """Generate a spiral initial configuration"""
        np.random.seed(42)
        points = []

        # Create a spiral pattern that spreads points well
        for i in range(16):
            if i == 0:
                points.append([0.5, 0.5])  # Center point
            else:
                angle = i * 2.5  # Angle in radians
                radius = min(0.4, i * 0.05)  # Radius increases gradually
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])

        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points

    def generate_grid_initial_config():
        """Generate a regular grid initial configuration"""
        np.random.seed(42)
        points = []

        # Regular 4x4 grid
        x_vals = np.linspace(0.1, 0.9, 4)
        y_vals = np.linspace(0.1, 0.9, 4)

        for i in range(4):
            for j in range(4):
                points.append([x_vals[i], y_vals[j]])

        points = np.array(points[:16])
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def generate_random_initial_config():
        """Generate a random initial configuration"""
        np.random.seed(42)
        return np.random.rand(16, 2)

    def generate_corner_initial_config():
        """Generate an initial configuration with points at corners and center"""
        np.random.seed(42)
        points = np.array([
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
            [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75],
            [0.33, 0.33], [0.33, 0.67], [0.67, 0.33], [0.67, 0.67]
        ])
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    # Generate multiple diverse initial configurations
    initial_configs = [
        generate_hexagonal_initial_config(),
        generate_spiral_initial_config(),
        generate_grid_initial_config(),
        generate_random_initial_config(),
        generate_corner_initial_config()
    ]

    best_ratio = float('inf')
    best_points = None

    # Try multiple initial configurations with optimized hybrid approach
    for i, initial_config in enumerate(initial_configs):
        try:
            # Flatten for optimization
            x0 = initial_config.flatten()

            # Define bounds for each coordinate (safe bounds to avoid numerical issues)
            bounds = [(1e-8, 1-1e-8) for _ in range(32)]

            # Phase 1: Global optimization with Differential Evolution
            de_result = differential_evolution(
                bounded_objective,
                bounds,
                seed=42+i,
                maxiter=150,  # More iterations for better global search
                popsize=25,   # Larger population for better exploration
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )

            # If DE finds a good solution, refine it locally
            if de_result.success and -de_result.fun > 0.15:
                x0 = de_result.x

            # Phase 2: Local optimization with adaptive iteration limits
            # Use SLSQP for better constraint handling
            result = minimize(
                bounded_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9},
                callback=None
            )

            # If SLSQP fails, try L-BFGS-B as fallback
            if not result.success:
                fallback_result = minimize(
                    bounded_objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300}
                )
                result = fallback_result

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                # Calculate actual ratio for this configuration
                distances = pdist(optimized_points)
                if len(distances) > 0 and np.max(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    ratio = min_dist / max_dist

                    if ratio < best_ratio:  # We want to maximize ratio, so minimize negative ratio
                        best_ratio = ratio
                        best_points = optimized_points.copy()

        except Exception as e:
            continue

    # If we still don't have a good solution, use a fallback approach
    if best_points is None:
        # Try one more time with a strong optimization approach
        fallback_config = generate_hexagonal_initial_config()
        x0 = fallback_config.flatten()
        bounds = [(1e-8, 1-1e-8) for _ in range(32)]

        # More aggressive optimization with careful parameter tuning
        try:
            result = minimize(
                bounded_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                # Final fallback - just return the best initial configuration
                best_points = generate_hexagonal_initial_config()
        except Exception:
            # If everything fails, return the hexagonal initial config
            best_points = generate_hexagonal_initial_config()

    # Final safety check - ensure points are within bounds
    if best_points is not None:
        best_points = np.clip(best_points, 1e-8, 1-1e-8)
    else:
        # Last resort: return a default configuration
        best_points = generate_hexagonal_initial_config()

    return best_points


# EVOLVE-BLOCK-END