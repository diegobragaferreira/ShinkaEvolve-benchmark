# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        distances = pdist(points)
        return np.min(distances) / np.max(distances)

    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        return -compute_min_max_ratio(points)

    def constraint_function(x_flat):
        """Constraint to keep points within unit square"""
        points = x_flat.reshape(-1, 2)
        # Return negative values for constraint violations (we want >= 0)
        violations = np.concatenate([
            np.minimum(points[:, 0], 0),
            np.minimum(points[:, 1], 0),
            np.maximum(points[:, 0] - 1, 0),
            np.maximum(points[:, 1] - 1, 0)
        ])
        return violations

    # Multi-start optimization with different initialization strategies
    best_ratio = -np.inf
    best_points = None

    # Strategy 1: Enhanced hexagonal grid with true hexagonal lattice
    def hexagonal_grid_init():
        # Create points in a true hexagonal lattice pattern
        # Using the mathematical relationship: spacing_y = spacing_x * sqrt(3)/2
        spacing = 1.0
        row_spacing = spacing * np.sqrt(3) / 2.0
        col_spacing = spacing

        points = []

        # Create a hexagonal grid that covers enough area for 16 points
        # Using 4 rows and 4 columns with appropriate offsets
        rows = 4
        cols = 4

        for row in range(rows):
            for col in range(cols):
                if len(points) >= 16:
                    break
                # Calculate position
                x = col * col_spacing
                # Offset odd rows
                if row % 2 == 1:
                    x += col_spacing / 2.0
                y = row * row_spacing
                points.append([x, y])

        # Convert to numpy array
        points = np.array(points[:16])

        # Normalize to fit within [0,1] x [0,1]
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        # Avoid division by zero
        if max_x > min_x and max_y > min_y:
            scale_x = 1.0 / (max_x - min_x) if max_x > min_x else 1.0
            scale_y = 1.0 / (max_y - min_y) if max_y > min_y else 1.0
            scale = min(scale_x, scale_y, 1.0)

            points[:, 0] = (points[:, 0] - min_x) * scale
            points[:, 1] = (points[:, 1] - min_y) * scale

        # Center in unit square
        center_shift = 0.5 - np.mean(points, axis=0)
        points = points + center_shift

        # Ensure within bounds
        points = np.clip(points, 0, 1)

        # Apply sophisticated perturbation strategy
        np.random.seed(42)

        # Apply different types of perturbations based on position
        for i in range(len(points)):
            # Position-dependent perturbation magnitudes
            row = i // 4
            col = i % 4

            # Different noise levels for different regions
            if (row == 0 or row == 3) and (col == 0 or col == 3):
                # Corners: less perturbation
                pert_magnitude = 0.005
            elif row == 0 or row == 3 or col == 0 or col == 3:
                # Edges: medium perturbation
                pert_magnitude = 0.01
            else:
                # Center: more perturbation
                pert_magnitude = 0.015

            # Add 2D Gaussian noise
            noise = np.random.normal(0, pert_magnitude, 2)
            points[i] += noise

        # Final clipping to ensure bounds
        points = np.clip(points, 0, 1)

        # Apply additional symmetry breaking with multiple small rotations
        center = np.mean(points, axis=0)
        # Rotate specific points to break rotational symmetry
        rotation_angles = [np.pi/24, np.pi/12, np.pi/8, np.pi/6]  # 7.5, 15, 22.5, 30 degrees
        for i in range(0, 16, 4):  # Every 4th point
            if i < len(points):
                angle = rotation_angles[i % len(rotation_angles)]
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                points[i] = rotation_matrix @ (points[i] - center) + center

        return points

    # Strategy 2: Random with clustering avoidance
    def random_init():
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return points

    # Strategy 3: Perturbed hexagonal grid (more structured)
    def perturbed_hexagonal_init():
        points = hexagonal_grid_init()
        np.random.seed(42)
        # Add small random perturbations
        points += np.random.normal(0, 0.02, points.shape)
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points

    initial_strategies = [
        hexagonal_grid_init,
        random_init,
        perturbed_hexagonal_init
    ]

    # Try multiple random restarts with adaptive optimization
    num_restarts = 5
    for restart in range(num_restarts):
        # Select initialization strategy
        init_func = initial_strategies[restart % len(initial_strategies)]
        points = init_func()

        # Flatten for optimization
        x0 = points.flatten()

        # Set up constraints
        bounds = [(0, 1) for _ in range(32)]
        constraints = {'type': 'ineq', 'fun': constraint_function}

        # Adaptive optimization with intelligent cooling schedule
        try:
            # Use different optimization methods based on restart count
            if restart < 3:
                # More thorough optimization for early restarts
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
                )
            else:
                # Faster optimization for later restarts
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
                )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()

        except Exception as e:
            continue

    # If none worked, fallback to simple approach
    if best_points is None:
        # Fallback to the original approach with better settings
        points = hexagonal_grid_init()
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        x0 = points.flatten()

        bounds = [(0, 1) for _ in range(32)]
        constraints = {'type': 'ineq', 'fun': constraint_function}

        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
            )

            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                # Final fallback to random points
                np.random.seed(42)
                best_points = np.random.rand(16, 2)
        except:
            # Final fallback
            np.random.seed(42)
            best_points = np.random.rand(16, 2)

    return best_points

# EVOLVE-BLOCK-END