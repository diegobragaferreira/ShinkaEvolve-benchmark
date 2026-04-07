# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return np.min(distances) / max_dist

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

    # Strategy 1: Enhanced hexagonal grid with proper hexagonal lattice
    def hexagonal_grid_init():
        # Create a more mathematically precise hexagonal lattice arrangement
        points = []

        # Parameters for hexagonal lattice
        # For 16 points, we can arrange them in a 4x4 grid but with proper hexagonal spacing
        # Using a hexagonal lattice where each point is at distance 1 from its neighbors
        row_spacing = np.sqrt(3) / 2.0  # Vertical spacing for hexagonal lattice
        col_spacing = 1.0               # Horizontal spacing

        # Create a 4x4 hexagonal grid
        for row in range(4):
            for col in range(4):
                if len(points) >= 16:
                    break
                # Calculate position in hexagonal lattice
                x = col * col_spacing
                # Offset odd rows
                if row % 2 == 1:
                    x += col_spacing / 2.0

                y = row * row_spacing

                points.append([x, y])

        # Convert to numpy array
        points = np.array(points[:16])

        # Scale to fit nicely within [0,1] square while maintaining hexagonal properties
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        # Handle case where all points are same (shouldn't happen but safety check)
        if max_x <= min_x:
            max_x = min_x + 1.0
        if max_y <= min_y:
            max_y = min_y + 1.0

        # Scale to fit within unit square without distortion
        scale_x = 1.0 / (max_x - min_x)
        scale_y = 1.0 / (max_y - min_y)
        scale = min(scale_x, scale_y) * 0.9  # Leave some margin

        points[:, 0] = (points[:, 0] - min_x) * scale
        points[:, 1] = (points[:, 1] - min_y) * scale

        # Center the points in the unit square
        center_shift_x = 0.5 - (np.max(points[:, 0]) + np.min(points[:, 0])) / 2.0
        center_shift_y = 0.5 - (np.max(points[:, 1]) + np.min(points[:, 1])) / 2.0

        points[:, 0] += center_shift_x
        points[:, 1] += center_shift_y

        # Ensure points are within bounds
        points = np.clip(points, 0, 1)

        # Apply symmetry breaking with more sophisticated approach
        # Add deterministic perturbations to break symmetry
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.01, points.shape)

        # Apply stronger perturbations to corner points to break symmetries
        corner_indices = [0, 3, 12, 15]  # Four corners of 4x4 grid
        for idx in corner_indices:
            if idx < len(points):
                perturbations[idx] *= 2.0  # Twice the perturbation for corners

        points += perturbations
        points = np.clip(points, 0, 1)

        # Additional symmetry breaking: rotate some points
        center = np.mean(points, axis=0)
        rotation_angles = [np.pi/12, -np.pi/12, np.pi/6, -np.pi/6]

        # Apply small rotations to every 4th point to break rotational symmetry
        for i in range(0, len(points), 4):
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
        points += np.random.normal(0, 0.01, points.shape)
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points

    # Strategy 4: Structured approach with adaptive perturbations
    def adaptive_hexagonal_init():
        points = hexagonal_grid_init()

        # Apply adaptive perturbations based on position
        center = np.mean(points, axis=0)
        distances_from_center = np.sqrt(np.sum((points - center)**2, axis=1))
        max_distance = np.max(distances_from_center)

        if max_distance > 0:
            normalized_distances = distances_from_center / max_distance
            # Apply stronger perturbations near center for better spread
            perturbation_magnitude = 0.01 * (1 - normalized_distances)

            np.random.seed(42)
            perturbations = np.random.normal(0, 0.005, points.shape)
            perturbations *= perturbation_magnitude.reshape(-1, 1)
            points += perturbations

        points = np.clip(points, 0, 1)
        return points

    initial_strategies = [
        hexagonal_grid_init,
        random_init,
        perturbed_hexagonal_init,
        adaptive_hexagonal_init
    ]

    # Multi-start optimization with different initialization strategies
    best_ratio = -np.inf
    best_points = None

    # Try multiple random restarts with differential evolution for global search
    num_restarts = 5
    for restart in range(num_restarts):
        # Select initialization strategy
        init_func = initial_strategies[restart % len(initial_strategies)]
        points = init_func()

        # Flatten for optimization
        x0 = points.flatten()

        # Set up bounds
        bounds = [(0, 1) for _ in range(32)]

        # Use differential evolution for global search first
        try:
            result_de = differential_evolution(
                objective_function,
                bounds,
                maxiter=300,
                popsize=15,
                tol=1e-8,
                seed=42 + restart,
                callback=None
            )

            if result_de.success:
                final_points = result_de.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception as e:
            # Fall back to local optimization if differential evolution fails
            try:
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_function},
                    options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
            except Exception as e2:
                continue

    # If none worked, fallback to simple approach
    if best_points is None:
        # Fallback to hexagonal grid with refinement
        points = hexagonal_grid_init()
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(32)]

        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_function},
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
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