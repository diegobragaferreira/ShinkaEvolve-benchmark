# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into 16 points
        points = x.reshape(-1, 2)
        # Calculate pairwise distances
        distances = pdist(points)
        # Avoid division by zero
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        # Minimize negative of min/max ratio (equivalent to maximizing min/max ratio)
        return -min_dist / max_dist

    def calculate_ratio(points):
        """Calculate min/max distance ratio for given points."""
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

    def generate_refined_hexagonal_initial():
        """Generate improved hexagonal lattice with optimized spacing"""
        # Use a slightly different approach for better point distribution
        points = []
        rows = 4
        cols = 4

        # More precise spacing calculation
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0

        # Adjust spacing to make it more uniform
        spacing_x *= 0.9
        spacing_y *= 0.9

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x_offset = spacing_x * 0.25 if i % 2 == 1 else 0.0
                x = (j * spacing_x) + x_offset
                y = i * spacing_y

                # Ensure points are within bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))

                points.append([x, y])

        return np.array(points)

    def generate_fibonacci_spiral():
        """Generate points using Fibonacci spiral for good distribution"""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            theta = math.acos(-1 + (2 * i) / 15)  # elevation angle
            phi_angle = (i * 2 * math.pi) / (phi * phi)  # azimuthal angle

            # Convert to cartesian coordinates
            x = math.sin(theta) * math.cos(phi_angle)
            y = math.sin(theta) * math.sin(phi_angle)

            # Map to [0.05, 0.95] range to avoid boundaries
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2

            points.append([x, y])

        return np.array(points)

    def generate_grid_initial():
        """Generate regular grid initial configuration"""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])
        return np.array(points)

    def generate_polar_initial():
        """Generate initial configuration in polar arrangement"""
        points = []
        # Place points in concentric circles
        radii = [0.2, 0.4, 0.6, 0.8]
        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        # Center point
        points.append([0.5, 0.5])

        # Add points in rings
        for i, radius in enumerate(radii):
            for angle in angles:
                if len(points) >= 16:
                    break
                rad = math.radians(angle)
                x = 0.5 + radius * math.cos(rad)
                y = 0.5 + radius * math.sin(rad)
                points.append([x, y])
            if len(points) >= 16:
                break

        # Fill remaining spots
        while len(points) < 16:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            points.append([x, y])

        return np.array(points[:16])

    def adaptive_perturbation(points, iteration=0):
        """Apply adaptive perturbation based on current configuration"""
        # Calculate current distribution statistics
        distances = pdist(points)
        if len(distances) > 0:
            avg_dist = np.mean(distances)
            std_dist = np.std(distances)
            # Perturbation magnitude decreases with iterations
            perturbation_std = 0.02 * (1.0 / (1.0 + iteration))
            # But increases if distribution is too uniform
            if std_dist / avg_dist < 0.1:  # Very uniform, increase perturbation
                perturbation_std *= 2.0

            perturbed = points + np.random.normal(0, perturbation_std, points.shape)
            # Clip to valid range
            perturbed = np.clip(perturbed, 0.001, 0.999)
            return perturbed
        return points

    def multi_start_optimization(initial_points_list):
        """Perform multi-start optimization with progressive refinement"""
        best_ratio = -np.inf
        best_points = None

        # Define bounds for coordinates
        bounds = [(0.001, 0.999) for _ in range(32)]

        # Try multiple starting configurations with different optimization methods
        for i, initial_config in enumerate(initial_points_list):
            # Try with L-BFGS-B
            try:
                x0 = initial_config.flatten()
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-5}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = calculate_ratio(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

            except Exception:
                pass

            # Also try with SLSQP for potentially better results
            try:
                x0 = initial_config.flatten()
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 150}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = calculate_ratio(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

            except Exception:
                pass

        return best_points, best_ratio

    # Generate multiple diverse initial configurations
    initial_configs = [
        generate_refined_hexagonal_initial(),
        generate_fibonacci_spiral(),
        generate_grid_initial(),
        generate_polar_initial()
    ]

    # Add adaptive perturbations to each initial configuration
    np.random.seed(42)
    perturbed_configs = []
    for i, config in enumerate(initial_configs):
        # Create multiple variations of each initial config
        for j in range(3):  # 3 variations per initial config
            if j == 0:
                # No perturbation for first variation
                perturbed = config.copy()
            elif j == 1:
                # Small perturbation
                perturbed = config + np.random.normal(0, 0.01, config.shape)
            else:
                # Medium perturbation
                perturbed = config + np.random.normal(0, 0.02, config.shape)

            # Clip to valid range
            perturbed = np.clip(perturbed, 0.001, 0.999)
            perturbed_configs.append(perturbed)

    # Perform multi-start optimization
    best_points, best_ratio = multi_start_optimization(perturbed_configs)

    # If no good solution was found, try a more aggressive refinement approach
    if best_points is None or best_ratio < 0.001:
        # Try a different approach with progressive refinement
        original_points = generate_refined_hexagonal_initial()
        current_points = original_points.copy()

        # Perform multiple rounds of optimization with adaptive perturbations
        for iteration in range(5):
            # Try optimization from current points
            try:
                bounds = [(0.001, 0.999) for _ in range(32)]
                x0 = current_points.flatten()

                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = calculate_ratio(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()

            except Exception:
                pass

            # Apply adaptive perturbation
            current_points = adaptive_perturbation(current_points, iteration)

    # If still no good solution, return the best from our initial attempts
    if best_points is None:
        # Return the most evenly distributed configuration from our initial attempts
        fallback_config = generate_grid_initial()
        # Add small random noise to break symmetry
        fallback_config += np.random.normal(0, 0.01, fallback_config.shape)
        fallback_config = np.clip(fallback_config, 0.001, 0.999)
        best_points = fallback_config

    return best_points

# EVOLVE-BLOCK-END