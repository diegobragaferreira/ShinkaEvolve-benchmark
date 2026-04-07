# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 2)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        if len(distances) == 0:
            return 0

        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max <= 1e-12:
            return 0

        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio

    def generate_hexagonal_grid():
        """Generate points in a hexagonal lattice pattern"""
        points = []
        rows = 4
        cols = 4

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x_offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + x_offset) / 3.0
                y = i / 3.0

                # Ensure points are within bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))

                points.append([x, y])

        return np.array(points)

    def generate_ring_distribution():
        """Generate points in concentric rings"""
        points = []
        # Two rings with 8 points each
        radii = [0.3, 0.7]
        angles_per_ring = [8, 8]

        for r_idx, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for i in range(num_angles):
                angle = 2 * math.pi * i / num_angles
                x = 0.5 + radius * math.cos(angle) * 0.4
                y = 0.5 + radius * math.sin(angle) * 0.4

                # Ensure within bounds
                x = max(0.001, min(0.999, x))
                y = max(0.001, min(0.999, y))

                points.append([x, y])

        return np.array(points)

    def generate_fibonacci_spiral():
        """Generate points using Fibonacci spiral-like arrangement"""
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio

        for i in range(16):
            # Modified Fibonacci approach for better distribution
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

    def generate_perturbed_grid():
        """Generate a regular grid with controlled perturbations"""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])
        return np.array(points)

    def generate_central_spread():
        """Generate points with central concentration and peripheral spread"""
        points = []
        # Central cluster
        for i in range(4):
            for j in range(4):
                x = 0.2 + 0.1 * (i - 1.5)
                y = 0.2 + 0.1 * (j - 1.5)
                points.append([x, y])

        # Peripheral points
        for i in range(12):
            angle = 2 * math.pi * i / 12
            radius = 0.8
            x = 0.5 + radius * math.cos(angle) * 0.4
            y = 0.5 + radius * math.sin(angle) * 0.4
            points.append([x, y])

        return np.array(points[:16])  # Take first 16 points

    def generate_spherical_arrangement():
        """Generate points arranged in a spherical pattern for good coverage"""
        points = []
        # Distribute 16 points on a sphere using Fibonacci method
        for i in range(16):
            # Use a variant that works well for 2D projection
            theta = math.acos(-1 + (2 * i) / 15)  # elevation angle
            phi = math.sqrt(16 * math.pi) * theta  # azimuthal angle

            # Project onto 2D plane
            x = 0.5 + 0.4 * math.sin(theta) * math.cos(phi)
            y = 0.5 + 0.4 * math.sin(theta) * math.sin(phi)

            # Ensure within bounds
            x = max(0.001, min(0.999, x))
            y = max(0.001, min(0.999, y))

            points.append([x, y])

        return np.array(points)

    def adaptive_perturbation(config, current_ratio):
        """Apply adaptive perturbation based on the current solution quality"""
        # Scale perturbation based on how well-distributed the points currently are
        # If ratio is low (poor distribution), apply larger perturbations
        # If ratio is high (good distribution), apply smaller perturbations
        if current_ratio < 0.1:
            perturbation_magnitude = 0.05
        elif current_ratio < 0.2:
            perturbation_magnitude = 0.03
        else:
            perturbation_magnitude = 0.015

        perturbed = config + np.random.normal(0, perturbation_magnitude, config.shape)
        # Clip to valid range
        perturbed = np.clip(perturbed, 0.001, 0.999)
        return perturbed

    # Generate multiple initial configurations
    initial_configs = [
        generate_hexagonal_grid(),
        generate_ring_distribution(),
        generate_fibonacci_spiral(),
        generate_perturbed_grid(),
        generate_central_spread(),
        generate_spherical_arrangement()
    ]

    # Add random perturbations to each configuration with adaptive scaling
    np.random.seed(42)
    perturbed_configs = []
    for config in initial_configs:
        # Apply adaptive perturbations
        perturbed = adaptive_perturbation(config, 0.0)
        perturbed_configs.append(perturbed)

    # Try optimization from different starting points using multi-stage approach
    best_ratio = -np.inf
    best_points = None

    # Define bounds for coordinates (slightly inside [0,1] to avoid edge issues)
    bounds = [(0.001, 0.999) for _ in range(32)]

    # Multi-stage optimization approach with progressive refinement
    for i, initial_config in enumerate(perturbed_configs):
        try:
            # Stage 1: Global search with differential evolution for broad exploration
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=25,  # Moderate iterations for speed
                popsize=10,   # Larger population for better exploration
                seed=42+i,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )

            # Stage 2: Local refinement with L-BFGS-B using coarse tolerances
            coarse_result = minimize(
                objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 75, 'ftol': 1e-8, 'gtol': 1e-8}
            )

            if coarse_result.success:
                coarse_points = coarse_result.x.reshape(-1, 2)
                distances = pdist(coarse_points)

                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)

                    if max_dist > 0:
                        coarse_ratio = min_dist / max_dist

                        # Stage 3: Fine refinement with tight tolerances
                        fine_result = minimize(
                            objective,
                            coarse_points.flatten(),
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10}
                        )

                        if fine_result.success:
                            fine_points = fine_result.x.reshape(-1, 2)
                            fine_distances = pdist(fine_points)

                            if len(fine_distances) > 0:
                                fine_min_dist = np.min(fine_distances)
                                fine_max_dist = np.max(fine_distances)

                                if fine_max_dist > 0:
                                    fine_ratio = fine_min_dist / fine_max_dist

                                    if fine_ratio > best_ratio:
                                        best_ratio = fine_ratio
                                        best_points = fine_points.copy()
                        else:
                            # If fine optimization fails, use coarse result as fallback
                            if coarse_ratio > best_ratio:
                                best_ratio = coarse_ratio
                                best_points = coarse_points.copy()

            # Even if optimization fails, keep trying other configurations
        except Exception as e:
            continue

    # If no good solution was found, return a good structured configuration
    if best_points is None:
        # Use the hexagonal grid as fallback since it's typically effective
        fallback_points = generate_hexagonal_grid()
        # Add a small amount of random noise to break possible symmetries
        fallback_points += np.random.normal(0, 0.005, fallback_points.shape)
        fallback_points = np.clip(fallback_points, 0.001, 0.999)
        best_points = fallback_points

    # Apply hill-climbing refinement to further improve the best solution
    if best_points is not None:
        # Hill-climbing parameters
        max_iterations = 500
        step_size = 0.001
        patience = 50
        current_points = best_points.copy()
        current_ratio = best_ratio

        patience_counter = 0
        for iteration in range(max_iterations):
            # Try small random perturbations to each point
            np.random.seed(None)  # Use system time for randomness
            perturbation = np.random.normal(0, step_size, current_points.shape)
            candidate_points = np.clip(current_points + perturbation, 0.001, 0.999)

            # Evaluate candidate
            distances = pdist(candidate_points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    candidate_ratio = min_dist / max_dist

                    # Accept if better
                    if candidate_ratio > current_ratio:
                        current_points = candidate_points
                        current_ratio = candidate_ratio
                        patience_counter = 0  # Reset patience
                    else:
                        patience_counter += 1

                    # Stop if no improvement for too long
                    if patience_counter >= patience:
                        break

        # Update best solution if improved
        if current_ratio > best_ratio:
            best_points = current_points

    return best_points

# EVOLVE-BLOCK-END