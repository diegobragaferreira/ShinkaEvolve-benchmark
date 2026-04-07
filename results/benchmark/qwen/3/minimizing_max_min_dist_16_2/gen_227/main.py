# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
import math
import time


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

        # Avoid division by zero
        if len(distances) == 0:
            return 0

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since minimize minimizes)
        if d_max == 0:
            return 0
        return -d_min / d_max

    def compute_ratio(points):
        """Compute the actual ratio for given points"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max

    def generate_initial_configurations():
        """Generate multiple diverse initial configurations"""
        configs = []
        np.random.seed(42)

        # 1. Grid configuration
        grid_points = []
        grid_size = 4  # 4x4 grid for 16 points
        spacing = 1.0 / (grid_size - 1) if grid_size > 1 else 1.0
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_points) < 16:
                    grid_points.append([i * spacing, j * spacing])
        configs.append(np.array(grid_points))

        # 2. Perturbed grid configuration
        perturbed_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(perturbed_points) < 16:
                    x = max(0, min(1, i * spacing + np.random.normal(0, 0.05 * spacing)))
                    y = max(0, min(1, j * spacing + np.random.normal(0, 0.05 * spacing)))
                    perturbed_points.append([x, y])
        configs.append(np.array(perturbed_points))

        # 3. Random configuration
        configs.append(np.random.rand(16, 2))

        # 4. Enhanced hexagonal configuration with better mathematical foundation
        hex_points = []

        # Create a more sophisticated hexagonal lattice with proper spacing
        # Using 4 rows and 4 columns arranged in hexagonal pattern
        rows = 4
        cols = 4

        # Calculate proper spacing for hexagonal packing in unit square
        # For optimal hexagonal packing, spacing should be calculated to fit 16 points
        spacing_x = 1.0 / (cols - 0.5)  # Adjusted for hexagonal packing
        spacing_y = spacing_x * np.sqrt(3) / 2

        # Generate hexagonal lattice points
        for i in range(rows):
            for j in range(cols):
                if len(hex_points) < 16:
                    # Position in hexagonal pattern
                    x = j * spacing_x + (i % 2) * spacing_x / 2
                    y = i * spacing_y

                    # Apply systematic asymmetry to break rotational symmetry
                    # Use prime-based perturbations for better distribution
                    prime_factor = (i * 7 + j * 11) % 13
                    asymmetry_x = 0.005 * np.sin(prime_factor * 0.3) * np.cos(prime_factor * 0.7)
                    asymmetry_y = 0.005 * np.cos(prime_factor * 0.4) * np.sin(prime_factor * 0.8)

                    # Add structured noise to avoid regular patterns
                    noise_x = 0.003 * np.sin(i * 0.5 + j * 0.3)
                    noise_y = 0.003 * np.cos(i * 0.2 + j * 0.6)

                    x += asymmetry_x + noise_x
                    y += asymmetry_y + noise_y

                    hex_points.append([x, y])

        # Normalize to [0,1] bounds properly
        if len(hex_points) > 0:
            hex_array = np.array(hex_points)
            x_range = np.max(hex_array[:, 0]) - np.min(hex_array[:, 0])
            y_range = np.max(hex_array[:, 1]) - np.min(hex_array[:, 1])

            if x_range > 0 and y_range > 0:
                # Scale and translate to fit within unit square
                hex_array[:, 0] = (hex_array[:, 0] - np.min(hex_array[:, 0])) / x_range
                hex_array[:, 1] = (hex_array[:, 1] - np.min(hex_array[:, 1])) / y_range

                # Rescale to make better use of available space
                hex_array[:, 0] = hex_array[:, 0] * 0.9 + 0.05
                hex_array[:, 1] = hex_array[:, 1] * 0.9 + 0.05

                # Convert back to list
                hex_points = [list(point) for point in hex_array]

        # Ensure we have exactly 16 points
        while len(hex_points) < 16:
            hex_points.append([np.random.rand(), np.random.rand()])

        configs.append(np.array(hex_points[:16]))

        # 5. Fibonacci-inspired pattern for better distribution
        fib_points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

        for i in range(16):
            y = 1 - (i / float(16 - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            # Map to 2D unit square
            x_mapped = (x + 1) / 2
            y_mapped = (z + 1) / 2

            fib_points.append([np.clip(x_mapped, 0, 1), np.clip(y_mapped, 0, 1)])

        configs.append(np.array(fib_points))

        return configs

    def simulated_annealing(points, max_iter=5000, initial_temp=1.0, cooling_rate=0.9995):
        """
        Simulated Annealing optimization for point dispersion
        """
        current_points = points.copy()
        current_ratio = compute_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio

        temp = initial_temp

        for iteration in range(max_iter):
            # Create neighbor by perturbing one random point
            neighbor_points = current_points.copy()
            idx = np.random.randint(0, len(neighbor_points))

            # Perturb the selected point with adaptive step size
            step_size = 0.02 if iteration < max_iter//2 else 0.005
            neighbor_points[idx, 0] += np.random.normal(0, step_size)
            neighbor_points[idx, 1] += np.random.normal(0, step_size)

            # Keep within bounds
            neighbor_points[idx, 0] = np.clip(neighbor_points[idx, 0], 0, 1)
            neighbor_points[idx, 1] = np.clip(neighbor_points[idx, 1], 0, 1)

            # Calculate neighbor ratio
            neighbor_ratio = compute_ratio(neighbor_points)

            # Accept or reject the neighbor
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                if neighbor_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                if delta < 0:  # Only accept worse solutions with probability
                    acceptance_prob = math.exp(delta / temp)
                    if np.random.random() < acceptance_prob:
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio

            # Cool down
            temp *= cooling_rate

            # Early stopping condition
            if temp < 1e-8:
                break

        return best_points, best_ratio

    def optimize_with_lbfgsb(initial_points, max_iter=2000):
        """Optimize using L-BFGS-B for better local convergence"""
        # Flatten initial guess
        x0 = initial_points.flatten()

        # Set up bounds (each coordinate must be between 0 and 1)
        bounds = [(0, 1) for _ in range(32)]

        # Optimize using L-BFGS-B which handles bounds well
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                return optimized_points, compute_ratio(optimized_points)
            else:
                warnings.warn(f"L-BFGS-B optimization failed: {result.message}")
                return initial_points, compute_ratio(initial_points)
        except Exception as e:
            warnings.warn(f"L-BFGS-B optimization error: {str(e)}")
            return initial_points, compute_ratio(initial_points)

    # Generate multiple initial configurations
    initial_configs = generate_initial_configurations()

    best_points = None
    best_ratio = -np.inf
    start_time = time.time()
    max_time_seconds = 180

    # Try all initial configurations with optimization
    for i, initial_config in enumerate(initial_configs):
        if time.time() - start_time > max_time_seconds - 5:  # Leave buffer for final processing
            break

        try:
            # First, try L-BFGS-B optimization
            lbfgsb_points, lbfgsb_ratio = optimize_with_lbfgsb(initial_config.copy())

            # Then refine with simulated annealing if time allows
            if time.time() - start_time < max_time_seconds - 10:
                sa_points, sa_ratio = simulated_annealing(lbfgsb_points.copy())
                final_points = sa_points if sa_ratio > lbfgsb_ratio else lbfgsb_points
                final_ratio = sa_ratio if sa_ratio > lbfgsb_ratio else lbfgsb_ratio
            else:
                final_points = lbfgsb_points
                final_ratio = lbfgsb_ratio

            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()

        except Exception as e:
            warnings.warn(f"Error optimizing initial config {i}: {str(e)}")
            continue

    # If no optimization succeeded, return the first configuration
    if best_points is None:
        return initial_configs[0]

    return best_points


# EVOLVE-BLOCK-END