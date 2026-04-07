# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Voronoi
import time
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
        
        try:
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0 or np.isinf(min_dist) or np.isnan(min_dist) or np.isnan(max_dist):
                return 0.0
                
            return min_dist / max_dist
        except Exception:
            return 0.0

    def voronoi_relaxation(points, max_iterations=50, tolerance=1e-6):
        """Improve point distribution using Voronoi relaxation."""
        current_points = points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        best_points = current_points.copy()

        for iteration in range(max_iterations):
            try:
                # Compute Voronoi diagram
                vor = Voronoi(current_points)

                # Calculate new positions as centroids of Voronoi cells
                new_points = np.zeros_like(current_points)
                converged = True

                # Process each point
                for i in range(len(current_points)):
                    # Get vertices of Voronoi cell for point i
                    region = vor.regions[vor.point_region[i]]

                    if -1 in region or len(region) < 3:
                        # Handle unbounded regions (use current position with slight adjustment)
                        new_points[i] = current_points[i] + np.random.normal(0, 0.001, 2)
                        continue

                    # Extract vertices of the Voronoi cell
                    vertices = np.array([vor.vertices[j] for j in region if j >= 0])

                    if len(vertices) < 3:
                        # Not enough vertices, use current position
                        new_points[i] = current_points[i]
                        continue

                    # Compute centroid of polygon (Voronoi cell)
                    centroid = np.mean(vertices, axis=0)

                    # Apply boundary constraints
                    centroid = np.clip(centroid, 0, 1)

                    # Update point position
                    new_points[i] = centroid

                    # Check for convergence
                    if np.linalg.norm(new_points[i] - current_points[i]) > tolerance:
                        converged = False

                # Apply cooling schedule for better convergence
                cooling_factor = 0.95 ** iteration
                current_points = current_points + cooling_factor * (new_points - current_points)

                # Ensure points stay within bounds
                current_points = np.clip(current_points, 0, 1)

                # Track best solution
                if iteration % 10 == 0:
                    ratio = compute_min_max_ratio(current_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = current_points.copy()

                # Early stopping if converged
                if converged:
                    break

            except Exception as e:
                # Fallback to simple perturbation
                current_points += np.random.normal(0, 0.001, current_points.shape)
                current_points = np.clip(current_points, 0, 1)

        return best_points

    def generate_initial_configurations():
        """Generate diverse initial configurations for multi-start optimization."""
        configs = []
        
        # Strategy 1: Hexagonal grid
        np.random.seed(42)
        hex_points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = np.sqrt(3) / 2 / (rows - 1) if rows > 1 else 1.0

        for i in range(rows):
            for j in range(cols):
                if len(hex_points) >= 16:
                    break
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                hex_points.append([x, y])

        # Normalize and clip
        hex_points = np.array(hex_points[:16])
        if len(hex_points) > 0:
            x_min, y_min = np.min(hex_points, axis=0)
            x_max, y_max = np.max(hex_points, axis=0)
            if x_max > x_min and y_max > y_min:
                hex_points[:, 0] = (hex_points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                hex_points[:, 1] = (hex_points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        configs.append(hex_points.copy())

        # Strategy 2: Perturbed hexagonal grid
        perturbed_hex = configs[0] + np.random.normal(0, 0.015, configs[0].shape)
        configs.append(np.clip(perturbed_hex, 0, 1))

        # Strategy 3: Golden spiral
        np.random.seed(123)
        golden_spiral_points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            angle = 2 * np.pi * i / phi
            radius = 0.4 * np.sqrt(i / 15) if i > 0 else 0
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            golden_spiral_points.append([x, y])
        configs.append(np.array(golden_spiral_points))

        # Strategy 4: Fibonacci sphere projection (2D approximation)
        sph_points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
        for i in range(16):
            y = 1 - (i / float(16 - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            sph_points.append([x, y])

        # Normalize to unit square
        sph_points = np.array(sph_points)
        if len(sph_points) > 0:
            x_min, y_min = np.min(sph_points, axis=0)
            x_max, y_max = np.max(sph_points, axis=0)
            if x_max > x_min and y_max > y_min:
                sph_points[:, 0] = (sph_points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                sph_points[:, 1] = (sph_points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        configs.append(sph_points.copy())

        # Strategy 5: Random uniform distribution
        random_points = np.random.rand(16, 2)
        configs.append(np.clip(random_points, 0, 1))

        # Strategy 6: Grid pattern with jitter
        grid_points = []
        for i in range(4):
            for j in range(4):
                if len(grid_points) >= 16:
                    break
                x = i * 0.25 + 0.125 + np.random.normal(0, 0.01)
                y = j * 0.25 + 0.125 + np.random.normal(0, 0.01)
                grid_points.append([x, y])
        grid_points = np.array(grid_points[:16])
        configs.append(np.clip(grid_points, 0, 1))

        return configs

    def adaptive_local_optimization(points, max_iter=1000):
        """Apply adaptive local optimization with multiple strategies."""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Strategy 1: L-BFGS-B with adaptive parameters
        try:
            # Start with less strict tolerance for faster convergence initially
            result = minimize(
                objective_function,
                current_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': max_iter // 3, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass

        # Strategy 2: SLSQP with tighter tolerances for high-quality refinement
        try:
            result = minimize(
                objective_function,
                best_points.flatten(),
                method='SLSQP',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': max_iter // 3, 'ftol': 1e-12}
            )
            if result.success:
                final_points = result.x.reshape(-1, 2)
                final_points = np.clip(final_points, 0, 1)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass

        # Strategy 3: Additional random perturbations followed by L-BFGS-B
        np.random.seed(999)
        for _ in range(2):
            perturbed = best_points + np.random.normal(0, 0.005, best_points.shape)
            perturbed = np.clip(perturbed, 0, 1)
            try:
                result = minimize(
                    objective_function,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': max_iter // 6, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
            except Exception:
                continue

        return best_points, best_ratio

    def simulated_annealing(points, max_iter=1000, initial_temp=1.0, cooling_rate=0.95):
        """Simulated annealing optimization to escape local optima."""
        current_points = points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        temperature = initial_temp

        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing current points
            neighbor_points = current_points + np.random.normal(0, temperature * 0.01, current_points.shape)
            neighbor_points = np.clip(neighbor_points, 0, 1)

            # Evaluate neighbor
            neighbor_ratio = compute_min_max_ratio(neighbor_points)

            # Accept or reject based on Metropolis criterion
            if neighbor_ratio > current_ratio:
                current_points = neighbor_points
                current_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                acceptance_prob = np.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio

            # Update best solution
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points.copy()

            # Cool down
            temperature *= cooling_rate

            # Early stopping if temperature gets too low
            if temperature < 1e-8:
                break

        return best_points, best_ratio

    def objective_function(x_flat):
        """Objective function to minimize (negative ratio to maximize ratio)."""
        # Reshape into points
        points = x_flat.reshape(-1, 2)
        
        # Ensure points are within bounds for numerical stability
        points = np.clip(points, 0, 1)
        
        # Calculate ratio
        ratio = compute_min_max_ratio(points)
        
        # Return negative ratio to minimize (we want to maximize ratio)
        return -ratio

    def run_multi_start_optimization():
        """Run optimization with multiple restarts and strategies."""
        # Generate initial configurations
        initial_configs = generate_initial_configurations()

        # Find the best initial configuration
        best_initial = None
        best_ratio = -np.inf

        for points in initial_configs:
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_initial = points.copy()

        # Apply Voronoi relaxation to the best initial as preprocessing
        if best_initial is not None:
            best_initial = voronoi_relaxation(best_initial, max_iterations=30)

        # Multi-start optimization
        best_points = best_initial.copy()
        current_best_ratio = best_ratio

        # Try different noise levels for diversification
        noise_levels = [0.02, 0.03, 0.04, 0.05, 0.06]
        
        for restart in range(len(noise_levels)):
            noise_level = noise_levels[restart % len(noise_levels)]
            
            # Generate new variation
            np.random.seed(restart + 1000)
            perturbed_points = best_points.copy()
            perturbed_points += np.random.normal(0, noise_level, best_points.shape)
            perturbed_points = np.clip(perturbed_points, 0, 1)

            # Adaptive optimization based on previous performance
            if current_best_ratio < 0.25:
                # For lower ratios, use more extensive optimization
                max_iter = 800
            else:
                # For higher ratios, use more focused optimization
                max_iter = 600

            # Apply local optimization
            optimized_points, optimized_ratio = adaptive_local_optimization(perturbed_points, max_iter=max_iter)

            # If optimization didn't help much, try simulated annealing as fallback
            if optimized_ratio < current_best_ratio * 0.99:
                sa_points, sa_ratio = simulated_annealing(optimized_points, max_iter=500)
                if sa_ratio > optimized_ratio:
                    optimized_points = sa_points
                    optimized_ratio = sa_ratio

            if optimized_ratio > current_best_ratio:
                current_best_ratio = optimized_ratio
                best_points = optimized_points.copy()

        # Final refinement passes
        final_points, final_ratio = adaptive_local_optimization(best_points, max_iter=500)
        
        # Try one more simulated annealing pass if still not optimal
        if final_ratio < current_best_ratio * 0.995:
            sa_points, sa_ratio = simulated_annealing(final_points, max_iter=300)
            if sa_ratio > final_ratio:
                final_points = sa_points
                final_ratio = sa_ratio

        return final_points

    # Main execution
    try:
        # Run multi-start optimization
        result = run_multi_start_optimization()
        return result
    except Exception as e:
        # Fallback to simple hexagonal grid if anything fails
        hex_points = []
        rows = 4
        cols = 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = np.sqrt(3) / 2 / (rows - 1) if rows > 1 else 1.0

        for i in range(rows):
            for j in range(cols):
                if len(hex_points) >= 16:
                    break
                x = j * spacing_x + (i % 2) * spacing_x / 2
                y = i * spacing_y
                hex_points.append([x, y])

        return np.array(hex_points[:16])

# EVOLVE-BLOCK-END