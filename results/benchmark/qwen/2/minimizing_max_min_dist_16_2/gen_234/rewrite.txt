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
        
        # Calculate pairwise distances efficiently using scipy
        distances = pdist(points)
        
        # Handle edge case with no distances
        if len(distances) == 0:
            return 0
            
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-12:
            return 0
            
        # Return negative ratio (we minimize negative ratio to maximize ratio)
        return -d_min / d_max

    def evaluate_solution(points):
        """Evaluate the quality of a solution efficiently"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 1e-12:
            return 0
        return d_min / d_max

    def generate_hexagonal_initial():
        """Generate initial configuration based on hexagonal lattice for better spacing"""
        points = []
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1)
        spacing_y = 1.0 / (rows - 1)

        for i in range(rows):
            for j in range(cols):
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else spacing_x * 0.5
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

    def generate_regular_grid():
        """Generate regular grid initial configuration"""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])
        return np.array(points)

    def generate_structured_grid():
        """Generate structured grid with adaptive perturbation"""
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                points.append([x, y])

        points = np.array(points)
        
        # Add adaptive perturbation based on distance distribution
        distances = pdist(points)
        if len(distances) > 0:
            current_ratio = np.min(distances) / np.max(distances) if np.max(distances) > 0 else 0
            # Scale perturbation inversely with current distribution balance
            perturbation_magnitude = max(0.005, 0.03 * (1.0 - current_ratio * 5))
        else:
            perturbation_magnitude = 0.02

        # Add controlled perturbation
        np.random.seed(42)
        perturbation = np.random.normal(0, perturbation_magnitude, points.shape)
        points += perturbation

        # Clip to valid range
        points = np.clip(points, 0.001, 0.999)
        return points

    def generate_multi_scale_grid():
        """Generate multiple grid configurations for diverse starting points"""
        configs = []
        
        # Regular grid
        grid_points = np.array([[i, j] for i in range(4) for j in range(4)]) / 3.0
        configs.append(grid_points)
        
        # Perturbed grid
        np.random.seed(42)
        perturbed = grid_points + np.random.normal(0, 0.02, (16, 2))
        perturbed = np.clip(perturbed, 0.001, 0.999)
        configs.append(perturbed)
        
        # Corner-perturbed grid
        corner_perturbed = grid_points.copy()
        corner_perturbed[0] = [0.1, 0.1]      # Bottom-left
        corner_perturbed[15] = [0.9, 0.9]     # Top-right
        corner_perturbed[3] = [0.9, 0.1]      # Bottom-right
        corner_perturbed[12] = [0.1, 0.9]     # Top-left
        configs.append(corner_perturbed)
        
        return configs

    def adaptive_local_search(initial_points, bounds, maxiter=100):
        """Adaptive local optimization with multiple phases"""
        # Phase 1: Coarse optimization (fast)
        try:
            result_coarse = minimize(
                objective,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter//3, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if result_coarse.success:
                coarse_points = result_coarse.x.reshape(-1, 2)
                
                # Phase 2: Medium optimization (moderate precision)
                result_medium = minimize(
                    objective,
                    coarse_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter//3, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if result_medium.success:
                    medium_points = result_medium.x.reshape(-1, 2)
                    
                    # Phase 3: Fine optimization (high precision)
                    result_fine = minimize(
                        objective,
                        medium_points.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': maxiter//3, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    if result_fine.success:
                        return result_fine.x.reshape(-1, 2)
                    return medium_points
                return coarse_points
        except Exception:
            pass
        return initial_points

    # Generate diverse initial configurations
    np.random.seed(42)
    initial_configs = [
        generate_hexagonal_initial(),
        generate_fibonacci_spiral(),
        generate_regular_grid(),
        generate_structured_grid()
    ]

    # Add multiple perturbed versions of structured configurations
    structured_configs = generate_multi_scale_grid()
    for config in structured_configs:
        # Multiple perturbed variants
        for i in range(2):
            np.random.seed(42 + i)
            perturbed = config + np.random.normal(0, 0.01 + i * 0.005, config.shape)
            perturbed = np.clip(perturbed, 0.001, 0.999)
            initial_configs.append(perturbed)

    # Define bounds for each coordinate (between 0.001 and 0.999 to avoid boundary issues)
    bounds = [(0.001, 0.999) for _ in range(32)]

    best_ratio = -np.inf
    best_points = None

    # Stage 1: Global search with differential evolution for rough exploration
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=20,  # Reduced iterations for speed
            popsize=8,   # Smaller population for faster execution
            seed=42,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7
        )

        # Refine with local optimization using tighter tolerances
        refined_points = adaptive_local_search(de_result.x.reshape(-1, 2), bounds, maxiter=100)
        ratio = evaluate_solution(refined_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = refined_points.copy()
    except Exception:
        pass

    # Stage 2: Multi-start local optimization from diverse initial points
    # Focus on the most promising configurations to save time
    initial_configs_subset = initial_configs[:6]  # Limit to first 6 configs
    
    for i, initial_config in enumerate(initial_configs_subset):
        try:
            # Apply adaptive optimization based on configuration type
            refined_points = adaptive_local_search(initial_config, bounds, maxiter=100)
            ratio = evaluate_solution(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except Exception:
            continue

    # Stage 3: If solution is still weak, try focused grid search
    if best_points is None or best_ratio < 0.20:  # Only if solution is poor
        # Try focused search on promising regions
        test_grid = np.linspace(0.15, 0.85, 5)  # Focus on interior region
        for i in range(len(test_grid)):
            for j in range(len(test_grid)):
                base_x = test_grid[i]
                base_y = test_grid[j]

                # Create structured perturbation based on position
                np.random.seed(42 + i * 5 + j)
                base_points = np.array([[base_x, base_y]] * 16)
                # Add position-dependent perturbation
                perturbation = np.random.normal(0, 0.02, (16, 2))
                perturbed_points = base_points + perturbation
                perturbed_points = np.clip(perturbed_points, 0.001, 0.999)

                try:
                    refined_points = adaptive_local_search(perturbed_points, bounds, maxiter=50)
                    ratio = evaluate_solution(refined_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
                except Exception:
                    continue

    # Stage 4: Final high-precision refinement if needed
    if best_points is not None and best_ratio < 0.25:  # Only if solution is moderate
        try:
            # Final high-precision optimization
            refined_points = adaptive_local_search(best_points, bounds, maxiter=200)
            ratio = evaluate_solution(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except Exception:
            pass

    # Return the best solution found
    if best_points is None:
        # Fallback to a robust structured configuration
        fallback_config = generate_structured_grid()
        # Add small random noise to break any remaining symmetries
        np.random.seed(42)
        fallback_points = fallback_config + np.random.normal(0, 0.005, fallback_config.shape)
        fallback_points = np.clip(fallback_points, 0.001, 0.999)
        best_points = fallback_points

    return best_points

# EVOLVE-BLOCK-END