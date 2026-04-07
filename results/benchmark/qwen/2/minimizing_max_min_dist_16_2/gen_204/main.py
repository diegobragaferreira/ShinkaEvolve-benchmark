# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def objective(x):
        # Reshape x back to points array
        points = x.reshape(-1, 2)

        # Compute pairwise distances efficiently using scipy
        distances = pdist(points)

        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero - return a very negative value for invalid cases
        if max_dist == 0:
            return -1e10

        return -min_dist / max_dist

    def evaluate_solution(points):
        """Evaluate the quality of a solution"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist

    def adaptive_perturb_points(points, base_magnitude, current_ratio, target_ratio=0.2786):
        """Add adaptive random perturbations based on solution quality"""
        # Adjust perturbation magnitude based on current solution quality
        if current_ratio > target_ratio * 0.9:
            # If we're close to target, use small perturbations
            magnitude = base_magnitude * 0.3
        elif current_ratio > target_ratio * 0.7:
            # Medium quality, use medium perturbations
            magnitude = base_magnitude * 0.6
        else:
            # Poor quality, use larger perturbations
            magnitude = base_magnitude * 1.0
            
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, magnitude, points.shape)
        perturbed = points + noise
        return np.clip(perturbed, 0, 1)

    def create_initial_grid():
        """Create initial 4x4 grid points"""
        grid_size = 4
        x_vals = np.linspace(0.05, 0.95, grid_size)
        y_vals = np.linspace(0.05, 0.95, grid_size)
        return np.array([[x, y] for x in x_vals for y in y_vals])

    def create_spiral_pattern():
        """Create a spiral-like initial pattern"""
        np.random.seed(42)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles) * 0.8
        y = 0.5 + radii * np.sin(angles) * 0.8
        spiral_points = np.column_stack([x, y])
        return spiral_points

    def create_hexagonal_pattern():
        """Create a hexagonal pattern"""
        np.random.seed(42)
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                # Add small random perturbation
                x += (np.random.random() - 0.5) * 0.05
                y += (np.random.random() - 0.5) * 0.05
                points.append([x, y])
        return np.array(points)

    def create_fibonacci_sphere():
        """Create points using Fibonacci sphere method for better distribution"""
        np.random.seed(42)
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            theta = np.arccos(-1 + (2 * i) / 15)  # elevation angle
            phi_angle = (i * 2 * np.pi) / (phi * phi)  # azimuthal angle

            # Convert to cartesian coordinates
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)

            # Map to [0.05, 0.95] range to avoid boundaries
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2

            points.append([x, y])

        return np.array(points)

    def multi_stage_optimization(initial_points, method='auto'):
        """Perform multi-stage optimization for better convergence"""
        best_points = initial_points.copy()
        best_ratio = evaluate_solution(best_points)
        
        # Stage 1: Coarse optimization (faster, less precise)
        try:
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            if method == 'auto' or method == 'L-BFGS-B':
                result_coarse = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result_coarse.success:
                    coarse_points = result_coarse.x.reshape(-1, 2)
                    coarse_ratio = evaluate_solution(coarse_points)
                    if coarse_ratio > best_ratio:
                        best_points = coarse_points.copy()
                        best_ratio = coarse_ratio
        except Exception:
            pass
        
        # Stage 2: Fine optimization (slower, more precise)
        try:
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Try both methods for robustness
            methods_to_try = ['L-BFGS-B', 'SLSQP'] if method == 'auto' else [method]
            
            for method_name in methods_to_try:
                try:
                    result_fine = minimize(
                        objective,
                        x0,
                        method=method_name,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if result_fine.success:
                        fine_points = result_fine.x.reshape(-1, 2)
                        fine_ratio = evaluate_solution(fine_points)
                        if fine_ratio > best_ratio:
                            best_points = fine_points.copy()
                            best_ratio = fine_ratio
                except Exception:
                    continue
                    
        except Exception:
            pass
            
        return best_points, best_ratio

    # Multi-start optimization with adaptive strategies
    best_ratio = 0
    best_points = None
    
    # Strategy 1: Adaptive grid with dynamic perturbations
    try:
        grid_points = create_initial_grid()
        # Evaluate baseline
        baseline_ratio = evaluate_solution(grid_points)
        
        # Try perturbations with adaptive magnitudes
        perturbation_levels = [0.005, 0.01, 0.02, 0.03]
        for mag in perturbation_levels:
            perturbed_points = adaptive_perturb_points(
                grid_points, 
                mag, 
                baseline_ratio
            )
            optimized_points, ratio = multi_stage_optimization(perturbed_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception as e:
        pass

    # Strategy 2: Fibonacci-style spiral pattern
    try:
        spiral_points = create_spiral_pattern()
        optimized_points, ratio = multi_stage_optimization(spiral_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except Exception as e:
        pass

    # Strategy 3: Hexagonal pattern
    try:
        hex_points = create_hexagonal_pattern()
        optimized_points, ratio = multi_stage_optimization(hex_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except Exception as e:
        pass

    # Strategy 4: Fibonacci sphere-like pattern
    try:
        fib_points = create_fibonacci_sphere()
        optimized_points, ratio = multi_stage_optimization(fib_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
    except Exception as e:
        pass

    # Strategy 5: Multiple random starts with adaptive perturbations
    try:
        for i in range(5):  # Multiple random starts
            np.random.seed(42 + i * 10)  # Different seed for each attempt
            random_points = np.random.rand(16, 2)
            # Apply adaptive perturbation based on evaluation
            evaluated_ratio = evaluate_solution(random_points)
            adapted_points = adaptive_perturb_points(
                random_points, 
                0.02, 
                evaluated_ratio
            )
            optimized_points, ratio = multi_stage_optimization(adapted_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception as e:
        pass

    # Strategy 6: Grid-based with higher precision optimization
    try:
        if best_points is None:
            grid_points = create_initial_grid()
            optimized_points, ratio = multi_stage_optimization(grid_points, 'SLSQP')
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception as e:
        pass

    # Final refinement with highest precision if needed
    if best_points is not None and best_ratio < 0.25:  # Only if solution is not very good
        try:
            # Perform high precision optimization
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )

            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = evaluate_solution(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
        except Exception:
            pass

    # If no good solution was found, return a default configuration
    if best_points is None:
        # Fallback to simple grid initialization
        grid_points = create_initial_grid()
        best_points = grid_points.copy()

    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)

    return best_points

# EVOLVE-BLOCK-END