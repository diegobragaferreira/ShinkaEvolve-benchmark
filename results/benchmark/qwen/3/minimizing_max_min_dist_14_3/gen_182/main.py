# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    np.random.seed(42)

    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def normalize_to_unit_sphere(points):
        """Normalize points to lie on the unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def spherical_constraint(x_flat):
        """Apply spherical constraint to keep points on unit sphere."""
        points = x_flat.reshape(-1, 3)
        return normalize_to_unit_sphere(points).flatten()

    def spherical_objective(points_flat):
        """Objective function that maximizes min/max distance ratio on sphere."""
        points = points_flat.reshape(-1, 3)
        # Ensure points are on unit sphere
        points = normalize_to_unit_sphere(points)
        
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max == 0:
            return 0.0

        return d_min / d_max

    def spherical_gradient_objective(points_flat):
        """Gradient-based objective for optimization."""
        points = points_flat.reshape(-1, 3)
        points = normalize_to_unit_sphere(points)
        
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max == 0:
            return 0.0

        return -d_min / d_max  # Negative because we minimize in scipy

    def generate_initial_configurations():
        """Generate multiple initial configurations using different geometric principles."""
        configs = []
        
        # Strategy 1: Fibonacci sphere (classic good distribution)
        n = 14
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # Golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        
        fib_points = np.array(points)
        fib_points = normalize_to_unit_sphere(fib_points)
        configs.append(("fibonacci", fib_points))
        
        # Strategy 2: Random points on sphere with normalization
        np.random.seed(42)
        random_points = np.random.randn(14, 3)
        random_points = normalize_to_unit_sphere(random_points)
        configs.append(("random_sphere", random_points))
        
        # Strategy 3: Polar distribution with some structure
        polar_points = np.zeros((14, 3))
        # Place points along latitude circles
        for i in range(14):
            if i < 6:
                # Upper hemisphere - evenly spaced
                lat = np.pi * (i / 5) / 2
                lon = 2 * np.pi * (i % 5) / 5
                polar_points[i] = [
                    np.sin(lat) * np.cos(lon),
                    np.sin(lat) * np.sin(lon),
                    np.cos(lat)
                ]
            else:
                # Lower hemisphere - evenly spaced
                lat = np.pi * ((i-6) / 5) / 2 + np.pi/2
                lon = 2 * np.pi * ((i-6) % 5) / 5
                polar_points[i] = [
                    np.sin(lat) * np.cos(lon),
                    np.sin(lat) * np.sin(lon),
                    np.cos(lat)
                ]
        polar_points = normalize_to_unit_sphere(polar_points)
        configs.append(("polar", polar_points))
        
        # Strategy 4: Perturbed Fibonacci
        np.random.seed(43)
        perturbed = fib_points + np.random.normal(0, 0.05, fib_points.shape)
        perturbed = normalize_to_unit_sphere(perturbed)
        configs.append(("perturbed_fib", perturbed))
        
        return configs

    def voronoi_based_refinement(initial_points, max_iterations=50):
        """Refine points using Voronoi-based optimization approach."""
        points = initial_points.copy()
        
        for iteration in range(max_iterations):
            # Create Voronoi diagram
            try:
                sv = SphericalVoronoi(points)
                
                # Compute Voronoi cell areas
                cell_areas = sv.calculate_areas()
                
                # Simple refinement: move points to centroids
                # This is a simplified version - in practice this would be more sophisticated
                # But serves as a good geometric update step
                
                # For now, just try small perturbations guided by geometric considerations
                old_ratio = compute_min_max_ratio(points)
                
                # Try perturbing each point slightly in a way that tends to increase minimum distances
                updated_points = points.copy()
                for i in range(len(points)):
                    # Try moving point away from neighbors
                    neighbor_indices = []
                    for j in range(len(points)):
                        if i != j:
                            dist = np.linalg.norm(points[i] - points[j])
                            if dist < 0.5:  # Close neighbors
                                neighbor_indices.append(j)
                    
                    if neighbor_indices:
                        # Move away from close neighbors
                        move_vector = np.zeros(3)
                        for j in neighbor_indices:
                            diff = points[i] - points[j]
                            norm_diff = np.linalg.norm(diff)
                            if norm_diff > 0:
                                move_vector += diff / norm_diff * (0.1 / (norm_diff + 0.01))
                        
                        if np.linalg.norm(move_vector) > 0:
                            updated_points[i] += move_vector * 0.01
                        
                # Normalize to unit sphere
                updated_points = normalize_to_unit_sphere(updated_points)
                
                new_ratio = compute_min_max_ratio(updated_points)
                if new_ratio > old_ratio:
                    points = updated_points
                else:
                    # Small random perturbation if no improvement
                    np.random.seed(iteration)  # For reproducibility
                    noise = np.random.normal(0, 0.001, points.shape)
                    points += noise
                    points = normalize_to_unit_sphere(points)
                    
            except Exception:
                # Fall back to simple geometric perturbation
                np.random.seed(iteration)  
                noise = np.random.normal(0, 0.002, points.shape)
                points += noise
                points = normalize_to_unit_sphere(points)
        
        return points

    # Phase 1: Generate diverse initial configurations
    initial_configs = generate_initial_configurations()
    
    # Phase 2: Optimize each configuration
    best_solution = None
    best_ratio = 0.0
    
    # Try each initial configuration with different optimization approaches
    for config_name, initial_points in initial_configs:
        # Apply Voronoi-based refinement first
        refined_points = voronoi_based_refinement(initial_points, max_iterations=20)
        
        # Then apply constrained optimization
        try:
            bounds = [(-1, 1)] * (14 * 3)
            
            # Use L-BFGS-B for local optimization with spherical constraint
            x0 = refined_points.flatten()
            
            # Define constraint to keep points on unit sphere
            result = minimize(
                spherical_gradient_objective, 
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                optimized_points = normalize_to_unit_sphere(optimized_points)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = optimized_points.copy()
        except:
            pass
            
        # Also try a few more optimization attempts with different initialization
        for attempt in range(3):
            try:
                # Add more perturbation and try again
                np.random.seed(42 + attempt)
                perturbed = refined_points + np.random.normal(0, 0.005, refined_points.shape)
                perturbed = normalize_to_unit_sphere(perturbed)
                
                bounds = [(-1, 1)] * (14 * 3)
                x0 = perturbed.flatten()
                
                result = minimize(
                    spherical_gradient_objective, 
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 3)
                    optimized_points = normalize_to_unit_sphere(optimized_points)
                    ratio = compute_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = optimized_points.copy()
            except:
                continue

    # Phase 3: Final refinement if we have a good solution
    if best_solution is not None:
        # Apply final Voronoi refinement
        final_points = voronoi_based_refinement(best_solution, max_iterations=10)
        
        # One final L-BFGS optimization
        try:
            bounds = [(-1, 1)] * (14 * 3)
            x0 = final_points.flatten()
            
            result = minimize(
                spherical_gradient_objective, 
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                final_points_opt = result.x.reshape(-1, 3)
                final_points_opt = normalize_to_unit_sphere(final_points_opt)
                final_ratio = compute_min_max_ratio(final_points_opt)
                
                if final_ratio > best_ratio:
                    best_solution = final_points_opt.copy()
        except:
            pass

    # Fallback to the best initial configuration if nothing worked
    if best_solution is None:
        # Select the best among initial configurations
        best_initial_points = None
        best_initial_ratio = 0.0
        
        for _, initial_points in initial_configs:
            ratio = compute_min_max_ratio(initial_points)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_initial_points = initial_points.copy()
                
        best_solution = best_initial_points

    return best_solution

# EVOLVE-BLOCK-END