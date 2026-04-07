# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Voronoi
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0

        # Compute pairwise distances using squareform for numerical stability
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        dmin = np.min(distances)
        dmax = np.max(distances)

        # Avoid division by zero
        if dmax == 0:
            return 0

        return dmin / dmax

    def generate_hexagonal_grid():
        """Generate a hexagonal grid initialization for 16 points with enhanced spacing."""
        np.random.seed(42)
        points = []

        # Create a more sophisticated hexagonal pattern with proper spacing
        rows, cols = 4, 4
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 0.5
        spacing_y = 1.0 / (rows - 1) if rows > 1 else 0.5

        for i in range(rows):
            for j in range(cols):
                # Hexagonal grid with alternating row offset
                x = j * spacing_x
                if i % 2 == 1:  # Offset odd rows
                    x += spacing_x * 0.5

                y = i * spacing_y

                # Add more substantial but controlled random perturbation
                x += (np.random.rand() - 0.5) * 0.15 * spacing_x
                y += (np.random.rand() - 0.5) * 0.15 * spacing_y

                # Ensure points stay within safe boundaries with padding
                x = np.clip(x, 0.02, 0.98)
                y = np.clip(y, 0.02, 0.98)

                points.append([x, y])

        return np.array(points[:16])

    def optimize_with_voronoi_guidance(points, max_iter=100):
        """Optimize points using Voronoi-based guidance for better convergence."""
        current_points = points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Precompute bounds for clipping
        bounds_min = 0.01
        bounds_max = 0.99
        
        for iteration in range(max_iter):
            # Analyze current Voronoi diagram
            try:
                vor = Voronoi(current_points)
                # Get Voronoi vertices and regions for each point
                n_points = len(current_points)
                
                # Gradient-based updates guided by Voronoi geometry
                new_points = current_points.copy()
                step_size = 0.001 * (1.0 - iteration/max_iter)  # Decreasing step size
                
                # For each point, move towards more optimal position based on Voronoi analysis
                for i in range(n_points):
                    # Find neighboring points that influence this point's Voronoi cell
                    # This is a simplified approximation for performance
                    distances_to_others = [np.linalg.norm(current_points[i] - current_points[j]) 
                                         for j in range(n_points) if i != j]
                    sorted_indices = np.argsort(distances_to_others)
                    
                    # Move point away from very close neighbors and towards far ones
                    # This creates a repulsion-attract mechanism
                    total_force = np.array([0.0, 0.0])
                    
                    # Repulsion from close points (up to 3 closest)
                    for j in range(min(3, len(sorted_indices))):
                        idx = sorted_indices[j]
                        dist = distances_to_others[idx]
                        if dist > 1e-6 and dist < 0.2:  # Only consider nearby points
                            force_direction = current_points[i] - current_points[idx]
                            force_magnitude = 1.0 / (dist * dist + 1e-8)
                            total_force += force_direction * force_magnitude
                    
                    # Attraction to far points (up to 3 furthest)
                    for j in range(min(3, len(sorted_indices))):
                        idx = sorted_indices[-(j+1)]
                        dist = distances_to_others[idx]
                        if dist > 0.2:  # Only consider distant points for attraction
                            force_direction = current_points[idx] - current_points[i]
                            force_magnitude = 0.1 / (dist * dist + 1e-8)
                            total_force += force_direction * force_magnitude
                    
                    # Apply force with damping
                    new_position = current_points[i] + total_force * step_size * 0.5
                    # Clip to bounds
                    new_position = np.clip(new_position, bounds_min, bounds_max)
                    new_points[i] = new_position
                
                # Evaluate new configuration
                new_ratio = compute_min_max_ratio(new_points)
                
                # Accept improvement or occasionally accept worse solutions for escape
                if new_ratio > best_ratio:
                    current_points = new_points
                    best_ratio = new_ratio
                    best_points = new_points.copy()
                elif np.random.rand() < 0.05:  # 5% chance to accept worse solutions
                    current_points = new_points
                    
            except Exception:
                # If Voronoi computation fails, do simple random perturbations
                new_points = current_points.copy()
                for i in range(len(current_points)):
                    new_points[i] += (np.random.rand(2) - 0.5) * 0.01
                    new_points[i] = np.clip(new_points[i], bounds_min, bounds_max)
                current_points = new_points
                
            # Periodic validation check
            if iteration % 20 == 0:
                ratio_check = compute_min_max_ratio(current_points)
                if ratio_check > best_ratio:
                    best_ratio = ratio_check
                    best_points = current_points.copy()
        
        return best_points

    def optimize_points_local(points):
        """Optimize the point configuration using local optimization with enhanced constraints."""
        n_points = len(points)
        total_vars = n_points * 2

        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            # Ensure points are within bounds with epsilon padding
            points = np.clip(points, 1e-6, 1-1e-6)

            # Calculate pairwise distances using squareform for numerical stability
            distances = squareform(pdist(points))

            # Mask diagonal elements (distance to itself)
            np.fill_diagonal(distances, np.inf)

            # Get min and max distances
            d_min = np.min(distances)
            d_max = np.max(distances)

            # Return negative ratio to maximize (since we're minimizing)
            if d_max <= 0:
                return -1.0
            return -d_min / d_max

        def bounds_constraint(x_flat):
            points = x_flat.reshape(-1, 2)
            # Lower bounds (negative for inequality constraints)
            lower = -points.flatten()
            # Upper bounds (values that should be <= 0 for inequality constraints)
            upper = points.flatten() - 1.0
            return np.concatenate([lower, upper])

        # Use L-BFGS-B for local optimization
        bounds = [(1e-6, 1-1e-6) for _ in range(total_vars)]
        cons = {'type': 'ineq', 'fun': bounds_constraint}

        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                return result.x.reshape(-1, 2)
        except Exception:
            pass

        return points

    def adaptive_voronoi_optimization(initial_points, max_iterations=100):
        """Run multiple rounds of Voronoi-guided optimization with different strategies."""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Try several optimization strategies
        strategies = [
            lambda p: optimize_with_voronoi_guidance(p, max_iter=max_iterations//3),
            lambda p: optimize_with_voronoi_guidance(p, max_iter=max_iterations//2),
            lambda p: optimize_with_voronoi_guidance(p, max_iter=max_iterations)
        ]
        
        for strategy in strategies:
            try:
                optimized_points = strategy(best_points)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
            except:
                continue
                
        return best_points

    # Generate initial hexagonal grid configuration
    initial_points = generate_hexagonal_grid()

    # Apply Voronoi-guided optimization for better convergence
    optimized_points = adaptive_voronoi_optimization(initial_points, max_iterations=150)

    # Final local refinement with L-BFGS-B
    final_points = optimize_points_local(optimized_points)

    # Ensure final results respect bounds
    final_points = np.clip(final_points, 1e-6, 1-1e-6)

    return final_points

# EVOLVE-BLOCK-END