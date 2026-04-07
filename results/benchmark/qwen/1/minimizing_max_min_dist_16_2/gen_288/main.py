# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Set seed for reproducibility
    np.random.seed(42)

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def initialize_fortune_grid():
        """Initialize points using a structured grid that mimics Voronoi optimal distribution."""
        # Create a 4x4 grid with slight perturbations to avoid perfect symmetry
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + 0.125
                y = i * 0.25 + 0.125
                # Add small random perturbation
                x += (np.random.rand() - 0.5) * 0.03
                y += (np.random.rand() - 0.5) * 0.03
                points.append([x, y])
        
        # Ensure points are within bounds
        points = np.array(points)
        points = np.clip(points, 0.01, 0.99)
        return points

    def compute_voronoi_quality(points):
        """Compute quality metric based on Voronoi cell uniformity."""
        try:
            vor = Voronoi(points)
            # Calculate areas of finite Voronoi cells
            areas = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    # Compute area of polygon
                    polygon_points = [vor.vertices[i] for i in region]
                    if len(polygon_points) >= 3:
                        # Simple polygon area calculation using shoelace formula
                        x_coords = [p[0] for p in polygon_points]
                        y_coords = [p[1] for p in polygon_points]
                        area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] 
                                           for i in range(len(x_coords)-1)) + 
                                         x_coords[-1] * y_coords[0] - x_coords[0] * y_coords[-1])
                        areas.append(area)
            
            if not areas:
                return 0.0
                
            # Return coefficient of variation of cell areas (lower is better)
            mean_area = np.mean(areas)
            if mean_area <= 1e-12:
                return 0.0
            cv = np.std(areas) / mean_area
            return 1.0 / (1.0 + cv)  # Invert so higher values indicate better uniformity
        except:
            return 0.0

    def optimize_with_voronoi_guidance(points, max_iter=100):
        """Optimize points using Voronoi-based guidance."""
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

    def initialize_hexagonal_grid():
        """Initialize points using an improved hexagonal grid pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Create hexagonal grid pattern with better distribution and spacing
        rows = 4
        cols = 4
        spacing_x = 0.25
        spacing_y = spacing_x * math.sqrt(3) / 2

        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx < n:
                    # Offset every other row for hexagonal packing
                    x = col * spacing_x + (row % 2) * spacing_x * 0.5
                    y = row * spacing_y
                    points[idx] = [x, y]
                    idx += 1

        # Scale to fit within [0.1,0.9]x[0.1,0.9] with some randomness
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) * 0.8 + 0.1
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min()) * 0.8 + 0.1

        # Add small random perturbation for better exploration
        points += np.random.normal(0, 0.008, points.shape)

        # Clamp to bounds
        points = np.clip(points, 0.01, 0.99)

        return points

    def initialize_spiral_pattern():
        """Initialize points using a spiral pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Create spiral pattern with better distribution
        angles = np.linspace(0, 4*np.pi, n, endpoint=False)
        radii = np.linspace(0.1, 0.4, n)

        for i in range(n):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

        return points

    def initialize_random():
        """Initialize points using random distribution."""
        return np.random.uniform(0.1, 0.9, (16, 2))

    def initialize_golden_spiral():
        """Initialize points using golden spiral pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Golden spiral with logarithmic growth
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            angle = i * 2 * np.pi / (phi * phi)
            radius = i * 0.3 / n
            points[i, 0] = 0.5 + radius * np.cos(angle)
            points[i, 1] = 0.5 + radius * np.sin(angle)

        return points

    def initialize_circle_packing():
        """Initialize points in a circular arrangement."""
        n = 16
        points = np.zeros((n, 2))

        # Arrange points evenly on a circle with some randomization
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii = 0.35 + np.random.normal(0, 0.04, n)  # Slight variations in radii

        for i in range(n):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

        return points

    def initialize_regular_grid():
        """Initialize points in a regular 4x4 grid."""
        n = 16
        points = np.zeros((n, 2))

        # Create regular grid with better spacing
        grid_size = 4
        spacing = 1.0 / (grid_size - 1)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx < n:
                    points[idx, 0] = j * spacing
                    points[idx, 1] = i * spacing
                    idx += 1

        # Add some randomness with reduced magnitude
        points += np.random.normal(0, 0.015, points.shape)
        points = np.clip(points, 0.01, 0.99)

        return points

    # Try multiple initialization strategies
    initial_configs = [
        initialize_fortune_grid(),      # From Voronoi-inspired approach
        initialize_hexagonal_grid(),
        initialize_spiral_pattern(),
        initialize_random(),
        initialize_golden_spiral(),
        initialize_circle_packing(),
        initialize_regular_grid()
    ]

    # Add more diverse initial configurations
    np.random.seed(123)  # Different seed for additional diversity
    for _ in range(3):  # Add 3 more random configurations
        initial_configs.append(np.random.uniform(0.1, 0.9, (16, 2)))

    best_points = None
    best_ratio = -np.inf

    # Run multiple optimization runs from each initialization (multi-start approach)
    num_runs_per_init = 3
    total_runs = len(initial_configs) * num_runs_per_init

    for init_idx, init_config in enumerate(initial_configs):
        for run in range(num_runs_per_init):
            points = init_config.copy()

            # Simulated Annealing parameters with optimized cooling schedule
            temp = 1.0
            min_temp = 1e-8
            max_iter = 30000  # Reduced iterations for faster execution

            # Track convergence for adaptive cooling
            current_ratio = compute_min_max_ratio(points)
            best_points_local = points.copy()
            best_ratio_local = current_ratio

            # Track recent improvements
            recent_improvements = []
            last_improvement = 0
            patience = 1500  # Reduced patience for faster convergence

            # Main optimization loop with optimized cooling schedule
            for iteration in range(max_iter):
                # Generate neighbor solution (random perturbation)
                neighbor_points = points.copy()
                # Pick a random point to move
                move_idx = np.random.randint(0, 16)

                # Adaptive displacement magnitude with more sophisticated scaling
                # Start with larger displacements, decrease as optimization progresses
                # Use a smoother decay curve
                base_displacement = 0.02
                temp_factor = temp
                iteration_factor = 1.0 - iteration/max_iter
                displacement_magnitude = max(0.0001, base_displacement * temp_factor * iteration_factor)
                
                displacement = np.random.normal(0, displacement_magnitude, 2)
                neighbor_points[move_idx] += displacement

                # Apply boundary constraints with epsilon padding
                epsilon = 1e-8
                neighbor_points[move_idx, 0] = np.clip(neighbor_points[move_idx, 0], 0+epsilon, 1-epsilon)
                neighbor_points[move_idx, 1] = np.clip(neighbor_points[move_idx, 1], 0+epsilon, 1-epsilon)

                # Compute ratio of neighbor
                neighbor_ratio = compute_min_max_ratio(neighbor_points)

                # Accept or reject move
                if neighbor_ratio > current_ratio:
                    # Always accept better solutions
                    points = neighbor_points
                    current_ratio = neighbor_ratio
                else:
                    # Accept worse solutions with probability based on temperature
                    delta = current_ratio - neighbor_ratio  # Note: negative is worse
                    if np.random.rand() < np.exp(-delta / temp):
                        points = neighbor_points
                        current_ratio = neighbor_ratio

                # Update best solution
                if current_ratio > best_ratio_local:
                    best_ratio_local = current_ratio
                    best_points_local = points.copy()
                    last_improvement = iteration
                    recent_improvements.append(iteration)
                    if len(recent_improvements) > 10:
                        recent_improvements.pop(0)

                # Enhanced adaptive cooling schedule
                # More aggressive initial cooling, slower later
                if iteration < 5000:
                    cooling_rate = 0.9992
                elif iteration < 15000:
                    cooling_rate = 0.9995
                else:
                    cooling_rate = 0.9997

                temp *= cooling_rate

                # Aggressive cooling if we haven't improved recently
                if iteration - last_improvement > patience // 2 and temp > min_temp:
                    temp *= 0.92

                # Early stopping based on convergence rate
                if iteration - last_improvement > patience:
                    # Check if recent improvements have stalled significantly
                    if len(recent_improvements) >= 5:
                        improvement_window = recent_improvements[-1] - recent_improvements[0]
                        if improvement_window < 500:  # Very slow improvement
                            break
                    else:
                        break

                # Stop if temperature gets too low
                if temp < min_temp:
                    break

            # Update global best if this run was better
            if best_ratio_local > best_ratio:
                best_ratio = best_ratio_local
                best_points = best_points_local.copy()

    # If we have a good solution, refine it with Voronoi-guided optimization
    if best_points is not None:
        # Apply Voronoi-guided refinement
        refined_points = optimize_with_voronoi_guidance(best_points, max_iter=100)
        refined_ratio = compute_min_max_ratio(refined_points)
        
        # Final gradient descent refinement
        try:
            x_flat = refined_points.flatten()
            
            def objective(x_flat):
                points = x_flat.reshape(-1, 2)
                # Ensure bounds
                points = np.clip(points, 0.01, 0.99)
                ratio = compute_min_max_ratio(points)
                # Minimize negative ratio (maximize ratio)
                return -ratio
            
            # Use L-BFGS-B for final refinement
            bounds = [(0.01, 0.99) for _ in range(32)]
            
            result = minimize(
                objective,
                x_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
            else:
                final_points = refined_points
                
        except:
            final_points = refined_points
    else:
        # Fallback to original best points if nothing worked
        final_points = best_points if best_points is not None else initialize_fortune_grid()

    # Ensure final results respect bounds
    final_points = np.clip(final_points, 0.01, 0.99)
    
    return final_points

# EVOLVE-BLOCK-END