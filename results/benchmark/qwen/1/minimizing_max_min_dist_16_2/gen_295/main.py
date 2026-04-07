# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
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

    def compute_energy_and_ratio(points):
        """Compute energy and ratio with bounds checking."""
        # Compute pairwise distances efficiently
        distances = squareform(pdist(points))

        # Mask diagonal elements (distance to self is 0)
        np.fill_diagonal(distances, np.inf)

        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            ratio = 0
        else:
            ratio = min_dist / max_dist

        # Energy is negative ratio (we want to maximize ratio, so minimize negative ratio)
        # Add penalty for points outside bounds with epsilon padding
        penalty = 0
        epsilon = 1e-8
        for pt in points:
            if pt[0] < 0+epsilon or pt[0] > 1-epsilon or pt[1] < 0+epsilon or pt[1] > 1-epsilon:
                penalty += 1000

        return -ratio + penalty, ratio

    def initialize_hexagonal_grid():
        """Initialize points using an enhanced hexagonal grid pattern."""
        points = []
        rows = 4
        cols = 4
        
        # Generate hexagonal grid points with alternating row offsets
        for i in range(rows):
            for j in range(cols):
                # Hexagonal grid with proper spacing
                x = (j + 0.5 * (i % 2)) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                
                # Add more substantial but controlled random perturbation
                x += (np.random.rand() - 0.5) * 0.15
                y += (np.random.rand() - 0.5) * 0.15
                
                # Ensure points stay within safe boundaries with padding
                x = np.clip(x, 0.02, 0.98)
                y = np.clip(y, 0.02, 0.98)
                
                points.append([x, y])
        
        return np.array(points[:16])

    def initialize_spiral_pattern():
        """Initialize points using a spiral pattern."""
        n = 16
        points = np.zeros((n, 2))

        # Create spiral pattern
        angles = np.linspace(0, 4*np.pi, n)
        radii = np.linspace(0.1, 0.4, n)

        for i in range(n):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

        return points

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
        radii = 0.35 + np.random.normal(0, 0.05, n)  # Slight variations in radii

        for i in range(n):
            points[i, 0] = 0.5 + radii[i] * np.cos(angles[i])
            points[i, 1] = 0.5 + radii[i] * np.sin(angles[i])

        return points

    def initialize_regular_grid():
        """Initialize points in a regular 4x4 grid."""
        n = 16
        points = np.zeros((n, 2))

        # Create regular grid
        grid_size = 4
        spacing = 1.0 / (grid_size - 1)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx < n:
                    points[idx, 0] = j * spacing
                    points[idx, 1] = i * spacing
                    idx += 1

        # Add some randomness
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0.01, 0.99)

        return points

    def initialize_known_optimal():
        """Initialize using a known good configuration inspired by optimal point arrangements."""
        # This is a manually crafted configuration that has shown good results
        # in similar point dispersion problems
        points = np.array([
            [0.5, 0.1],      # Top center
            [0.1, 0.3],      # Left middle
            [0.9, 0.3],      # Right middle
            [0.1, 0.7],      # Left bottom
            [0.9, 0.7],      # Right bottom
            [0.5, 0.9],      # Bottom center
            [0.3, 0.2],      # Upper left
            [0.7, 0.2],      # Upper right
            [0.2, 0.5],      # Middle left
            [0.8, 0.5],      # Middle right
            [0.3, 0.8],      # Lower left
            [0.7, 0.8],      # Lower right
            [0.15, 0.15],    # Corner
            [0.85, 0.15],    # Corner
            [0.15, 0.85],    # Corner
            [0.85, 0.85]     # Corner
        ])

        # Add small random perturbation
        points += np.random.normal(0, 0.01, points.shape)

        # Clamp to bounds
        points = np.clip(points, 0.01, 0.99)

        return points

    def optimize_with_voronoi_guidance(points, max_iter=100):
        """Optimize points using Voronoi-based guidance for better distribution."""
        current_points = points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Precompute bounds for clipping
        bounds_min = 0.01
        bounds_max = 0.99
        
        for iteration in range(max_iter):
            try:
                # Analyze current Voronoi diagram
                vor = Voronoi(current_points)
                n_points = len(current_points)
                
                # Gradient-based updates guided by Voronoi geometry
                new_points = current_points.copy()
                step_size = 0.001 * (1.0 - iteration/max_iter)  # Decreasing step size
                
                # For each point, move towards more optimal position based on Voronoi analysis
                for i in range(n_points):
                    # Find neighboring points that influence this point's Voronoi cell
                    distances_to_others = [np.linalg.norm(current_points[i] - current_points[j]) 
                                         for j in range(n_points) if i != j]
                    sorted_indices = np.argsort(distances_to_others)
                    
                    # Move point away from very close neighbors and towards far ones
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

    def local_refinement(points, max_iter=200):
        """Apply local optimization to refine the solution."""
        def objective(x_flat):
            points = x_flat.reshape(-1, 2)
            # Ensure points are within bounds
            points = np.clip(points, 0.01, 0.99)
            _, ratio = compute_energy_and_ratio(points)
            return -ratio  # Minimize negative ratio (maximize ratio)

        # Use L-BFGS-B for local refinement
        bounds = [(0.01, 0.99) for _ in range(32)]

        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
        return points

    def multi_stage_optimization(initial_points):
        """Multi-stage optimization approach combining global and local methods."""
        # Stage 1: Global optimization with Differential Evolution
        def de_objective(x_flat):
            points = x_flat.reshape(-1, 2)
            points = np.clip(points, 0.01, 0.99)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Minimize negative ratio

        bounds = [(0.01, 0.99) for _ in range(32)]
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=100,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False
            )
            de_points = de_result.x.reshape(-1, 2)
        except:
            de_points = initial_points.copy()
        
        # Stage 2: Local refinement with L-BFGS-B
        lbfgs_points = local_refinement(de_points, max_iter=150)
        
        # Stage 3: Voronoi-guided optimization for fine-tuning
        voronoi_points = optimize_with_voronoi_guidance(lbfgs_points, max_iter=50)
        
        # Return the best among all stages
        ratios = [
            compute_min_max_ratio(initial_points),
            compute_min_max_ratio(de_points),
            compute_min_max_ratio(lbfgs_points),
            compute_min_max_ratio(voronoi_points)
        ]
        
        best_idx = np.argmax(ratios)
        if best_idx == 0:
            return initial_points
        elif best_idx == 1:
            return de_points
        elif best_idx == 2:
            return lbfgs_points
        else:
            return voronoi_points

    # Try multiple initialization strategies
    initial_configs = [
        initialize_hexagonal_grid(),
        initialize_spiral_pattern(),
        initialize_golden_spiral(),
        initialize_circle_packing(),
        initialize_regular_grid(),
        initialize_known_optimal()
    ]

    best_points = None
    best_ratio = -np.inf

    # Run optimization from each initialization with multi-stage approach
    for init_config in initial_configs:
        try:
            # Apply multi-stage optimization
            optimized_points = multi_stage_optimization(init_config)
            
            # Final evaluation and comparison
            final_ratio = compute_min_max_ratio(optimized_points)
            
            # Update global best if this run was better
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = optimized_points.copy()
        except Exception as e:
            continue

    # If no successful optimization occurred, return the best initialization
    if best_points is None:
        best_points = initialize_hexagonal_grid()

    # Final validation and ensure bounds
    best_points = np.clip(best_points, 0.01, 0.99)
    
    return best_points

# EVOLVE-BLOCK-END