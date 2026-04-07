# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def objective(x):
        # Reshape x into points
        points = x.reshape(-1, 2)

        # Compute pairwise distances using squareform for better numerical stability
        distances = squareform(pdist(points))

        # Zero out diagonal elements (distance to self)
        np.fill_diagonal(distances, np.inf)

        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio to maximize (since we're minimizing the negative)
        if d_max == 0:
            return -1.0
        return -d_min / d_max

    def adaptive_objective(x):
        """
        Adaptive objective function with enhanced penalties for better convergence
        """
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -1.0
        
        # Calculate ratio to maximize
        ratio = d_min / d_max
        
        # Enhance penalty for boundary constraints
        boundary_penalty = 0.0
        margin = 0.01
        for point in points:
            if (point[0] < margin or point[0] > 1-margin or 
                point[1] < margin or point[1] > 1-margin):
                boundary_penalty += 1000.0 * (margin - min(point[0], 1-point[0], point[1], 1-point[1]))
        
        # Penalty for very small distances (clustering)
        min_distance_penalty = 0.0
        if d_min < 0.03:  # Tighter threshold for clustering penalty
            min_distance_penalty = 500.0 * (0.03 - d_min)
        
        # Penalty for extreme distance variance
        if len(distances[distances != np.inf]) > 0:
            valid_distances = distances[distances != np.inf]
            dist_mean = np.mean(valid_distances)
            dist_std = np.std(valid_distances)
            
            # Penalize high variance more heavily
            variance_penalty = 100.0 * (dist_std / (dist_mean + 1e-10))
            total_penalty = boundary_penalty + min_distance_penalty + variance_penalty
        else:
            total_penalty = boundary_penalty + min_distance_penalty
            
        return -(ratio - total_penalty / 1000.0)

    def generate_hexagonal_initialization():
        """Generate initial points using hexagonal grid pattern with optimized spacing"""
        np.random.seed(42)
        
        points = []
        rows, cols = 4, 4
        
        # Create hexagonal pattern with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                
                # Add slight randomization to avoid perfect symmetry
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                
                points.append([x, y])
        
        points = np.array(points[:16])
        # Clip to safe range
        points = np.clip(points, 0.05, 0.95)
        return points

    def generate_golden_spiral_initialization():
        """Generate initial points using golden spiral for even distribution"""
        np.random.seed(123)
        
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for i in range(16):
            angle = i * 2.4  # Modified golden angle for better distribution
            radius = 0.4 * np.sqrt(i / 15.0) if i > 0 else 0.05
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Add noise for diversity
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            points.append([x, y])
        
        points = np.array(points)
        # Clip to safe range
        points = np.clip(points, 0.05, 0.95)
        return points

    def generate_circle_initialization():
        """Generate initial points arranged in circles with radial variation"""
        np.random.seed(234)
        
        points = []
        # Create points in concentric circles
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radii = 0.4 + 0.1 * np.sin(np.arange(16) * np.pi / 8)
        center = np.array([0.5, 0.5])
        
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([x, y])
        
        # Add noise for diversity
        points = np.array(points) + np.random.normal(0, 0.01, (16, 2))
        # Clip to safe range
        points = np.clip(points, 0.05, 0.95)
        return points

    def generate_corner_initialization():
        """Generate initial points using corner-based positioning"""
        np.random.seed(345)
        
        # Start with corner positions
        corners = np.array([
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
        ])
        
        # Fill remaining positions with random points near center
        remaining = 16 - len(corners)
        center_points = np.random.rand(remaining, 2) * 0.6 + 0.2  # Center region
        
        points = np.vstack([corners, center_points])
        # Add noise for diversity
        points += np.random.normal(0, 0.01, points.shape)
        # Clip to safe range
        points = np.clip(points, 0.05, 0.95)
        return points

    def generate_random_initialization():
        """Generate random initial configuration"""
        np.random.seed(456)
        points = np.random.rand(16, 2) * 0.9 + 0.05  # Range [0.05, 0.95]
        return points

    # Generate multiple diversified initial configurations
    initial_configs = [
        generate_hexagonal_initialization(),
        generate_golden_spiral_initialization(),
        generate_circle_initialization(),
        generate_corner_initialization(),
        generate_random_initialization()
    ]

    # Define bounds with stricter padding
    bounds = [(0.05, 0.95) for _ in range(32)]  # 16 points * 2 coordinates each

    best_result = None
    best_value = float('inf')

    # Try multiple initial configurations with hybrid optimization
    for i, init_config in enumerate(initial_configs):
        try:
            x0 = init_config.flatten()
            
            # First stage: Use differential evolution for global optimization with better parameters
            de_result = differential_evolution(
                adaptive_objective,
                bounds,
                seed=42+i,
                maxiter=150,   # Increased iterations for better exploration
                popsize=30,    # Increased population size for better diversity
                tol=1e-9,      # Tighter tolerance
                recombination=0.9,
                mutation=(0.8, 1.0),
                disp=False
            )

            # Second stage: Local refinement with multiple strategies
            local_results = []
            
            # Try L-BFGS-B first
            lbfgs_result = minimize(
                adaptive_objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14},
                callback=None
            )
            local_results.append(lbfgs_result)
            
            # Try SLSQP as fallback if L-BFGS-B doesn't work well
            if not lbfgs_result.success or lbfgs_result.fun > -0.1:  # If not very good result
                slsqp_result = minimize(
                    adaptive_objective,
                    de_result.x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
                    callback=None
                )
                local_results.append(slsqp_result)
            
            # Choose the best local result
            best_local = min(local_results, key=lambda r: r.fun if r.success else float('inf'))
            
            # Keep track of the best result
            if best_local.success and best_local.fun < best_value:
                best_value = best_local.fun
                best_result = best_local

        except Exception as e:
            warnings.warn(f"Error in optimization attempt {i}: {e}")
            continue

    # If we found a valid result, return it; otherwise use the first configuration
    if best_result is not None:
        points = best_result.x.reshape(-1, 2)
    else:
        points = initial_configs[0].reshape(-1, 2)

    # Final refinement with standard objective
    try:
        final_result = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
            callback=None
        )
        points = final_result.x.reshape(-1, 2)
    except Exception as e:
        warnings.warn(f"Final refinement failed: {e}")
        pass

    # Ensure final points are within bounds
    points = np.clip(points, 0.05, 0.95)

    return points

# EVOLVE-BLOCK-END