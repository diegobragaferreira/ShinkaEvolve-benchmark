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

    def voronoi_penalty_objective(x):
        """
        Objective function incorporating Voronoi-like geometric constraints
        """
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -1.0
        
        # Base ratio to maximize
        ratio = d_min / d_max
        
        # Voronoi-inspired penalty based on distance variance
        # Penalize configurations where points cluster too closely or spread too far
        if len(distances[distances != np.inf]) > 0:
            valid_distances = distances[distances != np.inf]
            dist_mean = np.mean(valid_distances)
            dist_std = np.std(valid_distances)
            
            # Penalty for high variance (uneven spacing)
            variance_penalty = dist_std / (dist_mean + 1e-10)
            
            # Penalty for very small distances (clustering)
            small_dist_penalty = 0.0
            if d_min < 0.05:
                small_dist_penalty = (0.05 - d_min) * 1000.0
            
            # Combine penalties
            total_penalty = variance_penalty * 100.0 + small_dist_penalty
            return -(ratio - total_penalty / 1000.0)
        else:
            return -ratio

    def generate_voronoi_initialization():
        """Generate initial points using Voronoi-inspired geometric construction"""
        np.random.seed(42)
        
        # Start with a basic grid pattern and apply Voronoi-style adjustments
        points = []
        # Create a 4x4 grid-like structure
        for i in range(4):
            for j in range(4):
                if len(points) >= 16:
                    break
                x = j * 0.25 + 0.125
                y = i * 0.25 + 0.125
                
                # Add strategic perturbations based on Voronoi cell concepts
                if i % 2 == 1:
                    x += 0.05
                if j % 2 == 1:
                    y += 0.05
                    
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Apply Voronoi-inspired adjustment to make spacing more uniform
        for _ in range(20):  # Iterative adjustment
            # Add some randomness to encourage diversity
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
            
            # Constrain to valid region
            points = np.clip(points, 0.05, 0.95)
            
            # Adjust positions based on distance constraints
            if len(points) >= 2:
                # For each point, move away from neighbors that are too close
                for k in range(len(points)):
                    dist_to_others = []
                    for l in range(len(points)):
                        if k != l:
                            dist = np.linalg.norm(points[k] - points[l])
                            dist_to_others.append(dist)
                    
                    if len(dist_to_others) > 0:
                        avg_dist = np.mean(dist_to_others)
                        min_dist = np.min(dist_to_others)
                        
                        # If too clustered, push away
                        if min_dist < 0.15:
                            # Move away from nearest neighbor
                            nearest_idx = np.argmin(dist_to_others)
                            dx = points[k][0] - points[nearest_idx][0]
                            dy = points[k][1] - points[nearest_idx][1]
                            length = np.sqrt(dx*dx + dy*dy)
                            if length > 1e-8:
                                scale = 0.01 * (0.15 - min_dist) / length
                                points[k][0] += dx * scale
                                points[k][1] += dy * scale
        
        return points

    def generate_symmetric_initialization():
        """Generate symmetric initial configuration with geometric properties"""
        np.random.seed(123)
        
        # Create a more sophisticated initial layout based on symmetry principles
        points = []
        
        # Center point
        points.append([0.5, 0.5])
        
        # Points around the perimeter in a circular pattern
        angles = np.linspace(0, 2*np.pi, 15, endpoint=False)
        radius = 0.4
        
        for angle in angles:
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add some Voronoi-like perturbations
        for i in range(len(points)):
            # Add small perturbations to avoid perfect symmetry
            points[i] += np.random.normal(0, 0.02, 2)
        
        # Clip to valid range
        points = np.clip(points, 0.05, 0.95)
        
        return points[:16]

    def generate_optimized_grid():
        """Generate an optimized grid-like pattern with irregular spacing"""
        np.random.seed(234)
        
        points = []
        
        # Create a hexagonal-like grid with irregularities
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Hexagonal offset pattern
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                
                # Add irregularity based on Voronoi concepts
                irregularity_factor = 0.15
                x += np.random.uniform(-irregularity_factor, irregularity_factor) * 0.25
                y += np.random.uniform(-irregularity_factor, irregularity_factor) * 0.25
                
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize and clip
        points = np.clip(points, 0.05, 0.95)
        
        return points

    # Generate multiple diversified initial configurations
    initial_configs = [
        generate_voronoi_initialization(),
        generate_symmetric_initialization(),
        generate_optimized_grid()
    ]
    
    # Add random initialization for diversity
    np.random.seed(345)
    random_points = np.random.rand(16, 2) * 0.9 + 0.05  # [0.05, 0.95]
    initial_configs.append(random_points)
    
    # Add another Voronoi-inspired configuration
    np.random.seed(456)
    # Spiral pattern with geometric correction
    points_spiral = []
    for i in range(16):
        angle = i * 0.5
        radius = 0.4 * (i / 15.0) + 0.05
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        points_spiral.append([x, y])
    
    points_spiral = np.array(points_spiral)
    # Add some Voronoi-like adjustments
    for _ in range(10):
        noise = np.random.normal(0, 0.01, points_spiral.shape)
        points_spiral += noise
        points_spiral = np.clip(points_spiral, 0.05, 0.95)
    
    initial_configs.append(points_spiral)

    # Define bounds with stricter padding
    bounds = [(0.05, 0.95) for _ in range(32)]  # 16 points * 2 coordinates each

    best_result = None
    best_value = float('inf')

    # Try multiple initial configurations with hybrid optimization
    for i, init_config in enumerate(initial_configs):
        try:
            x0 = init_config.flatten()
            
            # First stage: Use differential evolution for global optimization
            de_result = differential_evolution(
                voronoi_penalty_objective,
                bounds,
                seed=42+i,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                recombination=0.8,
                mutation=(0.7, 1.0),
                disp=False
            )

            # Second stage: Local refinement with L-BFGS-B
            lbfgs_result = minimize(
                voronoi_penalty_objective,
                de_result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                callback=None
            )

            # Keep track of the best result
            if lbfgs_result.fun < best_value:
                best_value = lbfgs_result.fun
                best_result = lbfgs_result

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
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10},
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