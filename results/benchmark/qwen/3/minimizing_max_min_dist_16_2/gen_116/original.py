# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0.0
        return np.min(distances) / max_dist

    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        return -compute_min_max_ratio(points)

    def constraint_function(x_flat):
        """Constraint to keep points within unit square"""
        points = x_flat.reshape(-1, 2)
        # Return negative values for constraint violations (we want >= 0)
        violations = np.concatenate([
            np.minimum(points[:, 0], 0),
            np.minimum(points[:, 1], 0),
            np.maximum(points[:, 0] - 1, 0),
            np.maximum(points[:, 1] - 1, 0)
        ])
        return violations

    # Strategy 1: Enhanced hexagonal grid with proper hexagonal lattice
    def hexagonal_grid_init():
        # Create a proper hexagonal lattice arrangement
        points = []
        
        # Parameters for hexagonal lattice
        # In a hexagonal lattice, the vertical spacing is sqrt(3)/2 times the horizontal spacing
        hex_spacing = 1.0  # Normalize to unit square
        row_spacing = hex_spacing * np.sqrt(3) / 2.0
        col_spacing = hex_spacing

        # Place points in hexagonal pattern (4 rows, 4 columns)
        for row in range(4):
            for col in range(4):
                if len(points) >= 16:
                    break
                # Calculate position
                x = col * col_spacing
                # Offset odd rows
                if row % 2 == 1:
                    x += col_spacing / 2.0

                y = row * row_spacing

                points.append([x, y])

        # Convert to numpy array and adjust to fit within [0,1] bounds
        points = np.array(points[:16])

        # Normalize to fit within unit square [0,1] x [0,1]
        # Find bounding box
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        # Avoid division by zero
        if max_x > min_x and max_y > min_y:
            # Scale to fit within unit square
            scale_x = 1.0 / (max_x - min_x)
            scale_y = 1.0 / (max_y - min_y)
            scale = min(scale_x, scale_y, 1.0)
            
            points[:, 0] = (points[:, 0] - min_x) * scale
            points[:, 1] = (points[:, 1] - min_y) * scale

        # Center the points in the unit square
        center_shift = 0.5 - np.mean(points, axis=0)
        points = points + center_shift

        # Ensure points are within bounds
        points = np.clip(points, 0, 1)

        # Apply small random perturbations to break potential symmetries
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.005, points.shape)
        points += perturbations
        points = np.clip(points, 0, 1)

        return points

    # Strategy 2: Random with clustering avoidance
    def random_init():
        np.random.seed(42)
        points = np.random.rand(16, 2)
        return points

    # Strategy 3: Perturbed hexagonal grid (more structured)
    def perturbed_hexagonal_init():
        points = hexagonal_grid_init()
        np.random.seed(42)
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points

    # Strategy 4: Structured approach with adaptive perturbations
    def adaptive_hexagonal_init():
        points = hexagonal_grid_init()
        
        # Apply adaptive perturbations based on position
        center = np.mean(points, axis=0)
        distances_from_center = np.sqrt(np.sum((points - center)**2, axis=1))
        max_distance = np.max(distances_from_center)
        
        if max_distance > 0:
            normalized_distances = distances_from_center / max_distance
            # Apply stronger perturbations near center for better spread
            perturbation_magnitude = 0.01 * (1 - normalized_distances)
            
            np.random.seed(42)
            perturbations = np.random.normal(0, 0.005, points.shape)
            perturbations *= perturbation_magnitude.reshape(-1, 1)
            points += perturbations
        
        points = np.clip(points, 0, 1)
        return points

    # Strategy 5: Inspired by the differential evolution approach with better initialization
    def generate_initial_config():
        """Generate a better initial configuration for the points."""
        # Start with a more informed initial configuration
        # Using a quasi-uniform distribution pattern
        
        # Phase 1: Create a structured base pattern
        points = []
        
        # Create a 4x4 grid but make it more optimal with strategic spacing
        for i in range(4):
            for j in range(4):
                # Create slightly irregular pattern to avoid degeneracies
                x = j * 0.25 + (i % 2) * 0.125 + random.uniform(-0.01, 0.01)
                y = i * 0.25 + random.uniform(-0.01, 0.01)
                points.append([x, y])
        
        # Convert to numpy and add noise
        points = np.array(points)
        points += np.random.normal(0, 0.005, (16, 2))
        
        # Ensure all points are within [0,1] bounds
        points = np.clip(points, 0, 1)
        return points

    initial_strategies = [
        hexagonal_grid_init,
        random_init,
        perturbed_hexagonal_init,
        adaptive_hexagonal_init,
        generate_initial_config
    ]

    # Multi-start optimization with different initialization strategies
    best_ratio = -np.inf
    best_points = None

    # Try multiple random restarts with differential evolution for global search
    num_restarts = 5
    for restart in range(num_restarts):
        # Select initialization strategy
        init_func = initial_strategies[restart % len(initial_strategies)]
        points = init_func()

        # Flatten for optimization
        x0 = points.flatten()

        # Set up bounds
        bounds = [(0, 1) for _ in range(32)]

        # Use differential evolution for global search first
        try:
            # For first few restarts, use a more thorough search
            if restart < 3:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=500,
                    popsize=15,
                    tol=1e-8,
                    seed=42 + restart,
                    callback=None
                )
            else:
                # For later restarts, use less intensive search
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=200,
                    popsize=10,
                    tol=1e-6,
                    seed=42 + restart,
                    callback=None
                )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                current_ratio = compute_min_max_ratio(optimized_points)
                
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_points = optimized_points.copy()
                    
        except Exception as e:
            # Fall back to local optimization if differential evolution fails
            try:
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_function},
                    options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
                )

                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
            except Exception as e2:
                continue

    # If none worked, fallback to simple approach
    if best_points is None:
        # Fallback to hexagonal grid with refinement
        points = hexagonal_grid_init()
        x0 = points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_function},
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
            )

            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                # Final fallback to random points
                np.random.seed(42)
                best_points = np.random.rand(16, 2)
        except:
            # Final fallback
            np.random.seed(42)
            best_points = np.random.rand(16, 2)

    return best_points

# EVOLVE-BLOCK-END