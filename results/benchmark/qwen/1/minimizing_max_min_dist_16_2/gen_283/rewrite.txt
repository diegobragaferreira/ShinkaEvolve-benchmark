# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
import warnings


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
        
        distances = pdist(points)
        distances = distances[distances > 1e-12]  # Filter out near-zero distances
        
        if len(distances) == 0:
            return 0
            
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        if dmax == 0:
            return 0
            
        return dmin / dmax

    def compute_voronoi_uniformity(points):
        """Compute a measure of Voronoi cell uniformity to encourage even distribution."""
        try:
            vor = Voronoi(points)
            areas = []
            
            # Calculate area for each Voronoi cell (excluding infinite regions)
            for i, region in enumerate(vor.regions):
                if len(region) > 0 and -1 not in region:
                    # Calculate polygon area using shoelace formula
                    vertices = vor.vertices[region]
                    if len(vertices) >= 3:
                        x_vals = vertices[:, 0]
                        y_vals = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x_vals, np.roll(y_vals, 1)) - np.dot(y_vals, np.roll(x_vals, 1)))
                        areas.append(area)
            
            if len(areas) == 0:
                return 0
                
            # Lower variance in cell areas indicates better uniformity
            return 1.0 / (1.0 + np.var(areas))
        except:
            return 0

    def combined_objective(x_flat):
        """Combined objective function balancing distance ratio and uniformity."""
        points = x_flat.reshape(-1, 2)
        
        # Ensure bounds are respected
        points = np.clip(points, 1e-8, 1-1e-8)
        
        ratio = compute_min_max_ratio(points)
        uniformity = compute_voronoi_uniformity(points)
        
        # Combined objective: maximize ratio while promoting uniformity
        # Weight ratio higher (0.8) to prioritize distance spread
        combined = ratio * 0.8 + uniformity * 0.2
        
        # Return negative for minimization
        return -combined

    def constraint_func(x):
        # Ensure points are within [0+eps,1-eps] x [0+eps,1-eps] for numerical stability
        eps = 1e-8
        points = x.reshape(-1, 2)
        constraints = []

        # x coordinates in [eps, 1-eps]
        constraints.append(points[:, 0].min() - eps)  # x_min >= eps
        constraints.append(1 - eps - points[:, 0].max())  # x_max <= 1-eps

        # y coordinates in [eps, 1-eps]
        constraints.append(points[:, 1].min() - eps)  # y_min >= eps
        constraints.append(1 - eps - points[:, 1].max())  # y_max <= 1-eps

        return np.array(constraints)

    def bounded_objective(x):
        # Boundary checking with clamping to safe bounds
        eps = 1e-8
        points = np.clip(x.reshape(-1, 2), eps, 1-eps).flatten()
        return combined_objective(points)

    def generate_hexagonal_initial_config():
        """Generate a sophisticated hexagonal initial configuration"""
        np.random.seed(42)
        points = []

        # Create a precise hexagonal lattice pattern optimized for 16 points
        sqrt3 = np.sqrt(3)

        # Create hexagonal pattern with better spacing
        row_height = 0.8 * sqrt3 / 2  
        col_width = 0.8  

        for i in range(4):
            for j in range(4):
                # Hexagonal offset pattern
                x = 0.1 + j * 0.8 / 3.0
                y = 0.1 + i * row_height / 3.0

                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += 0.8 / 6.0

                points.append([x, y])

        # Convert to numpy array and add small random jitter to break symmetry
        points = np.array(points[:16])
        points += np.random.normal(0, 0.005, points.shape) 

        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)

        return points

    def generate_spiral_initial_config():
        """Generate a spiral initial configuration"""
        np.random.seed(42)
        points = []

        # Golden spiral pattern
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(16):
            if i == 0:
                points.append([0.5, 0.5])  
            else:
                angle = i * 2.5  # Angle in radians
                radius = min(0.4, i * 0.05)  
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])

        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points

    def generate_grid_initial_config():
        """Generate a regular grid initial configuration"""
        np.random.seed(42)
        points = []

        # Regular 4x4 grid
        x_vals = np.linspace(0.1, 0.9, 4)
        y_vals = np.linspace(0.1, 0.9, 4)

        for i in range(4):
            for j in range(4):
                points.append([x_vals[i], y_vals[j]])

        points = np.array(points[:16])
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    def generate_random_initial_config():
        """Generate a random initial configuration"""
        np.random.seed(42)
        return np.random.rand(16, 2)

    def generate_corner_initial_config():
        """Generate an initial configuration with points at corners and center"""
        np.random.seed(42)
        points = np.array([
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5],
            [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75],
            [0.33, 0.33], [0.33, 0.67], [0.67, 0.33], [0.67, 0.67]
        ])
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points

    # Generate multiple diverse initial configurations
    initial_configs = [
        generate_hexagonal_initial_config(),
        generate_spiral_initial_config(),
        generate_grid_initial_config(),
        generate_random_initial_config(),
        generate_corner_initial_config()
    ]

    best_ratio = float('-inf')
    best_points = None

    # Try multiple initial configurations with advanced hybrid approach
    for i, initial_config in enumerate(initial_configs):
        try:
            # Flatten for optimization
            x0 = initial_config.flatten()

            # Define bounds for each coordinate (safe bounds to avoid numerical issues)
            bounds = [(1e-8, 1-1e-8) for _ in range(32)]

            # Phase 1: Global optimization with Differential Evolution (enhanced)
            de_result = differential_evolution(
                bounded_objective,
                bounds,
                seed=42+i,
                maxiter=200,  # Increased iterations
                popsize=30,   # Larger population for better exploration
                tol=1e-8,     # Tighter tolerance
                mutation=(0.5, 1.0),
                recombination=0.8,  # Higher recombination rate
                disp=False
            )

            # If DE finds a reasonable solution, refine it locally
            if de_result.success and -de_result.fun > 0.1:
                x0 = de_result.x

            # Phase 2: Local optimization with adaptive iteration limits
            # Use SLSQP first for better constraint handling
            result = minimize(
                bounded_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-10},  # Tighter tolerances
                callback=None
            )

            # If SLSQP fails, try L-BFGS-B as fallback
            if not result.success:
                fallback_result = minimize(
                    bounded_objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                result = fallback_result

            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                # Calculate actual ratio for this configuration
                ratio = compute_min_max_ratio(optimized_points)

                if ratio > best_ratio:  # We want to maximize ratio
                    best_ratio = ratio
                    best_points = optimized_points.copy()

        except Exception as e:
            continue

    # If we still don't have a good solution, use a fallback approach
    if best_points is None:
        # Try one more time with a strong optimization approach
        fallback_config = generate_hexagonal_initial_config()
        x0 = fallback_config.flatten()
        bounds = [(1e-8, 1-1e-8) for _ in range(32)]

        # More aggressive optimization with careful parameter tuning
        try:
            result = minimize(
                bounded_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
            )

            if result.success:
                best_points = result.x.reshape(-1, 2)
            else:
                # Final fallback - just return the best initial configuration
                best_points = generate_hexagonal_initial_config()
        except Exception:
            # If everything fails, return the hexagonal initial config
            best_points = generate_hexagonal_initial_config()

    # Final safety check - ensure points are within bounds
    if best_points is not None:
        best_points = np.clip(best_points, 1e-8, 1-1e-8)
    else:
        # Last resort: return a default configuration
        best_points = generate_hexagonal_initial_config()

    return best_points


# EVOLVE-BLOCK-END