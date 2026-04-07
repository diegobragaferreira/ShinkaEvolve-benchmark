# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distances between all point pairs."""
        if len(points) < 2:
            return 0

        # Calculate pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)

        if dmax == 0:
            return 0

        return dmin / dmax

    def objective_function(points):
        """Objective function to maximize (negative because we minimize in scipy)."""
        return -calculate_min_max_ratio(points)

    def create_hexagonal_initialization():
        """Create a hexagonal-like arrangement of points."""
        points = np.zeros((16, 2))
        
        # Create a roughly hexagonal arrangement with better spacing
        rows = 4
        cols = 4
        spacing = 1.0 / (rows + 1)

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < 16:
                    # Offset every other row for better hexagonal packing
                    x = (j + 0.5 * (i % 2)) * spacing
                    y = i * spacing
                    points[idx] = [x, y]
                    idx += 1
        
        return points

    def create_perturbed_initialization(base_points, perturbation_magnitude=0.01):
        """Create a perturbed version of base initialization."""
        perturbed = base_points.copy()
        # Add random perturbation
        perturbed += np.random.normal(0, perturbation_magnitude, base_points.shape)
        # Ensure points stay within bounds
        perturbed = np.clip(perturbed, 0, 1)
        return perturbed

    def optimize_with_differential_evolution(initial_points):
        """Perform global optimization using differential evolution."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]
        
        # Use differential evolution for global search with tuned parameters
        de_result = differential_evolution(
            lambda x: objective_function(x.reshape(-1, 2)),
            bounds,
            seed=42,
            maxiter=500,  # Reduced iterations for faster execution
            popsize=20,   # Increased population size for better exploration
            mutation=(0.5, 1),
            recombination=0.7,
            tol=1e-8,     # Tighter tolerance
            disp=False
        )
        
        # Extract optimized points
        optimized_points = de_result.x.reshape(-1, 2)
        
        # Ensure all points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)
        
        return optimized_points

    def single_point_optimization(initial_points):
        """Perform local optimization refinement on initial configuration."""
        # Flatten for optimization
        initial_flat = initial_points.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(len(initial_flat))]
        
        # Optimize using L-BFGS-B method with tighter tolerances
        from scipy.optimize import minimize
        
        result = minimize(
            lambda flat_points: objective_function(flat_points.reshape(-1, 2)),
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10},
            callback=None
        )
        
        # Extract optimized points
        optimized_points = result.x.reshape(-1, 2)
        
        # Ensure all points are within bounds
        optimized_points = np.clip(optimized_points, 0, 1)
        
        return optimized_points

    # Main optimization routine
    np.random.seed(42)
    
    # Create multiple initial configurations and select the best one
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Hexagonal initialization with perturbation
    hex_initial = create_hexagonal_initialization()
    hex_perturbed = create_perturbed_initialization(hex_initial, 0.01)
    
    # Apply differential evolution to hexagonal configuration
    try:
        de_result = optimize_with_differential_evolution(hex_perturbed)
        ratio = calculate_min_max_ratio(de_result)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = de_result
    except Exception:
        pass
    
    # Strategy 2: Slightly different hexagonal pattern
    try:
        # Create a more spread out hexagonal pattern
        hex_initial_v2 = create_hexagonal_initialization()
        # Scale and shift slightly for variety
        hex_initial_v2 = hex_initial_v2 * 0.9 + 0.05
        hex_perturbed_v2 = create_perturbed_initialization(hex_initial_v2, 0.015)
        de_result_v2 = optimize_with_differential_evolution(hex_perturbed_v2)
        ratio_v2 = calculate_min_max_ratio(de_result_v2)
        if ratio_v2 > best_ratio:
            best_ratio = ratio_v2
            best_points = de_result_v2
    except Exception:
        pass
    
    # Strategy 3: Random initialization with hex pattern influence
    try:
        random_initial = np.random.rand(16, 2)
        # Add some hexagonal influence to the random points
        hex_base = create_hexagonal_initialization()
        random_initial = random_initial * 0.5 + hex_base * 0.5
        random_initial = np.clip(random_initial, 0, 1)
        de_result_v3 = optimize_with_differential_evolution(random_initial)
        ratio_v3 = calculate_min_max_ratio(de_result_v3)
        if ratio_v3 > best_ratio:
            best_ratio = ratio_v3
            best_points = de_result_v3
    except Exception:
        pass
    
    # If no good results from DE, fallback to local optimization with hexagonal
    if best_points is None:
        try:
            hex_initial = create_hexagonal_initialization()
            perturbed_hex = create_perturbed_initialization(hex_initial, 0.005)
            best_points = single_point_optimization(perturbed_hex)
        except Exception:
            # Final fallback to simple hexagonal grid
            best_points = create_hexagonal_initialization()
            # Add very small perturbation
            best_points += np.random.normal(0, 0.001, best_points.shape)
            best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END