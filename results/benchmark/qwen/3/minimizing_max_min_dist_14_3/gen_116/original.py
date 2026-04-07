# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)

    def penalty_objective(x, penalty_weight=1e6):
        """Objective with penalty for boundary violations"""
        points = x.reshape(-1, 3)
        
        # Apply penalty for points outside bounds
        penalty = 0
        for i in range(14):
            for j in range(3):
                if points[i,j] < 0:
                    penalty += penalty_weight * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += penalty_weight * (points[i,j] - 1)**2
        
        # Original objective
        original_obj = objective(x)
        
        return original_obj + penalty

    def initialize_spherical_points(n_points):
        """Initialize points on a unit sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n_points):
            # Latitude
            phi = np.arccos(1 - 2*i/(n_points-1))
            # Longitude
            theta = 2 * np.pi * i / golden_ratio

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)

    def initialize_cube_grid_points(n_points):
        """Initialize points in a 3D cube grid"""
        # Find appropriate grid size
        grid_size = int(np.ceil(n_points**(1/3)))
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(grid_points) < n_points:
                        grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:n_points])

    def evaluate_initialization(points):
        """Fast evaluation of initialization quality"""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max > 1e-12:
            return d_min / d_max
        return 0

    def adaptive_differential_evolution(objective_func, bounds, initial_popsize=20, maxiter=300):
        """Enhanced differential evolution with adaptive population sizing and early stopping"""
        current_popsize = initial_popsize
        prev_best = -np.inf
        stagnation_count = 0
        improvement_threshold = 1e-8
        min_improvement = 1e-12
        
        # Track improvement for early stopping
        recent_improvements = []
        
        for iteration in range(maxiter // 10):  # Reduced iterations per batch
            # Adjust population size based on convergence
            if stagnation_count > 3 and current_popsize < 30:
                current_popsize = min(current_popsize + 5, 30)
            
            # Run differential evolution with current parameters
            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=42 + iteration,
                    maxiter=10,  # Fewer iterations per batch
                    popsize=current_popsize,
                    tol=1e-12,
                    mutation=(0.5, 1.0),  # Adaptive mutation range
                    recombination=0.9,    # Higher recombination for better exploration
                    disp=False
                )
            except:
                # Fall back to smaller population if needed
                try:
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=42 + iteration,
                        maxiter=10,
                        popsize=max(5, current_popsize - 5),
                        tol=1e-12,
                        mutation=(0.5, 1.0),
                        recombination=0.9,
                        disp=False
                    )
                except:
                    # Last resort - use basic differential evolution
                    result = differential_evolution(
                        objective_func,
                        bounds,
                        seed=42 + iteration,
                        maxiter=10,
                        popsize=10,
                        tol=1e-12,
                        mutation=(0.5, 1.0),
                        recombination=0.7,
                        disp=False
                    )
            
            # Check for improvement
            current_best = -result.fun
            improvement = current_best - prev_best
            
            recent_improvements.append(improvement)
            if len(recent_improvements) > 5:
                recent_improvements.pop(0)
            
            # Early stopping if improvement is minimal
            if len(recent_improvements) == 5 and all(abs(impr) < min_improvement for impr in recent_improvements):
                break
                
            if improvement > improvement_threshold:
                stagnation_count = 0
            else:
                stagnation_count += 1
                
            prev_best = current_best
            
        return result

    # Try multiple initialization strategies
    strategies = []

    # Strategy 1: Spherical Fibonacci points
    fib_points = initialize_spherical_points(14)
    fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("fibonacci", fib_points))

    # Strategy 2: Cube grid points
    cube_points = initialize_cube_grid_points(14)
    strategies.append(("cube_grid", cube_points))

    # Strategy 3: Random points
    np.random.seed(42)
    random_points = np.random.rand(14, 3)
    strategies.append(("random", random_points))

    # Strategy 4: Perturbed spherical points
    np.random.seed(42)
    perturbed_points = fib_points + np.random.normal(0, 0.03, (14, 3))
    perturbed_points = np.clip(perturbed_points, 0, 1)
    strategies.append(("perturbed", perturbed_points))

    # Evaluate all strategies and select the best
    best_initialization = None
    best_ratio = -np.inf

    for name, points in strategies:
        ratio = evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()

    # Use the best initialization as starting point
    x0 = best_initialization.flatten()

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Run adaptive differential evolution optimization
    best_result = None
    best_ratio = -np.inf

    # Try 3 different random seeds for better exploration
    for seed_val in [42, 123, 456]:
        np.random.seed(seed_val)
        
        # Use adaptive differential evolution
        result = adaptive_differential_evolution(
            penalty_objective, 
            bounds, 
            initial_popsize=20, 
            maxiter=200
        )
        
        # Check if this result is better
        if -result.fun > best_ratio:
            best_ratio = -result.fun
            best_result = result

    # Extract optimized points
    optimized_points = best_result.x.reshape(-1, 3)

    # Apply local refinement with L-BFGS-B for final tuning
    def lbfgs_refinement(points):
        """Apply local refinement using L-BFGS-B"""
        x0_refine = points.flatten()
        
        # Objective with stricter tolerance
        def obj_for_lbfgs(x):
            points_refined = x.reshape(-1, 3)
            distances = pdist(points_refined)
            
            if len(distances) == 0:
                return -np.inf
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 1e-12:
                return -(d_min / d_max)
            else:
                return -np.inf
                
        # Refine with L-BFGS-B
        try:
            result_refine = minimize(
                obj_for_lbfgs,
                x0_refine,
                method='L-BFGS-B',
                bounds=[(0, 1)] * 42,
                options={'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            refined_points = result_refine.x.reshape(-1, 3)
            # Ensure bounds are respected
            refined_points = np.clip(refined_points, 0, 1)
            return refined_points
        except:
            # If L-BFGS fails, return original
            return points
    
    # Apply refinement
    optimized_points = lbfgs_refinement(optimized_points)

    # Final clipping to ensure bounds are respected
    optimized_points = np.clip(optimized_points, 0, 1)

    return optimized_points

# EVOLVE-BLOCK-END