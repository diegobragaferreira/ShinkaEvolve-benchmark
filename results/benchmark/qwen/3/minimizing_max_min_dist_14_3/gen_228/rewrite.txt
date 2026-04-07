# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import time

# Core functional modules

def initialize_fibonacci_points(n_points: int = 14) -> np.ndarray:
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

def initialize_cube_grid_points(n_points: int = 14) -> np.ndarray:
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

def initialize_random_points(n_points: int = 14) -> np.ndarray:
    """Initialize random points in 3D space"""
    np.random.seed(42)
    return np.random.rand(n_points, 3)

def initialize_voronoi_uniform_points(n_points: int = 14) -> np.ndarray:
    """Initialize points with good Voronoi uniformity using iterative approach"""
    # Start with random points
    points = np.random.rand(n_points, 3)

    # Simple iterative improvement: move points to increase minimum distance
    for _ in range(50):  # Limited iterations to avoid excessive computation
        distances = pdist(points)
        if len(distances) > 0:
            # Move each point away from its nearest neighbor
            for i in range(n_points):
                # Find closest neighbor
                dist_row = distances[i*(n_points-1):(i+1)*(n_points-1)]
                if len(dist_row) > 0:
                    closest_idx = np.argmin(dist_row)
                    # Move point away from neighbor
                    if closest_idx < i:
                        neighbor = points[closest_idx]
                    else:
                        neighbor = points[closest_idx + 1]

                    direction = points[i] - neighbor
                    norm_dir = np.linalg.norm(direction)
                    if norm_dir > 1e-12:
                        points[i] += 0.01 * direction / norm_dir
            # Keep within bounds
            points = np.clip(points, 0, 1)

    return points

def initialize_spherical_cap_points(n_points: int = 14) -> np.ndarray:
    """Initialize points on a spherical cap for better distribution"""
    points = []
    # Generate points along a spherical cap
    for i in range(n_points):
        # Distribute points more evenly across the surface
        phi = np.arccos(1 - 2 * (i / (n_points - 1)))
        theta = 2 * np.pi * i * (1 + np.sqrt(5)) / 2
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append([x, y, z])
    return np.array(points)

def initialize_golden_spiral_points(n_points: int = 14) -> np.ndarray:
    """Initialize points using golden spiral method for better uniformity"""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # Golden angle
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def initialize_hybrid_points(n_points: int = 14) -> np.ndarray:
    """Initialize hybrid points combining multiple strategies"""
    # Start with Fibonacci points
    fib_points = initialize_fibonacci_points(n_points)
    fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
    
    # Apply slight perturbations to create hybrid
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, fib_points.shape)
    perturbed = fib_points + noise
    perturbed = np.clip(perturbed, 0, 1)
    
    # Apply KMeans clustering for better distribution
    try:
        kmeans = KMeans(n_clusters=n_points, random_state=42, n_init=5)
        kmeans.fit(perturbed)
        kmeans_points = kmeans.cluster_centers_
        return kmeans_points
    except:
        return perturbed

def project_to_unit_cube(points: np.ndarray) -> np.ndarray:
    """Project points to [0,1]^3 bounds"""
    return np.clip(points, 0, 1)

def compute_distance_ratio(points: np.ndarray) -> float:
    """Compute the minimum/maximum distance ratio"""
    if len(points) < 2:
        return 0

    distances = pdist(points)
    if len(distances) == 0:
        return 0

    d_min = np.min(distances)
    d_max = np.max(distances)

    # Avoid division by zero
    if d_max <= 1e-12:
        return 0

    return d_min / d_max

def evaluate_initialization(points: np.ndarray) -> float:
    """Fast evaluation of initialization quality with uniformity consideration"""
    # Primary metric: distance ratio
    ratio = compute_distance_ratio(points)

    # Secondary metric: estimate uniformity using pairwise distance variance
    # Lower variance in distances suggests more uniform distribution
    try:
        distances = pdist(points)
        if len(distances) > 0:
            # Better uniformity metric: inverse of distance variance normalized by mean
            distance_mean = np.mean(distances)
            if distance_mean > 1e-12:
                distance_variance = np.var(distances)
                # Uniformity score (higher is better): inverse of variance scaled by mean
                uniformity_score = 1.0 / (distance_variance + 1e-12) * distance_mean
                # Combine with ratio (weight 0.8 for ratio, 0.2 for uniformity)
                return 0.8 * ratio + 0.2 * uniformity_score
            else:
                return ratio
        else:
            return ratio
    except:
        return ratio

def objective_function(x: np.ndarray) -> float:
    """Objective function that returns negative ratio to minimize"""
    points = x.reshape(-1, 3)

    # Ensure points are within bounds [0,1]^3
    points = project_to_unit_cube(points)

    # Compute distances
    distances = pdist(points)

    if len(distances) == 0:
        return -np.inf

    d_min = np.min(distances)
    d_max = np.max(distances)

    # Avoid division by zero
    if d_max <= 1e-12:
        return -np.inf

    # Return negative because we want to maximize the ratio
    return -(d_min / d_max)

def penalty_objective(x: np.ndarray, penalty_weight: float = 1e7) -> float:
    """Objective with penalty for boundary violations"""
    points = x.reshape(-1, 3)

    # Apply penalty for points outside bounds using vectorized operations
    penalty = 0
    penalty += np.sum(np.maximum(0, -points)**2) * penalty_weight  # Below 0
    penalty += np.sum(np.maximum(0, points - 1)**2) * penalty_weight  # Above 1

    # Original objective
    original_obj = objective_function(x)

    return original_obj + penalty

def adaptive_differential_evolution(objective_func, bounds, maxiter=300, seed=42):
    """Enhanced differential evolution with adaptive parameters and improved early stopping"""
    current_popsize = 25  # Increased initial population size
    prev_best = -np.inf
    stagnation_count = 0
    improvement_threshold = 1e-8
    min_improvement = 1e-10  # Tighter criterion
    recent_improvements = []
    improvement_window = 10  # Larger window for better trend detection
    
    for iteration in range(maxiter // 15):  # Reduced iterations per batch
        # Adjust population size based on convergence
        if stagnation_count > 2 and current_popsize < 40:
            current_popsize = min(current_popsize + 5, 40)
        
        try:
            result = differential_evolution(
                objective_func,
                bounds,
                seed=seed + iteration,
                maxiter=15,  # Increased iterations per call
                popsize=current_popsize,
                tol=1e-15,  # Stricter tolerance
                mutation=(0.7, 1.0),  # Wider mutation range for better exploration
                recombination=0.8,   # Higher recombination for better exploitation
                disp=False
            )
        except Exception:
            # Fallback to smaller population if needed
            try:
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=seed + iteration,
                    maxiter=15,
                    popsize=max(5, current_popsize - 5),
                    tol=1e-15,
                    mutation=(0.7, 1.0),
                    recombination=0.8,
                    disp=False
                )
            except Exception:
                # Last resort - use basic differential evolution
                result = differential_evolution(
                    objective_func,
                    bounds,
                    seed=seed + iteration,
                    maxiter=15,
                    popsize=15,
                    tol=1e-15,
                    mutation=(0.7, 1.0),
                    recombination=0.7,
                    disp=False
                )
        
        # Check for improvement
        current_best = -result.fun
        improvement = current_best - prev_best
        
        recent_improvements.append(improvement)
        if len(recent_improvements) > improvement_window:
            recent_improvements.pop(0)
        
        # More aggressive early stopping
        if len(recent_improvements) == improvement_window:
            avg_improvement = np.mean(recent_improvements)
            if abs(avg_improvement) < min_improvement:
                break
                
        if improvement > improvement_threshold:
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        prev_best = current_best
        
    return result

def local_refinement(points: np.ndarray) -> np.ndarray:
    """Apply local refinement to improve final solution with enhanced controls"""
    def objective_local(x):
        points_local = x.reshape(-1, 3)
        distances = pdist(points_local)

        if len(distances) == 0:
            return -np.inf

        d_min = np.min(distances)
        d_max = np.max(distances)

        if d_max > 1e-12:
            return -(d_min / d_max)
        else:
            return -np.inf

    try:
        x0_refine = points.flatten()
        bounds = [(0, 1)] * len(points.flatten())
        
        # First refinement with very strict tolerances
        result_refine = minimize(
            objective_local,
            x0_refine,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-15, 'gtol': 1e-15},  # Very tight tolerances
            tol=1e-15
        )
        
        refined_points = result_refine.x.reshape(-1, 3)
        refined_points = project_to_unit_cube(refined_points)
        
        # Second refinement step if improvement is significant
        final_ratio = compute_distance_ratio(refined_points)
        original_ratio = compute_distance_ratio(points)
        if final_ratio > 0.95 * original_ratio:
            # Re-run with even stricter parameters
            result_refine2 = minimize(
                objective_local,
                refined_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-16, 'gtol': 1e-16},
                tol=1e-16
            )
            refined_points = result_refine2.x.reshape(-1, 3)
            refined_points = project_to_unit_cube(refined_points)
        
        return refined_points
    except Exception:
        return points

def generate_initial_strategies(n_points: int = 14) -> list:
    """Generate multiple initialization strategies"""
    strategies = []
    
    # Strategy 1: Spherical Fibonacci points
    fib_points = initialize_fibonacci_points(n_points)
    fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
    strategies.append(("fibonacci", fib_points))
    
    # Strategy 2: Cube grid points
    cube_points = initialize_cube_grid_points(n_points)
    strategies.append(("cube_grid", cube_points))
    
    # Strategy 3: Random points
    random_points = initialize_random_points(n_points)
    strategies.append(("random", random_points))
    
    # Strategy 4: Voronoi uniform points
    voronoi_uniform_points = initialize_voronoi_uniform_points(n_points)
    strategies.append(("voronoi_uniform", voronoi_uniform_points))
    
    # Strategy 5: Spherical cap points
    spherical_cap_points = initialize_spherical_cap_points(n_points)
    spherical_cap_points = (spherical_cap_points + 1) / 2
    strategies.append(("spherical_cap", spherical_cap_points))
    
    # Strategy 6: Golden spiral points
    golden_spiral_points = initialize_golden_spiral_points(n_points)
    golden_spiral_points = (golden_spiral_points + 1) / 2
    strategies.append(("golden_spiral", golden_spiral_points))
    
    # Strategy 7: Hybrid points
    hybrid_points = initialize_hybrid_points(n_points)
    strategies.append(("hybrid", hybrid_points))
    
    return strategies

def select_best_initialization(strategies: list) -> tuple:
    """Select best initialization based on evaluation"""
    best_initialization = None
    best_ratio = -np.inf

    for name, points in strategies:
        ratio = evaluate_initialization(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initialization = points.copy()

    return best_initialization, best_ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Generate multiple initialization strategies
    strategies = generate_initial_strategies(14)
    
    # Select the best initialization
    best_initialization, _ = select_best_initialization(strategies)
    
    # Use the best initialization as starting point
    x0 = best_initialization.flatten()

    # Bounds for each coordinate: [0, 1] for all 14 points × 3 coordinates
    bounds = [(0, 1)] * 14 * 3

    # Run adaptive differential evolution optimization
    best_result = None
    best_ratio = -np.inf

    # Try 5 different random seeds for better exploration (increased from 3)
    for seed_val in [42, 123, 456, 789, 101]:
        # Reset numpy seed for reproducibility in each attempt
        np.random.seed(seed_val)

        # Use adaptive differential evolution
        result = adaptive_differential_evolution(
            penalty_objective,
            bounds,
            maxiter=300,  # Increased iterations
            seed=seed_val
        )

        # Check if this result is better
        if -result.fun > best_ratio:
            best_ratio = -result.fun
            best_result = result

    # Extract optimized points
    optimized_points = best_result.x.reshape(-1, 3)

    # Apply local refinement
    final_points = local_refinement(optimized_points)

    # Final clipping to ensure bounds are respected
    final_points = project_to_unit_cube(final_points)

    return final_points

# EVOLVE-BLOCK-END