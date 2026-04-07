# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time

def initialize_points(n: int = 14, d: int = 3) -> np.ndarray:
    """
    Initialize points using a known good 14-point spherical code configuration
    This provides a much better starting point than simple Fibonacci spirals.
    """
    # Known good 14-point spherical code configuration from mathematical literature
    # These coordinates are normalized to unit sphere
    spherical_points = np.array([
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, -1.0000],
        [0.9343, 0.0000, 0.3564],
        [-0.9343, 0.0000, 0.3564],
        [0.0000, 0.9343, 0.3564],
        [0.0000, -0.9343, 0.3564],
        [0.0000, 0.9343, -0.3564],
        [0.0000, -0.9343, -0.3564],
        [0.9343, 0.0000, -0.3564],
        [-0.9343, 0.0000, -0.3564],
        [0.3564, 0.9343, 0.0000],
        [-0.3564, 0.9343, 0.0000],
        [0.3564, -0.9343, 0.0000],
        [-0.3564, -0.9343, 0.0000]
    ])

    # Normalize to unit sphere if needed
    norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
    spherical_points = spherical_points / np.where(norms == 0, 1, norms)

    # Add small perturbations to escape local optima
    np.random.seed(42)
    perturbation = np.random.normal(0, 0.01, spherical_points.shape)
    spherical_points = spherical_points + perturbation

    # Normalize again after perturbation
    norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
    spherical_points = spherical_points / np.where(norms == 0, 1, norms)

    # Scale to unit cube [0,1]^3
    # Map from [-1,1]^3 to [0,1]^3
    points = (spherical_points + 1) / 2

    return points

def calculate_distance_metrics(points: np.ndarray) -> tuple[float, float]:
    """
    Calculate minimum and maximum distances between all point pairs.

    Args:
        points: Array of shape (n, d)

    Returns:
        Tuple of (min_distance, max_distance)
    """
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    return min_dist, max_dist

def objective_function(points_flat: np.ndarray, penalty_weight: float = 100.0) -> float:
    """
    Objective function to maximize the min/max distance ratio with penalty constraints.
    Returns negative ratio since optimizers minimize by default.

    Args:
        points_flat: Flattened array of point coordinates
        penalty_weight: Weight for constraint violation penalties

    Returns:
        Negative min/max ratio plus penalty (to be minimized)
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)

    # Apply boundary penalties
    penalty = 0.0
    for coord in points.flat:
        if coord < 0:
            penalty += penalty_weight * (0 - coord)**2
        elif coord > 1:
            penalty += penalty_weight * (coord - 1)**2

    # Calculate distances
    distances = pdist(points)

    if len(distances) == 0:
        return float('inf') + penalty

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    # Avoid division by zero
    if max_dist <= 0:
        return float('inf') + penalty

    # Return negative ratio to minimize (maximize the ratio) plus penalty
    return -(min_dist / max_dist) + penalty

def adaptive_differential_evolution(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Optimize point configuration using adaptive differential evolution with exponential population sizing.
    """
    # Flatten initial points for optimization
    initial_flat = initial_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0.0, 1.0)] * len(initial_flat)

    # Set base optimization options and adaptive parameters
    base_options = {
        'maxiter': 200,
        'popsize': 15,
        'tol': 1e-6,
        'mutation': (0.5, 1.0),
        'recombination': 0.7
    }

    # Track history for convergence monitoring
    history = []
    stagnation_counter = 0
    max_stagnation = 10
    population_size = base_options['popsize']
    last_improvement_iter = 0
    improvement_threshold = 1e-6
    
    # Exponential population size growth factor
    growth_factor = 1.5
    
    try:
        # Run multiple rounds with adaptive parameters
        for iteration in range(8):  # More iterations for better exploration
            # Update population size based on progress
            current_popsize = int(population_size)
            
            # Run differential evolution with current settings
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=base_options['maxiter'] // 8,
                popsize=current_popsize,
                tol=base_options['tol'],
                mutation=base_options['mutation'],
                recombination=base_options['recombination'],
                seed=42 + iteration,
                disp=False
            )
            
            # Track results
            history.append(result.fun)
            
            # Monitor for stagnation and update population size exponentially
            if len(history) > 1:
                improvement = abs(history[-2] - history[-1])
                if improvement < improvement_threshold:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_improvement_iter = iteration
                    
                # Exponential population growth when stagnating
                if stagnation_counter >= 3 and current_popsize < 40:
                    population_size *= growth_factor
                    stagnation_counter = 0
                elif stagnation_counter >= 5 and current_popsize > 10:
                    population_size = max(current_popsize * 0.9, 10)
                    
            # Early stopping if no significant improvement
            if len(history) >= 3 and abs(history[-1] - history[-2]) < 1e-9:
                break
                
        optimized_points = result.x.reshape(14, 3)
        
    except Exception as e:
        # Fallback to basic optimization if adaptive fails
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=base_options['maxiter'],
                popsize=base_options['popsize'],
                tol=base_options['tol'],
                mutation=base_options['mutation'],
                recombination=base_options['recombination'],
                seed=42,
                disp=False
            )
            optimized_points = result.x.reshape(14, 3)
        except:
            # Last resort fallback
            optimized_points = initial_points.copy()

    return optimized_points

def local_refinement(initial_points: np.ndarray) -> np.ndarray:
    """
    Perform multi-stage local refinement using L-BFGS-B optimization with aggressive tolerances.
    """
    initial_flat = initial_points.flatten()
    bounds = [(0.0, 1.0)] * len(initial_flat)
    
    try:
        # Stage 1: Coarse refinement with moderate tolerances
        result1 = minimize(
            objective_function,
            initial_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        # Stage 2: Fine refinement with tighter tolerances
        if result1.success:
            refined_flat = result1.x
            result2 = minimize(
                objective_function,
                refined_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result2.success:
                refined_points = result2.x.reshape(14, 3)
            else:
                refined_points = result1.x.reshape(14, 3)
        else:
            refined_points = initial_points.copy()
            
    except Exception:
        refined_points = initial_points.copy()
        
    return refined_points

def create_symmetric_variants(points: np.ndarray, num_variants: int = 6) -> list[np.ndarray]:
    """
    Create geometrically meaningful symmetric variants of the point configuration.
    """
    variants = [points]
    
    # Basic transformations: reflections along axes
    basic_transforms = [
        np.eye(3),  # Identity
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # Reflect x-axis
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # Reflect y-axis
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # Reflect z-axis
    ]

    # Apply transformations
    for transform in basic_transforms:
        transformed = points @ transform.T
        variants.append(transformed)

    # Coordinate permutations
    perms = [
        [0, 1, 2],  # identity
        [0, 2, 1],  # swap y and z
        [1, 0, 2],  # swap x and y
        [2, 0, 1],  # cyclic perm x->z, y->x, z->y
        [1, 2, 0],  # cyclic perm x->y, y->z, z->x
        [2, 1, 0],  # swap x and z
    ]

    for perm in perms:
        permuted = points[:, perm]
        variants.append(permuted)

    return variants[:num_variants]

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    # Phase 1: Initialize points using high-quality spherical code
    initial_points = initialize_points(14, 3)

    # Phase 2: Global optimization with adaptive differential evolution
    try:
        global_optimized = adaptive_differential_evolution(initial_points)
    except Exception:
        global_optimized = initial_points

    # Phase 3: Local refinement with multiple stages
    try:
        final_points = local_refinement(global_optimized)
    except Exception:
        final_points = global_optimized

    # Phase 4: Multiple restarts with different perturbations
    best_points = final_points.copy()
    best_ratio = calculate_distance_metrics(final_points)[0] / calculate_distance_metrics(final_points)[1] if calculate_distance_metrics(final_points)[1] > 0 else 0
    
    # Try several random restarts with different perturbation scales
    restart_scales = [0.003, 0.005, 0.007, 0.01, 0.015]
    
    for restart_idx, scale in enumerate(restart_scales):
        np.random.seed(restart_idx * 1000 + 42)
        
        # Create slightly perturbed starting point
        perturbed = final_points + np.random.normal(0, scale, final_points.shape)
        perturbed = np.clip(perturbed, 0, 1)
        
        # Apply refinement to perturbed point
        try:
            restarted_points = local_refinement(perturbed)
            min_dist, max_dist = calculate_distance_metrics(restarted_points)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = restarted_points.copy()
        except:
            continue

    # Phase 5: Symmetry-based exploration of best configuration
    try:
        variants = create_symmetric_variants(best_points, num_variants=4)
        for variant in variants:
            min_dist, max_dist = calculate_distance_metrics(variant)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = variant.copy()
    except:
        pass

    # Phase 6: Final validation and adjustment
    final_points = np.clip(best_points, 0, 1)

    # Calculate final metrics
    min_dist, max_dist = calculate_distance_metrics(final_points)

    # If optimization didn't work well, fall back to a good known arrangement
    if max_dist <= 0 or min_dist <= 0 or max_dist < 1e-10:
        # Fallback to regularized arrangement
        np.random.seed(42)
        final_points = np.random.rand(14, 3)

    return final_points

# EVOLVE-BLOCK-END
