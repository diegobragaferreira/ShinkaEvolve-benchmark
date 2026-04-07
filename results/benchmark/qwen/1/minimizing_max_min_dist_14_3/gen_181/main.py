# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time

def sobol_points_sphere(n_points, seed=42):
    """Generate points on sphere using 3D Sobol sequence for superior space-filling properties"""
    try:
        # Try to import sobol sequence generator
        from sobol_seq import i4_sobol_generate

        # Generate Sobol points in [0,1]^3
        np.random.seed(seed)
        sobol_points = i4_sobol_generate(3, n_points)

        # Convert to sphere using spherical coordinates
        points = np.zeros((n_points, 3))

        # Use the Sobol points to create well-distributed points on sphere
        for i in range(n_points):
            # Map to sphere using similar approach as Fibonacci
            u = sobol_points[i, 0]  # Uniform random in [0,1]
            v = sobol_points[i, 1]  # Uniform random in [0,1]

            # Use these as parameters for spherical coordinates
            theta = 2 * np.pi * u  # azimuthal angle
            phi = np.arccos(2 * v - 1)  # polar angle

            # Convert to Cartesian
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points[i] = [x, y, z]

        return points

    except ImportError:
        # Fallback to fibonacci if sobol not available
        return fibonacci_spiral_sphere(n_points, seed)

def fibonacci_spiral_sphere(n_points, seed=42):
    """Generate points on a sphere using Fibonacci spiral method."""
    np.random.seed(seed)
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def icosahedron_points(n=14, seed=42):
    """Generate points using icosahedron-based construction"""
    np.random.seed(seed)
    # Vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # If we need more than 12 points, distribute additional points
    if n <= 12:
        # Just return subset of vertices
        return vertices[:n]
    else:
        # For 14 points, we'll start with icosahedron vertices and add two more
        points = vertices.copy()

        # Add two more points that are well-distributed
        # Add points along major axes
        points = np.vstack([points, [[0, 0, 1], [0, 0, -1]]])

        # Apply slight random perturbation to ensure good distribution
        points += np.random.normal(0, 0.05, (points.shape[0], 3))

        # Normalize again to maintain unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / np.maximum(norms[:, np.newaxis], 1e-12)

        return points[:n]

def min_max_dist_ratio(points):
    """Calculate the ratio of minimum to maximum distance."""
    if len(points) < 2:
        return 0.0
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist < 1e-12:
        return 0.0
    return min_dist / max_dist

def compute_min_max_ratio_with_penalty(points, penalty_weight=0.1):
    """Modified objective that includes penalty terms for extreme distributions"""
    if len(points) < 2:
        return 0.0

    distances = pdist(points)
    if len(distances) == 0:
        return 0.0

    min_dist = np.min(distances)
    max_dist = np.max(distances)

    if max_dist < 1e-12:
        return 0.0

    # Base ratio
    ratio = min_dist / max_dist

    # Penalty for very large or very small ratios
    if ratio < 0.1:
        ratio -= penalty_weight * (0.1 - ratio)  # Penalize very small ratios

    return ratio

def spherical_voronoi_initialization(n_points, seed=42):
    """Improved initialization using Sobol sequence for better space-filling properties"""
    # Use Sobol sequence points as base
    sobol_points = sobol_points_sphere(n_points, seed)

    # Apply geometric refinement
    refined_points = sobol_points.copy()

    # Add small random perturbation to break symmetry
    np.random.seed(seed)
    noise = np.random.normal(0, 0.02, (n_points, 3))
    refined_points += noise

    # Project back to sphere
    norms = np.linalg.norm(refined_points, axis=1, keepdims=True)
    refined_points = refined_points / np.maximum(norms, 1e-12)

    return refined_points

def adaptive_constraint_tightening(x_flat, iteration_step=0, max_steps=100, original_ratio=None):
    """Adaptive constraint tightening to gradually enforce tighter constraints during optimization"""
    n = 14
    d = 3
    points = x_flat.reshape(n, d)

    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    normalized_points = points / np.maximum(norms, 1e-12)

    # Calculate distances
    distances = pdist(normalized_points)
    if len(distances) == 0:
        return 1e10

    d_min = np.min(distances)
    d_max = np.max(distances)

    if d_max < 1e-12:
        return 1e10

    # Compute ratio
    ratio = d_min / d_max

    # Apply adaptive constraint tightening
    # Gradually tighten the constraints as optimization progresses
    if iteration_step < max_steps:
        # Allow gradual tightening of maximum distance constraint
        max_distance_target = 2.8 - (iteration_step / max_steps) * 0.8
        if d_max > max_distance_target:
            # Apply penalty for exceeding target
            penalty = (d_max - max_distance_target) * 1000
            return -ratio + penalty
        else:
            # No penalty when within constraint
            return -ratio

    return -ratio

def hybrid_multi_stage_optimization(initial_points, max_iter=1000, seed=42):
    """Hybrid optimization approach combining multiple methods for better convergence"""
    n, d = initial_points.shape

    # Set up optimization with fixed seed for reproducibility
    np.random.seed(seed)

    # Initial optimization using SLSQP with constraints
    def objective_slqp(x_flat):
        points = x_flat.reshape(n, d)
        # Ensure points are on unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / np.maximum(norms, 1e-12)

        # Return negative ratio since we're minimizing
        ratio = compute_min_max_ratio_with_penalty(normalized_points)
        return -ratio if ratio > 0 else 1e10

    def constraint_sphere(x_flat):
        points = x_flat.reshape(n, d)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    constraints = {'type': 'eq', 'fun': constraint_sphere}
    bounds = [(-2, 2) for _ in range(n * d)]

    try:
        # First stage: SLSQP for global optimization with constraints
        x0 = initial_points.flatten()
        result1 = minimize(
            objective_slqp,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter // 4, 'ftol': 1e-8, 'gtol': 1e-8}
        )

        if result1.success:
            # Second stage: L-BFGS-B for local refinement with tightened tolerances
            points_after_slsqp = result1.x.reshape(n, d)
            norms = np.linalg.norm(points_after_slsqp, axis=1, keepdims=True)
            points_after_slsqp = points_after_slsqp / np.maximum(norms, 1e-12)

            # Optimize again with L-BFGS-B
            x0_lbfgs = points_after_slsqp.flatten()
            result2 = minimize(
                objective_slqp,
                x0_lbfgs,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter // 4, 'ftol': 1e-12, 'gtol': 1e-12}
            )

            # Third stage: Trust-Constr for even tighter convergence
            if result2.success:
                points_after_lbfgs = result2.x.reshape(n, d)
                norms = np.linalg.norm(points_after_lbfgs, axis=1, keepdims=True)
                points_after_lbfgs = points_after_lbfgs / np.maximum(norms, 1e-12)

                # Final optimization with trust-constr
                x0_trust = points_after_lbfgs.flatten()
                result3 = minimize(
                    objective_slqp,
                    x0_trust,
                    method='trust-constr',
                    bounds=bounds,
                    options={'maxiter': max_iter // 2, 'xtol': 1e-14, 'gtol': 1e-14}
                )

                if result3.success:
                    final_points = result3.x.reshape(n, d)
                    return final_points
                else:
                    return points_after_lbfgs
            else:
                return points_after_slsqp
        else:
            return initial_points

    except Exception:
        # Fallback to just using the initial points
        return initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced initialization with 3D Sobol sequence and hybrid optimization approach.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3

    # Multi-start approach with diverse initialization strategies
    best_ratio = -np.inf
    best_points = None

    # Strategy 1: Various initialization methods with multiple restarts for better exploration
    init_strategies = [
        ("sobol", lambda s: sobol_points_sphere(n, s)),
        ("icosahedron", lambda s: icosahedron_points(n, s)),
        ("fibonacci", lambda s: fibonacci_spiral_sphere(n, s)),
        ("spherical_voronoi", lambda s: spherical_voronoi_initialization(n, s))
    ]

    # Run multiple restarts for each strategy with different seeds
    num_restarts_per_strategy = 5
    total_restarts = num_restarts_per_strategy * len(init_strategies)

    for restart in range(total_restarts):
        # Select initialization strategy based on restart index
        strategy_idx = restart % len(init_strategies)
        strategy_name, init_func = init_strategies[strategy_idx]

        # Set seed for reproducibility with some diversity
        seed = restart * 100 + 42

        # Get initial points
        initial_points = init_func(seed)

        # Add slight random perturbation to break symmetries
        np.random.seed(seed)
        noise = np.random.normal(0, 0.02, (n, d))
        initial_points += noise

        # Project back to sphere
        norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
        initial_points = initial_points / np.maximum(norms, 1e-12)

        # Optimize using hybrid multi-stage approach
        try:
            optimized_points = hybrid_multi_stage_optimization(initial_points, max_iter=800, seed=seed)

            # Evaluate the result
            ratio = min_max_dist_ratio(optimized_points)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

        except Exception:
            # If optimization fails, continue to next restart
            continue

    # Strategy 2: Additional random initialization with Sobol-inspired points
    if best_points is None or best_ratio < 0.4:
        # Generate points with Sobol-inspired distribution
        np.random.seed(42)
        # Create Sobol-like distribution with additional randomization
        sobol_like_points = np.random.rand(n, d)
        # Transform to sphere
        sobol_like_points = sobol_like_points * 2 - 1  # [-1, 1]
        norms = np.linalg.norm(sobol_like_points, axis=1, keepdims=True)
        normalized_points = sobol_like_points / np.maximum(norms, 1e-12)

        # Optimize this configuration
        try:
            optimized_points = hybrid_multi_stage_optimization(normalized_points, max_iter=600, seed=1001)

            ratio = min_max_dist_ratio(optimized_points)

            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()

        except Exception:
            pass

    # Final safeguard - return random points if nothing worked
    if best_points is None:
        np.random.seed(42)
        points = np.random.rand(n, d) * 2 - 1
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        best_points = points / np.maximum(norms, 1e-12)

    # Convert to unit cube [0,1]^3
    # Center and scale appropriately
    centered = best_points - np.mean(best_points, axis=0)
    max_coord = np.max(np.abs(centered))
    if max_coord > 0:
        scaled = centered / max_coord * 0.5
    else:
        scaled = centered
    # Shift to [0,1]^3
    final_points = scaled + 0.5

    return final_points

# EVOLVE-BLOCK-END