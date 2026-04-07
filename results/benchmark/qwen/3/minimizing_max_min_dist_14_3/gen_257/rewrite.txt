# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import warnings
from numba import jit

@jit(nopython=True)
def fast_pdist_matrix(points):
    """Fast computation of pairwise distances using Numba."""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def fibonacci_sphere(n):
    """Generate n points evenly distributed on a unit sphere using Fibonacci spiral method."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle

    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def icosahedral_initialization(n_points):
    """Initialize points using icosahedral symmetry for better uniformity."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Vertices of icosahedron scaled to unit sphere
    vertices = []
    # Add vertices at (±1, 0, ±φ) and permutations
    for i in [1, -1]:
        for j in [0, 1, -1]:
            for k in [0, 1, -1]:
                if i*j*k != 0:
                    vertices.append([i, j*phi, k/phi])
                elif i*j != 0:
                    vertices.append([i, j, k*phi])
                elif i*k != 0:
                    vertices.append([i, j*phi, k])
                elif j*k != 0:
                    vertices.append([i*phi, j, k])

    # Normalize to unit sphere and take first n_points
    vertices = np.array(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms

    # For 14 points, we'll use a modified approach
    # Generate points on icosahedron faces
    points = []

    # Add vertices of icosahedron
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])

    # Normalize to unit sphere
    norms = np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    ico_vertices = ico_vertices / norms

    # For 14 points, we can start with 12 vertices and add 2 more
    # Or use a carefully chosen subset of icosahedral points
    selected_vertices = ico_vertices[[0,2,4,6,8,10,1,3,5,7,9,11]]

    # Add additional points in strategic locations
    additional_points = []
    for i in range(n_points - len(selected_vertices)):
        # Add points along great circles
        angle = 2 * np.pi * i / (n_points - len(selected_vertices))
        point = [np.cos(angle), np.sin(angle), 0]
        additional_points.append(point)

    points = np.vstack([selected_vertices, additional_points])

    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms

    # Apply slight random perturbations to avoid perfect symmetry
    np.random.seed(42)
    perturbations = np.random.normal(0, 0.02, (len(points), 3))
    points += perturbations
    # Project back to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms

    # Scale to unit cube [0,1]^3
    points = (points + 1) / 2
    return points

def spherical_voronoi_initialization(n_points):
    """Create initial points using spherical Voronoi uniformity principle."""
    # Start with Fibonacci distribution
    points = fibonacci_sphere(n_points)

    # Apply iterative adjustment to improve uniformity
    np.random.seed(42)
    for _ in range(20):
        # Perturb points slightly
        perturbations = np.random.normal(0, 0.01, (n_points, 3))
        points += perturbations

        # Project back to sphere surface
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points = points / norms

    # Scale to unit cube [0,1]^3
    points = (points + 1) / 2
    return points

def spherical_map(points):
    """Map points from 3D space to unit sphere using normalization."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return points / norms

def spherical_voronoi_quality(sphere_points):
    """Calculate quality based on Voronoi cell areas on sphere."""
    if len(sphere_points) < 2:
        return 0

    # Create spherical Voronoi diagram
    try:
        sv = SphericalVoronoi(sphere_points)
        # Calculate total area of Voronoi cells
        cell_areas = sv.calculate_areas()
        # Quality is inversely related to variance of cell areas
        # More uniform areas indicate better distribution
        if len(cell_areas) > 0:
            mean_area = np.mean(cell_areas)
            if mean_area > 0:
                variance = np.var(cell_areas)
                # Return inverse variance (higher is better)
                return 1.0 / (1.0 + variance / mean_area**2)
    except Exception:
        pass
    return 0

def min_max_ratio(points):
    """Calculate the ratio of minimum to maximum pairwise distances."""
    if len(points) < 2:
        return 0

    # Calculate pairwise distances efficiently with numba
    distances = fast_pdist_matrix(points)
    
    # Get min and max distances (excluding diagonal zeros)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    masked_distances = distances[mask]
    
    if len(masked_distances) == 0:
        return 0
        
    d_min = np.min(masked_distances)
    d_max = np.max(masked_distances)

    # Avoid division by zero
    if d_max == 0:
        return 0

    return d_min / d_max

def objective_function(x_flat, use_spherical_quality=True):
    """Objective function to maximize the min/max distance ratio.

    Args:
        x_flat: Flattened array of point coordinates [x1, y1, z1, x2, y2, z2, ...]
        use_spherical_quality: Whether to include spherical voronoi quality term

    Returns:
        Negative of combined objective (since we minimize in scipy.optimize)
    """
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Normalize points to unit sphere
    sphere_points = spherical_map(points)

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Calculate spherical Voronoi quality
    voronoi_quality = spherical_voronoi_quality(sphere_points)

    # Combine objectives: prioritize min/max ratio but also consider geometric distribution
    combined = ratio + 0.1 * voronoi_quality

    # Return negative because we want to maximize (minimize negative)
    return -combined

def constraint_penalty(points, penalty_weight=1000.0):
    """Calculate penalty for constraint violations."""
    penalty = 0
    for i in range(len(points)):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2
    return penalty

def objective_with_penalty(x_flat, penalty_weight=1000.0):
    """Objective function combining min/max ratio with penalties."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Calculate penalty for constraints
    penalty = constraint_penalty(points, penalty_weight)

    # Return negative ratio plus penalty (since we minimize)
    return -ratio + penalty

def adaptive_differential_evolution(objective_func, bounds, max_iter=200, popsize=None, seed=42):
    """Run differential evolution with adaptive population sizing and early stopping."""
    if popsize is None:
        popsize = 25
    
    # Run differential evolution
    result = differential_evolution(
        objective_func,
        bounds,
        seed=seed,
        maxiter=max_iter,
        popsize=popsize,
        tol=1e-7,
        mutation=(0.5, 1),
        recombination=0.8,
        disp=False
    )
    
    return result

def create_initial_placement():
    """Create initial point placement using hybrid approach."""
    # Method: Use icosahedral initialization as primary strategy
    initial_points = icosahedral_initialization(14)
    
    # Secondary: Improve with Fibonacci-based refinement
    fib_points = fibonacci_sphere(14)
    fib_points = (fib_points + 1) / 2
    
    # Compare and select better starting configuration
    ratio1 = min_max_ratio(initial_points)
    ratio2 = min_max_ratio(fib_points)
    
    if ratio2 > ratio1:
        initial_points = fib_points
    
    return initial_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Try multiple initialization strategies and select the best starting point
    initial_strategies = [
        lambda: create_initial_placement(),
        lambda: spherical_voronoi_initialization(14),
        lambda: (fibonacci_sphere(14) + 1) / 2,
        lambda: np.random.rand(14, 3)
    ]

    best_points = None
    best_ratio = 0

    for i, init_func in enumerate(initial_strategies):
        try:
            initial_points = init_func()
            ratio = min_max_ratio(initial_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = initial_points.copy()
        except Exception:
            continue

    if best_points is None:
        # Fallback to spherical Voronoi if all strategies fail
        best_points = spherical_voronoi_initialization(14)

    # Phase 1: Global optimization with differential evolution
    try:
        # Flatten initial points for optimization
        x0 = best_points.flatten()

        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(14 * 3)]

        # Global optimization with differential evolution
        result = adaptive_differential_evolution(
            objective_function,
            bounds,
            max_iter=150,
            popsize=25,
            seed=42
        )

        # Extract optimized points
        optimized_points = result.x.reshape((14, 3))

        # Calculate final ratio
        final_ratio = min_max_ratio(optimized_points)

        # Store better result
        if final_ratio > best_ratio:
            best_points = optimized_points
            best_ratio = final_ratio

    except Exception:
        pass  # Continue with fallback

    # Phase 2: Local refinement with L-BFGS-B
    try:
        # Flatten the best points
        x0_refine = best_points.flatten()

        # Refinement with L-BFGS-B using tighter tolerances
        result_refined = minimize(
            objective_function,
            x0_refine,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000},
            tol=1e-10
        )

        refined_points = result_refined.x.reshape((14, 3))
        final_ratio = min_max_ratio(refined_points)

        # Update if improved
        if final_ratio > best_ratio:
            best_points = refined_points

    except Exception:
        pass  # Keep original best points if refinement fails

    return best_points

# EVOLVE-BLOCK-END