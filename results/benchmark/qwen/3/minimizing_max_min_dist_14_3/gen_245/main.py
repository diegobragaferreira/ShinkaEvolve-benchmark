# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
from numba import jit
import warnings

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

    # Calculate pairwise distances
    distances = pdist(points)

    # Get min and max distances
    d_min = np.min(distances)
    d_max = np.max(distances)

    # Avoid division by zero
    if d_max == 0:
        return 0

    return d_min / d_max

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

def energy_based_refinement(points, max_iter=1000, learning_rate=0.01):
    """Refine point distribution using an energy-based approach with repulsive forces."""
    # Create a copy to avoid modifying original points
    current_points = points.copy()
    
    for iteration in range(max_iter):
        # Calculate pairwise distances
        distances = fast_pdist_matrix(current_points)
        
        # Calculate forces (repulsive between all points)
        forces = np.zeros_like(current_points)
        n_points = len(current_points)
        
        # Compute repulsive forces
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Calculate vector from point j to point i
                    diff = current_points[i] - current_points[j]
                    dist_sq = np.sum(diff ** 2)
                    
                    # Avoid division by zero and very close points
                    if dist_sq > 1e-10:
                        force_magnitude = 1.0 / dist_sq
                        forces[i] += force_magnitude * diff
        
        # Apply forces with learning rate
        current_points -= learning_rate * forces
        
        # Constrain points to [0,1]^3
        current_points = np.clip(current_points, 0, 1)
        
        # Simple convergence check
        if iteration > 10 and np.all(np.abs(forces) < 1e-8):
            break
            
    return current_points

def combine_objectives(x_flat, weights=(1.0, 0.1, 0.05)):
    """
    Combined objective function that balances multiple criteria:
    1. Min/max ratio 
    2. Spherical Voronoi quality
    3. Energy-based uniformity
    """
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Calculate individual components
    ratio = min_max_ratio(points)
    sphere_points = spherical_map(points)
    voronoi_quality = spherical_voronoi_quality(sphere_points)
    
    # Basic penalty for constraint violations
    penalty = constraint_penalty(points)

    # Combine objectives
    # We want to maximize ratio, maximize voronoi quality, and minimize penalty
    # But we also want to ensure points are well-distributed
    combined = weights[0] * ratio + weights[1] * voronoi_quality - weights[2] * penalty
    
    # Return negative because we minimize in scipy.optimize
    return -combined

def adaptive_penalty_objective(x_flat, penalty_weight=1e6, iteration=0):
    """Objective function with adaptive penalty for out-of-bounds points."""
    # Reshape flat array back to points
    n_points = 14
    points = x_flat.reshape((n_points, 3))

    # Apply penalty for constraint violations
    penalty = 0
    for i in range(n_points):
        for j in range(3):  # x, y, z coordinates
            if points[i, j] < 0:
                penalty += penalty_weight * (0 - points[i, j])**2 * (1 + iteration * 0.1)
            elif points[i, j] > 1:
                penalty += penalty_weight * (points[i, j] - 1)**2 * (1 + iteration * 0.1)

    # Calculate min/max ratio
    ratio = min_max_ratio(points)

    # Return value to minimize (negative ratio + penalty)
    return -ratio + penalty

def create_initial_placement():
    """Create initial point placement using enhanced spherical code approach."""
    # Start with icosahedral initialization for better distribution
    points = icosahedral_initialization(14)
    
    # Improve distribution by applying multiple small perturbations
    np.random.seed(42)
    for _ in range(20):  # More perturbations for better uniformity
        # Add small random perturbations
        perturbation = np.random.normal(0, 0.01, (14, 3))
        points += perturbation

        # Project back to sphere surface
        points = spherical_map(points)

    # Scale to unit cube [0,1]^3
    points = (points + 1) / 2
    
    return points

def multi_stage_optimization(initial_points, max_time_seconds=360):
    """Perform multi-stage optimization with progressive refinement."""
    current_points = initial_points.copy()
    
    # Stage 1: Coarse global optimization with differential evolution
    try:
        x0 = current_points.flatten()
        bounds = [(0, 1) for _ in range(14 * 3)]
        
        # Coarse optimization with fewer iterations
        result = differential_evolution(
            combine_objectives,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False
        )
        
        current_points = result.x.reshape((14, 3))
        
    except Exception:
        pass

    # Stage 2: Energy-based refinement (physics-inspired)  
    try:
        current_points = energy_based_refinement(current_points, max_iter=500, learning_rate=0.02)
    except Exception:
        pass

    # Stage 3: Fine-grained local optimization with L-BFGS-B
    try:
        x0 = current_points.flatten()
        bounds = [(0, 1) for _ in range(14 * 3)]
        
        # Tighter tolerances for final refinement
        result = minimize(
            adaptive_penalty_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-10, 'gtol': 1e-10},
            tol=1e-10
        )
        
        current_points = result.x.reshape((14, 3))
        
    except Exception:
        pass

    # Stage 4: Final energy refinement to polish
    try:
        current_points = energy_based_refinement(current_points, max_iter=300, learning_rate=0.01)
    except Exception:
        pass

    return current_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    # Try multiple initialization strategies
    initial_strategies = [
        lambda: create_initial_placement(),  # Our custom enhanced strategy
        lambda: (fibonacci_sphere(14) + 1) / 2,  # Standard Fibonacci
        lambda: np.random.rand(14, 3),  # Random
        lambda: icosahedral_initialization(14)  # Icosahedral
    ]
    
    best_points = None
    best_ratio = 0
    
    # Test different initialization strategies
    for i, init_func in enumerate(initial_strategies):
        try:
            initial_points = init_func()
            ratio = min_max_ratio(initial_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = initial_points.copy()
        except Exception as e:
            continue
    
    # If no good initialization worked, use fallback
    if best_points is None:
        best_points = create_initial_placement()
    
    # Apply multi-stage optimization
    optimized_points = multi_stage_optimization(best_points)
    optimized_ratio = min_max_ratio(optimized_points)
    
    # Return the better of the two
    if optimized_ratio > best_ratio:
        return optimized_points
    else:
        return best_points

# EVOLVE-BLOCK-END