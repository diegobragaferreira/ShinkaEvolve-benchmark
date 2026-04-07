# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math
from typing import Tuple, Optional

def _generate_icosahedron_points() -> np.ndarray:
    """Generate initial points using icosahedron vertices."""
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    vertices = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]
    points = np.array(vertices, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    return points / norms[:, np.newaxis]

def _add_remaining_points(base_points: np.ndarray, num_points: int) -> np.ndarray:
    """Add remaining points using spherical coordinate distribution."""
    points = base_points.copy()
    remaining = num_points - len(base_points)
    
    if remaining <= 0:
        return points[:num_points]
        
    for i in range(remaining):
        theta = math.acos(1 - 2 * (i / (remaining - 1)))
        phi = math.sqrt(num_points * math.pi) * theta
        
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points = np.vstack([points, [x, y, z]])
    return points

def _initialize_points(num_points: int = 14) -> np.ndarray:
    """Create initial point configuration using hybrid approach."""
    base_points = _generate_icosahedron_points()
    points = _add_remaining_points(base_points, num_points)
    
    # Add jitter to break symmetry
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, points.shape)
    points += noise
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1)
    return points / norms[:, np.newaxis]

def _calculate_distance_ratio(points_flat: np.ndarray) -> float:
    """Calculate the ratio of minimum to maximum distance."""
    points = points_flat.reshape(-1, 3)
    distances = squareform(pdist(points))
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    if max_dist == 0:
        return 0
    return min_dist / max_dist

def _objective_function(points_flat: np.ndarray) -> float:
    """Minimize negative of distance ratio (since we want to maximize)."""
    return -_calculate_distance_ratio(points_flat)

def _create_constraints(num_points: int) -> list:
    """Create constraint functions for unit sphere."""
    def constraint_func(x):
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    constraints = []
    for i in range(num_points):
        constraints.append({'type': 'eq', 'fun': lambda x, i=i: constraint_func(x)[i]})
    return constraints

def _optimize_phase(x0: np.ndarray, bounds: list, maxiter: int, ftol: float, gtol: float) -> np.ndarray:
    """Perform optimization with specific parameters."""
    constraints = _create_constraints(len(x0) // 3)
    
    try:
        result = minimize(
            _objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
            tol=ftol
        )
        return result.x if result.success else x0
    except Exception:
        return x0

def _progressive_optimization(x0: np.ndarray) -> np.ndarray:
    """Perform multi-phase optimization with progressive tightening."""
    # Phase 1: Coarse optimization
    bounds = [(-1.2, 1.2)] * len(x0)
    x1 = _optimize_phase(x0, bounds, 200, 1e-4, 1e-4)
    
    # Phase 2: Medium optimization
    bounds = [(-1.1, 1.1)] * len(x0)
    x2 = _optimize_phase(x1, bounds, 300, 1e-6, 1e-6)
    
    # Phase 3: Fine optimization
    bounds = [(-1.05, 1.05)] * len(x0)
    x3 = _optimize_phase(x2, bounds, 500, 1e-8, 1e-8)
    
    return x3

def _run_single_optimization(restart_seed: int) -> Tuple[float, np.ndarray]:
    """Run a single optimization with given seed."""
    np.random.seed(restart_seed)
    
    # Initialize points
    initial_points = _initialize_points(14)
    x0 = initial_points.flatten()
    
    # Add small random perturbation
    perturbation = np.random.normal(0, 0.01, x0.shape)
    x0 += perturbation
    
    # Optimize
    try:
        optimized_points = _progressive_optimization(x0)
        
        # Calculate final ratio
        ratio = _calculate_distance_ratio(optimized_points)
        return ratio, optimized_points
    except Exception:
        return -np.inf, x0

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    best_ratio = -np.inf
    best_points = None
    
    # Multi-start optimization with different initializations
    for restart in range(8):
        ratio, points = _run_single_optimization(42 + restart)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Fallback to initialization if no good solution found
    if best_points is None:
        initial_points = _initialize_points(14)
        best_points = initial_points.flatten()
    
    return best_points.reshape(-1, 3)

# EVOLVE-BLOCK-END
