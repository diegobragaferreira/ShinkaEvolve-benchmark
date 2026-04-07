# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import SphericalVoronoi
from scipy.linalg import norm

def spherical_voronoi_initialization(n_points: int = 14) -> np.ndarray:
    """
    Initialize points using spherical Voronoi-based approach for better distribution.
    Creates points that approximately maximize the minimum distance on a sphere.
    """
    # Start with random points on unit sphere
    np.random.seed(42)
    points = np.random.randn(n_points, 3)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.where(norms == 0, 1, norms)
    
    # Apply iterative improvement using approximate Voronoi construction
    # This creates points that are roughly equidistributed
    for _ in range(50):
        # Compute pairwise distances
        distances = pdist(points)
        
        # If we have too few points or all points are identical, add some randomness
        if len(distances) < 1 or np.allclose(distances, 0):
            points = points + np.random.normal(0, 0.01, points.shape)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / np.where(norms == 0, 1, norms)
            continue
            
        # Simple force-based relaxation (repulsive forces)
        forces = np.zeros_like(points)
        for i in range(n_points):
            for j in range(i+1, n_points):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq > 1e-10:
                    force_magnitude = 1.0 / dist_sq
                    forces[i] += force_magnitude * diff
                    forces[j] -= force_magnitude * diff
        
        # Apply updates and project back to sphere
        points += 0.01 * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms == 0, 1, norms)
    
    # Scale to unit cube [0,1]^3
    points = (points + 1) / 2
    return points

def improved_fibonacci_initialization(n_points: int = 14) -> np.ndarray:
    """Improved Fibonacci sphere initialization with better spatial distribution."""
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    
    for i in range(n_points):
        # Distribute points more uniformly in z-direction
        z = 1 - (i / float(n_points - 1)) * 2  
        radius = np.sqrt(1 - z*z)
        
        # Add slight variation to avoid perfect periodicity
        theta = phi * i + (np.sin(i) * 0.1)
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / np.where(norms == 0, 1, norms)
    
    # Scale to unit cube [0,1]^3
    points = (points + 1) / 2
    return points

def initialize_points(n: int = 14, d: int = 3, num_starts: int = 8) -> np.ndarray:
    """
    Initialize points using multiple strategies for better starting configuration.
    """
    best_points = None
    best_ratio = -float('inf')
    
    strategies = [
        lambda: spherical_voronoi_initialization(n),
        lambda: improved_fibonacci_initialization(n),
        lambda: np.random.rand(n, d),
        lambda: np.random.rand(n, d) * 0.8 + 0.1,  # Centered random
        lambda: spherical_voronoi_initialization(n) * 0.8 + 0.1,  # Scaled and shifted
        lambda: improved_fibonacci_initialization(n) * 0.8 + 0.1,  # Scaled and shifted
        lambda: np.random.rand(n, d) * 0.5 + 0.25,  # Centralized
        lambda: spherical_voronoi_initialization(n) * 0.5 + 0.25  # Centered spherical
    ]
    
    for start_idx, strategy in enumerate(strategies[:num_starts]):
        try:
            points = strategy()
            distances = pdist(points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
        except Exception as e:
            continue
    
    # Fallback to random if nothing worked
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.rand(n, d)
    
    return best_points

def calculate_distance_metrics(points: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate minimum, maximum, and ratio of distances between all point pairs.
    """
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0, 0.0, 0.0
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist <= 0:
        return 0.0, 0.0, 0.0
    
    ratio = min_dist / max_dist
    return min_dist, max_dist, ratio

def voronoi_constraint_penalty(points: np.ndarray, threshold: float = 0.05) -> float:
    """
    Add penalty for Voronoi cell degeneracy (very small cells).
    """
    penalty = 0.0
    
    # Only compute if we have enough points
    if len(points) >= 4:
        try:
            # For small sets, estimate minimum distance directly
            distances = pdist(points)
            min_dist = np.min(distances) if len(distances) > 0 else 1.0
            
            # Penalize configurations where distances are too small
            if min_dist < threshold:
                penalty += 1000 * (threshold - min_dist)**2
                
        except Exception:
            pass
    
    return penalty

def objective_with_constraints(points_flat: np.ndarray) -> float:
    """
    Enhanced objective function with Voronoi-based constraints and penalty terms.
    """
    n, d = 14, 3
    points = points_flat.reshape(n, d)
    
    # Ensure points are within bounds [0,1]^3
    points = np.clip(points, 0, 1)

    # Check for degenerate configurations
    min_dist, max_dist = 0.0, 1.0
    try:
        distances = pdist(points)
        
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
        else:
            # If no distances computed, return high penalty
            return float('inf')
            
    except Exception:
        return float('inf')
    
    # Avoid division by zero
    if max_dist <= 0:
        return float('inf')

    # Base negative ratio
    base_ratio = -min_dist / max_dist
    
    # Add penalties
    penalty = 0.0
    
    # Boundary constraint penalty
    for i in range(n):
        for j in range(d):
            if points[i,j] < 0:
                penalty += 1000 * (0 - points[i,j])**2
            elif points[i,j] > 1:
                penalty += 1000 * (points[i,j] - 1)**2
    
    # Voronoi constraint penalty
    penalty += voronoi_constraint_penalty(points)
    
    # Return total objective
    return base_ratio + penalty

def adaptive_multi_stage_optimization(initial_points: np.ndarray, max_time: float = 350.0) -> np.ndarray:
    """
    Perform adaptive multi-stage optimization with alternating search strategies.
    """
    start_time = time.time()
    
    # Flatten initial points for optimization
    initial_flat = initial_points.flatten()
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0.0, 1.0)] * len(initial_flat)
    
    # Stage 1: Coarse global search
    try:
        result1 = differential_evolution(
            objective_with_constraints,
            bounds,
            maxiter=150,
            popsize=30,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=False
        )
        current_result = result1
    except Exception:
        current_result = None

    # Stage 2: Finer global search with smaller popsize
    if current_result is not None and (time.time() - start_time) < max_time * 0.6:
        try:
            result2 = differential_evolution(
                objective_with_constraints,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-6,
                mutation=(0.7, 1.0),
                recombination=0.8,
                seed=43,
                disp=False
            )
            current_result = result2
        except Exception:
            pass

    # Stage 3: Local refinement with quadratic programming approach
    if current_result is not None and (time.time() - start_time) < max_time * 0.8:
        def local_objective(x_flat):
            points = x_flat.reshape(-1, 3)
            points = np.clip(points, 0, 1)
            distances = pdist(points)
            if len(distances) == 0:
                return float('inf')
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist <= 0:
                return float('inf')
            return -min_dist / max_dist
        
        try:
            # Try L-BFGS-B first
            local_result = minimize(
                local_objective,
                current_result.x,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            current_result = local_result
        except Exception:
            pass

    # Stage 4: Alternate local search with different method
    if current_result is not None and (time.time() - start_time) < max_time * 0.9:
        def local_objective(x_flat):
            points = x_flat.reshape(-1, 3)
            points = np.clip(points, 0, 1)
            distances = pdist(points)
            if len(distances) == 0:
                return float('inf')
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist <= 0:
                return float('inf')
            return -min_dist / max_dist
        
        try:
            # Try Nelder-Mead as backup
            local_result = minimize(
                local_objective,
                current_result.x,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-8, 'xatol': 1e-8}
            )
            if local_result.fun < current_result.fun:
                current_result = local_result
        except Exception:
            pass

    # Final reshape and validation
    if current_result is not None:
        optimized_points = current_result.x.reshape(14, 3)
    else:
        optimized_points = initial_points.copy()

    # Ensure all points are within valid range
    optimized_points = np.clip(optimized_points, 0, 1)
    
    # Final check for numerical stability
    final_min_dist, final_max_dist, final_ratio = calculate_distance_metrics(optimized_points)
    
    # If ratios are very poor, do one final fallback optimization
    if final_ratio < 0.1 and (time.time() - start_time) < max_time * 0.95:
        try:
            bounds = [(0.0, 1.0)] * (14 * 3)
            final_result = differential_evolution(
                objective_with_constraints,
                bounds,
                maxiter=250,
                popsize=25,
                tol=1e-9,
                mutation=(0.8, 1.0),
                recombination=0.9,
                seed=44,
                disp=False
            )
            optimized_points = final_result.x.reshape(14, 3)
            optimized_points = np.clip(optimized_points, 0, 1)
        except Exception:
            pass

    return optimized_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a novel spherical Voronoi-based evolutionary approach.
    """
    # Phase 1: Initialize points with diverse strategies
    initial_points = initialize_points(14, 3, num_starts=8)
    
    # Phase 2: Apply geometric relaxation for better distribution
    # This uses a simplified version of the relaxation used in the previous programs
    points = initial_points.copy()
    for _ in range(20):
        n = len(points)
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq > 1e-10:
                    force_magnitude = 1.0 / dist_sq
                    forces[i] += force_magnitude * diff
                    forces[j] -= force_magnitude * diff
        points += 0.005 * forces
        points = np.clip(points, 0, 1)
    
    # Phase 3: Adaptive multi-stage optimization
    optimized_points = adaptive_multi_stage_optimization(points)
    
    # Phase 4: Final validation and adjustment
    final_points = optimized_points.copy()
    
    # Final validation
    min_dist, max_dist, ratio = calculate_distance_metrics(final_points)
    
    # If optimization didn't work well, try one more focused optimization
    if max_dist <= 0 or min_dist <= 0 or ratio < 0.15:
        try:
            bounds = [(0.0, 1.0)] * (14 * 3)
            result = differential_evolution(
                objective_with_constraints,
                bounds,
                maxiter=300,
                popsize=35,
                tol=1e-10,
                mutation=(0.8, 1.0),
                recombination=0.9,
                seed=42,
                disp=False
            )
            final_points = result.x.reshape(14, 3)
            final_points = np.clip(final_points, 0, 1)
        except Exception:
            pass

    # Final validation check
    _, _, final_ratio = calculate_distance_metrics(final_points)
    if final_ratio < 0.05:
        np.random.seed(42)
        final_points = np.random.rand(14, 3)
    
    return final_points

# EVOLVE-BLOCK-END