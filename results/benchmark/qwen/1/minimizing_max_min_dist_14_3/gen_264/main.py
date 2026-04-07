# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
from scipy.stats import qmc
import time
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def spherical_voronoi_initialization(n_points: int) -> np.ndarray:
        """Initialize points using spherical Voronoi distribution principles."""
        # Start with a basic icosahedral distribution
        # Vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        # Take first 12 vertices and add 2 more points
        points = vertices[:12].copy()
        
        # Add two more points (north and south poles) for 14 total
        points = np.vstack([points, [0, 0, 1], [0, 0, -1]])
        
        # Add small random perturbation to break symmetry
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.02, points.shape)
        points = points + perturbation
        
        # Re-normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        return points

    def sobol_initialization(n_points: int, seed: int = 42) -> np.ndarray:
        """Initialize points using Sobol sequence for better space-filling properties."""
        try:
            sampler = qmc.Sobol(d=3, seed=seed)
            points = sampler.random(n=n_points)
            # Scale to [-1, 1]^3
            points = points * 2 - 1
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            points = points / norms
            return points
        except Exception:
            # Fallback to random
            return np.random.uniform(-1, 1, (n_points, 3))

    def fibonacci_spiral_on_sphere(n_points: int) -> np.ndarray:
        """Generate points on sphere using Fibonacci spiral method."""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def voronoi_uniformity_score(points: np.ndarray) -> float:
        """Calculate uniformity score based on Voronoi cell area variance."""
        try:
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points, radius=1.0, center=np.zeros(3))
            cell_areas = sv.voronoi_regions_area()
            # Lower variance means more uniform distribution
            return np.var(cell_areas)
        except:
            return np.inf

    def hybrid_objective(x, phase=1):
        """Hybrid objective function combining distance ratio with uniformity."""
        points = x.reshape(-1, 3)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)
        
        # Compute distance matrix
        dist_matrix = cdist(points, points)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_dist = np.min(dist_matrix)
        max_dist = np.max(dist_matrix)
        
        if max_dist == 0:
            ratio = 0.0
        else:
            ratio = min_dist / max_dist
        
        # Calculate uniformity penalty
        uniformity_penalty = voronoi_uniformity_score(points)
        
        # Phase-dependent weights
        if phase == 1:  # Initial coarse phase
            ratio_weight = 1.0
            uniformity_weight = 0.01
        elif phase == 2:  # Middle refinement
            ratio_weight = 1.0
            uniformity_weight = 0.05
        else:  # Fine-tuning phase
            ratio_weight = 1.0
            uniformity_weight = 0.1
            
        # Combined objective (minimize negative of combined value)
        return -(ratio * ratio_weight - uniformity_weight * uniformity_penalty)

    def sphere_constraint(x):
        """Constraint function ensuring all points lie on unit sphere."""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0

    def multi_stage_optimization(initial_points: np.ndarray) -> tuple:
        """Perform multi-stage optimization with progressive refinement."""
        # Stage 1: Coarse optimization with relaxed tolerances
        x0 = initial_points.flatten()
        cons = {'type': 'eq', 'fun': sphere_constraint}
        
        result1 = minimize(
            lambda x: hybrid_objective(x, phase=1),
            x0,
            method='L-BFGS-B',
            options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': 200}
        )
        
        # Stage 2: Medium refinement 
        if result1.success:
            x0_refine = result1.x
            result2 = minimize(
                lambda x: hybrid_objective(x, phase=2),
                x0_refine,
                method='L-BFGS-B',
                options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 300}
            )
        else:
            result2 = result1
            
        # Stage 3: Fine optimization
        if result2.success:
            x0_fine = result2.x
            result3 = minimize(
                lambda x: hybrid_objective(x, phase=3),
                x0_fine,
                method='L-BFGS-B',
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 500}
            )
        else:
            result3 = result2
            
        return result3.x.reshape(-1, 3) if result3.success else initial_points

    # Multi-start optimization with diverse initializations
    best_ratio = -np.inf
    best_points = None
    
    # Different initialization strategies
    initial_strategies = [
        ("voronoi", lambda: spherical_voronoi_initialization(14)),
        ("sobol", lambda: sobol_initialization(14, seed=42)),
        ("fibonacci", lambda: fibonacci_spiral_on_sphere(14)),
        ("random", lambda: np.random.uniform(-1, 1, (14, 3)))
    ]
    
    # Add perturbed versions for diversity
    np.random.seed(42)
    for i, (name, func) in enumerate(initial_strategies):
        if name == "voronoi":
            base_points = func()
        elif name == "sobol":
            base_points = func()
        elif name == "fibonacci":
            base_points = func()
        else:  # random
            base_points = func()
            
        # Create perturbed version
        perturbed = base_points + np.random.normal(0, 0.03, base_points.shape)
        # Normalize
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.where(norms > 0, norms, 1)
        
        # Optimize both original and perturbed versions
        for j, points in enumerate([base_points, perturbed]):
            try:
                optimized_points = multi_stage_optimization(points)
                
                # Calculate final ratio
                dist_matrix = cdist(optimized_points, optimized_points)
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)
                max_dist = np.max(dist_matrix)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception as e:
                warnings.warn(f"Optimization failed for {name}_perturbed{j}: {str(e)}")
                continue
    
    # If no good solution found, fall back to Voronoi initialization
    if best_points is None:
        fallback_points = spherical_voronoi_initialization(14)
        return fallback_points
        
    return best_points

# EVOLVE-BLOCK-END