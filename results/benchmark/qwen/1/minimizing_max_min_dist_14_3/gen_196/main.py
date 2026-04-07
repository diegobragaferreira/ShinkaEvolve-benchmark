# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import SphericalVoronoi
import time
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical Voronoi-based initialization combined with evolutionary refinement.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    random.seed(42)

    n = 14
    d = 3

    # Generate initial points using spherical Voronoi approach for better geometric distribution
    def spherical_voronoi_init(samples=14):
        # Generate points on unit sphere using spherical Voronoi principles
        # Start with random points and use a variant of Lloyd's algorithm
        points = np.random.rand(samples, 3) - 0.5
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Apply a few iterations of Lloyd relaxation to improve distribution
        for _ in range(10):
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(points, radius=1.0)
            
            # Compute centroids of Voronoi cells (Lloyd relaxation step)
            new_points = []
            for cell in sv:
                # Get indices of points in this cell
                indices = cell.vertices
                if len(indices) > 0:
                    # Compute centroid of the polygon on sphere
                    centroid = np.mean(cell.vertices, axis=0)
                    # Project back to sphere
                    centroid = centroid / np.linalg.norm(centroid)
                    new_points.append(centroid)
                else:
                    # Fallback to original point
                    new_points.append(points[len(new_points)])
            
            points = np.array(new_points)
        
        return points

    # Alternative initialization using spherical Fibonacci with enhancement
    def enhanced_fibonacci_sphere(samples=14):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(samples):
            # Distribute points more uniformly across the sphere
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Add controlled jitter for better distribution
            theta = phi * i + np.random.normal(0, 0.08)
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)

    # Enhanced initialization with multiple strategies
    def enhanced_initialization(samples=14):
        points = []
        
        # Strategy 1: Spherical Voronoi for better geometric properties
        voronoi_points = spherical_voronoi_init(samples // 2)
        points.extend(voronoi_points)
        
        # Strategy 2: Enhanced Fibonacci for diversity
        fib_points = enhanced_fibonacci_sphere(samples // 2)
        points.extend(fib_points)
        
        # Strategy 3: Random points for global exploration
        random_points = np.random.rand(samples - len(points), 3) - 0.5
        random_points = random_points / np.linalg.norm(random_points, axis=1, keepdims=True)
        points.extend(random_points)
        
        return np.array(points[:samples])

    # Normalize points to unit cube [0,1]^3
    def normalize_to_cube(points):
        centered = points - np.mean(points, axis=0)
        max_coord = np.max(np.abs(centered))
        if max_coord > 0:
            scaled = centered / max_coord * 0.5
        else:
            scaled = centered
        normalized = scaled + 0.5
        return normalized

    # Calculate ratio with proper validation
    def calculate_ratio(points):
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        # Filter out near-zero distances
        distances = distances[distances > 1e-12]
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max > 0:
            ratio = d_min / d_max
        else:
            ratio = 0.0
            
        return ratio, d_min, d_max

    # Objective function with geometric regularization
    def objective(x_flat):
        points = x_flat.reshape((n, d))
        
        # Ensure points remain on unit sphere for geometric consistency
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / np.where(norms > 0, norms, 1)
        
        distances = pdist(normalized_points)
        # Filter out near-zero distances
        distances = distances[distances > 1e-12]
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        d_mean = np.mean(distances)
        d_var = np.var(distances)
        
        if d_max == 0:
            return -np.inf
            
        # Regularized objective: maximize ratio while penalizing distance variance
        ratio = d_min / d_max
        variance_penalty = 0.1 * (d_var / (d_mean * d_mean + 1e-12))  # Normalized variance penalty
        
        # Return negative because we're minimizing
        return -(ratio - variance_penalty)

    # Simulated annealing-like perturbation with adaptive cooling
    def adaptive_perturbation(current_points, iteration, max_iterations):
        # Cooling schedule for perturbation magnitude
        cooling_factor = 0.95 ** iteration
        base_magnitude = 0.05 * cooling_factor
        
        # Add controlled perturbation
        perturbation = np.random.normal(0, base_magnitude, current_points.shape)
        perturbed_points = current_points + perturbation
        
        # Project back to sphere
        norms = np.linalg.norm(perturbed_points, axis=1, keepdims=True)
        perturbed_points = perturbed_points / np.where(norms > 0, norms, 1)
        
        return perturbed_points

    # Multi-start optimization with evolutionary refinement
    best_ratio = -np.inf
    best_points = None

    # Multiple initialization strategies
    init_strategies = [
        lambda s: spherical_voronoi_init(s),
        lambda s: enhanced_fibonacci_sphere(s),
        lambda s: enhanced_initialization(s),
        lambda s: np.random.rand(s, 3) - 0.5,
    ]
    
    # Normalize random points to sphere
    init_strategies[3] = lambda s: init_strategies[3](s) / np.linalg.norm(init_strategies[3](s), axis=1, keepdims=True)

    # Number of optimization runs with adaptive parameters
    num_runs = 35  # Increased for better exploration
    max_iterations_per_run = 1000

    for run_idx in range(num_runs):
        # Select initialization strategy
        if run_idx < len(init_strategies):
            init_func = init_strategies[run_idx]
        else:
            init_func = lambda s: np.random.rand(s, 3) - 0.5
            init_func = lambda s: init_func(s) / np.linalg.norm(init_func(s), axis=1, keepdims=True)

        # Get initial points
        initial_points = init_func(n)

        # Normalize to unit cube [0,1]^3
        initial_points = normalize_to_cube(initial_points)

        # Flatten for optimization
        initial_flat = initial_points.flatten()

        # Optimization bounds: [0,1] for all coordinates
        bounds = [(0, 1) for _ in range(n * d)]

        # Phase 1: Coarse global optimization with L-BFGS-B
        try:
            # Coarse optimization
            result_coarse = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            # Extract intermediate points
            if result_coarse.success:
                intermediate_points = result_coarse.x.reshape((n, d))
                # Keep points on sphere
                norms = np.linalg.norm(intermediate_points, axis=1, keepdims=True)
                intermediate_points = intermediate_points / np.where(norms > 0, norms, 1)
                intermediate_points = normalize_to_cube(intermediate_points)
            else:
                # Fallback to initial points
                intermediate_points = initial_points.copy()
                
        except Exception:
            intermediate_points = initial_points.copy()

        # Phase 2: Fine-grained refinement using hybrid approach
        current_points = intermediate_points.copy()
        current_flat = current_points.flatten()
        
        try:
            # Fine-tune with SLSQP
            result_fine = minimize(
                objective,
                current_flat,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result_fine.success:
                optimized_points = result_fine.x.reshape((n, d))
                # Keep points on sphere
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                optimized_points = optimized_points / np.where(norms > 0, norms, 1)
                optimized_points = normalize_to_cube(optimized_points)
            else:
                optimized_points = current_points.copy()
                
        except Exception:
            optimized_points = current_points.copy()

        # Calculate the actual ratio for this optimization run
        ratio, _, _ = calculate_ratio(optimized_points)

        # Update best solution
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()

    # If we didn't find a good solution, return the best initialization
    if best_points is None:
        initial_points = enhanced_initialization(n)
        initial_points = normalize_to_cube(initial_points)
        return initial_points

    # Apply final evolution refinement using simulated annealing-inspired approach
    refined_points = best_points.copy()
    
    # Evolutionary cooling schedule
    for iteration in range(100):  # Limited iterations to maintain time budget
        # Adaptive perturbation with cooling
        perturbed_points = adaptive_perturbation(refined_points, iteration, 100)
        
        # Test this perturbed version
        try:
            perturbed_flat = perturbed_points.flatten()
            test_result = minimize(
                objective,
                perturbed_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-11}
            )
            
            if test_result.success:
                test_points = test_result.x.reshape((n, d))
                # Keep points on sphere
                norms = np.linalg.norm(test_points, axis=1, keepdims=True)
                test_points = test_points / np.where(norms > 0, norms, 1)
                test_points = normalize_to_cube(test_points)
                
                # Evaluate improvement
                test_ratio, _, _ = calculate_ratio(test_points)
                if test_ratio > best_ratio:
                    refined_points = test_points.copy()
                    best_ratio = test_ratio
                    
        except Exception:
            pass

        # Break early if convergence is good
        if iteration > 30 and best_ratio > 0.45:
            break

    # Final validation to make sure we have a valid solution
    final_distances = pdist(refined_points)
    final_distances = final_distances[final_distances > 1e-12]

    if len(final_distances) > 0 and np.min(final_distances) > 1e-12:
        return refined_points
    else:
        return best_points

# EVOLVE-BLOCK-END