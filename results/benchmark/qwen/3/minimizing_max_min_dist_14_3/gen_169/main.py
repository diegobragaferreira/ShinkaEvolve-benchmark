# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective_function(x):
        # Reshape flat array back to 14x3 points
        points = x.reshape((14, 3))

        # Compute pairwise distances efficiently
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -min_dist / max_dist

    def penalty_objective(x, penalty_weight=1e6):
        """Objective function with penalty for boundary violations"""
        points = x.reshape((14, 3))

        # Calculate base objective
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return -np.inf

        ratio = -min_dist / max_dist

        # Add penalty for points outside [0,1]^3 bounds
        penalty = 0
        for i in range(14):
            for j in range(3):
                coord = points[i, j]
                if coord < 0:
                    penalty += penalty_weight * (0 - coord) ** 2
                elif coord > 1:
                    penalty += penalty_weight * (coord - 1) ** 2

        return ratio - penalty

    def fibonacci_sphere_sampling(n):
        """Generate points on a unit sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # golden angle

        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def spherical_to_cube_projection(spherical_points):
        """Project spherical points to unit cube [0,1]^3"""
        # Normalize to unit sphere
        norms = np.linalg.norm(spherical_points, axis=1, keepdims=True)
        normalized = spherical_points / norms
        
        # Project to cube: map from [-1,1]^3 to [0,1]^3
        cube_points = (normalized + 1) / 2
        return cube_points

    def initialize_multiple_strategies():
        """Initialize points using multiple strategies and choose the best"""
        strategies = []

        # Strategy 1: Fibonacci sphere
        np.random.seed(42)
        points1 = fibonacci_sphere_sampling(14)
        perturbations = np.random.normal(0, 0.02, points1.shape)
        points1 += perturbations
        norms = np.linalg.norm(points1, axis=1, keepdims=True)
        points1 = points1 / norms
        points1 *= 0.8
        points1 = spherical_to_cube_projection(points1)
        strategies.append(("fibonacci", points1))

        # Strategy 2: Random uniform
        np.random.seed(42)
        points2 = np.random.rand(14, 3)
        strategies.append(("random_uniform", points2))

        # Strategy 3: K-means clustering approach (spread out)
        np.random.seed(42)
        points3 = np.random.rand(14, 3)
        try:
            kmeans = KMeans(n_clusters=14, random_state=42, n_init=10)
            labels = kmeans.fit_predict(points3)
            # Use centroids as initial points
            points3 = kmeans.cluster_centers_
        except:
            pass
        strategies.append(("kmeans", points3))

        # Evaluate all strategies
        best_strategy = None
        best_ratio = -np.inf

        for name, points in strategies:
            try:
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_strategy = points.copy()
            except:
                continue

        return best_strategy.flatten() if best_strategy is not None else np.random.rand(42)

    def adaptive_differential_evolution(initial_points, max_iter=1000):
        """Perform differential evolution with adaptive population sizing"""
        bounds = [(0, 1)] * 14 * 3

        # Try different population sizes to adaptively improve results
        pop_sizes = [15, 20, 25]
        best_result = None
        best_ratio = -np.inf

        for popsize in pop_sizes:
            try:
                result = differential_evolution(
                    penalty_objective,
                    bounds,
                    seed=42,
                    maxiter=min(max_iter, 1000),
                    popsize=popsize,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    tol=1e-8,
                    callback=None
                )

                # Calculate ratio for this result
                points = result.x.reshape((14, 3))
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = result

            except Exception:
                continue

        if best_result is not None:
            return best_result.x.reshape((14, 3))
        else:
            return initial_points.reshape((14, 3))

    def local_refinement(points):
        """Apply local optimization refinement"""
        try:
            # Use L-BFGS-B for fine-tuning with stricter tolerances
            result = minimize(
                objective_function,
                points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * 14 * 3,
                options={'ftol': 1e-9, 'gtol': 1e-9}
            )

            refined_points = result.x.reshape((14, 3))
            return refined_points

        except Exception:
            return points

    def validate_and_correct_bounds(points):
        """Ensure all points are within [0,1]^3 bounds"""
        corrected_points = np.clip(points, 0, 1)
        return corrected_points

    # Alternative approach: Sphere-focused optimization
    def sphere_optimization_approach():
        """Use spherical optimization approach for better distribution"""
        # Start with spherical Fibonacci points
        np.random.seed(42)
        sph_points = fibonacci_sphere_sampling(14)
        
        # Perturb slightly to break symmetries
        perturb = np.random.normal(0, 0.05, sph_points.shape)
        sph_points += perturb
        
        # Normalize to unit sphere
        norms = np.linalg.norm(sph_points, axis=1, keepdims=True)
        sph_points = sph_points / norms
        
        # Project to cube
        cube_points = spherical_to_cube_projection(sph_points)
        
        # Refine with evolutionary approach
        bounds = [(0, 1)] * 14 * 3
        
        # Run multiple differential evolution optimizations with different settings
        best_result = None
        best_ratio = -np.inf
        
        # Different population sizes for diversity
        pop_sizes = [15, 20, 25]
        for popsize in pop_sizes:
            try:
                result = differential_evolution(
                    penalty_objective,
                    bounds,
                    seed=42,
                    maxiter=200,
                    popsize=popsize,
                    mutation=(0.5, 1.0),
                    recombination=0.7,
                    tol=1e-10,
                    disp=False
                )
                
                # Calculate ratio for this result
                points = result.x.reshape((14, 3))
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = result
                        
            except Exception:
                continue
        
        if best_result is not None:
            final_points = best_result.x.reshape((14, 3))
        else:
            final_points = cube_points
            
        # Final local refinement
        final_points = local_refinement(final_points)
        final_points = validate_and_correct_bounds(final_points)
        
        return final_points

    # Initialize with better strategies
    initial_points = initialize_multiple_strategies()

    # Multiple optimization attempts with different settings
    best_points = None
    best_ratio = -np.inf

    # Try the sphere optimization approach
    try:
        sphere_optimized = sphere_optimization_approach()
        
        # Calculate ratio for sphere approach
        distances = pdist(sphere_optimized)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist > 0:
            ratio = min_dist / max_dist
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = sphere_optimized.copy()
                
    except Exception:
        pass

    # Try adaptive differential evolution multiple times
    for attempt in range(3):  # Three different optimization attempts
        try:
            # Global optimization
            global_optimized = adaptive_differential_evolution(initial_points, max_iter=500)

            # Local refinement
            local_optimized = local_refinement(global_optimized)

            # Validate bounds
            final_points = validate_and_correct_bounds(local_optimized)

            # Calculate final ratio
            distances = pdist(final_points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()

        except Exception:
            continue

    # If no good result was found, return initial points
    if best_points is None:
        return initial_points.reshape((14, 3))

    return best_points

# EVOLVE-BLOCK-END