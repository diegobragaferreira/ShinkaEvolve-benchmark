# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def objective(x):
        # Reshape x into points array
        points = x.reshape(-1, 3)

        # Calculate distance matrix
        distances = cdist(points, points)

        # Set diagonal to large value to ignore self-distances
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Return negative ratio to maximize (since we minimize in scipy)
        # Handle case where max_dist is 0 (should never happen with distinct points)
        if max_dist > 1e-10:
            return -min_dist / max_dist
        else:
            return 0

    def spherical_fibonacci_points(n):
        """Generate points on sphere using Fibonacci spiral method"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2

        for i in range(n):
            # Latitude
            phi = np.arccos(1 - 2*i/(n-1))
            # Longitude
            theta = 2 * np.pi * i / golden_ratio

            # Convert to Cartesian coordinates
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            points.append([x, y, z])

        return np.array(points)

    def create_cube_grid_points():
        """Create points arranged in a 3D grid pattern"""
        # Create a roughly uniform distribution in cube
        grid_size = 3  # 3x3x3 grid gives 27 points, we'll take 14
        coords = np.linspace(0, 1, grid_size)
        grid_points = []

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    grid_points.append([coords[i], coords[j], coords[k]])

        return np.array(grid_points[:14])

    def spherical_voronoi_points(n):
        """Generate points using spherical Voronoi diagram for even distribution"""
        # Start with random points on sphere
        points = np.random.randn(n, 3)
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Use spherical Voronoi to get more uniform distribution
        try:
            sv = SphericalVoronoi(points)
            # Get the centers of the Voronoi cells as new candidates
            voronoi_centers = sv.vertices
            # Normalize to unit sphere again
            voronoi_centers = voronoi_centers / np.linalg.norm(voronoi_centers, axis=1, keepdims=True)

            # Take first n points, or generate more if needed
            if len(voronoi_centers) >= n:
                selected = voronoi_centers[:n]
            else:
                # If not enough, use a combination of original and Voronoi points
                selected = np.vstack([voronoi_centers, points[:n-len(voronoi_centers)]])

            return selected
        except:
            # Fallback to fibonacci if spherical voronoi fails
            return spherical_fibonacci_points(n)

    def initialize_multiple_strategies():
        """Initialize using multiple strategies and return the best"""
        strategies = []

        # Strategy 1: Spherical Fibonacci (original)
        np.random.seed(42)
        fib_points = spherical_fibonacci_points(14)
        fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("fibonacci", fib_points))

        # Strategy 2: Spherical Voronoi
        np.random.seed(42)
        voronoi_points = spherical_voronoi_points(14)
        voronoi_points = (voronoi_points + 1) / 2  # Normalize to [0,1]^3
        strategies.append(("voronoi", voronoi_points))

        # Strategy 3: Random uniform points
        np.random.seed(42)
        random_points = np.random.rand(14, 3)
        strategies.append(("random", random_points))

        # Strategy 4: Cube grid points
        cube_points = create_cube_grid_points()
        strategies.append(("cube_grid", cube_points))

        # Strategy 5: KMeans clustering approach
        np.random.seed(42)
        kmeans_points = np.random.rand(100, 3)
        kmeans = KMeans(n_clusters=14, random_state=42, n_init=10)
        kmeans.fit(kmeans_points)
        kmeans_centers = kmeans.cluster_centers_
        strategies.append(("kmeans", kmeans_centers))

        # Strategy 6: Perturbed Fibonacci
        np.random.seed(42)
        perturbed = fib_points + np.random.normal(0, 0.03, (14, 3))
        # Clamp to [0,1]
        perturbed = np.clip(perturbed, 0, 1)
        strategies.append(("perturbed", perturbed))

        # Evaluate all strategies using a fast approximation
        best_strategy = None
        best_ratio = -np.inf

        for name, points in strategies:
            # Calculate ratio for this initialization (fast version)
            distances = cdist(points, points)
            np.fill_diagonal(distances, np.inf)

            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist > 1e-10:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_strategy = points.copy()

        return best_strategy

    def adaptive_local_refinement(points, max_iter=50):
        """Apply multiple rounds of local refinement to polish the solution"""
        best_points = points.copy()
        best_ratio = -np.inf

        # Try multiple local optimization approaches
        for attempt in range(3):
            # Random perturbation for diversity
            np.random.seed(42 + attempt)
            perturbed = points + np.random.normal(0, 0.01, points.shape)
            perturbed = np.clip(perturbed, 0, 1)

            # Try L-BFGS-B optimization
            try:
                x0_refine = perturbed.flatten()

                def obj_for_lbfgs(x):
                    points_refined = x.reshape(-1, 3)
                    distances = cdist(points_refined, points_refined)
                    np.fill_diagonal(distances, np.inf)

                    min_dist = np.min(distances)
                    max_dist = np.max(distances)

                    if max_dist > 1e-10:
                        return -min_dist / max_dist
                    else:
                        return 0

                result_refine = minimize(
                    obj_for_lbfgs,
                    x0_refine,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * 42,
                    options={'ftol': 1e-8, 'gtol': 1e-8},
                    tol=1e-8
                )

                refined_points = result_refine.x.reshape(-1, 3)
                refined_points = np.clip(refined_points, 0, 1)

                # Calculate ratio for refined points
                distances = cdist(refined_points, refined_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 1e-10:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()

            except:
                pass  # Continue if optimization fails

        return best_points

    # Initialize multiple strategies and pick the best
    init_points = initialize_multiple_strategies()

    # Flatten initial points for optimization
    x0 = init_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1)] * 42  # 14 points * 3 coordinates each

    # Run differential evolution optimization with enhanced parameters
    start_time = time.time()

    # Multiple restarts with different seeds for better exploration
    best_result = None
    best_ratio = -np.inf

    # Try 5 different random seeds for better exploration
    for seed_val in [42, 123, 456, 789, 999]:
        np.random.seed(seed_val)

        # Adaptive population sizing based on convergence monitoring
        def adaptive_de_callback(xk, convergence):
            # This would normally be used to monitor progress, but
            # we'll implement a simpler version that adjusts popsize
            pass

        # Set initial population size
        current_popsize = 20

        # Try different population sizes if needed
        result = differential_evolution(
            objective,
            bounds,
            seed=seed_val,
            maxiter=100,  # Reduced iterations since we'll refine locally
            popsize=current_popsize,   # Larger population for better exploration
            tol=1e-9,
            mutation=(0.5, 1),
            recombination=0.7,
            callback=None
        )

        # Check if this result is better
        if -result.fun > best_ratio:
            best_ratio = -result.fun
            best_result = result

    # Extract optimized points
    optimized_points = best_result.x.reshape(-1, 3)

    # Apply adaptive local refinement
    refined_points = adaptive_local_refinement(optimized_points)

    # Final check and selection between original and refined
    distances_orig = cdist(optimized_points, optimized_points)
    np.fill_diagonal(distances_orig, np.inf)
    orig_min = np.min(distances_orig)
    orig_max = np.max(distances_orig)
    orig_ratio = orig_min / orig_max if orig_max > 1e-10 else 0

    distances_refined = cdist(refined_points, refined_points)
    np.fill_diagonal(distances_refined, np.inf)
    refined_min = np.min(distances_refined)
    refined_max = np.max(distances_refined)
    refined_ratio = refined_min / refined_max if refined_max > 1e-10 else 0

    final_points = refined_points if refined_ratio > orig_ratio else optimized_points

    # Final clipping to ensure bounds are respected
    final_points = np.clip(final_points, 0, 1)

    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Final ratio: {max(orig_ratio, refined_ratio):.6f}")

    return final_points

# EVOLVE-BLOCK-END