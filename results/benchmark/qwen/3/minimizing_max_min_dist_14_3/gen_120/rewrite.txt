# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import time
import warnings
warnings.filterwarnings('ignore')
import numba
from numba import jit

@jit(nopython=True)
def compute_distances_numba(points):
    """Optimized distance computation using Numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

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

    def penalty_objective(x, penalty_weight=1e6):
        """Objective function with penalty for boundary violations"""
        points = x.reshape((-1, 3))

        # Calculate base objective
        ratio = -objective(x)
        base_obj = ratio

        # Add penalty for points outside [0,1]^3 bounds
        penalty = 0
        for i in range(14):
            for j in range(3):
                coord = points[i, j]
                if coord < 0:
                    penalty += penalty_weight * (0 - coord) ** 2
                elif coord > 1:
                    penalty += penalty_weight * (coord - 1) ** 2

        return base_obj + penalty

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

        # Strategy 7: Improved perturbed random
        np.random.seed(42)
        perturbed_random = random_points + np.random.normal(0, 0.02, (14, 3))
        perturbed_random = np.clip(perturbed_random, 0, 1)
        strategies.append(("perturbed_random", perturbed_random))

        # Strategy 8: Modified Fibonacci with better spread
        np.random.seed(42)
        modified_fib = spherical_fibonacci_points(14)
        # Scale and shift to avoid edge cases
        modified_fib = modified_fib * 0.8 + 0.1
        strategies.append(("modified_fib", modified_fib))

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

        # Try multiple local optimization approaches with different settings
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
                    options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 500},
                    tol=1e-9
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

    def adaptive_de_strategy(best_points):
        """Adaptive differential evolution with multiple configurations"""
        bounds = [(0, 1)] * 42  # 14 points * 3 coordinates each
        
        # Multiple DE configurations to try
        configs = [
            {'popsize': 15, 'mutation': (0.5, 1.0), 'recombination': 0.7, 'maxiter': 120},
            {'popsize': 20, 'mutation': (0.7, 1.0), 'recombination': 0.8, 'maxiter': 120},
            {'popsize': 25, 'mutation': (0.8, 1.0), 'recombination': 0.9, 'maxiter': 150},
            {'popsize': 30, 'mutation': (0.9, 1.0), 'recombination': 0.95, 'maxiter': 150}
        ]

        best_solution = best_points.copy()
        best_ratio = -np.inf

        for i, config in enumerate(configs):
            try:
                result = differential_evolution(
                    penalty_objective,
                    bounds,
                    seed=42 + i,
                    maxiter=config['maxiter'],
                    popsize=config['popsize'],
                    mutation=config['mutation'],
                    recombination=config['recombination'],
                    tol=1e-9,
                    callback=None
                )

                # Evaluate result
                points = result.x.reshape((14, 3))
                distances = cdist(points, points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)

                if max_dist > 1e-10:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_solution = points.copy()

            except Exception as e:
                continue

        return best_solution

    # Initialize multiple strategies and pick the best
    init_points = initialize_multiple_strategies()

    # Flatten initial points for optimization
    x0 = init_points.flatten()

    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1)] * 42  # 14 points * 3 coordinates each

    # Run adaptive differential evolution optimization
    start_time = time.time()
    
    # First stage: Global optimization with adaptive DE
    global_optimized = adaptive_de_strategy(init_points)

    # Second stage: Local refinement
    local_optimized = adaptive_local_refinement(global_optimized)

    # Third stage: Final polishing with different local optimization
    try:
        # Try one more round with very tight tolerances
        final_x0 = local_optimized.flatten()
        
        def final_obj(x):
            points_final = x.reshape(-1, 3)
            distances = cdist(points_final, points_final)
            np.fill_diagonal(distances, np.inf)
            
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist > 1e-10:
                return -min_dist / max_dist
            else:
                return 0
                
        final_result = minimize(
            final_obj,
            final_x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * 42,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000},
            tol=1e-12
        )
        
        final_points = final_result.x.reshape(-1, 3)
        final_points = np.clip(final_points, 0, 1)
        
    except:
        final_points = local_optimized

    # Final validation and bound checking
    final_points = np.clip(final_points, 0, 1)

    # Final evaluation
    distances_final = cdist(final_points, final_points)
    np.fill_diagonal(distances_final, np.inf)
    final_min = np.min(distances_final)
    final_max = np.max(distances_final)
    final_ratio = final_min / final_max if final_max > 1e-10 else 0

    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    print(f"Final ratio: {final_ratio:.6f}")

    return final_points

# EVOLVE-BLOCK-END