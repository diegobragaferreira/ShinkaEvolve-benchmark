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

    def spherical_voronoi_points(n):
        """Generate points using spherical Voronoi diagram for even distribution"""
        # Start with random points on sphere
        np.random.seed(42)
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
            # Fallback to random points if spherical voronoi fails
            return points

    def fibonacci_sphere_points(n):
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

    def improved_initialization():
        """Create multiple initialization strategies and select the best"""
        strategies = []
        np.random.seed(42)
        
        # Strategy 1: Spherical Voronoi points
        voronoi_points = spherical_voronoi_points(14)
        # Normalize to [0,1]^3
        voronoi_points = (voronoi_points + 1) / 2
        strategies.append(("voronoi", voronoi_points))
        
        # Strategy 2: Fibonacci sphere points
        fib_points = fibonacci_sphere_points(14)
        # Normalize to [0,1]^3
        fib_points = (fib_points + 1) / 2
        strategies.append(("fibonacci", fib_points))
        
        # Strategy 3: Random uniform points
        random_points = np.random.rand(14, 3)
        strategies.append(("random", random_points))
        
        # Strategy 4: KMeans clustering approach
        kmeans_points = np.random.rand(100, 3)
        kmeans = KMeans(n_clusters=14, random_state=42, n_init=10)
        kmeans.fit(kmeans_points)
        kmeans_centers = kmeans.cluster_centers_
        strategies.append(("kmeans", kmeans_centers))

        # Strategy 5: Perturbed Voronoi
        perturbed_voronoi = voronoi_points + np.random.normal(0, 0.02, (14, 3))
        perturbed_voronoi = np.clip(perturbed_voronoi, 0, 1)
        strategies.append(("perturbed_voronoi", perturbed_voronoi))
        
        # Evaluate all strategies
        best_strategy = None
        best_ratio = -np.inf

        for name, points in strategies:
            # Calculate ratio for this initialization
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

        # Multiple refinement attempts
        for attempt in range(4):
            # Different perturbation strategies
            np.random.seed(42 + attempt)
            
            # Strategy 1: Small random perturbation
            perturbed = points + np.random.normal(0, 0.005, points.shape)
            
            # Strategy 2: Mixed perturbation  
            if attempt % 2 == 0:
                perturbed = points + np.random.normal(0, 0.01, points.shape)
            else:
                perturbed = points * (1 + np.random.uniform(-0.01, 0.01, points.shape))

            # Ensure bounds
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

    # Phase 1: Multiple initialization strategies
    init_points = improved_initialization()

    # Phase 2: Global optimization with enhanced DE parameters
    x0 = init_points.flatten()
    bounds = [(0, 1)] * 42  # 14 points * 3 coordinates each

    # Try multiple DE configurations for robust optimization
    best_result = None
    best_ratio = -np.inf
    
    # Different DE configurations to try
    de_configs = [
        {'popsize': 20, 'mutation': (0.5, 1.0), 'recombination': 0.7},
        {'popsize': 25, 'mutation': (0.7, 1.0), 'recombination': 0.8}, 
        {'popsize': 30, 'mutation': (0.8, 1.0), 'recombination': 0.9}
    ]
    
    for config in de_configs:
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=150,
                popsize=config['popsize'],
                mutation=config['mutation'],
                recombination=config['recombination'],
                tol=1e-10,
                callback=None
            )
            
            if -result.fun > best_ratio:
                best_ratio = -result.fun
                best_result = result
                
        except:
            continue

    # Extract optimized points
    if best_result is not None:
        optimized_points = best_result.x.reshape(-1, 3)
    else:
        optimized_points = init_points

    # Phase 3: Adaptive local refinement
    refined_points = adaptive_local_refinement(optimized_points)

    # Phase 4: Final polishing with additional local search
    try:
        # Try one more round with different parameters
        x0_final = refined_points.flatten()
        
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
            x0_final,
            method='L-BFGS-B',
            bounds=[(0, 1)] * 42,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000},
            tol=1e-12
        )
        
        final_points = final_result.x.reshape(-1, 3)
        final_points = np.clip(final_points, 0, 1)
        
    except:
        final_points = refined_points

    # Final validation and bound checking
    final_points = np.clip(final_points, 0, 1)

    return final_points

# EVOLVE-BLOCK-END
