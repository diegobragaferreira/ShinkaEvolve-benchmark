# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import SphericalVoronoi
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = pdist(points)

        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return -np.inf

        # Return negative because we want to maximize the ratio
        return -(d_min / d_max)

    def distance_weighted_objective(x):
        """Objective that weights the contribution of each pair based on distance"""
        points = x.reshape(-1, 3)
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return -np.inf
            
        # Weighted approach: focus more on improving the minimum distance
        # by weighting smaller distances more heavily
        weights = 1.0 / (distances + 1e-12)  # Avoid division by zero
        weighted_sum = np.sum(weights)
        
        # Return negative weighted ratio to maximize 
        return -(d_min / d_max) * (weighted_sum / len(distances))

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
            # Fallback to fibonacci if spherical voronoi fails
            return spherical_fibonacci_points(n)

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

    def spherical_embedding_optimization(initial_points, maxiter=100):
        """
        Optimize points using spherical embedding approach:
        1. Project points onto unit sphere
        2. Optimize on sphere (using spherical geometry)
        3. Map back to 3D space
        """
        # Project initial points onto unit sphere
        points_sphere = initial_points.copy()
        norms = np.linalg.norm(points_sphere, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        points_sphere = points_sphere / norms
        
        # Create spherical Voronoi for reference
        try:
            sv = SphericalVoronoi(points_sphere)
            voronoi_centers = sv.vertices
        except:
            voronoi_centers = points_sphere.copy()
            
        # Optimization using spherical approach
        def sphere_objective(x):
            sphere_points = x.reshape(-1, 3)
            # Normalize to sphere
            norms = np.linalg.norm(sphere_points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            sphere_points = sphere_points / norms
            
            distances = pdist(sphere_points)
            if len(distances) == 0:
                return -np.inf
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # Penalize if too close to other points on sphere
            if d_max == 0:
                return -np.inf
                
            return -(d_min / d_max)
            
        # Try optimizing points on sphere
        try:
            bounds = [(-1, 1) for _ in range(42)]
            result = differential_evolution(
                sphere_objective,
                bounds,
                seed=42,
                maxiter=maxiter//4,
                popsize=25,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.8,
                disp=False
            )
            
            optimized_sphere = result.x.reshape(-1, 3)
            # Normalize to unit sphere
            norms = np.linalg.norm(optimized_sphere, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            optimized_sphere = optimized_sphere / norms
            
            return optimized_sphere
        except:
            return points_sphere

    def multi_phase_refinement(initial_points):
        """Apply multi-phase refinement to improve the solution"""
        current_points = initial_points.copy()
        
        # Phase 1: Global optimization with spherical embedding
        try:
            embedded_points = spherical_embedding_optimization(current_points, maxiter=50)
            current_points = embedded_points
        except:
            pass
            
        # Phase 2: Local optimization with L-BFGS-B
        def local_objective(x):
            points_local = x.reshape(-1, 3)
            distances = pdist(points_local)
            
            if len(distances) == 0:
                return -np.inf
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 1e-12:
                return -(d_min / d_max)
            else:
                return -np.inf
                
        try:
            x0_local = current_points.flatten()
            bounds_local = [(0, 1) for _ in range(42)]
            
            result_local = minimize(
                local_objective,
                x0_local,
                method='L-BFGS-B',
                bounds=bounds_local,
                options={'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            current_points = result_local.x.reshape(-1, 3)
            current_points = np.clip(current_points, 0, 1)
        except:
            pass
            
        # Phase 3: Final improvement with direct optimization
        try:
            # Try a few more rounds of direct optimization
            for _ in range(3):
                # Sample around current solution
                sample_points = current_points + np.random.normal(0, 0.01, current_points.shape)
                sample_points = np.clip(sample_points, 0, 1)
                
                # Optimization
                x0_final = sample_points.flatten()
                bounds_final = [(0, 1) for _ in range(42)]
                
                result_final = minimize(
                    local_objective,
                    x0_final,
                    method='L-BFGS-B',
                    bounds=bounds_final,
                    options={'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if result_final.success:
                    candidate_points = result_final.x.reshape(-1, 3)
                    candidate_points = np.clip(candidate_points, 0, 1)
                    
                    # Compare with current
                    current_distance = pdist(current_points)
                    candidate_distance = pdist(candidate_points)
                    
                    if len(candidate_distance) > 0 and len(current_distance) > 0:
                        current_min = np.min(current_distance)
                        current_max = np.max(current_distance)
                        candidate_min = np.min(candidate_distance)
                        candidate_max = np.max(candidate_distance)
                        
                        if (candidate_min / candidate_max) > (current_min / current_max):
                            current_points = candidate_points
        except:
            pass
            
        return current_points

    def create_better_initializations():
        """Create multiple improved initial configurations"""
        configs = []
        
        # Configuration 1: Spherical Voronoi points
        voronoi_points = spherical_voronoi_points(14)
        voronoi_points = (voronoi_points + 1) / 2  # Normalize to [0,1]^3
        configs.append(("voronoi", voronoi_points.copy()))
        
        # Configuration 2: Spherical Fibonacci points
        fib_points = spherical_fibonacci_points(14)
        fib_points = (fib_points + 1) / 2  # Normalize to [0,1]^3
        configs.append(("fibonacci", fib_points.copy()))
        
        # Configuration 3: KMeans clustering points
        np.random.seed(42)
        kmeans_points = np.random.rand(50, 3)
        kmeans = KMeans(n_clusters=14, random_state=42, n_init=20)
        kmeans.fit(kmeans_points)
        kmeans_centers = kmeans.cluster_centers_
        configs.append(("kmeans", kmeans_centers.copy()))
        
        # Configuration 4: Grid with jitter
        grid = np.mgrid[0:1:4j, 0:1:4j, 0:1:4j].reshape(3, -1).T
        grid_points = grid[:14] + np.random.normal(0, 0.02, (14, 3))
        grid_points = np.clip(grid_points, 0, 1)
        configs.append(("grid_jitter", grid_points.copy()))
        
        # Configuration 5: Random with spherical bias
        rand_points = np.random.rand(14, 3)
        sphere_points = spherical_fibonacci_points(14)
        sphere_points = (sphere_points + 1) / 2
        # Mix both approaches
        mixed_points = rand_points * 0.4 + sphere_points * 0.6
        mixed_points = np.clip(mixed_points, 0, 1)
        configs.append(("mixed", mixed_points.copy()))
        
        return configs

    def evaluate_configurations(configs):
        """Evaluate all initialization configurations"""
        best_config = None
        best_ratio = -np.inf
        
        for name, points in configs:
            try:
                distances = pdist(points)
                if len(distances) == 0:
                    continue
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_config = points.copy()
            except:
                continue
                
        return best_config if best_config is not None else np.random.rand(14, 3)

    # Create and evaluate initial configurations
    initial_configs = create_better_initializations()
    initial_points = evaluate_configurations(initial_configs)
    
    # Apply multi-phase refinement
    refined_points = multi_phase_refinement(initial_points)
    
    # Final optimization step with DE
    def final_objective(x):
        points = x.reshape(-1, 3)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return -np.inf
        return -(d_min / d_max)
    
    try:
        bounds = [(0, 1) for _ in range(42)]
        result = differential_evolution(
            final_objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-12,
            mutation=(0.5, 1.0),
            recombination=0.9,
            disp=False
        )
        
        final_points = result.x.reshape(-1, 3)
        final_points = np.clip(final_points, 0, 1)
        
        # Compare with our previous solution
        distances_prev = pdist(refined_points)
        distances_final = pdist(final_points)
        
        if len(distances_final) > 0 and len(distances_prev) > 0:
            d_min_prev = np.min(distances_prev)
            d_max_prev = np.max(distances_prev)
            d_min_final = np.min(distances_final)
            d_max_final = np.max(distances_final)
            
            prev_ratio = d_min_prev / d_max_prev if d_max_prev > 1e-12 else 0
            final_ratio = d_min_final / d_max_final if d_max_final > 1e-12 else 0
            
            if final_ratio > prev_ratio:
                refined_points = final_points
    except:
        pass
    
    return refined_points

# EVOLVE-BLOCK-END