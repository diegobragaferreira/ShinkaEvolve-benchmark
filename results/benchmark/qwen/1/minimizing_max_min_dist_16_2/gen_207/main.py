# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import SphericalVoronoi, ConvexHull
import time
from sklearn.cluster import KMeans
import warnings

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses spherical geometry-inspired evolutionary approach with multi-resolution optimization.
    """
    
    np.random.seed(42)
    n_points = 16
    max_time = 180.0
    start_time = time.time()
    
    def spherical_fibonacci_points(n):
        """Generate n points on unit sphere using Fibonacci spiral"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            # Latitude
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Longitude
            theta = phi * i
            
            # Convert to Cartesian coordinates
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def stereographic_project(points_3d):
        """Project 3D points to 2D using stereographic projection from south pole"""
        points_2d = []
        for x, y, z in points_3d:
            # Stereographic projection from south pole (0,0,-1)
            w = 1 / (1 + z)
            proj_x = x * w
            proj_y = y * w
            points_2d.append([proj_x, proj_y])
        return np.array(points_2d)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance"""
        if len(points) < 2:
            return 0.0
            
        try:
            # Use more stable distance calculation
            distances = squareform(pdist(points))
            np.fill_diagonal(distances, np.inf)
            finite_distances = distances[np.isfinite(distances)]
            
            if len(finite_distances) == 0:
                return 0.0
                
            d_min = np.min(finite_distances)
            d_max = np.max(finite_distances)
            
            if d_max <= 0:
                return 0.0
                
            return d_min / d_max
        except Exception:
            return 0.0
    
    def adaptive_mutate(individual, generation, max_generations, mutation_strength=0.05):
        """Adaptive mutation that decreases over generations with smart perturbations"""
        mutated = individual.copy()
        
        # Decreasing mutation rate
        adaptive_rate = 0.3 * (1 - generation/max_generations)
        
        # Apply mutations
        for i in range(len(mutated)):
            if np.random.random() < adaptive_rate:
                # Use adaptive noise based on generation
                noise_magnitude = mutation_strength * (1 - generation/max_generations) * 0.1
                mutated[i] += np.random.normal(0, noise_magnitude)
                
        # Keep within bounds
        mutated = np.clip(mutated, 0, 1)
        return mutated
    
    def geometric_initialization():
        """Generate high-quality initial points using multiple geometric strategies"""
        initial_configs = []
        
        # Strategy 1: Spherical Fibonacci points with projection
        try:
            sph_points = spherical_fibonacci_points(n_points)
            proj_points = stereographic_project(sph_points)
            # Normalize to [0,1] with margin
            if len(proj_points) > 0:
                x_min, y_min = np.min(proj_points, axis=0)
                x_max, y_max = np.max(proj_points, axis=0)
                if x_max > x_min and y_max > y_min:
                    proj_points[:, 0] = (proj_points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                    proj_points[:, 1] = (proj_points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
            # Add noise and clip
            noise = np.random.normal(0, 0.005, proj_points.shape)
            proj_points += noise
            proj_points = np.clip(proj_points, 0, 1)
            initial_configs.append(proj_points.flatten())
        except Exception:
            pass
            
        # Strategy 2: Cluster-based initialization  
        try:
            # Generate points using k-means clustering on random points
            random_points = np.random.rand(100, 2)
            kmeans = KMeans(n_clusters=n_points, random_state=42, n_init=10)
            kmeans.fit(random_points)
            cluster_centers = kmeans.cluster_centers_
            # Add small noise
            noise = np.random.normal(0, 0.01, cluster_centers.shape)
            cluster_centers += noise
            cluster_centers = np.clip(cluster_centers, 0, 1)
            initial_configs.append(cluster_centers.flatten())
        except Exception:
            pass
            
        # Strategy 3: Hexagonal grid with perturbation
        try:
            points = []
            rows = 4
            cols = 4
            for i in range(rows):
                for j in range(cols):
                    if len(points) >= n_points:
                        break
                    x = j * 0.25 + (i % 2) * 0.125
                    y = i * 0.25
                    points.append([x, y])
            points = np.array(points[:n_points])
            # Normalize and add noise
            if len(points) > 0:
                x_range = np.max(points[:, 0]) - np.min(points[:, 0])
                y_range = np.max(points[:, 1]) - np.min(points[:, 1])
                if x_range > 0:
                    points[:, 0] = (points[:, 0] - np.min(points[:, 0])) / x_range * 0.9 + 0.05
                if y_range > 0:
                    points[:, 1] = (points[:, 1] - np.min(points[:, 1])) / y_range * 0.9 + 0.05
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
            points = np.clip(points, 0, 1)
            initial_configs.append(points.flatten())
        except Exception:
            pass
            
        # Strategy 4: Random with boundary padding
        try:
            random_points = np.random.rand(n_points, 2)
            # Pad away from boundaries
            random_points = np.clip(random_points, 0.05, 0.95)
            initial_configs.append(random_points.flatten())
        except Exception:
            pass
            
        return initial_configs
    
    def multi_start_local_search(initial_points, max_iter=500):
        """Multi-start local optimization with different strategies"""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points.reshape(-1, 2))
        
        # Strategy 1: L-BFGS-B
        try:
            def objective(x):
                points = x.reshape(-1, 2)
                ratio = compute_min_max_ratio(points)
                return -ratio  # Minimize negative to maximize ratio
            
            bounds = [(0, 1) for _ in range(len(initial_points))]
            result = minimize(
                objective, 
                initial_points, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter//2, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.flatten()
                    
        except Exception:
            pass
            
        # Strategy 2: Nelder-Mead for diversity
        try:
            def objective(x):
                points = x.reshape(-1, 2)
                ratio = compute_min_max_ratio(points)
                return -ratio
            
            result = minimize(
                objective, 
                initial_points, 
                method='Nelder-Mead',
                options={'maxiter': max_iter//4, 'adaptive': True}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.flatten()
                    
        except Exception:
            pass
            
        # Strategy 3: Differential evolution for global search
        try:
            def bounded_objective(x):
                points = np.clip(x.reshape(-1, 2), 0, 1)
                ratio = compute_min_max_ratio(points)
                return -ratio
            
            bounds = [(0, 1) for _ in range(len(initial_points))]
            result = differential_evolution(
                bounded_objective,
                bounds,
                maxiter=max_iter//4,
                popsize=10,
                seed=42,
                disp=False
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.flatten()
                    
        except Exception:
            pass
            
        return best_points.reshape(-1, 2)
    
    def multi_resolution_optimization(initial_points, max_iterations=500):
        """Progressive refinement at different resolutions"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Phase 1: Coarse grained search
        try:
            coarse_points = multi_start_local_search(current_points, max_iter=max_iterations//3)
            coarse_ratio = compute_min_max_ratio(coarse_points)
            if coarse_ratio > current_ratio:
                current_points = coarse_points
                current_ratio = coarse_ratio
        except Exception:
            pass
        
        # Phase 2: Medium resolution
        try:
            medium_points = multi_start_local_search(current_points, max_iter=max_iterations//3)
            medium_ratio = compute_min_max_ratio(medium_points)
            if medium_ratio > current_ratio:
                current_points = medium_points
                current_ratio = medium_ratio
        except Exception:
            pass
            
        # Phase 3: Fine tuning
        try:
            fine_points = multi_start_local_search(current_points, max_iter=max_iterations//3)
            fine_ratio = compute_min_max_ratio(fine_points)
            if fine_ratio > current_ratio:
                current_points = fine_points
                current_ratio = fine_ratio
        except Exception:
            pass
            
        return current_points
    
    # Generate diverse initial configurations
    initial_configs = geometric_initialization()
    
    # Multi-start optimization
    best_overall_ratio = -np.inf
    best_overall_points = None
    
    # Try each initial configuration with multiple optimization runs
    for i, flat_config in enumerate(initial_configs):
        if time.time() - start_time > max_time - 10:
            break
            
        # Multiple optimization attempts from same initial config
        for attempt in range(3):
            np.random.seed(42 + i * 10 + attempt)
            
            # Create fresh copy of initial config
            initial_points = flat_config.reshape(-1, 2)
            
            # Apply multi-resolution optimization
            optimized_points = multi_resolution_optimization(initial_points, max_iterations=200)
            
            # Final evaluation
            final_ratio = compute_min_max_ratio(optimized_points)
            
            if final_ratio > best_overall_ratio:
                best_overall_ratio = final_ratio
                best_overall_points = optimized_points.copy()
                
        # Early termination check
        if time.time() - start_time > max_time - 10:
            break
    
    # Final verification and refinement
    if best_overall_points is not None:
        # Apply final multi-start optimization
        final_points = multi_resolution_optimization(best_overall_points, max_iterations=200)
        final_ratio = compute_min_max_ratio(final_points)
        
        if final_ratio > best_overall_ratio:
            best_overall_points = final_points
    
    # Fallback if nothing worked
    if best_overall_points is None:
        # Use simple heuristic: hexagonal grid with small perturbations
        points = []
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                x = j * 0.25 + (i % 2) * 0.125
                y = i * 0.25
                points.append([x, y])
        
        points = np.array(points[:n_points])
        # Normalize to [0.05, 0.95] range
        if len(points) > 0:
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            if x_max > x_min and y_max > y_min:
                points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 0.9 + 0.05
                points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 0.9 + 0.05
        # Add small random noise
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        best_overall_points = points
    
    return best_overall_points

# EVOLVE-BLOCK-END
