# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
from multiprocessing import Pool
import multiprocessing as mp

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses hierarchical octree-based evolutionary optimization with multi-scale refinement.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    
    def generate_octree_initialization():
        """Generate initial points using octree-based spatial distribution"""
        # Start with a basic spherical arrangement
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Distribute points using golden spiral on sphere
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y*y)
            
            theta = np.arctan2(y, radius) + (i * 2 * np.pi / golden_ratio)
            
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            
            points.append([x, y, z])
        
        points = np.array(points)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-10)
        
        # Project to unit cube [0,1]^3
        points = (points + 1) / 2
        
        # Add structured perturbation to break symmetries
        np.random.seed(42)
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    def compute_min_max_ratio(points_flat):
        """Compute negative of min/max distance ratio for optimization"""
        points = points_flat.reshape(n, d)
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        if max_dist == 0:
            return -np.inf

        return -min_dist / max_dist
    
    def objective_with_penalty(x):
        """Objective function with penalty for boundary violations"""
        points = x.reshape(n, d)

        # Apply soft penalty for boundary violations
        penalty = 0
        for i in range(n):
            for j in range(d):
                if points[i,j] < 0:
                    penalty += 1e6 * (0 - points[i,j])**2
                elif points[i,j] > 1:
                    penalty += 1e6 * (points[i,j] - 1)**2

        # Calculate distance matrix
        distances = cdist(points, points, 'euclidean')
        np.fill_diagonal(distances, np.inf)

        # Find min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Avoid division by zero
        if max_dist == 0:
            return -np.inf + penalty

        # Return negative ratio plus penalty (to minimize)
        return -(min_dist / max_dist) + penalty
    
    def create_octree_partitions(points, depth=2):
        """Create octree partitions of the point set"""
        # Determine bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        ranges = max_coords - min_coords
        
        # If range is zero, just return the points as one partition
        if np.all(ranges == 0):
            return [points]
        
        # Create partitions based on depth
        partitions = []
        if depth <= 0:
            return [points]
        
        # Create 8 octants
        mid = (min_coords + max_coords) / 2
        octants = [
            points[(points[:, 0] <= mid[0]) & (points[:, 1] <= mid[1]) & (points[:, 2] <= mid[2])],
            points[(points[:, 0] > mid[0]) & (points[:, 1] <= mid[1]) & (points[:, 2] <= mid[2])],
            points[(points[:, 0] <= mid[0]) & (points[:, 1] > mid[1]) & (points[:, 2] <= mid[2])],
            points[(points[:, 0] > mid[0]) & (points[:, 1] > mid[1]) & (points[:, 2] <= mid[2])],
            points[(points[:, 0] <= mid[0]) & (points[:, 1] <= mid[1]) & (points[:, 2] > mid[2])],
            points[(points[:, 0] > mid[0]) & (points[:, 1] <= mid[1]) & (points[:, 2] > mid[2])],
            points[(points[:, 0] <= mid[0]) & (points[:, 1] > mid[1]) & (points[:, 2] > mid[2])],
            points[(points[:, 0] > mid[0]) & (points[:, 1] > mid[1]) & (points[:, 2] > mid[2])]
        ]
        
        # Filter out empty octants and recursively partition non-empty ones
        for octant in octants:
            if len(octant) > 0:
                if len(octant) <= 4 or depth <= 1:
                    partitions.append(octant)
                else:
                    partitions.extend(create_octree_partitions(octant, depth-1))
        
        return partitions
    
    def cluster_based_optimization(partitions, max_iters=20):
        """Optimize each cluster independently with constraint enforcement"""
        optimized_partitions = []
        
        def optimize_single_partition(partition_data):
            partition_points, partition_idx = partition_data
            
            # Only optimize if there are points in the partition
            if len(partition_points) == 0:
                return partition_points
            
            # For single point, just return it
            if len(partition_points) == 1:
                return partition_points
            
            # Create an optimization problem for this partition
            def obj(x):
                points = x.reshape(-1, 3)
                # Ensure bounds
                points = np.clip(points, 0, 1)
                distances = cdist(points, points, 'euclidean')
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 1e10
                return -min_dist / max_dist  # Negative for maximization
            
            # Try different optimization strategies
            best_points = partition_points.copy()
            best_ratio = compute_min_max_ratio(best_points.flatten())
            
            # Strategy 1: L-BFGS-B
            try:
                bounds = [(0, 1) for _ in range(len(partition_points) * 3)]
                res = minimize(obj, partition_points.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
                if res.success:
                    new_points = res.x.reshape(-1, 3)
                    new_ratio = compute_min_max_ratio(new_points.flatten())
                    if new_ratio < best_ratio:
                        best_points = new_points
                        best_ratio = new_ratio
            except:
                pass
            
            # Strategy 2: Nelder-Mead
            try:
                res = minimize(obj, partition_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 100, 'disp': False})
                if res.success:
                    new_points = res.x.reshape(-1, 3)
                    new_ratio = compute_min_max_ratio(new_points.flatten())
                    if new_ratio < best_ratio:
                        best_points = new_points
                        best_ratio = new_ratio
            except:
                pass
            
            return best_points
        
        # Process partitions in parallel
        partition_data = [(partition, i) for i, partition in enumerate(partitions)]
        
        with Pool(min(mp.cpu_count(), len(partitions))) as pool:
            optimized_partitions = pool.map(optimize_single_partition, partition_data)
        
        return optimized_partitions
    
    def combine_partitions(partitions):
        """Combine optimized partitions back into a single point set"""
        combined = np.vstack(partitions)
        return combined
    
    def multi_scale_evolution(initial_points, max_time=280):
        """Perform multi-scale evolutionary optimization"""
        start_time = time.time()
        
        # Level 1: Coarse optimization (smaller points sets)
        current_points = initial_points.copy()
        level = 0
        
        while time.time() - start_time < max_time * 0.9 and level < 3:
            # Create octree partitions
            partitions = create_octree_partitions(current_points, depth=max(1, 3-level))
            
            # Optimize each partition independently
            optimized_partitions = cluster_based_optimization(partitions, max_iters=10)
            
            # Combine partitions
            current_points = combine_partitions(optimized_partitions)
            
            # Refine globally
            def global_obj(x):
                points = x.reshape(n, d)
                distances = cdist(points, points, 'euclidean')
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist == 0:
                    return 1e10
                return -min_dist / max_dist  # Negative for maximization
            
            bounds = [(0, 1) for _ in range(n * d)]
            try:
                res = minimize(global_obj, current_points.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
                if res.success:
                    current_points = res.x.reshape(n, d)
            except:
                pass
            
            level += 1
        
        return current_points
    
    def adaptive_local_refinement(points):
        """Apply adaptive refinement strategies"""
        def obj(x):
            points_temp = x.reshape(n, d)
            distances = cdist(points_temp, points_temp, 'euclidean')
            np.fill_diagonal(distances, np.inf)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 1e10
            return -min_dist / max_dist  # Negative for maximization

        # Strategy 1: L-BFGS-B with tight tolerances
        bounds = [(0, 1) for _ in range(n * d)]
        try:
            res = minimize(obj, points.flatten(), method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9})
            if res.success:
                return res.x.reshape(n, d)
        except:
            pass

        # Strategy 2: Nelder-Mead as fallback
        try:
            res = minimize(obj, points.flatten(), method='Nelder-Mead',
                         options={'maxiter': 500, 'disp': False})
            if res.success:
                return res.x.reshape(n, d)
        except:
            pass

        return points
    
    # Phase 1: Generate initial configuration using octree-inspired approach
    np.random.seed(42)
    initial_points = generate_octree_initialization()
    
    # Phase 2: Multi-scale evolutionary optimization
    refined_points = multi_scale_evolution(initial_points, max_time=280)
    
    # Phase 3: Adaptive local refinement
    final_points = adaptive_local_refinement(refined_points)
    
    # Phase 4: Multiple restarts for better exploration
    best_points = final_points.copy()
    best_ratio = compute_min_max_ratio(final_points.flatten())
    
    # Try several random restarts with octree-based initialization
    for restart in range(5):
        np.random.seed(restart * 1000 + 42)
        
        # Create a slightly different octree-based initialization
        restart_points = generate_octree_initialization()
        
        # Apply multi-scale evolution to this restart
        restart_optimized = multi_scale_evolution(restart_points, max_time=50)
        
        # Local refinement
        restart_final = adaptive_local_refinement(restart_optimized)
        restart_ratio = compute_min_max_ratio(restart_final.flatten())
        
        if restart_ratio < best_ratio:  # Better ratio
            best_ratio = restart_ratio
            best_points = restart_final.copy()
    
    # Final verification
    final_points = best_points
    
    # Ensure bounds are respected
    final_points = np.clip(final_points, 0, 1)
    
    # Verify we have correct shape
    assert final_points.shape == (14, 3), f"Expected shape (14, 3), got {final_points.shape}"
    
    return final_points

# EVOLVE-BLOCK-END