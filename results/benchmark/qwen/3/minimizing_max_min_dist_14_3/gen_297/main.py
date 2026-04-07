# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import SphericalVoronoi
import time
from sklearn.cluster import KMeans

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points."""
        if len(points) < 2:
            return 0.0

        # Compute pairwise distances efficiently
        distances = pdist(points)

        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Avoid division by zero
        if d_max == 0:
            return 0.0

        return d_min / d_max

    def fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral."""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = golden_angle * i  # Golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def spherical_constraint(points):
        """Normalize points to lie on the unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms

    def voronoi_based_refinement(points, max_iter=100):
        """Refine points using Voronoi-based geometric principles."""
        current_points = points.copy()
        
        for _ in range(max_iter):
            # Create spherical Voronoi diagram
            sv = SphericalVoronoi(current_points, radius=1.0)
            
            # Get Voronoi cells and areas
            cells = sv.cells
            total_area = 4 * np.pi
            
            # Find points that are in small cells (dense regions) and move them outward
            # and points in large cells (sparse regions) and move them inward
            new_points = current_points.copy()
            moved_count = 0
            
            for i, cell in enumerate(cells):
                # Simple heuristic: move points that are in small cells outward
                cell_area = np.abs(np.sum(np.cross(cell[:-1], cell[1:])))
                if cell_area < 0.5:  # Threshold for sparse region
                    # Move point outward
                    direction = current_points[i] / np.linalg.norm(current_points[i])
                    new_points[i] += 0.005 * direction
                    moved_count += 1
                elif cell_area > 3.0:  # Threshold for dense region
                    # Move point inward
                    direction = -current_points[i] / np.linalg.norm(current_points[i])
                    new_points[i] += 0.005 * direction
                    moved_count += 1
            
            # Normalize new points to sphere
            new_points = spherical_constraint(new_points)
            
            # Only accept changes if they improve the ratio
            old_ratio = compute_min_max_ratio(current_points)
            new_ratio = compute_min_max_ratio(new_points)
            
            if new_ratio > old_ratio:
                current_points = new_points
            else:
                # Try local optimization on one point at a time
                for point_idx in range(14):
                    best_point = current_points[point_idx].copy()
                    best_ratio = compute_min_max_ratio(current_points)
                    best_move = False
                    
                    # Try small moves in each dimension
                    step_sizes = [0.002, 0.005, 0.01]
                    for step_size in step_sizes:
                        for dim in range(3):
                            # Positive move
                            test_points = current_points.copy()
                            test_points[point_idx, dim] += step_size
                            test_points[point_idx] = spherical_constraint(test_points[point_idx:point_idx+1])[0]
                            test_ratio = compute_min_max_ratio(test_points)
                            
                            if test_ratio > best_ratio:
                                best_ratio = test_ratio
                                best_point = test_points[point_idx].copy()
                                best_move = True
                                
                            # Negative move
                            test_points = current_points.copy()
                            test_points[point_idx, dim] -= step_size
                            test_points[point_idx] = spherical_constraint(test_points[point_idx:point_idx+1])[0]
                            test_ratio = compute_min_max_ratio(test_points)
                            
                            if test_ratio > best_ratio:
                                best_ratio = test_ratio
                                best_point = test_points[point_idx].copy()
                                best_move = True
                    
                    if best_move:
                        current_points[point_idx] = best_point
            
            # Normalize again to keep on sphere
            current_points = spherical_constraint(current_points)
            
        return current_points

    def hierarchical_optimization(initial_points, max_iter=50):
        """Perform hierarchical optimization from coarse to fine resolution."""
        points = initial_points.copy()
        
        # Coarse optimization - cluster-based adjustment
        try:
            # Group points into clusters to identify dense/sparse regions
            kmeans = KMeans(n_clusters=min(4, len(points)), random_state=42, n_init=10)
            labels = kmeans.fit_predict(points)
            
            # Adjust clusters to spread out
            cluster_centers = kmeans.cluster_centers_
            adjusted_points = []
            
            for i in range(len(points)):
                cluster_id = labels[i]
                center = cluster_centers[cluster_id]
                
                # Move point away from cluster center
                direction = points[i] - center
                magnitude = np.linalg.norm(direction)
                if magnitude > 0:
                    direction = direction / magnitude
                    new_point = points[i] + 0.02 * direction
                    adjusted_points.append(new_point)
                else:
                    adjusted_points.append(points[i])
            
            points = np.array(adjusted_points)
            points = spherical_constraint(points)
            
        except:
            pass

        # Medium resolution refinement
        try:
            # Apply Voronoi-based refinement for medium scale
            points = voronoi_based_refinement(points, 30)
        except:
            pass

        # Fine optimization with local search
        try:
            # Local optimization using scipy minimize on a subset for better performance
            def local_objective(x_flat):
                points = x_flat.reshape(-1, 3)
                points = spherical_constraint(points)
                ratio = compute_min_max_ratio(points)
                return -ratio
            
            x0 = points.flatten()
            bounds = [(-1, 1)] * (14 * 3)
            
            # Use L-BFGS-B with tight tolerances for fine-tuning
            result = minimize(
                local_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                points = result.x.reshape(-1, 3)
                points = spherical_constraint(points)
                
        except:
            pass

        # Final Voronoi refinement
        points = voronoi_based_refinement(points, 20)
        
        return points

    def generate_diverse_initializations():
        """Generate multiple diverse initial point sets with geometric awareness."""
        initial_sets = []

        # Strategy 1: Fibonacci sphere with moderate variance
        fib_points = fibonacci_sphere(14)
        perturbed = fib_points + np.random.normal(0, 0.03, fib_points.shape)
        initial_sets.append(spherical_constraint(perturbed))

        # Strategy 2: Random points on sphere with clustering consideration
        random_points = np.random.randn(14, 3)
        initial_sets.append(spherical_constraint(random_points))

        # Strategy 3: Structured with geometric principles
        struct_points = np.zeros((14, 3))
        for i in range(14):
            if i < 4:
                # Along axes
                struct_points[i] = [1 if j==i else 0 for j in range(3)]
            elif i < 8:
                # Opposite axes
                struct_points[i] = [-1 if j==i-4 else 0 for j in range(3)]
            elif i < 12:
                # Diagonal combinations (more complex pattern)
                j = i - 8
                struct_points[i] = [1 if k==j else -1 if k==(j+1)%3 else 0 for k in range(3)]
            else:
                # Random with normalization
                struct_points[i] = np.random.randn(3)
        initial_sets.append(spherical_constraint(struct_points))

        # Strategy 4: High variance Fibonacci
        fib_perturbed = fib_points + np.random.normal(0, 0.08, fib_points.shape)
        initial_sets.append(spherical_constraint(fib_perturbed))

        # Strategy 5: Clustered distribution
        clustered = np.zeros((14, 3))
        # Place points in a way that creates some clustering but maintains spread
        for i in range(14):
            # Most points in a pattern, some scattered
            if i < 10:
                # Create a structured pattern, slightly perturbed
                angle = (i / 10) * 2 * np.pi
                radius = 0.8 + 0.2 * np.sin(angle)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = 0.2 * np.sin(3 * angle)
                clustered[i] = [x, y, z]
            else:
                # Random points
                clustered[i] = np.random.randn(3)
        initial_sets.append(spherical_constraint(clustered))

        # Strategy 6: Symmetric pattern around equator
        symmetric = []
        for i in range(14):
            if i < 7:
                # Equatorial ring
                angle = (i / 7) * 2 * np.pi
                symmetric.append([np.cos(angle) * 0.9, np.sin(angle) * 0.9, 0])
            else:
                # Upper hemisphere with some structure
                angle = ((i-7) / 7) * 2 * np.pi
                height = 0.5 + 0.3 * np.sin(angle)
                radius = np.sqrt(1 - height * height)
                symmetric.append([np.cos(angle) * radius, np.sin(angle) * radius, height])
        initial_sets.append(spherical_constraint(np.array(symmetric)))

        # Strategy 7: Interleaved pattern
        interleaved = []
        for i in range(14):
            if i % 2 == 0:
                # Even indices along a great circle
                angle = (i // 2) * (2 * np.pi / 7)
                interleaved.append([np.cos(angle), np.sin(angle), 0])
            else:
                # Odd indices in another plane
                angle = ((i-1) // 2) * (2 * np.pi / 7)
                interleaved.append([0, np.cos(angle), np.sin(angle)])
        initial_sets.append(spherical_constraint(np.array(interleaved)))

        return initial_sets

    def evolutionary_spatial_search(initial_sets):
        """Use evolutionary approach with spatial constraints."""
        best_solution = None
        best_ratio = 0.0
        
        # Evaluate all initial configurations
        for i, initial_points in enumerate(initial_sets):
            # Hierarchical optimization for each initialization
            optimized_points = hierarchical_optimization(initial_points, max_iter=30)
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_solution = optimized_points.copy()
        
        # Additional refinement on the best solution
        if best_solution is not None:
            # Try multiple refinement passes with different strategies
            refined_candidates = []
            
            # Pass 1: Enhanced Voronoi refinement
            try:
                refined1 = voronoi_based_refinement(best_solution, 50)
                refined_candidates.append(refined1)
            except:
                refined_candidates.append(best_solution.copy())
                
            # Pass 2: Different local optimization
            try:
                def local_obj(x_flat):
                    points = x_flat.reshape(-1, 3)
                    points = spherical_constraint(points)
                    ratio = compute_min_max_ratio(points)
                    return -ratio
                
                x0 = best_solution.flatten()
                bounds = [(-1, 1)] * (14 * 3)
                result = minimize(
                    local_obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
                )
                if result.success:
                    refined2 = result.x.reshape(-1, 3)
                    refined2 = spherical_constraint(refined2)
                    refined_candidates.append(refined2)
            except:
                pass
                
            # Select the best among candidates
            for candidate in refined_candidates:
                ratio = compute_min_max_ratio(candidate)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = candidate.copy()
        
        return best_solution

    # Main optimization pipeline
    initial_sets = generate_diverse_initializations()
    best_solution = evolutionary_spatial_search(initial_sets)
    
    # Fallback if needed
    if best_solution is None:
        # Fallback to Fibonacci with small perturbation
        fib_points = fibonacci_sphere(14)
        fib_points = fib_points + np.random.normal(0, 0.05, fib_points.shape)
        best_solution = spherical_constraint(fib_points)

    return best_solution

# EVOLVE-BLOCK-END