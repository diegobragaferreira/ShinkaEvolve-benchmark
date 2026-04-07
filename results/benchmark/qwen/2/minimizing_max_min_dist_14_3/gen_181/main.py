# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import math

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """

    def sphere_tiling_initialization(n):
        """Initialize points using a deterministic sphere tiling approach"""
        # Use a modified icosahedral tiling for better coverage
        # Generate points on sphere using a combination of icosahedral structure and refinement
        points = []
        
        # Generate vertices of icosahedron
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        ico_vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        
        # Normalize to unit sphere
        ico_vertices = np.array(ico_vertices)
        norms = np.linalg.norm(ico_vertices, axis=1)
        ico_vertices = ico_vertices / norms[:, np.newaxis]
        
        # Add some additional points for 14 total
        # Take edges and face centers for refinement
        additional_points = []
        
        # Add edge midpoints  
        for i in range(len(ico_vertices)):
            for j in range(i+1, len(ico_vertices)):
                if np.dot(ico_vertices[i], ico_vertices[j]) < 0.5:  # roughly 60 degree angle
                    midpoint = (ico_vertices[i] + ico_vertices[j]) / 2
                    norm = np.linalg.norm(midpoint)
                    if norm > 0:
                        midpoint = midpoint / norm
                        additional_points.append(midpoint)
                        
        # Take face centers (approximation)
        for i in range(0, len(ico_vertices), 3):
            if i+2 < len(ico_vertices):
                face_center = (ico_vertices[i] + ico_vertices[i+1] + ico_vertices[i+2]) / 3
                norm = np.linalg.norm(face_center)
                if norm > 0:
                    face_center = face_center / norm
                    additional_points.append(face_center)
        
        # Combine and select 14 points ensuring good distribution
        all_points = np.vstack([ico_vertices, additional_points[:14-len(ico_vertices)]])
        if len(all_points) > 14:
            # Use k-means style selection to avoid clustering
            selected = []
            remaining = list(range(len(all_points)))
            
            # Start with first point
            selected.append(remaining.pop(0))
            
            # Greedily select points with maximum minimum distance to existing selection
            while len(selected) < 14 and remaining:
                max_min_dist = -1
                best_idx = -1
                
                for idx in remaining:
                    min_dist = float('inf')
                    for sel_idx in selected:
                        dist = np.linalg.norm(all_points[idx] - all_points[sel_idx])
                        min_dist = min(min_dist, dist)
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = idx
                
                if best_idx != -1:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
            
            points = all_points[selected]
        else:
            points = all_points[:14]
            
        return np.array(points)

    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
        
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            return 0.0
            
        return d_min / d_max

    def constrained_optimization_step(initial_points):
        """Use constrained optimization to refine the point configuration"""
        n_points = len(initial_points)
        points = initial_points.copy()
        
        # Flatten initial points for optimization
        initial_flat = points.flatten()
        
        def objective(x_flat):
            # Reshape points
            current_points = x_flat.reshape(n_points, 3)
            
            # Ensure they're on unit sphere
            norms = np.linalg.norm(current_points, axis=1)
            norms = np.where(norms == 0, 1, norms)
            normalized_points = current_points / norms[:, np.newaxis]
            
            # Compute distances
            distances = pdist(normalized_points)
            
            if len(distances) == 0:
                return 0.0
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # Return negative ratio (we want to maximize)
            if d_max == 0:
                return -np.inf
            return -d_min / d_max
        
        def sphere_constraint(x_flat):
            # Constraint: all points must lie on unit sphere
            points = x_flat.reshape(n_points, 3)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0  # Should equal zero for unit sphere
            
        # Define constraints
        constraints = {'type': 'eq', 'fun': sphere_constraint}
        
        # Use L-BFGS-B optimizer which handles constraints well
        try:
            result = minimize(
                objective, 
                initial_flat,
                method='L-BFGS-B',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                optimized_points = result.x.reshape(n_points, 3)
                # Re-normalize to ensure they're on unit sphere
                norms = np.linalg.norm(optimized_points, axis=1)
                norms = np.where(norms == 0, 1, norms)
                return optimized_points / norms[:, np.newaxis]
        except:
            pass
            
        # Return original if optimization fails
        return points

    def progressive_refinement(initial_points, max_iterations=50):
        """Progressively refine the solution with multiple stages"""
        points = initial_points.copy()
        best_ratio = compute_min_max_ratio(points)
        best_points = points.copy()
        
        # Stage 1: Coarse optimization
        for _ in range(10):
            refined = constrained_optimization_step(points)
            ratio = compute_min_max_ratio(refined)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined.copy()
                points = refined.copy()
        
        # Stage 2: Fine refinement with different initializations
        for stage in range(2):  # Two refinement stages
            # Perturb slightly and optimize
            perturbed = points + np.random.normal(0, 0.001, points.shape)
            # Normalize
            norms = np.linalg.norm(perturbed, axis=1)
            norms = np.where(norms == 0, 1, norms)
            perturbed = perturbed / norms[:, np.newaxis]
            
            refined = constrained_optimization_step(perturbed)
            ratio = compute_min_max_ratio(refined)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined.copy()
                points = refined.copy()
        
        return best_points

    # Multi-start approach with different initialization strategies
    best_points = None
    best_ratio = 0.0
    
    # Strategy 1: Sphere tiling initialization
    np.random.seed(42)
    tiling_points = sphere_tiling_initialization(14)
    refined_tiling = progressive_refinement(tiling_points)
    ratio = compute_min_max_ratio(refined_tiling)
    
    if ratio > best_ratio:
        best_ratio = ratio
        best_points = refined_tiling.copy()
    
    # Strategy 2: Random initialization with constraint handling
    np.random.seed(123)
    random_points = np.random.randn(14, 3)
    norms = np.linalg.norm(random_points, axis=1)
    norms = np.where(norms == 0, 1, norms)
    random_points = random_points / norms[:, np.newaxis]
    
    refined_random = progressive_refinement(random_points)
    ratio = compute_min_max_ratio(refined_random)
    
    if ratio > best_ratio:
        best_ratio = ratio
        best_points = refined_random.copy()
    
    # Strategy 3: Fibonacci-inspired initialization
    np.random.seed(456)
    fib_points = fibonacci_sphere(14)
    norms = np.linalg.norm(fib_points, axis=1)
    norms = np.where(norms == 0, 1, norms)
    fib_points = fib_points / norms[:, np.newaxis]
    
    refined_fib = progressive_refinement(fib_points)
    ratio = compute_min_max_ratio(refined_fib)
    
    if ratio > best_ratio:
        best_ratio = ratio
        best_points = refined_fib.copy()

    # Final optimization on best solution
    if best_points is not None:
        final_points = constrained_optimization_step(best_points)
        final_ratio = compute_min_max_ratio(final_points)
        
        if final_ratio > best_ratio:
            return final_points
        else:
            return best_points

    # Fallback to basic approach
    initial_points = fibonacci_sphere(14)
    norms = np.linalg.norm(initial_points, axis=1)
    norms = np.where(norms == 0, 1, norms)
    initial_points = initial_points / norms[:, np.newaxis]
    
    return progressive_refinement(initial_points)

    def fibonacci_sphere(n):
        """Generate n points distributed approximately uniformly on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            # Distribute points more evenly
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius = np.sqrt(1 - z*z)

            # Better distribution using Fibonacci sequence
            theta = np.arctan2(np.sin(i * 2 * np.pi / phi), np.cos(i * 2 * np.pi / phi))
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

# EVOLVE-BLOCK-END