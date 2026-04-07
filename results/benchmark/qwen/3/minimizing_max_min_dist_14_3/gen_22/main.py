# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, SphericalVoronoi
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
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
        """Generate points on sphere using Fibonacci spiral (good initial distribution)"""
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
    
    def voronoi_based_objective(points_flat):
        """Objective function based on Voronoi cell properties and distance ratios"""
        points = points_flat.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return float('inf')
            
        # Also incorporate Voronoi cell volume properties for better geometric balance
        try:
            # Compute Voronoi diagram (this will be approximated due to bounded space)
            # We'll use a modified approach that focuses on geometric properties
            vor = Voronoi(points)
            
            # Calculate average Voronoi cell volume (normalized by number of cells)
            # This encourages uniform distribution and prevents clustering
            cell_volumes = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    # Approximate volume of convex hull for each cell
                    try:
                        # For simplicity, we use a proxy for cell uniformity
                        pass  # Not computing exact volumes for efficiency
                    except:
                        pass
            
            # Primary objective: maximize min/max ratio
            ratio = min_dist / max_dist
            
            # Secondary objective: encourage more uniform distribution
            # This is done implicitly through the geometric structure
            
            return -ratio  # Negative because we minimize
            
        except:
            # Fallback to just the distance ratio
            return -min_dist / max_dist
    
    def geometric_optimization(initial_points):
        """Use geometric constraints and optimization to refine point placement"""
        points = initial_points.copy()
        
        # Convert to unit sphere for better numerical properties
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-10)  # Normalize but avoid division by zero
        
        # Apply optimization to maximize min/max ratio
        # Using scipy minimize with trust-constr method which handles constraints well
        bounds = [(-1, 1)] * (14 * 3)
        
        def obj_func(x_flat):
            points = x_flat.reshape(-1, 3)
            # Ensure points remain on unit sphere (very important for this problem)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            normalized_points = points / np.maximum(norms, 1e-10)
            
            distances = pdist(normalized_points)
            if len(distances) == 0:
                return float('inf')
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist == 0:
                return float('inf')
                
            # Use a combination: ratio + penalty for extreme distances
            ratio = min_dist / max_dist
            penalty = 0.0
            
            # Add penalty for very small minimum distances (avoid degenerate cases)
            if min_dist < 1e-6:
                penalty += 1000.0
            
            return -(ratio - penalty)  # Negative because we minimize
            
        # Initial optimization step
        try:
            # Use trust-constr for better handling of constraints
            x0 = points.flatten()
            result = minimize(obj_func, x0, method='trust-constr', bounds=bounds, 
                            options={'maxiter': 500, 'disp': False})
            
            if result.success:
                points = result.x.reshape(-1, 3)
            else:
                # If optimization fails, use original points
                pass
        except:
            # If optimization fails entirely, return the initial points
            pass
        
        # Post-processing: ensure unit sphere constraint and normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-10) * 0.99  # Keep inside unit sphere to avoid boundaries
        
        # Final local refinement with hill climbing on the specific geometric properties
        for _ in range(100):
            current_ratio = compute_min_max_ratio(points)
            best_points = points.copy()
            best_ratio = current_ratio
            
            # Try small perturbations in various directions
            step_sizes = [0.01, 0.005, 0.001]
            
            for step_size in step_sizes:
                for i in range(14):
                    for dim in range(3):
                        # Try moving in positive direction
                        test_points = points.copy()
                        test_points[i, dim] += step_size
                        
                        # Project back to unit sphere
                        norm = np.linalg.norm(test_points[i])
                        if norm > 0:
                            test_points[i] = test_points[i] / norm * 0.99
                        
                        test_ratio = compute_min_max_ratio(test_points)
                        
                        if test_ratio > best_ratio:
                            best_ratio = test_ratio
                            best_points = test_points.copy()
                        
                        # Try moving in negative direction
                        test_points = points.copy()
                        test_points[i, dim] -= step_size
                        
                        # Project back to unit sphere
                        norm = np.linalg.norm(test_points[i])
                        if norm > 0:
                            test_points[i] = test_points[i] / norm * 0.99
                        
                        test_ratio = compute_min_max_ratio(test_points)
                        
                        if test_ratio > best_ratio:
                            best_ratio = test_ratio
                            best_points = test_points.copy()
            
            # If no improvement, stop
            if best_ratio <= current_ratio:
                break
                
            points = best_points
        
        return points
    
    def generate_strategic_initial_points():
        """Generate highly strategic initial points"""
        # Start with Fibonacci sphere distribution
        fib_points = fibonacci_sphere(14)
        
        # Slightly perturb to avoid symmetries that might cause local optima
        np.random.seed(42)
        perturbation = np.random.normal(0, 0.05, fib_points.shape)
        points = fib_points + perturbation
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-10) * 0.99
        
        return points
    
    # Multiple restart strategy with better geometric initialization
    best_solution = None
    best_ratio = 0.0
    
    # Try multiple restart configurations using different strategies
    for restart in range(15):  # More restarts for better exploration
        # Different initialization strategies
        if restart == 0:
            # Fibonacci spiral base
            points = generate_strategic_initial_points()
        elif restart < 5:
            # Random with some structured elements
            np.random.seed(42 + restart)
            points = np.random.rand(14, 3) * 2 - 1  # [-1, 1] range
            # Push to sphere surface for better conditioning
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / np.maximum(norms, 1e-10) * 0.99
        else:
            # More structured approach - try to distribute along axes
            np.random.seed(42 + restart)
            points = np.random.randn(14, 3)
            points = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-10) * 0.99
        
        # Optimize this configuration
        optimized_points = geometric_optimization(points)
        ratio = compute_min_max_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_solution = optimized_points.copy()
    
    # Final refinement of the best solution
    if best_solution is not None:
        final_points = geometric_optimization(best_solution)
        return final_points
    
    # Fallback to best initial configuration if all else fails
    return generate_strategic_initial_points()

# EVOLVE-BLOCK-END
