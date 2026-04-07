# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import math
from itertools import combinations

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def distance_ratio(points_flat):
        """Calculate the ratio of minimum to maximum distance"""
        points = points_flat.reshape(-1, 3)
        distances = squareform(pdist(points))
        # Set diagonal to large value so it doesn't affect min/max
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def objective_function(points_flat):
        """Minimize negative of distance ratio (since we want to maximize)"""
        points = points_flat.reshape(-1, 3)
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        ratio = min_dist / max_dist
        # Return negative for minimization
        return -ratio
    
    def create_icosahedron():
        """Create vertices of regular icosahedron"""
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        # Normalize to unit sphere
        norm_vertices = []
        for v in vertices:
            norm = math.sqrt(sum(x*x for x in v))
            norm_vertices.append([x/norm for x in v])
        return np.array(norm_vertices)
    
    def refine_tessellation(points, iterations=2):
        """Refine point set by subdividing faces iteratively"""
        # Start with icosahedron
        if len(points) == 12:
            # Already an icosahedron - we'll do subdivision
            pass
        else:
            # Assume we have a base set and subdivide
            pass
            
        # For the 14-point case, we'll start with icosahedron and add one point
        # Then refine it using vertex subdivision
        refined = points.copy()
        
        # Add one more point that's well-distributed
        # Start with icosahedron
        ico = create_icosahedron()
        
        # Take first 13 vertices of icosahedron  
        base_points = ico[:13].copy()
        
        # Add one point that we'll optimize, but position it in a way that improves distribution
        # Place it at a point that maximizes minimum distance from existing points
        # This heuristic approach avoids complex optimization for now
        farthest_point = np.array([0, 0, 1])  # Start with north pole
        
        # Find the best farthest point from existing
        min_dists = []
        for i in range(len(base_points)):
            dist = np.linalg.norm(farthest_point - base_points[i])
            min_dists.append(dist)
        
        # Add the 14th point at antipodal position to maximize distance
        antipodal = -farthest_point
        base_points = np.vstack([base_points, antipodal])
        
        # Now we have 14 points in a reasonable configuration
        return base_points
    
    def initialize_points(n):
        """Initialize points using sphere tessellation approach"""
        # Start with icosahedron
        ico_points = create_icosahedron()
        
        # We want 14 points total, so we'll take 13 from icosahedron and add one more
        # But we can also do a better approach: use icosahedral subdivision to generate good points
        
        # Let's use a simpler approach: start with icosahedron vertices and add one more point
        # Place the 14th point at a position that maximizes the minimum distance
        
        # First 13 points from icosahedron (normalized)
        points = ico_points[:13].copy()
        
        # Add the 14th point strategically - place it at the antipode of the average of first 13 points
        avg_point = np.mean(points, axis=0)
        # Normalize the average to get a unit vector
        avg_norm = np.linalg.norm(avg_point)
        if avg_norm > 0:
            avg_unit = avg_point / avg_norm
        else:
            avg_unit = np.array([0, 0, 1])
        
        # Place 14th point at antipode  
        antipodal = -avg_unit
        points = np.vstack([points, antipodal])
        
        # Apply small perturbations for better optimization
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        
        # Normalize again to keep on unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        return points
    
    def multi_stage_optimization(initial_points):
        """Perform multi-stage optimization to find better configuration"""
        
        # Stage 1: Coarse optimization with relaxed constraints
        x0 = initial_points.flatten()
        
        # Define constraint that points lie on unit sphere
        def constraint_sphere(x):
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0  # Should be zero for unit sphere
        
        constraints = []
        for i in range(14):
            constraints.append({'type': 'eq', 'fun': lambda x, i=i: constraint_sphere(x)[i]})
        
        bounds = [(-1.2, 1.2)] * len(x0)
        
        # Coarse optimization with relatively loose tolerances
        try:
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-4, 'gtol': 1e-4}
            )
            if result.success:
                x0 = result.x
        except:
            pass
        
        # Stage 2: Medium optimization with tighter constraints
        bounds = [(-1.1, 1.1)] * len(x0)
        try:
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            if result.success:
                x0 = result.x
        except:
            pass
        
        # Stage 3: Fine optimization with very tight constraints  
        bounds = [(-1.05, 1.05)] * len(x0)
        try:
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if result.success:
                x0 = result.x
        except:
            pass
        
        return x0.reshape(-1, 3)
    
    def sphere_refinement(points, max_iterations=5):
        """Apply iterative refinement to improve point distribution"""
        current_points = points.copy()
        best_ratio = -np.inf
        best_points = points.copy()
        
        # Try multiple refinements
        for refinement_step in range(max_iterations):
            # Do optimization for this configuration
            optimized_points = multi_stage_optimization(current_points)
            
            # Check ratio
            ratio = distance_ratio(optimized_points.flatten())
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
            
            # Update current points for next iteration
            current_points = optimized_points.copy()
            
            # Early termination if improvement is minimal
            if refinement_step > 0 and abs(ratio - best_ratio) < 1e-10:
                break
                
        return best_points
    
    # Main optimization procedure
    np.random.seed(42)
    
    # Initialize points using tessellation approach
    initial_points = initialize_points(14)
    
    # Refine the points through multiple optimization stages
    final_points = sphere_refinement(initial_points)
    
    # Final validation and normalization
    norms = np.linalg.norm(final_points, axis=1)
    final_points = final_points / norms[:, np.newaxis]
    
    return final_points

# EVOLVE-BLOCK-END
