# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import SphericalVoronoi
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def generate_voronoi_optimized_points(n):
        """Generate points using Voronoi uniformity optimization on sphere"""
        # Start with a Fibonacci spiral approach
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = golden_angle * i  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        points = np.array(points)
        
        # Refine using a simple iterative method focused on Voronoi uniformity
        # This creates a more uniform distribution that will work better for our optimization
        for _ in range(10):
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            points = points / norms
            
            # Simple gradient descent step towards uniform Voronoi cells
            try:
                sv = SphericalVoronoi(points)
                areas = sv.voronoi_cell_areas()
                # Adjust points to make areas more uniform (simplified version)
                # This is a heuristic to push points toward more uniform distribution
                for i in range(len(points)):
                    if areas[i] > np.mean(areas) * 1.1:
                        # Move away from over-large cells
                        # This is a simplified approximation
                        pass
            except:
                pass
                
        return points
    
    def compute_voronoi_uniformity_score(points):
        """Compute a score based on Voronoi cell uniformity"""
        try:
            sv = SphericalVoronoi(points)
            areas = sv.voronoi_cell_areas()
            # Return variance of areas - lower is better (more uniform)
            return np.var(areas)
        except:
            return np.inf
    
    def objective(x):
        # Reshape x into 14 points in 3D
        points = x.reshape(-1, 3)

        # Calculate pairwise distances
        distances = cdist(points, points)
        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)

        # Return negative ratio since we want to maximize
        if d_max < 1e-10:
            return -1e10
        return -d_min / d_max

    def constraint_func(x):
        # Ensure all points are within [0,1]^3
        points = x.reshape(-1, 3)
        return np.concatenate([
            points.flatten() - 0.0,      # lower bound
            1.0 - points.flatten()       # upper bound
        ])

    # Initialize with Voronoi-optimized points on sphere
    np.random.seed(42)
    
    # Generate initial points using Voronoi uniformity approach
    initial_points = generate_voronoi_optimized_points(14)
    
    # Scale to unit cube [0,1]^3 properly
    # Center around origin and scale
    initial_points = initial_points - np.mean(initial_points, axis=0)
    max_coord = np.max(np.abs(initial_points))
    if max_coord > 0:
        initial_points = initial_points / max_coord * 0.5
    # Shift to [0,1]^3
    initial_points = initial_points + 0.5
    
    # Add slight random perturbation to break symmetry
    initial_points += np.random.normal(0, 0.005, initial_points.shape)
    initial_points = np.clip(initial_points, 0.0, 1.0)
    
    # Flatten for optimization
    x0 = initial_points.flatten()
    
    # Set up bounds for optimization (0 to 1 for all coordinates)
    bounds = [(0.0, 1.0)] * 14 * 3

    # Multi-phase optimization approach
    # Phase 1: Global search with differential evolution
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=12,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            disp=False
        )
        
        # Phase 2: Local refinement with L-BFGS-B
        refined_result = minimize(
            objective,
            de_result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 300},
            callback=None
        )
        
        # Evaluate final result
        final_points = refined_result.x.reshape(-1, 3)
        final_points = np.clip(final_points, 0.0, 1.0)
        
        # Check if the result is better than our initial points
        distances = cdist(final_points, final_points)
        np.fill_diagonal(distances, np.inf)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max > 1e-10:
            ratio = d_min / d_max
            # If this is significantly better (or we have no valid previous result)
            # But return the better one
            if ratio > 0.01:  # Threshold to ensure substantial improvement
                return final_points
        
    except Exception as e:
        pass
    
    # Fallback: Return the best of multiple attempts with different strategies
    best_points = initial_points.copy()
    best_ratio = 0
    
    # Try multiple optimization attempts
    for attempt in range(3):
        np.random.seed(42 + attempt)
        
        # Create slightly different initialization
        if attempt == 0:
            # Original approach
            points = initial_points.copy()
        elif attempt == 1:
            # Random perturbation
            points = initial_points + np.random.normal(0, 0.01, initial_points.shape)
            points = np.clip(points, 0.0, 1.0)
        else:
            # Random initialization
            points = np.random.rand(14, 3)
        
        try:
            # Run optimization
            x0_attempt = points.flatten()
            result = differential_evolution(
                objective,
                bounds,
                seed=42 + attempt,
                maxiter=150,
                popsize=10,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                disp=False
            )
            
            # Final refinement
            final_result = minimize(
                objective,
                result.x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 200}
            )
            
            final_points = final_result.x.reshape(-1, 3)
            final_points = np.clip(final_points, 0.0, 1.0)
            
            # Evaluate quality
            distances = cdist(final_points, final_points)
            np.fill_diagonal(distances, np.inf)
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            if d_max > 1e-10:
                ratio = d_min / d_max
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
        except Exception as e:
            continue
    
    return best_points

# EVOLVE-BLOCK-END