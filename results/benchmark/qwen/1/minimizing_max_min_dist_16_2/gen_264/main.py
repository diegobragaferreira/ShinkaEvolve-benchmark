# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n_points = 16
    dimension = 2
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        distances = distances[distances > 1e-12]  # Filter out near-zero distances
        
        if len(distances) == 0:
            return 0
            
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        if dmax == 0:
            return 0
            
        return dmin / dmax
    
    def compute_voronoi_uniformity(points):
        """Compute a measure of Voronoi cell uniformity to encourage even distribution."""
        try:
            vor = Voronoi(points)
            areas = []
            
            # Calculate area for each Voronoi cell (excluding infinite regions)
            for i, region in enumerate(vor.regions):
                if len(region) > 0 and -1 not in region:
                    # Calculate polygon area using shoelace formula
                    vertices = vor.vertices[region]
                    if len(vertices) >= 3:
                        x_vals = vertices[:, 0]
                        y_vals = vertices[:, 1]
                        area = 0.5 * np.abs(np.dot(x_vals, np.roll(y_vals, 1)) - np.dot(y_vals, np.roll(x_vals, 1)))
                        areas.append(area)
            
            if len(areas) == 0:
                return 0
                
            # Lower variance in cell areas indicates better uniformity
            return 1.0 / (1.0 + np.var(areas))
        except:
            return 0
    
    def combined_objective(x_flat, weight_ratio=0.8, weight_uniformity=0.2):
        """Combined objective function balancing distance ratio and uniformity."""
        points = x_flat.reshape(-1, 2)
        
        # Ensure bounds are respected
        points = np.clip(points, 1e-8, 1-1e-8)
        
        ratio = compute_min_max_ratio(points)
        uniformity = compute_voronoi_uniformity(points)
        
        # Combined objective: maximize ratio while promoting uniformity
        combined = ratio * weight_ratio + uniformity * weight_uniformity
        
        # Return negative for minimization
        return -combined
    
    def generate_hexagonal_initial():
        """Generate a high-quality initial configuration based on hexagonal packing."""
        # Create a structured hexagonal arrangement
        points = []
        rows, cols = 4, 4
        
        # Hexagonal spacing
        spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
        spacing_y = np.sqrt(3) / 2 / (rows - 1) if rows > 1 else 1.0
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                # Offset every other row for hexagonal pattern
                x = j * spacing_x + (i % 2) * spacing_x * 0.5
                y = i * spacing_y
                points.append([x, y])
        
        points = np.array(points[:n_points])
        
        # Add small random perturbations to break symmetry and improve optimization
        points += np.random.normal(0, 0.01, points.shape)
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        
        return points
    
    def voronoi_refinement_step(points, iterations=3):
        """Apply iterative Voronoi refinement to improve point distribution."""
        current_points = points.copy()
        
        for _ in range(iterations):
            try:
                vor = Voronoi(current_points)
                new_points = []
                
                for i in range(len(current_points)):
                    region = vor.regions[vor.point_region[i]]
                    if not region or -1 in region:
                        # Keep original point if Voronoi region is invalid
                        new_points.append(current_points[i])
                        continue
                    
                    # Calculate centroid of Voronoi cell
                    vertices = [vor.vertices[j] for j in region if j >= 0]
                    if len(vertices) > 0:
                        vertices = np.array(vertices)
                        centroid = np.mean(vertices, axis=0)
                        new_points.append(centroid)
                    else:
                        new_points.append(current_points[i])
                
                # Update points and keep within bounds
                current_points = np.array(new_points)
                current_points = np.clip(current_points, 0, 1)
                
            except Exception:
                # If Voronoi computation fails, keep current points
                break
                
        return current_points
    
    def adaptive_local_optimization(initial_points, max_iter=500):
        """Apply adaptive local optimization with multiple strategies."""
        bounds = [(0, 1) for _ in range(n_points * dimension)]
        
        # Strategy 1: L-BFGS-B optimization
        try:
            result = minimize(
                combined_objective,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
        
        # Strategy 2: SLSQP as fallback
        try:
            result = minimize(
                combined_objective,
                initial_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter//2, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
            
        # Return original points if all optimization attempts fail
        return initial_points
    
    def multi_stage_optimization(initial_points):
        """Apply multi-stage optimization for better convergence."""
        # Stage 1: Voronoi refinement for global improvement
        refined_points = voronoi_refinement_step(initial_points, iterations=3)
        
        # Stage 2: Local optimization with combined objective
        final_points = adaptive_local_optimization(refined_points, max_iter=300)
        
        # Stage 3: Additional Voronoi refinement
        final_points = voronoi_refinement_step(final_points, iterations=2)
        
        # Stage 4: Final local optimization
        final_points = adaptive_local_optimization(final_points, max_iter=200)
        
        return final_points
    
    # Generate initial configuration
    initial_points = generate_hexagonal_initial()
    
    # Apply multi-stage optimization
    optimized_points = multi_stage_optimization(initial_points)
    
    # Final bounds checking
    optimized_points = np.clip(optimized_points, 0, 1)
    
    return optimized_points

# EVOLVE-BLOCK-END
