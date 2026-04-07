# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
import warnings
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the minimum to maximum distance ratio"""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = distance.pdist(points)
        
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def voronoi_centroid_adjustment(points, max_iterations=5):
        """Adjust points based on Voronoi region centroids to improve uniformity"""
        adjusted_points = points.copy()
        
        for iteration in range(max_iterations):
            try:
                # Create Voronoi diagram
                vor = Voronoi(adjusted_points)
                
                # Compute centroids of Voronoi regions
                new_points = []
                valid_regions = 0
                
                for i in range(len(adjusted_points)):
                    # Get vertices of Voronoi region for point i
                    region_indices = np.where(vor.point_region == i)[0]
                    if len(region_indices) > 0:
                        region_id = region_indices[0]
                        vertices = vor.vertices[vor.region[region_id]]
                        
                        # Skip unbounded or degenerate regions
                        if len(vertices) >= 3:
                            # Compute centroid of polygon
                            centroid = np.mean(vertices, axis=0)
                            
                            # Move point towards centroid with damping
                            damping = 0.3
                            new_point = adjusted_points[i] + damping * (centroid - adjusted_points[i])
                            
                            # Ensure point stays within bounds
                            new_point = np.clip(new_point, 1e-8, 1-1e-8)
                            new_points.append(new_point)
                            valid_regions += 1
                        else:
                            new_points.append(adjusted_points[i])
                    else:
                        new_points.append(adjusted_points[i])
                
                # Update points
                if len(new_points) == len(adjusted_points):
                    adjusted_points = np.array(new_points)
                    
            except Exception:
                # Fall back to simple smoothing if Voronoi fails
                break
                
        return adjusted_points
    
    def initialize_voronoi_based_configuration():
        """Initialize points using Voronoi-inspired geometric construction"""
        np.random.seed(42)
        
        # Start with a regular pattern and add Voronoi-based refinement
        points = []
        
        # Create a hexagonal-like pattern with controlled spacing
        rows, cols = 4, 4
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Regular grid position with offset
                x = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
                y = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
                if i % 2 == 1:  # Offset odd rows
                    x += 0.4 / (cols - 1) if cols > 1 else 0.2
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Add Voronoi-based perturbations for better uniformity
        for _ in range(10):
            points = voronoi_centroid_adjustment(points, max_iterations=3)
            # Add small random noise to avoid plateaus
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
            points = np.clip(points, 0.05, 0.95)
        
        return points
    
    def local_improvement_step(points, max_iters=20):
        """Local optimization using gradient-based methods"""
        def objective(x):
            points_arr = x.reshape(-1, 2)
            distances = distance.pdist(points_arr)
            if len(distances) == 0:
                return 1e10
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 1e10
            return -min_dist / max_dist  # Negative for minimization
        
        # Try multiple local optimization methods
        bounds = [(1e-8, 1-1e-8) for _ in range(32)]
        
        # Try L-BFGS-B first
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
            
        # Fallback to simpler approach if needed
        return points
    
    def evolutionary_voronoi_refinement(initial_points, max_generations=20):
        """Main evolutionary refinement using Voronoi structures"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        best_ratio = compute_min_max_ratio(current_points)
        
        # Evolutionary loop
        for generation in range(max_generations):
            # Apply Voronoi-based adjustment
            updated_points = voronoi_centroid_adjustment(current_points, max_iterations=3)
            
            # Apply local improvement
            improved_points = local_improvement_step(updated_points)
            
            # Refine with additional Voronoi steps
            refined_points = voronoi_centroid_adjustment(improved_points, max_iterations=2)
            
            # Calculate ratio for this configuration
            ratio = compute_min_max_ratio(refined_points)
            
            # Accept better solutions
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
            
            # Continue evolution with current points
            current_points = refined_points.copy()
        
        return best_points
    
    # Main algorithm execution
    try:
        # Step 1: Initialize with Voronoi-based configuration
        initial_config = initialize_voronoi_based_configuration()
        
        # Step 2: Apply evolutionary Voronoi refinement
        final_points = evolutionary_voronoi_refinement(initial_config, max_generations=15)
        
        # Step 3: Final local optimization
        final_points = local_improvement_step(final_points)
        
        # Step 4: Additional Voronoi refinement
        final_points = voronoi_centroid_adjustment(final_points, max_iterations=3)
        
        # Final validation
        final_points = np.clip(final_points, 1e-8, 1-1e-8)
        
        # Ensure we have valid output
        if len(final_points) != 16:
            # Fallback to default initialization
            final_points = np.random.rand(16, 2)
            final_points = np.clip(final_points, 1e-8, 1-1e-8)
            
    except Exception as e:
        # Last resort fallback
        warnings.warn(f"Voronoi-evolve algorithm failed: {e}")
        final_points = np.random.rand(16, 2)
        final_points = np.clip(final_points, 1e-8, 1-1e-8)
    
    return final_points

# EVOLVE-BLOCK-END