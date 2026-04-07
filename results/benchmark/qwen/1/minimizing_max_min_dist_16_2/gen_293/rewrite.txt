# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import Voronoi
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x):
        # Reshape x into points
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Compute min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing the negative)
        if d_max == 0:
            return -1.0
        return -d_min / d_max
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max
    
    def generate_hexagonal_initial():
        """Generate a highly structured hexagonal initial configuration"""
        np.random.seed(42)
        points = []
        
        # Use a 4x4 grid with proper hexagonal spacing
        rows, cols = 4, 4
        sqrt3 = np.sqrt(3)
        
        # Create hexagonal pattern with mathematical precision
        row_height = 0.8 * sqrt3 / 2  # Vertical spacing for hexagons
        col_width = 0.8  # Horizontal spacing
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Hexagonal offset pattern
                x = 0.1 + j * col_width / 3.0
                y = 0.1 + i * row_height / 3.0
                
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += col_width / 6.0
                
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Add small random jitter to break symmetry
        points += np.random.normal(0, 0.005, points.shape)
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        
        return points
    
    def generate_spiral_initial():
        """Generate a spiral initial configuration"""
        np.random.seed(42)
        points = []
        
        # Create a spiral pattern
        for i in range(16):
            if i == 0:
                points.append([0.5, 0.5])  # Center point
            else:
                angle = i * 2.5  # Angle in radians
                radius = min(0.4, i * 0.03)  # Radius increases gradually
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_grid_initial():
        """Generate a regular grid initial configuration"""
        np.random.seed(42)
        points = []
        
        # Regular 4x4 grid
        x_vals = np.linspace(0.1, 0.9, 4)
        y_vals = np.linspace(0.1, 0.9, 4)
        
        for i in range(4):
            for j in range(4):
                points.append([x_vals[i], y_vals[j]])
        
        points = np.array(points[:16])
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def voronoi_refinement(points, iterations=3):
        """Apply iterative Voronoi refinement to improve point distribution"""
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
                        # Ensure centroid is within bounds
                        centroid = np.clip(centroid, 0, 1)
                        new_points.append(centroid)
                    else:
                        new_points.append(current_points[i])
                
                # Update points
                current_points = np.array(new_points)
                
            except Exception:
                # If Voronoi computation fails, keep current points
                break
                
        return current_points
    
    def adaptive_local_optimization(initial_points, max_iter=300):
        """Apply adaptive local optimization with multiple strategies"""
        bounds = [(0, 1) for _ in range(32)]
        
        # Strategy 1: SLSQP optimization (best for constrained problems)
        try:
            result = minimize(
                objective,
                initial_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
        
        # Strategy 2: L-BFGS-B as fallback
        try:
            result = minimize(
                objective,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                return result.x.reshape(-1, 2)
        except:
            pass
            
        # Return original points if all optimization attempts fail
        return initial_points
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        generate_hexagonal_initial(),
        generate_spiral_initial(),
        generate_grid_initial()
    ]
    
    best_ratio = -np.inf
    best_points = None
    
    # Try multiple initial configurations
    for i, initial_config in enumerate(initial_configs):
        try:
            # Flatten for optimization
            x0 = initial_config.flatten()
            
            # Define bounds for each coordinate (0 to 1 for both x and y)
            bounds = [(0, 1) for _ in range(32)]
            
            # Phase 1: Global optimization with Differential Evolution - improved parameters
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42+i,
                maxiter=300,  # More iterations for better search
                popsize=30,   # Larger population for better diversity
                tol=1e-10,    # Tighter tolerance
                mutation=(0.7, 1.2),  # Wider mutation range
                recombination=0.9,    # Higher recombination rate
                disp=False
            )
            
            # Phase 2: Local refinement with adaptive optimization
            refined_points = adaptive_local_optimization(de_result.x.reshape(-1, 2), max_iter=300)
            
            # Phase 3: Voronoi refinement for better distribution
            final_points = voronoi_refinement(refined_points, iterations=3)
            
            # Calculate final ratio
            final_ratio = compute_min_max_ratio(final_points)
            
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()
                
        except Exception as e:
            continue
    
    # Fallback to hexagonal initial if nothing worked
    if best_points is None:
        initial_points = generate_hexagonal_initial()
        final_points = voronoi_refinement(initial_points, iterations=2)
        final_points = adaptive_local_optimization(final_points, max_iter=200)
        best_points = final_points
    
    # Ensure final points are within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END