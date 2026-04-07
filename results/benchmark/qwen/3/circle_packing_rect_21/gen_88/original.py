# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
import math
from itertools import combinations

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    rect_width = 1.2
    rect_height = 0.8
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    
    # Enhanced initial configuration using hexagonal grid pattern
    num_points = 21
    points = []
    
    # Create hexagonal grid pattern for better spatial distribution
    rows, cols = 5, 5
    grid_x = np.linspace(0.1, rect_width - 0.1, cols)
    grid_y = np.linspace(0.1, rect_height - 0.1, rows)
    
    # Hexagonal offset pattern with controlled randomness
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            if len(points) < num_points:
                # Add controlled random perturbation
                pert_x = np.random.uniform(-0.025, 0.025)
                pert_y = np.random.uniform(-0.025, 0.025)
                points.append([x + pert_x, y + pert_y])
    
    # Trim to exactly 21 points
    points = points[:num_points]
    points = np.array(points)
    
    # Optimization parameters
    best_sum = 0
    best_circles = None
    last_improvement = 0
    patience = 50
    
    # Main evolutionary loop
    for iteration in range(1000):
        try:
            # Construct Voronoi diagram
            vor = Voronoi(points)
        except:
            # Fallback if Voronoi construction fails
            vor = Voronoi(points[:len(points)-1] if len(points) > 1 else points)
        
        # Calculate circle parameters for each Voronoi cell
        circles = []
        
        # Process each Voronoi region
        for i, point in enumerate(points):
            try:
                # Get vertices of the Voronoi cell for this generator
                region_indices = np.where(vor.point_region == i)[0]
                
                if len(region_indices) > 0 and region_indices[0] >= 0:
                    # Find vertices belonging to this region
                    region_vertices = vor.vertices[vor.regions[region_indices[0]]]
                else:
                    raise IndexError("Invalid Voronoi region")
            except (IndexError, ValueError):
                # Fallback to simple calculation if Voronoi geometry is problematic
                region_vertices = []
            
            # Filter vertices that are within the rectangle bounds
            valid_vertices = []
            for vertex in region_vertices:
                if (0 <= vertex[0] <= rect_width and 0 <= vertex[1] <= rect_height):
                    valid_vertices.append(vertex)
            
            # Calculate center and radius
            center_x, center_y = point
            
            # Calculate distances to rectangle edges
            dist_to_edges = [
                center_x,
                rect_width - center_x,
                center_y,
                rect_height - center_y
            ]
            
            # Calculate minimum distance to other circles
            min_dist_to_others = float('inf')
            for j, other_point in enumerate(points):
                if i != j:
                    dist = distance.euclidean(point, other_point)
                    min_dist_to_others = min(min_dist_to_others, dist)
            
            # Compute maximum possible radius
            max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
            radius = max(0.001, max_radius)
            circles.append([center_x, center_y, radius])
        
        # Convert to numpy array
        circles = np.array(circles)
        
        # Validate configuration - check for overlaps efficiently
        valid = True
        total_radius = np.sum(circles[:, 2])
        
        # Early termination for overlap checking
        if len(circles) > 1:
            # Use faster pairwise comparison with early stopping
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    dist = distance.euclidean(circles[i][:2], circles[j][:2])
                    if dist < (circles[i][2] + circles[j][2]):
                        valid = False
                        break
                if not valid:
                    break
        
        # Accept better configurations with patience mechanism
        if valid and total_radius > best_sum:
            best_sum = total_radius
            best_circles = circles.copy()
            last_improvement = iteration
        
        # Early stopping if no improvement for too long
        if iteration - last_improvement > patience:
            break
        
        # Apply adaptive perturbations to points for evolution
        new_points = points.copy()
        for i in range(num_points):
            # Adaptive perturbation based on iteration progress
            scale = max(0.005, 0.02 * (1 - iteration/1000.0))
            new_points[i, 0] += np.random.normal(0, scale)
            new_points[i, 1] += np.random.normal(0, scale)
            
            # Keep within bounds with buffer
            new_points[i, 0] = np.clip(new_points[i, 0], 0.01, rect_width - 0.01)
            new_points[i, 1] = np.clip(new_points[i, 1], 0.01, rect_height - 0.01)
        
        points = new_points
    
    # Final refinement step with enhanced local optimization
    if best_circles is not None:
        refined_circles = best_circles.copy()
        
        # More aggressive local optimization
        for _ in range(300):
            # Select a random circle to optimize
            idx = np.random.randint(0, len(refined_circles))
            
            # Get current circle
            cx, cy, cr = refined_circles[idx]
            
            # Try to find a better position using grid search with adaptive steps
            best_cx, best_cy, best_cr = cx, cy, cr
            best_radius = cr
            
            # Adaptive search space based on current radius
            search_range = max(0.01, cr * 0.5)
            step_sizes = [search_range * 0.5, search_range * 0.25, search_range * 0.1, 0, -search_range * 0.1, -search_range * 0.25, -search_range * 0.5]
            
            # Sample nearby positions
            for dx in step_sizes:
                for dy in step_sizes:
                    ncx = cx + dx
                    ncy = cy + dy
                    
                    # Check bounds
                    if (ncx < 0.01 or ncx > rect_width - 0.01 or 
                        ncy < 0.01 or ncy > rect_height - 0.01):
                        continue
                    
                    # Compute max radius at new position
                    max_r = min(ncx, rect_width - ncx, ncy, rect_height - ncy)
                    
                    # Check overlap with others
                    overlap = False
                    for j in range(len(refined_circles)):
                        if j != idx:
                            dist = math.sqrt((ncx - refined_circles[j, 0])**2 + (ncy - refined_circles[j, 1])**2)
                            if dist < max_r + refined_circles[j, 2]:  # Overlap
                                overlap = True
                                break
                    
                    if not overlap and max_r > best_radius:
                        best_radius = max_r
                        best_cx, best_cy = ncx, ncy
            
            # Update if improvement found
            if best_radius > refined_circles[idx, 2]:
                refined_circles[idx, 0] = best_cx
                refined_circles[idx, 1] = best_cy
                refined_circles[idx, 2] = best_radius
    
    # Ensure we return exactly 21 circles
    if best_circles is None:
        # Fallback to a simple initial configuration
        circles = np.zeros((21, 3))
        # Place in a simple grid pattern
        row_size = int(np.ceil(np.sqrt(21)))
        col_size = int(np.ceil(21 / row_size))
        
        spacing_x = rect_width / (col_size + 1)
        spacing_y = rect_height / (row_size + 1)
        
        count = 0
        for i in range(row_size):
            for j in range(col_size):
                if count < 21:
                    x = spacing_x * (j + 1)
                    y = spacing_y * (i + 1)
                    # Set radius to be proportional to available space
                    radius = min(x, rect_width - x, y, rect_height - y) * 0.4
                    circles[count] = [x, y, max(radius, 0.001)]
                    count += 1
        
        return circles
    
    return refined_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")