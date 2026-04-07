# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
import random
from itertools import combinations

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.0, 1.0

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Phase 1: Voronoi-based initialization with strategic seed placement
    n_circles = 21
    
    # Generate strategic seed points using a combination of regular grid and random sampling
    seed_points = []
    
    # Add corner points
    seed_points.extend([(0.05, 0.05), (width-0.05, 0.05), (0.05, height-0.05), (width-0.05, height-0.05)])
    
    # Add edge center points
    seed_points.extend([(width/2, 0.05), (width/2, height-0.05), (0.05, height/2), (width-0.05, height/2)])
    
    # Add center point
    seed_points.append((width/2, height/2))
    
    # Fill remaining slots with grid-based sampling
    grid_density = 5
    for i in range(grid_density):
        for j in range(grid_density):
            if len(seed_points) < n_circles:
                x = 0.05 + (i + 0.5) * (width - 0.1) / grid_density
                y = 0.05 + (j + 0.5) * (height - 0.1) / grid_density
                seed_points.append((x, y))
    
    # Trim to exact number needed
    if len(seed_points) > n_circles:
        seed_points = seed_points[:n_circles]
    elif len(seed_points) < n_circles:
        # Add random points if needed
        while len(seed_points) < n_circles:
            x = random.uniform(0.05, width - 0.05)
            y = random.uniform(0.05, height - 0.05)
            seed_points.append((x, y))
    
    # Phase 2: Voronoi-guided optimization
    # Create initial configuration
    circles = np.zeros((n_circles, 3))
    for i, (x, y) in enumerate(seed_points):
        circles[i] = [x, y, 0.02]
    
    # Optimize using Voronoi-based approach
    best_sum = -1
    best_circles = None
    
    # Multiple optimization runs with different random seeds
    for run in range(3):
        # Set different seed for each run
        np.random.seed(42 + run)
        random.seed(42 + run)
        
        # Create copy for this run
        current_circles = circles.copy()
        
        # Generate Voronoi diagram for current configuration
        points = current_circles[:, :2]
        
        try:
            vor = Voronoi(points)
            
            # Extract potential improvement points from Voronoi structure
            improvement_points = []
            
            # Add Voronoi vertices
            for vertex in vor.vertices:
                x, y = vertex
                if 0.05 <= x <= width - 0.05 and 0.05 <= y <= height - 0.05:
                    improvement_points.append((x, y))
            
            # Add edge centers for better coverage
            for simplex in vor.simplices:
                # Compute centroid of simplex
                centroid_x = np.mean([points[i][0] for i in simplex])
                centroid_y = np.mean([points[i][1] for i in simplex])
                if 0.05 <= centroid_x <= width - 0.05 and 0.05 <= centroid_y <= height - 0.05:
                    improvement_points.append((centroid_x, centroid_y))
            
            # Use a hybrid approach: first optimize some circles, then refine
            # Optimized circles in batches
            batch_size = min(5, n_circles)
            
            for batch_start in range(0, n_circles, batch_size):
                batch_end = min(batch_start + batch_size, n_circles)
                
                # For each circle in batch, compute improved position
                for i in range(batch_start, batch_end):
                    # Get current position and radius
                    current_x, current_y, current_r = current_circles[i]
                    
                    # Compute maximum possible radius at current position
                    max_radius = min(current_x, width - current_x, current_y, height - current_y)
                    
                    # Check overlap with other circles
                    for j in range(n_circles):
                        if i != j:
                            x2, y2, r2 = current_circles[j]
                            dist = distance.euclidean((current_x, current_y), (x2, y2))
                            if dist > 0:
                                max_radius = min(max_radius, dist - r2)
                    
                    # Clip to valid range
                    max_radius = max(0.001, max_radius)
                    current_circles[i] = [current_x, current_y, max_radius]
            
            # Local refinement using optimization around Voronoi structure
            refinement_attempts = 100
            for attempt in range(refinement_attempts):
                improved = False
                
                # Try to improve each circle
                for i in range(n_circles):
                    current_x, current_y, current_r = current_circles[i]
                    
                    # Calculate max radius without overlaps
                    max_radius = min(current_x, width - current_x, current_y, height - current_y)
                    
                    for j in range(n_circles):
                        if i != j:
                            x2, y2, r2 = current_circles[j]
                            dist = distance.euclidean((current_x, current_y), (x2, y2))
                            if dist > 0:
                                max_radius = min(max_radius, dist - r2)
                    
                    max_radius = max(0.001, max_radius)
                    
                    # If radius can be increased, do it
                    if max_radius > current_r:
                        current_circles[i] = [current_x, current_y, max_radius]
                        improved = True
                
                if not improved:
                    break
                    
        except Exception:
            # Fallback to basic optimization if Voronoi fails
            for _ in range(200):
                improved = False
                for i in range(n_circles):
                    current_x, current_y, current_r = current_circles[i]
                    
                    # Compute maximum possible radius
                    max_radius = min(current_x, width - current_x, current_y, height - current_y)
                    
                    # Check conflicts with other circles
                    for j in range(n_circles):
                        if i != j:
                            x2, y2, r2 = current_circles[j]
                            dist = distance.euclidean((current_x, current_y), (x2, y2))
                            if dist > 0:
                                max_radius = min(max_radius, dist - r2)
                    
                    max_radius = max(0.001, max_radius)
                    
                    if max_radius > current_r:
                        current_circles[i] = [current_x, current_y, max_radius]
                        improved = True
                
                if not improved:
                    break
        
        # Validate final configuration
        if _validate_configuration(current_circles, width, height):
            current_sum = np.sum(current_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = current_circles.copy()
    
    # If no good solution found, use simple grid approach
    if best_circles is None:
        # Generate simple grid configuration
        rows, cols = 4, 6
        x_spacing = width / (cols + 1)
        y_spacing = height / (rows + 1)
        
        circles = np.zeros((n_circles, 3))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * x_spacing + random.uniform(-x_spacing*0.05, x_spacing*0.05)
                y = (i + 1) * y_spacing + random.uniform(-y_spacing*0.05, y_spacing*0.05)
                circles[idx] = [x, y, 0.01]
                idx += 1
        
        # Final optimization pass
        for _ in range(100):
            improved = False
            for i in range(n_circles):
                x, y, r = circles[i]
                max_radius = min(x, width - x, y, height - y)
                
                for j in range(n_circles):
                    if i != j:
                        px, py, pr = circles[j]
                        dist = distance.euclidean((x, y), (px, py))
                        if dist > 0:
                            max_radius = min(max_radius, dist - pr)
                
                max_radius = max(0.001, max_radius)
                if max_radius > r:
                    circles[i, 2] = max_radius
                    improved = True
            
            if not improved:
                break
        
        best_circles = circles
    
    # Final boundary correction
    for i in range(n_circles):
        x, y, r = best_circles[i]
        # Ensure circles are within bounds and radius is reasonable
        r = min(r, x, width - x, y, height - y)
        if r <= 0.001:
            r = 0.01
        best_circles[i] = [x, y, r]

    return best_circles

def _validate_configuration(circles, width, height):
    """Validate that all circles are within bounds and non-overlapping."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Check boundary conditions
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            return False

        # Check overlap with other circles
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
            # Overlap occurs when distance < sum of radii
            if distance < r + r2 - 1e-8:
                return False

    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")