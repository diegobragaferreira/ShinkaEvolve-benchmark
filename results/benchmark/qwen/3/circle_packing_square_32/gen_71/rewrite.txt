# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses Voronoi-based refinement for efficient optimization.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 32
    
    # Initialize with random positions and small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        # Ensure initial positions are within bounds with buffer
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = 0.02
        circles[i] = [x, y, r]
    
    # Helper function to compute Voronoi diagram and cell areas
    def compute_voronoi(cells):
        try:
            vor = Voronoi(cells)
            return vor
        except:
            # Fallback for degenerate cases
            return None
    
    # Helper function to check if a point is inside the unit square
    def is_valid_point(x, y):
        return 0 <= x <= 1 and 0 <= y <= 1
    
    # Helper function to get Voronoi cell vertices
    def get_voronoi_cell_vertices(vor, point_idx):
        if vor is None:
            return []
        try:
            # Get indices of ridges belonging to this point
            ridge_indices = [i for i, ridge in enumerate(vor.ridge_vertices) if point_idx in vor.ridge_points[i]]
            vertices = []
            for idx in ridge_indices:
                if not isinstance(vor.ridge_vertices[idx], list):
                    continue
                for vertex_idx in vor.ridge_vertices[idx]:
                    if vertex_idx >= 0 and vertex_idx < len(vor.vertices):
                        vertices.append(vor.vertices[vertex_idx])
            return vertices
        except:
            return []
    
    # Main optimization loop
    max_iterations = 500
    improvement_threshold = 1e-6
    
    for iteration in range(max_iterations):
        improved = False
        
        # Compute Voronoi diagram for current configuration
        positions = circles[:, :2]
        vor = compute_voronoi(positions)
        
        # Try to expand each circle based on Voronoi constraints
        for i in range(n):
            x, y, r = circles[i]
            old_r = r
            
            # Find maximum possible radius considering:
            # 1. Boundary constraints
            max_r_boundary = min(x, 1-x, y, 1-y)
            
            # 2. Voronoi-based constraints (approximate)
            max_r_voronoi = float('inf')
            if vor is not None:
                # Get neighboring points for this circle
                neighbors = []
                try:
                    # Find ridges involving this point
                    ridge_indices = [idx for idx, points in enumerate(vor.ridge_points) if i in points]
                    for idx in ridge_indices:
                        other_point_idx = vor.ridge_points[idx][0] if vor.ridge_points[idx][0] != i else vor.ridge_points[idx][1]
                        if other_point_idx >= 0 and other_point_idx < n:
                            neighbors.append(other_point_idx)
                except:
                    pass
                
                # For each neighbor, calculate minimum distance constraint
                min_neighbor_distance = float('inf')
                for j in neighbors:
                    if i != j:
                        dist = np.sqrt((x - circles[j, 0])**2 + (y - circles[j, 1])**2)
                        # This is a simplified approximation; actual Voronoi would be more precise
                        min_neighbor_distance = min(min_neighbor_distance, dist)
                
                if min_neighbor_distance < float('inf') and min_neighbor_distance > 0:
                    # Maximum radius such that r + r_neighbor <= distance
                    # But also account for current radius of neighbor
                    max_r_voronoi = min_neighbor_distance - circles[i, 2]  # Simplified
                    max_r_voronoi = max(0, max_r_voronoi)
            
            # Take the minimum of all constraints
            max_radius = min(max_r_boundary, max_r_voronoi if max_r_voronoi != float('inf') else max_r_boundary)
            
            # Increase radius slightly
            new_r = min(max_radius, old_r + 0.002)  # Small incremental updates
            
            if new_r > old_r + improvement_threshold:
                circles[i, 2] = new_r
                improved = True
        
        # Validate and adjust positions to prevent overlaps and boundary violations
        # Project circles back into valid positions
        for i in range(n):
            x, y, r = circles[i]
            
            # Ensure boundary constraints
            r = min(r, x, 1-x, y, 1-y)
            
            # Ensure valid center positions
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[i] = [x, y, r]
        
        # Refine by checking overlaps and reducing radii if needed
        distances = cdist(circles[:, :2], circles[:, :2])
        overlap_detected = False
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r_i, r_j = circles[i, 2], circles[j, 2]
                
                if dist < r_i + r_j:
                    # Overlap detected - reduce both radii
                    overlap_amount = (r_i + r_j - dist) * 0.5
                    new_r_i = max(0.001, r_i - overlap_amount)
                    new_r_j = max(0.001, r_j - overlap_amount)
                    
                    circles[i, 2] = new_r_i
                    circles[j, 2] = new_r_j
                    overlap_detected = True
        
        # Early stopping if no significant improvement
        if not improved and not overlap_detected:
            break
    
    # Final validation step
    def final_validation(circles_array):
        # Ensure all constraints are satisfied
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Fix boundary violations
            r = min(r, x, 1-x, y, 1-y)
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles_array[i] = [x, y, r]
        
        # Resolve any remaining overlaps
        distances = cdist(circles_array[:, :2], circles_array[:, :2])
        for _ in range(200):  # max iterations for overlap correction
            changed = False
            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    dist = distances[i, j]
                    r_i, r_j = circles_array[i, 2], circles_array[j, 2]
                    if dist < r_i + r_j:
                        # Reduce both radii
                        overlap = (r_i + r_j - dist) * 0.3
                        if r_i > overlap and r_j > overlap:
                            circles_array[i, 2] = max(0.001, r_i - overlap)
                            circles_array[j, 2] = max(0.001, r_j - overlap)
                            changed = True
            if not changed:
                break
        
        # Final boundary check
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            r = min(r, x, 1-x, y, 1-y)
            circles_array[i] = [x, y, r]
        
        return circles_array
    
    circles = final_validation(circles)
    
    return circles

# EVOLVE-BLOCK-END