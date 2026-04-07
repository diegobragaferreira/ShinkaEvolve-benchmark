# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, setting to 1x1 for simplicity (perimeter = 4)
    width, height = 1.0, 1.0
    
    # Initialize parameters
    n_circles = 21
    random.seed(42)
    
    def initialize_circles_adaptive():
        """Initialize circles with adaptive grid placement considering aspect ratio and optimal distribution"""
        circles = np.zeros((n_circles, 3))

        # Adaptive grid based on aspect ratio and circle count
        aspect_ratio = width / height
        # For 21 circles, find best rectangular grid layout
        best_cols = 1
        best_rows = n_circles
        min_aspect_diff = float('inf')

        # Test various grid configurations to find most balanced one
        for cols in range(1, n_circles + 1):
            rows = math.ceil(n_circles / cols)
            if cols * rows >= n_circles:
                grid_aspect = cols / rows
                aspect_diff = abs(grid_aspect - aspect_ratio)
                if aspect_diff < min_aspect_diff:
                    min_aspect_diff = aspect_diff
                    best_cols = cols
                    best_rows = rows

        # Ensure we don't exceed the required number of circles
        actual_n = best_cols * best_rows
        if actual_n > n_circles:
            # Reduce grid size if necessary
            while best_cols * best_rows > n_circles:
                best_cols = max(1, best_cols - 1)
                best_rows = math.ceil(n_circles / best_cols)

        # Calculate spacing
        spacing_x = width / (best_cols + 1)
        spacing_y = height / (best_rows + 1)

        # Create grid with hexagonal offset for better packing
        idx = 0
        for i in range(best_cols):
            for j in range(best_rows):
                if idx >= n_circles:
                    break
                # Offset every other row for better packing
                offset = (j % 2) * spacing_x * 0.5
                x = (i + 1) * spacing_x + offset + random.uniform(-spacing_x*0.05, spacing_x*0.05)
                y = (j + 1) * spacing_y + random.uniform(-spacing_y*0.05, spacing_y*0.05)
                # Initial radius based on spacing but adjusted for aspect ratio
                base_radius = min(spacing_x, spacing_y) * 0.12
                # Adjust radius based on position - center positions can have larger radii
                center_x, center_y = width/2, height/2
                dist_to_center = math.sqrt((x-center_x)**2 + (y-center_y)**2)
                max_dist_to_center = math.sqrt((width/2)**2 + (height/2)**2)
                adjustment_factor = 0.7 + 0.3 * (1.0 - dist_to_center/max_dist_to_center) if max_dist_to_center > 0 else 0.7
                r = base_radius * adjustment_factor
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n_circles:
                break

        # Adjust initial radii to ensure they fit within bounds properly
        for i in range(n_circles):
            x, y, r = circles[i]
            min_dist = min(x, y, width - x, height - y)
            # Allow larger radii for central regions, smaller for edges
            center_x, center_y = width/2, height/2
            dist_to_center = math.sqrt((x-center_x)**2 + (y-center_y)**2)
            max_dist_to_center = math.sqrt((width/2)**2 + (height/2)**2)
            adjustment_factor = 0.8 + 0.2 * (1.0 - dist_to_center/max_dist_to_center) if max_dist_to_center > 0 else 0.8
            circles[i][2] = min(r, min_dist * adjustment_factor * 0.9)

        return circles
    
    def get_spatial_index(circles_array):
        """Create spatial index for fast neighbor lookups"""
        points = circles_array[:, :2]
        return cKDTree(points)
    
    def is_valid_position(circle):
        """Check if circle position is valid (within bounds)"""
        x, y, r = circle
        return (r <= x <= width - r and 
                r <= y <= height - r)
    
    def check_overlap(circle, existing_circles):
        """Check if circle overlaps with any existing circles"""
        x, y, r = circle
        for cx, cy, cr in existing_circles:
            dx = x - cx
            dy = y - cy
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < (r + cr):
                return True
        return False
    
    def calculate_total_penalty(circles_array):
        """Calculate combined penalty for boundary and overlap violations"""
        penalty = 0.0
        
        # Boundary penalties
        for circle in circles_array:
            x, y, r = circle
            boundary_dist = min(x, y, width - x, height - y)
            if boundary_dist < r:
                penalty += (r - boundary_dist) ** 3  # Cubic penalty for boundary violations
        
        # Overlap penalties
        for i in range(len(circles_array)):
            x1, y1, r1 = circles_array[i]
            for j in range(i):
                x2, y2, r2 = circles_array[j]
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)
                overlap = max(0, (r1 + r2) - distance)
                if overlap > 0:
                    penalty += overlap ** 3  # Cubic penalty for overlaps
        
        return penalty
    
    # Physics-based optimization with adaptive parameters
    def apply_adaptive_physics_update(circles_array, max_iter=1000, stage="coarse"):
        """Apply physics-based optimization with adaptive parameters based on optimization stage"""
        tree = get_spatial_index(circles_array)
        
        # Stage-specific parameters
        if stage == "coarse":
            repulsion_strength = 3.0
            attraction_strength = 0.05
            step_size = 0.02
            max_neighbors = 5
        elif stage == "medium":
            repulsion_strength = 4.0
            attraction_strength = 0.1
            step_size = 0.01
            max_neighbors = 8
        else:  # fine
            repulsion_strength = 5.0
            attraction_strength = 0.15
            step_size = 0.005
            max_neighbors = 12
        
        for iteration in range(max_iter):
            forces = np.zeros_like(circles_array)
            
            # Calculate repulsion forces
            for i in range(len(circles_array)):
                x1, y1, r1 = circles_array[i]
                
                # Find neighbors efficiently
                neighbors = tree.query_ball_point([x1, y1], 3 * (r1 + 0.01), p=2)
                # Limit neighbors for performance
                neighbors = neighbors[:max_neighbors]
                
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles_array[j]
                        
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0.001:
                            overlap_distance = max(0, (r1 + r2) - distance)
                            if overlap_distance > 0:
                                # Adaptive force calculation
                                force_magnitude = repulsion_strength * overlap_distance / (distance ** 2)
                                
                                forces[i, 0] -= force_magnitude * dx / distance
                                forces[i, 1] -= force_magnitude * dy / distance
            
            # Boundary and center attraction forces
            for i in range(len(circles_array)):
                x, y, r = circles_array[i]
                
                # Attract to center of rectangle
                center_x, center_y = width/2, height/2
                dx = center_x - x
                dy = center_y - y
                forces[i, 0] += attraction_strength * dx
                forces[i, 1] += attraction_strength * dy
                
                # Boundary forces (stronger near edges)
                boundary_force = 1.0
                
                if x - r < 0:
                    forces[i, 0] += boundary_force * (r - x) * 2
                if x + r > width:
                    forces[i, 0] -= boundary_force * (x + r - width) * 2
                if y - r < 0:
                    forces[i, 1] += boundary_force * (r - y) * 2
                if y + r > height:
                    forces[i, 1] -= boundary_force * (y + r - height) * 2
            
            # Apply updates
            for i in range(len(circles_array)):
                circles_array[i, 0] += forces[i, 0] * step_size
                circles_array[i, 1] += forces[i, 1] * step_size
                
                # Maintain positive radii
                if circles_array[i, 2] < 0.0001:
                    circles_array[i, 2] = 0.0001
                    
                # Enforce valid positions
                if not is_valid_position(circles_array[i]):
                    x, y, r = circles_array[i]
                    x = max(r, min(width - r, x))
                    y = max(r, min(height - r, y))
                    circles_array[i] = [x, y, r]
            
            # Early termination based on improvement rate
            if iteration % 50 == 0 and iteration > 0:
                penalty = calculate_total_penalty(circles_array)
                if penalty < 1e-4:
                    break
        
        return circles_array
    
    # Evolutionary refinement with multiple strategies
    def refine_with_evolution(circles_array):
        """Refine using multiple evolutionary approaches"""
        best_circles = circles_array.copy()
        best_sum = np.sum(best_circles[:, 2])
        
        # Multiple rounds of enhancement
        for round_num in range(3):
            # Try different mutation strategies
            for _ in range(150):
                test_circles = best_circles.copy()
                mutation_type = random.choice(["radius_increase", "position_shift", "combined"])
                
                # Select circle to modify
                idx = random.randint(0, len(test_circles) - 1)
                x, y, r = test_circles[idx]
                
                if mutation_type == "radius_increase":
                    # Increase radius
                    new_r = min(r * 1.03, 0.25)
                    test_circles[idx, 2] = new_r
                    
                elif mutation_type == "position_shift":
                    # Shift position slightly
                    shift_x = random.uniform(-0.01, 0.01)
                    shift_y = random.uniform(-0.01, 0.01)
                    test_circles[idx, 0] = max(r, min(width - r, x + shift_x))
                    test_circles[idx, 1] = max(r, min(height - r, y + shift_y)) 
                    
                else:  # combined
                    # Combined mutation
                    new_r = min(r * 1.02, 0.25)
                    shift_x = random.uniform(-0.005, 0.005)
                    shift_y = random.uniform(-0.005, 0.005)
                    test_circles[idx, 2] = new_r
                    test_circles[idx, 0] = max(r, min(width - r, x + shift_x))
                    test_circles[idx, 1] = max(r, min(height - r, y + shift_y))
                
                # Validate constraints
                if not is_valid_position(test_circles[idx]):
                    continue
                    
                # Check overlap
                temp_circles = np.delete(test_circles, idx, axis=0)
                if not check_overlap(test_circles[idx], temp_circles):
                    new_sum = np.sum(test_circles[:, 2])
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = test_circles.copy()
        
        return best_circles
    
    # Multi-stage optimization approach
    # Stage 1: Coarse optimization
    circles = initialize_circles_adaptive()
    circles = apply_adaptive_physics_update(circles, max_iter=500, stage="coarse")
    
    # Stage 2: Medium refinement
    circles = apply_adaptive_physics_update(circles, max_iter=300, stage="medium")
    
    # Stage 3: Evolutionary refinement
    circles = refine_with_evolution(circles)
    
    # Stage 4: Fine optimization
    circles = apply_adaptive_physics_update(circles, max_iter=200, stage="fine")
    
    # Stage 5: Final polishing
    # Attempt to maximize individual radii with constrained adjustments
    for _ in range(50):
        improvement_made = False
        for i in range(n_circles):
            original_circle = circles[i].copy()
            x, y, r = original_circle
            
            # Try to slightly increase radius
            new_r = min(r * 1.015, 0.3)  # Slightly more aggressive than before
            test_circle = [x, y, new_r]
            
            if is_valid_position(test_circle) and not check_overlap(test_circle, np.delete(circles, i, axis=0)):
                circles[i] = test_circle
                improvement_made = True
                
        if not improvement_made:
            break
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
