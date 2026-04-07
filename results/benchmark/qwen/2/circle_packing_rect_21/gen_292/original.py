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
    # Rectangle dimensions: width + height = 2
    # Using 1.2 x 0.8 rectangle which provides better packing efficiency
    width, height = 1.2, 0.8
    
    # Initialize parameters
    n_circles = 21
    random.seed(42)
    
    class CirclePackingOptimizer:
        def __init__(self, width, height, n_circles):
            self.width = width
            self.height = height
            self.n_circles = n_circles
            self.rect_area = width * height
            
        def initialize_circles(self):
            """Initialize circles with adaptive grid placement based on rectangle aspect ratio"""
            circles = np.zeros((self.n_circles, 3))
            
            # Calculate optimal spacing using area-based approach
            area_per_circle = self.rect_area / self.n_circles
            optimal_spacing = math.sqrt(area_per_circle)
            
            # Determine grid dimensions that best match the rectangle aspect ratio
            aspect_ratio = self.width / self.height
            
            # Find suitable grid dimensions
            cols = max(1, int(self.width / optimal_spacing))
            rows = max(1, int(self.height / optimal_spacing))
            
            # Adjust grid to accommodate all circles
            while cols * rows < self.n_circles:
                if aspect_ratio >= 1:  # landscape
                    cols += 1
                else:  # portrait
                    rows += 1
            
            # Ensure we don't create too many cells
            while cols * rows > self.n_circles and cols > 1 and rows > 1:
                if aspect_ratio >= 1:
                    cols -= 1
                else:
                    rows -= 1
            
            # Recalculate spacing based on final grid
            spacing_x = self.width / (cols + 1) if cols > 0 else self.width
            spacing_y = self.height / (rows + 1) if rows > 0 else self.height
            
            # Create hexagonal grid with offset rows
            idx = 0
            for i in range(cols):
                for j in range(rows):
                    if idx >= self.n_circles:
                        break
                    # Offset every other row for hexagonal packing
                    offset = (j % 2) * spacing_x * 0.5
                    x = (i + 1) * spacing_x + offset + random.uniform(-spacing_x*0.05, spacing_x*0.05)
                    y = (j + 1) * spacing_y + random.uniform(-spacing_y*0.05, spacing_y*0.05)
                    
                    # Calculate initial radius using adaptive approach
                    base_radius = min(spacing_x, spacing_y) * 0.25
                    r = base_radius
                    
                    # Adjust radius based on position for better distribution
                    center_x, center_y = self.width/2, self.height/2
                    dist_to_center = math.sqrt((x-center_x)**2 + (y-center_y)**2)
                    max_dist_to_center = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                    if max_dist_to_center > 0:
                        center_adjustment = 1.0 + 0.5 * (1.0 - dist_to_center/max_dist_to_center)
                        r *= center_adjustment
                    else:
                        r *= 1.0
                    
                    circles[idx] = [x, y, r]
                    idx += 1
                if idx >= self.n_circles:
                    break
            
            # Adjust initial radii to ensure proper boundary constraints
            for i in range(self.n_circles):
                x, y, r = circles[i]
                min_dist = min(x, y, self.width - x, self.height - y)
                center_x, center_y = self.width/2, self.height/2
                dist_to_center = math.sqrt((x-center_x)**2 + (y-center_y)**2)
                max_dist_to_center = math.sqrt((self.width/2)**2 + (self.height/2)**2)
                if max_dist_to_center > 0:
                    edge_adjustment = 0.7 + 0.3 * (1.0 - dist_to_center/max_dist_to_center)
                    circles[i][2] = min(r, min_dist * edge_adjustment * 0.8)
                else:
                    circles[i][2] = min(r, min_dist * 0.7)
            
            return circles
        
        def is_valid_position(self, circle):
            """Check if circle position is valid (within bounds)"""
            x, y, r = circle
            return (r <= x <= self.width - r and 
                    r <= y <= self.height - r)
        
        def calculate_constraint_penalty(self, circles_array):
            """Calculate total penalty for boundary and overlap violations"""
            penalty = 0.0
            
            # Check boundary violations
            for circle in circles_array:
                x, y, r = circle
                boundary_dist = min(x, y, self.width - x, self.height - y)
                if boundary_dist < r:
                    penalty += (r - boundary_dist) ** 2
            
            # Check overlap violations using efficient spatial indexing
            points = circles_array[:, :2]
            tree = cKDTree(points)
            
            # Query pairs efficiently using spatial indexing
            max_radius = np.max(circles_array[:, 2])
            try:
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')
                for i, j in pairs:
                    x1, y1, r1 = circles_array[i]
                    x2, y2, r2 = circles_array[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx*dx + dy*dy)
                    overlap = max(0, (r1 + r2) - distance)
                    if overlap > 0:
                        penalty += overlap ** 2
            except:
                # Fallback to brute force if spatial indexing fails
                for i in range(len(circles_array)):
                    x1, y1, r1 = circles_array[i]
                    for j in range(i):
                        x2, y2, r2 = circles_array[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)
                        overlap = max(0, (r1 + r2) - distance)
                        if overlap > 0:
                            penalty += overlap ** 2
            
            return penalty
        
        def get_spatial_index(self, circles_array):
            """Create spatial index for fast neighbor lookups"""
            points = circles_array[:, :2]
            return cKDTree(points)
        
        def check_overlap(self, circle, existing_circles):
            """Check if circle overlaps with any existing circles"""
            x, y, r = circle
            for cx, cy, cr in existing_circles:
                dx = x - cx
                dy = y - cy
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < (r + cr):
                    return True
            return False
        
        def apply_physics_update(self, circles_array, max_iter=800):
            """Apply physics-based optimization to improve packing"""
            tree = self.get_spatial_index(circles_array)
            repulsion_strength = 2.0
            attraction_strength = 0.1
            
            for iteration in range(max_iter):
                forces = np.zeros_like(circles_array)
                
                # Calculate repulsion forces using spatial indexing
                for i in range(len(circles_array)):
                    x1, y1, r1 = circles_array[i]
                    
                    # Find neighbors within reasonable range
                    neighbors = tree.query_ball_point([x1, y1], 3 * (r1 + 0.01), p=2)
                    for j in neighbors:
                        if i != j:
                            x2, y2, r2 = circles_array[j]
                            
                            dx = x2 - x1
                            dy = y2 - y1
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            if distance > 0.001:
                                overlap_distance = (r1 + r2) - distance
                                if overlap_distance > 0:
                                    force_magnitude = repulsion_strength * overlap_distance / (distance ** 2)
                                    
                                    forces[i, 0] -= force_magnitude * dx / distance
                                    forces[i, 1] -= force_magnitude * dy / distance
                
                # Apply boundary and center attraction forces
                for i in range(len(circles_array)):
                    x, y, r = circles_array[i]
                    
                    # Attract to center of rectangle
                    center_x, center_y = self.width/2, self.height/2
                    dx = center_x - x
                    dy = center_y - y
                    forces[i, 0] += attraction_strength * dx
                    forces[i, 1] += attraction_strength * dy
                    
                    # Boundary forces
                    boundary_force = 0.5
                    
                    if x - r < 0:
                        forces[i, 0] += boundary_force * (r - x)
                    if x + r > self.width:
                        forces[i, 0] -= boundary_force * (x + r - self.width)
                    if y - r < 0:
                        forces[i, 1] += boundary_force * (r - y)
                    if y + r > self.height:
                        forces[i, 1] -= boundary_force * (y + r - self.height)
                
                # Update positions
                step_size = 0.01
                for i in range(len(circles_array)):
                    circles_array[i, 0] += forces[i, 0] * step_size
                    circles_array[i, 1] += forces[i, 1] * step_size
                    
                    # Maintain positive radii
                    if circles_array[i, 2] < 0.0001:
                        circles_array[i, 2] = 0.0001
                        
                    # Enforce valid positions
                    if not self.is_valid_position(circles_array[i]):
                        x, y, r = circles_array[i]
                        x = max(r, min(self.width - r, x))
                        y = max(r, min(self.height - r, y))
                        circles_array[i] = [x, y, r]
                
                # Early termination check
                if iteration % 100 == 0:
                    penalty = self.calculate_constraint_penalty(circles_array)
                    if penalty < 1e-5:
                        break
            
            return circles_array
        
        def refine_with_evolution(self, circles_array):
            """Use evolutionary approach to refine circle radii"""
            best_circles = circles_array.copy()
            best_sum = np.sum(best_circles[:, 2])
            
            # Generate variations and test
            for _ in range(200):
                test_circles = best_circles.copy()
                
                # Randomly select one circle to modify
                idx = random.randint(0, len(test_circles) - 1)
                x, y, r = test_circles[idx]
                
                # Try to slightly increase radius
                old_r = r
                new_r = min(old_r * 1.05, 0.2)
                test_circles[idx, 2] = new_r
                
                # Verify constraint satisfaction
                if not self.is_valid_position(test_circles[idx]):
                    continue
                    
                # Remove this circle for overlap testing
                temp_circles = np.delete(test_circles, idx, axis=0)
                if not self.check_overlap(test_circles[idx], temp_circles):
                    # Test the modified configuration
                    new_sum = np.sum(test_circles[:, 2])
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_circles = test_circles.copy()
            
            return best_circles
        
        def maximize_individual_radii(self, circles_array):
            """Try to maximize individual radii after main optimization"""
            for _ in range(100):
                improvement_made = False
                for i in range(len(circles_array)):
                    original_circle = circles_array[i].copy()
                    x, y, r = original_circle
                    
                    # Try to increase radius by small amount
                    new_r = min(r * 1.02, 0.2)
                    test_circle = [x, y, new_r]
                    
                    # Check if valid and causes no overlaps
                    if self.is_valid_position(test_circle) and not self.check_overlap(test_circle, np.delete(circles_array, i, axis=0)):
                        circles_array[i] = test_circle
                        improvement_made = True
                
                if not improvement_made:
                    break
            
            return circles_array
    
    # Create optimizer instance
    optimizer = CirclePackingOptimizer(width, height, n_circles)
    
    # Execute multi-phase optimization
    # Phase 1: Initialization
    circles = optimizer.initialize_circles()
    
    # Phase 2: Physics-based optimization
    circles = optimizer.apply_physics_update(circles, max_iter=1500)
    
    # Phase 3: Evolutionary refinement
    circles = optimizer.refine_with_evolution(circles)
    
    # Phase 4: Final polishing
    circles = optimizer.apply_physics_update(circles, max_iter=500)
    
    # Phase 5: Individual radius maximization
    circles = optimizer.maximize_individual_radii(circles)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
