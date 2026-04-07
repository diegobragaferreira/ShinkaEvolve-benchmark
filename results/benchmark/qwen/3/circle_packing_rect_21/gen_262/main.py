# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance_matrix
from scipy.spatial.distance import cdist
import random
from scipy.optimize import minimize
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Use 1.5 x 0.5 for good aspect ratio based on previous experiments
    rect_width, rect_height = 1.5, 0.5

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def generate_voronoi_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Generate initial configuration using Voronoi-based sampling"""
        # Generate random points with padding for boundary consideration
        padding = 0.1
        points = np.random.rand(n_circles * 3, 2) * [width - 2*padding, height - 2*padding] + [padding, padding]
        
        # Add corner points for better boundary coverage
        corner_points = np.array([
            [padding, padding],
            [width - padding, padding],
            [padding, height - padding],
            [width - padding, height - padding]
        ])
        points = np.vstack([points, corner_points])
        
        # Ensure we have enough points 
        if len(points) < n_circles:
            # Fill with additional random points
            extra_points = np.random.rand(n_circles - len(points), 2) * [width - 2*padding, height - 2*padding] + [padding, padding]
            points = np.vstack([points, extra_points])
        
        # Limit to exactly n_circles points
        points = points[:n_circles]
        
        # Generate Voronoi diagram
        try:
            vor = Voronoi(points)
            # Use Voronoi vertices as initial circle centers
            # But filter out points that are too close to boundaries
            valid_centers = []
            for vertex in vor.vertices:
                x, y = vertex
                if padding < x < width - padding and padding < y < height - padding:
                    valid_centers.append([x, y])
            
            # If we don't have enough valid vertices, fall back to points
            if len(valid_centers) < n_circles:
                # Use the original points as centers
                centers = points[:n_circles]
            else:
                # Use Voronoi vertices as centers
                centers = np.array(valid_centers[:n_circles])
            
            # Initialize with small radii
            circles = np.zeros((n_circles, 3))
            for i, (x, y) in enumerate(centers):
                circles[i] = [x, y, 0.01]
                
            return circles
            
        except:
            # Fallback to simple random initialization
            circles = np.zeros((n_circles, 3))
            for i in range(n_circles):
                x = random.uniform(padding, width - padding)
                y = random.uniform(padding, height - padding)
                circles[i] = [x, y, 0.01]
            return circles

    def compute_max_radius_at_position(x: float, y: float, existing_circles: np.ndarray,
                                     rect_width: float, rect_height: float) -> float:
        """Compute maximum possible radius for a circle at given position"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)
        
        # Distance to all other circles (excluding self)
        min_dist_to_others = float('inf')
        for circle in existing_circles:
            if circle[2] > 0:  # Only consider placed circles
                cx, cy, cr = circle
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                min_dist_to_others = min(min_dist_to_others, dist - cr)
        
        # Return minimum of boundary and overlapping distances
        max_radius = min(min_bound, min_dist_to_others)
        return max(0.001, max_radius)

    def is_valid_configuration(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Check if configuration is valid"""
        # Check boundary constraints
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False
        
        # Check overlap constraints
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                ci = circles[i]
                cj = circles[j]
                if ci[2] > 0 and cj[2] > 0:
                    dist = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
                    if dist < ci[2] + cj[2]:
                        return False
        
        return True

    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def voronoi_optimization_phase1(circles: np.ndarray, rect_width: float, rect_height: float, 
                                  iterations: int = 50) -> np.ndarray:
        """Phase 1: Voronoi-guided repositioning"""
        current = circles.copy()
        
        for iteration in range(iterations):
            # Create Voronoi diagram of current circle centers
            positions = current[:, :2]
            
            # Add boundary points to make Voronoi more meaningful
            boundary_points = np.array([
                [rect_width/2, 0],      # top
                [rect_width/2, rect_height],  # bottom  
                [0, rect_height/2],     # left
                [rect_width, rect_height/2]   # right
            ])
            all_points = np.vstack([positions, boundary_points])
            
            try:
                vor = Voronoi(all_points)
                
                # For each circle center, find the Voronoi cell it belongs to
                # and move to the centroid of that cell (if valid)
                new_positions = []
                for i in range(len(positions)):
                    # Find the Voronoi cell for this point
                    cell_index = None
                    for j, region in enumerate(vor.regions):
                        if len(region) > 0:
                            # Check if point i is in this region
                            if j < len(vor.point_region) and vor.point_region[j] == i:
                                cell_index = j
                                break
                    
                    # If we found a proper cell, try to move to centroid
                    if cell_index is not None and cell_index < len(vor.regions):
                        region = vor.regions[cell_index]
                        if len(region) > 0:
                            # Get vertices of this cell
                            vertices = [vor.vertices[k] for k in region if k >= 0]
                            if len(vertices) >= 2:
                                # Compute centroid of the cell polygon
                                vertices = np.array(vertices)
                                centroid_x = np.mean(vertices[:, 0])
                                centroid_y = np.mean(vertices[:, 1])
                                
                                # Check if centroid is within bounds
                                if (rect_width*0.01 < centroid_x < rect_width*0.99 and 
                                    rect_height*0.01 < centroid_y < rect_height*0.99):
                                    new_positions.append([centroid_x, centroid_y])
                                else:
                                    new_positions.append(positions[i])
                            else:
                                new_positions.append(positions[i])
                        else:
                            new_positions.append(positions[i])
                    else:
                        new_positions.append(positions[i])
                
                # Update positions
                for i in range(len(new_positions)):
                    x, y = new_positions[i]
                    # Compute max radius at new position
                    temp_circles = current.copy()
                    temp_circles[i, :2] = [x, y]
                    max_r = compute_max_radius_at_position(x, y, temp_circles, rect_width, rect_height)
                    current[i] = [x, y, max_r]
                    
            except:
                # If Voronoi fails, just run standard local search
                for i in range(len(current)):
                    old_x, old_y, old_r = current[i]
                    
                    # Simple local search around current position
                    best_x, best_y, best_r = old_x, old_y, old_r
                    best_sum = calculate_radius_sum(current)
                    
                    # Try small movements
                    for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                        for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                            test_x = max(0.01, min(rect_width - 0.01, old_x + dx))
                            test_y = max(0.01, min(rect_height - 0.01, old_y + dy))
                            
                            temp_circles = current.copy()
                            temp_circles[i, :2] = [test_x, test_y]
                            max_r = compute_max_radius_at_position(test_x, test_y, temp_circles, rect_width, rect_height)
                            temp_circles[i, 2] = max_r
                            
                            if is_valid_configuration(temp_circles, rect_width, rect_height):
                                new_sum = calculate_radius_sum(temp_circles)
                                if new_sum > best_sum:
                                    best_sum = new_sum
                                    best_x, best_y, best_r = test_x, test_y, max_r
                    
                    current[i] = [best_x, best_y, best_r]
        
        return current

    def voronoi_optimization_phase2(circles: np.ndarray, rect_width: float, rect_height: float, 
                                  iterations: int = 100) -> np.ndarray:
        """Phase 2: Gradient-based refinement using Voronoi-derived directions"""
        current = circles.copy()
        
        # For each circle, compute Voronoi-based gradient information
        for iteration in range(iterations):
            improved = False
            
            # Process circles in random order for better exploration
            indices = list(range(len(current)))
            random.shuffle(indices)
            
            for i in indices:
                old_x, old_y, old_r = current[i]
                
                # Get current Voronoi information
                positions = current[:, :2]
                
                # Add boundary points for stable Voronoi
                boundary_points = np.array([
                    [rect_width/2, 0],      # top
                    [rect_width/2, rect_height],  # bottom  
                    [0, rect_height/2],     # left
                    [rect_width, rect_height/2]   # right
                ])
                all_points = np.vstack([positions, boundary_points])
                
                # Try to get Voronoi info
                try:
                    vor = Voronoi(all_points)
                    
                    # Find neighbors in Voronoi diagram
                    neighbor_indices = []
                    for j, pos in enumerate(positions):
                        if j != i:
                            # Simple distance-based neighbor selection
                            dist = np.sqrt((old_x - pos[0])**2 + (old_y - pos[1])**2)
                            if dist < rect_width/3:  # Some reasonable threshold
                                neighbor_indices.append(j)
                    
                    # Move towards Voronoi-based direction if we have neighbors
                    if len(neighbor_indices) > 0:
                        # Simple gradient approximation: move toward average of neighbors
                        avg_x = np.mean([current[j][0] for j in neighbor_indices])
                        avg_y = np.mean([current[j][1] for j in neighbor_indices])
                        
                        # Direction from current position to average neighbor
                        dx = avg_x - old_x
                        dy = avg_y - old_y
                        
                        # Normalize and scale
                        norm = np.sqrt(dx*dx + dy*dy)
                        if norm > 0.001:
                            dx = dx/norm * 0.02
                            dy = dy/norm * 0.02
                            
                            test_x = max(0.01, min(rect_width - 0.01, old_x + dx))
                            test_y = max(0.01, min(rect_height - 0.01, old_y + dy))
                            
                            # Compute max radius at new position
                            temp_circles = current.copy()
                            temp_circles[i, :2] = [test_x, test_y]
                            max_r = compute_max_radius_at_position(test_x, test_y, temp_circles, rect_width, rect_height)
                            
                            if max_r > 0:
                                temp_circles[i, 2] = max_r
                                
                                if is_valid_configuration(temp_circles, rect_width, rect_height):
                                    new_sum = calculate_radius_sum(temp_circles)
                                    if new_sum > calculate_radius_sum(current):
                                        current[i] = [test_x, test_y, max_r]
                                        improved = True
                    else:
                        # If no clear neighbors, do local search
                        best_x, best_y, best_r = old_x, old_y, old_r
                        best_sum = calculate_radius_sum(current)
                        
                        # Try several movements
                        for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                            for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                                test_x = max(0.01, min(rect_width - 0.01, old_x + dx))
                                test_y = max(0.01, min(rect_height - 0.01, old_y + dy))
                                
                                temp_circles = current.copy()
                                temp_circles[i, :2] = [test_x, test_y]
                                max_r = compute_max_radius_at_position(test_x, test_y, temp_circles, rect_width, rect_height)
                                temp_circles[i, 2] = max_r
                                
                                if is_valid_configuration(temp_circles, rect_width, rect_height):
                                    new_sum = calculate_radius_sum(temp_circles)
                                    if new_sum > best_sum:
                                        best_sum = new_sum
                                        best_x, best_y, best_r = test_x, test_y, max_r
                        
                        current[i] = [best_x, best_y, best_r]
                        if best_sum > calculate_radius_sum(current):
                            improved = True
                            
                except:
                    # Fallback to basic local search if Voronoi fails
                    best_x, best_y, best_r = old_x, old_y, old_r
                    best_sum = calculate_radius_sum(current)
                    
                    # Try several movements
                    for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                        for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                            test_x = max(0.01, min(rect_width - 0.01, old_x + dx))
                            test_y = max(0.01, min(rect_height - 0.01, old_y + dy))
                            
                            temp_circles = current.copy()
                            temp_circles[i, :2] = [test_x, test_y]
                            max_r = compute_max_radius_at_position(test_x, test_y, temp_circles, rect_width, rect_height)
                            temp_circles[i, 2] = max_r
                            
                            if is_valid_configuration(temp_circles, rect_width, rect_height):
                                new_sum = calculate_radius_sum(temp_circles)
                                if new_sum > best_sum:
                                    best_sum = new_sum
                                    best_x, best_y, best_r = test_x, test_y, max_r
                    
                    current[i] = [best_x, best_y, best_r]
                    if best_sum > calculate_radius_sum(current):
                        improved = True
            
            # Early stopping if no improvement
            if not improved:
                break
                
        return current

    def multi_start_voronoi_optimization(n_starts: int = 3) -> np.ndarray:
        """Run multiple optimization starts with Voronoi-based approaches"""
        best_circles = None
        best_sum = -float('inf')

        for start_num in range(n_starts):
            # Different initializations
            if start_num == 0:
                # Voronoi-based initialization
                circles = generate_voronoi_initialization(21, rect_width, rect_height)
            elif start_num == 1:
                # Regular grid initialization
                circles = np.zeros((21, 3))
                rows, cols = 4, 6
                x_spacing = rect_width / (cols + 1)
                y_spacing = rect_height / (rows + 1)
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if idx >= 21:
                            break
                        x = (j + 1) * x_spacing
                        y = (i + 1) * y_spacing
                        if i % 2 == 1:
                            x += x_spacing * 0.5
                        circles[idx] = [x, y, 0.02]
                        idx += 1
                        if idx >= 21:
                            break
            else:
                # Random initialization
                circles = np.zeros((21, 3))
                for i in range(21):
                    x = random.uniform(0.01, rect_width - 0.01)
                    y = random.uniform(0.01, rect_height - 0.01)
                    circles[i] = [x, y, 0.02]

            # Phase 1: Voronoi-guided repositioning
            refined_1 = voronoi_optimization_phase1(circles, rect_width, rect_height, 50)

            # Phase 2: Gradient refinement  
            refined_2 = voronoi_optimization_phase2(refined_1, rect_width, rect_height, 100)

            final_sum = calculate_radius_sum(refined_2)

            if final_sum > best_sum:
                best_sum = final_sum
                best_circles = refined_2.copy()

        return best_circles

    # Main optimization workflow
    final_circles = multi_start_voronoi_optimization(3)

    # Final validation and cleanup
    if final_circles is not None:
        # Ensure all circles are valid and within bounds
        for i in range(len(final_circles)):
            x, y, r = final_circles[i]
            # Ensure within bounds
            r = min(r, x, rect_width - x, y, rect_height - y)
            if r <= 0:
                r = 0.01
            final_circles[i] = [x, y, r]

        # Validate final configuration
        while not is_valid_configuration(final_circles, rect_width, rect_height):
            # If invalid, regenerate from scratch
            final_circles = generate_voronoi_initialization(21, rect_width, rect_height)
            final_circles = voronoi_optimization_phase1(final_circles, rect_width, rect_height, 30)
            final_circles = voronoi_optimization_phase2(final_circles, rect_width, rect_height, 50)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")