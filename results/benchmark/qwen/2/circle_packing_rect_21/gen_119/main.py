# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist
import math
import random
from scipy.optimize import minimize
from sklearn.cluster import KMeans

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up container dimensions (perimeter = 4, so width + height = 2)
    container_width = 1.0
    container_height = 1.0
    
    # Parameters
    n_circles = 21
    max_iterations = 3000
    
    # Phase 1: Generate initial configuration using Voronoi-based approach
    def generate_initial_voronoi_placement():
        # Create initial points in a structured way that promotes good packing
        # Use a combination of grid points with perturbation and strategic seed points
        
        # Create a more sophisticated initial point distribution
        points = []
        
        # Add corner points to encourage boundary packing
        corners = [
            (0.1, 0.1), (0.1, container_height - 0.1),
            (container_width - 0.1, 0.1), (container_width - 0.1, container_height - 0.1)
        ]
        points.extend(corners)
        
        # Add center points
        center_points = [
            (container_width/2, container_height/2),
            (container_width/4, container_height/4),
            (3*container_width/4, 3*container_height/4),
            (container_width/4, 3*container_height/4),
            (3*container_width/4, container_height/4)
        ]
        points.extend(center_points)
        
        # Add grid points for remaining slots
        remaining_slots = n_circles - len(points)
        if remaining_slots > 0:
            grid_size = int(math.ceil(math.sqrt(remaining_slots)))
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(points) < n_circles:
                        x = (j + 0.5) * container_width / grid_size
                        y = (i + 0.5) * container_height / grid_size
                        # Add noise to avoid perfect grid alignment
                        x += random.uniform(-0.05, 0.05) * container_width
                        y += random.uniform(-0.05, 0.05) * container_height
                        # Clamp to boundaries
                        x = max(0.05, min(container_width - 0.05, x))
                        y = max(0.05, min(container_height - 0.05, y))
                        points.append((x, y))
        
        # Ensure we have exactly n_circles points
        while len(points) < n_circles:
            x = random.uniform(0.05, container_width - 0.05)
            y = random.uniform(0.05, container_height - 0.05)
            points.append((x, y))
        
        points = points[:n_circles]
        return np.array(points)
    
    # Phase 2: Compute Voronoi diagram and get cell information
    def compute_voronoi_info(points):
        # Remove duplicate points
        unique_points = np.unique(points, axis=0)
        
        try:
            # Compute Voronoi diagram
            vor = Voronoi(unique_points)
            
            # Get centroids of Voronoi cells
            centroids = []
            for region in vor.point_region:
                if region != -1:  # Valid region
                    vertices = vor.vertices[vor.regions[region]]
                    if len(vertices) > 0:
                        # Compute centroid of the region
                        centroid = np.mean(vertices, axis=0)
                        centroids.append(centroid)
                    else:
                        # Fallback to point itself if no vertices
                        centroids.append(vor.points[region])
                else:
                    # Invalid region - use point directly
                    centroids.append(vor.points[region])
            
            return np.array(centroids), vor
        except:
            # Fallback for Voronoi computation issues
            return points, None
    
    # Phase 3: Initialize circles with Voronoi-based radii estimation
    def initialize_circles_from_voronoi(points, voronoi_centroids, container_width, container_height):
        circles = np.zeros((len(points), 3))
        
        # Estimate initial radii based on Voronoi cell areas
        for i, (point, centroid) in enumerate(zip(points, voronoi_centroids)):
            # Compute distance to nearest neighbors (approximate cell size)
            distances = []
            for j, other_point in enumerate(points):
                if i != j:
                    dist = math.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)
                    distances.append(dist)
            
            # Use minimum distance as proxy for cell size
            if distances:
                avg_neighbor_dist = min(distances) * 0.5
            else:
                avg_neighbor_dist = 0.1
                
            # Initial radius should be proportional to cell size but bounded
            initial_radius = min(0.2, avg_neighbor_dist * 0.8, container_width/8, container_height/8)
            
            # Ensure it's within bounds
            initial_radius = max(0.01, min(initial_radius, 0.2))
            
            circles[i] = [point[0], point[1], initial_radius]
        
        return circles
    
    # Phase 4: Constraint checking
    def check_constraints(circles, container_width, container_height):
        # Check boundary constraints
        for circle in circles:
            x, y, r = circle
            if x - r < 0 or x + r > container_width or y - r < 0 or y + r > container_height:
                return False
        
        # Check overlap constraints (simple pairwise check for now)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < (r1 + r2):
                    return False
        return True
    
    # Phase 5: Optimizer with Voronoi-guided updates
    def optimize_with_voronoi(circles, container_width, container_height, max_iter=2000):
        # Create Voronoi structure once
        points = circles[:, :2]
        try:
            vor = Voronoi(points)
        except:
            vor = None
        
        # Main optimization loop
        best_sum = np.sum(circles[:, 2])
        best_circles = circles.copy()
        
        for iteration in range(max_iter):
            # Every 10 iterations, rebuild Voronoi for better guidance
            if iteration % 10 == 0 and vor is not None:
                try:
                    vor = Voronoi(circles[:, :2])
                except:
                    pass
            
            # Select circle to optimize
            selected_idx = random.randint(0, len(circles) - 1)
            
            # Get current circle info
            current_x, current_y, current_r = circles[selected_idx]
            
            # Use Voronoi-based heuristic for improvement
            # Find nearby circles and adjust accordingly
            nearby_circles = []
            for i in range(len(circles)):
                if i != selected_idx:
                    x, y, r = circles[i]
                    distance = math.sqrt((current_x - x)**2 + (current_y - y)**2)
                    # Consider circles within 3x current radius
                    if distance < (current_r + r) * 3:
                        nearby_circles.append((i, x, y, r, distance))
            
            # Compute potential improvements
            best_radius = current_r
            best_position = [current_x, current_y]
            best_valid = False
            
            # Try expanding the radius to maximum possible
            max_possible_radius = min(
                current_x, 
                container_width - current_x,
                current_y,
                container_height - current_y,
                0.2
            )
            
            # Try multiple radius values
            radius_steps = np.linspace(current_r, max_possible_radius, 10)
            for proposed_r in radius_steps:
                # Try different positions near current location
                for dx in [-0.1, -0.05, 0, 0.05, 0.1]:
                    for dy in [-0.1, -0.05, 0, 0.05, 0.1]:
                        trial_x = current_x + dx
                        trial_y = current_y + dy
                        
                        # Ensure within bounds
                        trial_x = max(proposed_r, min(container_width - proposed_r, trial_x))
                        trial_y = max(proposed_r, min(container_height - proposed_r, trial_y))
                        
                        # Check validity of this new configuration
                        test_circles = circles.copy()
                        test_circles[selected_idx] = [trial_x, trial_y, proposed_r]
                        
                        # Full constraint checking
                        valid = True
                        for i in range(len(test_circles)):
                            x1, y1, r1 = test_circles[i]
                            # Check boundary
                            if x1 - r1 < 0 or x1 + r1 > container_width or y1 - r1 < 0 or y1 + r1 > container_height:
                                valid = False
                                break
                            
                            # Check overlap with others
                            if i != selected_idx:
                                for j in range(len(test_circles)):
                                    if j != i and j != selected_idx:
                                        x2, y2, r2 = test_circles[j]
                                        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                                        if dist < (r1 + r2):
                                            valid = False
                                            break
                            
                            if not valid:
                                break
                        
                        if valid and proposed_r > best_radius:
                            best_radius = proposed_r
                            best_position = [trial_x, trial_y]
                            best_valid = True
            
            # Apply improvement if found
            if best_valid:
                circles[selected_idx] = [best_position[0], best_position[1], best_radius]
            
            # Periodic evaluation
            if iteration % 50 == 0:
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
        
        return best_circles
    
    # Phase 6: Final refinement
    def finalize_circles(circles, container_width, container_height):
        # Ensure boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Fix boundary violations
            if x - r < 0:
                circles[i, 0] = r
            elif x + r > container_width:
                circles[i, 0] = container_width - r
                
            if y - r < 0:
                circles[i, 1] = r
            elif y + r > container_height:
                circles[i, 1] = container_height - r
        
        # Resolve overlaps through iterative shrinking
        for _ in range(100):
            improved = False
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Try to reduce radius if overlapping
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                        if distance < (r + r2):
                            new_r = max(1e-6, (distance - 0.001) / 2)
                            if new_r < r:
                                circles[i, 2] = new_r
                                improved = True
                                break
            if not improved:
                break
        
        return circles
    
    # Main execution sequence
    # Generate initial points using Voronoi-inspired approach
    initial_points = generate_initial_voronoi_placement()
    
    # Compute Voronoi information and create initial circles
    centroids, voronoi_structure = compute_voronoi_info(initial_points)
    circles = initialize_circles_from_voronoi(initial_points, centroids, container_width, container_height)
    
    # Run optimization phase
    optimized_circles = optimize_with_voronoi(circles.copy(), container_width, container_height, max_iter=max_iterations)
    
    # Final refinement
    final_circles = finalize_circles(optimized_circles.copy(), container_width, container_height)
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
