# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    rect_width = 1.2
    rect_height = 0.8
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    def compute_circle_radius(point, points, rect_width, rect_height):
        """Compute maximum possible radius for a circle centered at 'point'"""
        center_x, center_y = point
        # Distance to rectangle edges
        dist_to_edges = [
            center_x,                    # distance to left edge
            rect_width - center_x,       # distance to right edge
            center_y,                    # distance to bottom edge
            rect_height - center_y       # distance to top edge
        ]
        
        # Distance to other circles (excluding self)
        min_dist_to_others = float('inf')
        for i, other_point in enumerate(points):
            if not (abs(other_point[0] - center_x) < 1e-10 and abs(other_point[1] - center_y) < 1e-10):
                dist = distance.euclidean(point, other_point)
                min_dist_to_others = min(min_dist_to_others, dist)
        
        # Maximum radius is limited by both edges and other circles
        max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
        return max(0.001, max_radius)
    
    def evaluate_configuration(points, rect_width, rect_height):
        """Evaluate a configuration by computing sum of radii"""
        total_radius = 0
        circles = []
        for point in points:
            radius = compute_circle_radius(point, points, rect_width, rect_height)
            circles.append([point[0], point[1], radius])
            total_radius += radius
        return total_radius, np.array(circles)
    
    def generate_voronoi_based_initialization(rect_width, rect_height):
        """Generate initial points using Voronoi geometric principles"""
        points = []
        
        # Strategy 1: Corner placements with padding
        corner_positions = [
            (rect_width * 0.1, rect_height * 0.1),
            (rect_width * 0.9, rect_height * 0.1),
            (rect_width * 0.1, rect_height * 0.9),
            (rect_width * 0.9, rect_height * 0.9),
            (rect_width / 2, rect_height / 2)
        ]
        
        # Add corners with slight perturbations
        for x, y in corner_positions:
            pert_x = np.random.normal(0, 0.03)
            pert_y = np.random.normal(0, 0.03)
            points.append([x + pert_x, y + pert_y])
        
        # Strategy 2: Edge midpoint placements
        edge_positions = [
            (rect_width/2, rect_height * 0.1),  # top center
            (rect_width/2, rect_height * 0.9),  # bottom center
            (rect_width * 0.1, rect_height/2),  # left center
            (rect_width * 0.9, rect_height/2),  # right center
        ]
        
        for x, y in edge_positions:
            pert_x = np.random.normal(0, 0.02)
            pert_y = np.random.normal(0, 0.02)
            points.append([x + pert_x, y + pert_y])
        
        # Strategy 3: Grid placements in interior
        grid_x = np.linspace(rect_width * 0.2, rect_width * 0.8, 3)
        grid_y = np.linspace(rect_height * 0.2, rect_height * 0.8, 3)
        
        for x in grid_x:
            for y in grid_y:
                if len(points) < 21:
                    pert_x = np.random.normal(0, 0.02)
                    pert_y = np.random.normal(0, 0.02)
                    points.append([x + pert_x, y + pert_y])
        
        # Strategy 4: Fill remaining with random but well-distributed points
        while len(points) < 21:
            x = np.random.uniform(0.05, rect_width - 0.05)
            y = np.random.uniform(0.05, rect_height - 0.05)
            points.append([x, y])
        
        return np.array(points[:21])
    
    def voronoi_optimization_step(points, rect_width, rect_height, step_size=0.05):
        """Optimize positions using Voronoi-based guidance"""
        new_points = points.copy()
        updated = False
        
        # For each point, determine optimal position based on Voronoi influence
        for i in range(len(points)):
            current_point = points[i]
            
            # Sample nearby positions in a structured manner
            best_point = current_point.copy()
            best_radius = compute_circle_radius(current_point, points, rect_width, rect_height)
            best_score = best_radius
            
            # Search in a grid around current position
            search_range = 0.1
            steps = int(search_range / step_size)
            
            for dx in np.linspace(-search_range, search_range, steps):
                for dy in np.linspace(-search_range, search_range, steps):
                    new_x = current_point[0] + dx
                    new_y = current_point[1] + dy
                    
                    # Keep within bounds
                    if (0.05 <= new_x <= rect_width - 0.05 and 
                        0.05 <= new_y <= rect_height - 0.05):
                        
                        test_point = np.array([new_x, new_y])
                        test_radius = compute_circle_radius(test_point, points, rect_width, rect_height)
                        
                        # Consider both radius and spatial distribution benefits
                        # Higher radius is preferred, but also prefer positions that maintain good separation
                        score = test_radius
                        
                        if score > best_score:
                            best_score = score
                            best_point = test_point
                            updated = True
            
            new_points[i] = best_point
            
        return new_points, updated
    
    def adaptive_local_refinement(points, rect_width, rect_height, max_iterations=300):
        """Perform adaptive local refinement with progressive search intensity"""
        current_points = points.copy()
        best_points = points.copy()
        best_radius_sum = compute_circle_radius(current_points[0], current_points[1:], rect_width, rect_height)
        
        # Progressive refinement with decreasing step sizes
        for iteration in range(max_iterations):
            # Adaptive step size - start with larger steps, shrink as we converge
            step_size = max(0.005, 0.1 * (1.0 - iteration/max_iterations))
            
            # Sample and update points
            updated = False
            for i in range(len(current_points)):
                current_point = current_points[i]
                best_point = current_point.copy()
                best_radius = compute_circle_radius(current_point, current_points, rect_width, rect_height)
                
                # Local search around the point
                for dx in [-step_size*2, -step_size, 0, step_size, step_size*2]:
                    for dy in [-step_size*2, -step_size, 0, step_size, step_size*2]:
                        new_x = current_point[0] + dx
                        new_y = current_point[1] + dy
                        
                        # Keep within bounds
                        if (0.05 <= new_x <= rect_width - 0.05 and 
                            0.05 <= new_y <= rect_height - 0.05):
                            
                            test_point = np.array([new_x, new_y])
                            test_radius = compute_circle_radius(test_point, current_points, rect_width, rect_height)
                            
                            if test_radius > best_radius:
                                best_radius = test_radius
                                best_point = test_point
                                updated = True
                
                current_points[i] = best_point
            
            # Check for improvement in total radius
            new_sum = sum(compute_circle_radius(p, current_points, rect_width, rect_height) for p in current_points)
            if new_sum > best_radius_sum:
                best_radius_sum = new_sum
                best_points = current_points.copy()
            
            # Early stopping criteria
            if not updated and iteration > 50:
                break
                
        return best_points
    
    # Generate initial configuration using Voronoi-based strategy
    points = generate_voronoi_based_initialization(rect_width, rect_height)
    
    # Multi-stage optimization approach
    # Stage 1: Coarse Voronoi-guided optimization
    for stage in range(3):
        points, _ = voronoi_optimization_step(points, rect_width, rect_height, step_size=0.1)
    
    # Stage 2: Progressive local refinement with adaptive search
    points = adaptive_local_refinement(points, rect_width, rect_height, max_iterations=200)
    
    # Stage 3: Fine-grained optimization with very small steps
    for _ in range(50):
        points, _ = voronoi_optimization_step(points, rect_width, rect_height, step_size=0.01)
    
    # Final refinement using precise grid search
    final_points = points.copy()
    for i in range(len(final_points)):
        current_point = final_points[i]
        best_point = current_point.copy()
        best_radius = compute_circle_radius(current_point, final_points, rect_width, rect_height)
        
        # Very fine grid search
        for dx in np.arange(-0.02, 0.025, 0.005):
            for dy in np.arange(-0.02, 0.025, 0.005):
                new_x = current_point[0] + dx
                new_y = current_point[1] + dy
                
                # Keep within bounds
                if (0.05 <= new_x <= rect_width - 0.05 and 
                    0.05 <= new_y <= rect_height - 0.05):
                    
                    test_point = np.array([new_x, new_y])
                    test_radius = compute_circle_radius(test_point, final_points, rect_width, rect_height)
                    
                    if test_radius > best_radius:
                        best_radius = test_radius
                        best_point = test_point
        
        final_points[i] = best_point
    
    # Evaluate final configuration
    _, circles = evaluate_configuration(final_points, rect_width, rect_height)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
