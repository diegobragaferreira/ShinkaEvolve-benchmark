# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    
    # Constants
    N_CIRCLES = 26
    BOUNDARY_MARGIN = 0.01
    MAX_ITERATIONS = 100
    
    def create_voronoi_lattice_initialization(n_circles: int) -> np.ndarray:
        """Create initial configuration using Voronoi-lattice hybrid approach."""
        # Create structured points using enhanced grid with lattice awareness
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        points = []
        
        # Generate main grid points with hexagonal offset
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n_circles:
                    # Hexagonal offset pattern
                    offset = (j % 2) * spacing_x / 2
                    x = (j + 1) * spacing_x + offset
                    y = (i + 1) * spacing_y
                    points.append([x, y])
        
        # Add boundary points for better edge coverage
        boundary_points = []
        for _ in range(20):
            side = np.random.randint(0, 4)
            if side == 0:  # Top
                boundary_points.append([np.random.uniform(0.05, 0.95), 1.0 - BOUNDARY_MARGIN])
            elif side == 1:  # Bottom
                boundary_points.append([np.random.uniform(0.05, 0.95), BOUNDARY_MARGIN])
            elif side == 2:  # Left
                boundary_points.append([BOUNDARY_MARGIN, np.random.uniform(0.05, 0.95)])
            else:  # Right
                boundary_points.append([1.0 - BOUNDARY_MARGIN, np.random.uniform(0.05, 0.95)])
        
        points.extend(boundary_points)
        points = points[:n_circles]
        
        # Convert to numpy array and add some randomness
        points = np.array(points)
        points += np.random.normal(0, 0.01, points.shape) * 0.5
        
        # Clip to bounds
        points = np.clip(points, BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN)
        
        # Create Voronoi structure to understand spatial relationships
        try:
            vor = Voronoi(points)
            # Use Voronoi cell centers as primary positions
            centroids = vor.points[vor.point_region[:-1]]
            centroids = centroids[:n_circles]
        except:
            # Fallback to direct points
            centroids = points[:n_circles]
        
        # Create initial circle configuration
        circles = np.zeros((n_circles, 3))
        
        # Compute average distances to neighbors for radius estimation
        for i in range(n_circles):
            x, y = centroids[i]
            
            # Calculate distances to all neighbors
            distances = []
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt((x - centroids[j][0])**2 + (y - centroids[j][1])**2)
                    distances.append(dist)
            
            # Estimate appropriate initial radius
            if distances:
                avg_distance = np.min(distances) * 0.4
                radius = min(avg_distance, 0.2)
            else:
                radius = 0.1
                
            # Ensure it's within bounds
            radius = min(radius, x - BOUNDARY_MARGIN, 1 - x - BOUNDARY_MARGIN,
                        y - BOUNDARY_MARGIN, 1 - y - BOUNDARY_MARGIN)
            radius = max(0.005, min(0.15, radius))
            
            circles[i] = [x, y, radius]
            
        return circles
    
    def is_valid_configuration(circles: np.ndarray) -> bool:
        """Check if configuration is valid."""
        n = len(circles)
        
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if (r <= 0 or x < r + BOUNDARY_MARGIN or x > 1-r - BOUNDARY_MARGIN or 
                y < r + BOUNDARY_MARGIN or y > 1-r - BOUNDARY_MARGIN):
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    return False
                    
        return True
    
    def calculate_total_radius(circles: np.ndarray) -> float:
        """Calculate sum of all radii."""
        return np.sum(circles[:, 2])
    
    def optimize_radii_for_positions(circles: np.ndarray) -> np.ndarray:
        """Optimize radii for fixed positions using geometric approach."""
        n = len(circles)
        result = circles.copy()
        
        # Try to maximize each radius while respecting constraints
        for _ in range(30):  # Multiple passes
            improved = False
            
            # Try to increase radii iteratively
            for i in range(n):
                # Find minimum distance to other centers
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        x1, y1, _ = result[i]
                        x2, y2, _ = result[j]
                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        min_dist = min(min_dist, dist)
                
                if min_dist < float('inf'):
                    max_possible_radius = min_dist * 0.5  # Safety factor
                    
                    # Consider boundary constraints
                    x, y, current_r = result[i]
                    boundary_radius = min(x, 1-x, y, 1-y)
                    max_possible_radius = min(max_possible_radius, boundary_radius - 0.001)
                    
                    # Increase radius if beneficial and feasible
                    if max_possible_radius > current_r and max_possible_radius > 0:
                        # Use more conservative increase
                        new_r = min(current_r + (max_possible_radius - current_r) * 0.3, max_possible_radius)
                        result[i, 2] = new_r
                        improved = True
            
            if not improved:
                break
                
        return result
    
    def constraint_adjust_positions(circles: np.ndarray) -> np.ndarray:
        """Adjust positions to satisfy constraints."""
        result = circles.copy()
        
        # First handle containment
        for i in range(len(result)):
            x, y, r = result[i]
            # Ensure within boundary
            x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
            y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
            result[i] = [x, y, r]
        
        # Handle overlaps through iterative adjustment
        for _ in range(100):
            improved = False
            for i in range(len(result)):
                x1, y1, r1 = result[i]
                for j in range(i+1, len(result)):
                    x2, y2, r2 = result[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    
                    if dist < (r1 + r2):
                        # Separate them
                        dx = x2 - x1
                        dy = y2 - y1
                        length = np.sqrt(dx*dx + dy*dy) + 1e-8
                        
                        # Normalize
                        dx /= length
                        dy /= length
                        
                        # Calculate separation needed
                        separation = (r1 + r2) - dist
                        
                        # Adjust both circles (push apart)
                        adjustment_factor = 0.5
                        result[i, 0] -= dx * separation * adjustment_factor
                        result[i, 1] -= dy * separation * adjustment_factor
                        result[j, 0] += dx * separation * adjustment_factor
                        result[j, 1] += dy * separation * adjustment_factor
                        
                        improved = True
            
            if not improved:
                break
        
        # Boundary clipping after adjustment
        for i in range(len(result)):
            x, y, r = result[i]
            x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
            y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
            result[i] = [x, y, r]
            
        return result
    
    def lattice_refinement(circles: np.ndarray) -> np.ndarray:
        """Use lattice-based refinement to optimize the configuration."""
        # Convert to lattice representation for geometric optimization
        result = circles.copy()
        
        # Create a simple optimization loop
        for iteration in range(50):
            improved = False
            
            # Phase 1: Optimize radii for current positions
            temp_result = optimize_radii_for_positions(result)
            
            # Phase 2: Adjust positions to maximize radii and reduce overlaps
            temp_result = constraint_adjust_positions(temp_result)
            
            # Check if there's improvement
            if abs(calculate_total_radius(temp_result) - calculate_total_radius(result)) > 1e-6:
                result = temp_result
                improved = True
            
            if not improved:
                break
                
        return result
    
    # Main optimization process
    # Step 1: Create initial configuration
    circles = create_voronoi_lattice_initialization(N_CIRCLES)
    
    # Step 2: Validate and initialize
    if not is_valid_configuration(circles):
        # Fallback to simple grid if initial is invalid
        grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        
        circles = np.zeros((N_CIRCLES, 3))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < N_CIRCLES:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    # Adjust for boundary constraints
                    x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    circles[count] = [x, y, r]
                    count += 1
    
    # Step 3: Apply lattice-based refinement
    refined_circles = lattice_refinement(circles)
    
    # Step 4: Final validation and optimization
    final_circles = constraint_adjust_positions(refined_circles)
    
    # Additional local optimization for better results
    for _ in range(20):
        improved = False
        # Try to slightly adjust positions to increase radii
        for i in range(N_CIRCLES):
            original_x, original_y, original_r = final_circles[i]
            best_r = original_r
            best_x = original_x
            best_y = original_y
            
            # Test small neighborhood moves
            for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                    test_x = max(BOUNDARY_MARGIN + original_r, min(1 - BOUNDARY_MARGIN - original_r, original_x + dx))
                    test_y = max(BOUNDARY_MARGIN + original_r, min(1 - BOUNDARY_MARGIN - original_r, original_y + dy))
                    
                    # Check if this position is valid
                    valid = True
                    temp_circles = final_circles.copy()
                    temp_circles[i] = [test_x, test_y, original_r]
                    
                    # Check overlap with others
                    for j in range(N_CIRCLES):
                        if i != j:
                            x1, y1, r1 = temp_circles[i]
                            x2, y2, r2 = temp_circles[j]
                            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                            if dist < (r1 + r2):
                                valid = False
                                break
                    
                    if valid:
                        # Try to increase the radius
                        test_r = original_r
                        max_r = min(test_x, 1-test_x, test_y, 1-test_y) - BOUNDARY_MARGIN
                        
                        if max_r > test_r:
                            # Binary search for maximum safe radius
                            low, high = test_r, max_r
                            best_test_r = test_r
                            for _ in range(10):
                                mid = (low + high) / 2
                                temp_circles[i] = [test_x, test_y, mid]
                                
                                valid_test = True
                                for k in range(N_CIRCLES):
                                    if i != k:
                                        x1, y1, r1 = temp_circles[i]
                                        x2, y2, r2 = temp_circles[k]
                                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                                        if dist < (r1 + r2):
                                            valid_test = False
                                            break
                                
                                if valid_test:
                                    best_test_r = mid
                                    low = mid
                                else:
                                    high = mid
                            
                            if best_test_r > best_r:
                                best_r = best_test_r
                                best_x = test_x
                                best_y = test_y
                                improved = True
            
            if improved:
                final_circles[i] = [best_x, best_y, best_r]
        
        if not improved:
            break
    
    # Final cleanup
    final_circles = constraint_adjust_positions(final_circles)
    
    # Check if final configuration is valid
    if not is_valid_configuration(final_circles):
        # Last resort: fallback to grid configuration
        grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        
        final_circles = np.zeros((N_CIRCLES, 3))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < N_CIRCLES:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    x = np.clip(x, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    y = np.clip(y, r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN)
                    final_circles[count] = [x, y, r]
                    count += 1
    
    return final_circles

# EVOLVE-BLOCK-END