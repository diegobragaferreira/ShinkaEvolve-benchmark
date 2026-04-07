# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random
from copy import deepcopy
from scipy.optimize import minimize
import warnings
from scipy.spatial import Voronoi
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - perimeter = 4, so width + height = 2
    # Optimized rectangle dimensions for maximum packing efficiency
    rect_width = 1.3
    rect_height = 0.7
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Parameters
    n_circles = 21
    max_iterations = 300
    
    # Hexagonal lattice initialization for superior packing
    def create_hexagonal_layout(n, width, height):
        """
        Create an initial layout using hexagonal close packing pattern.
        This provides a much better starting configuration than regular grids.
        """
        # Determine hexagonal grid dimensions
        side_length = np.sqrt(width * height / (n * np.sqrt(3)/2))
        hex_radius = side_length * 0.8  # Slightly smaller for padding
        
        # Create hexagonal grid
        rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        cols = int(np.ceil(n / rows))
        
        # Adjust for aspect ratio
        if width > height:
            cols = max(1, int(cols * width / height))
        else:
            rows = max(1, int(rows * height / width))
            
        circles = []
        y_offset = hex_radius * 0.5  # Vertical offset for hexagonal pattern
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Alternate row offset for hexagonal pattern
                x_offset = (j + (i % 2) * 0.5) * hex_radius * 2
                y_pos = i * hex_radius * np.sqrt(3) + y_offset
                
                # Ensure within bounds
                if x_offset < hex_radius or x_offset > width - hex_radius:
                    continue
                if y_pos < hex_radius or y_pos > height - hex_radius:
                    continue
                    
                circles.append([x_offset, y_pos, hex_radius])
                
        # If we don't have enough circles, add random ones
        while len(circles) < n:
            x = np.random.uniform(hex_radius, width - hex_radius)
            y = np.random.uniform(hex_radius, height - hex_radius)
            r = np.random.uniform(0.01, hex_radius * 0.8)
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Enhanced collision detection with early exits
    def enhanced_collision_check(circles, width, height):
        """
        Efficient collision detection using multiple strategies:
        1. Quick boundary check
        2. KDTree for spatial indexing
        3. Early termination on violations
        """
        # Quick boundary check
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        if np.any((coords[:, 0] - radii < 0) | 
                  (coords[:, 0] + radii > width) |
                  (coords[:, 1] - radii < 0) | 
                  (coords[:, 1] + radii > height)):
            return False
            
        # Use KDTree for neighbor search
        tree = cKDTree(coords)
        
        # Query pairs within 2 * max_radius to quickly identify potential conflicts
        pairs = tree.query_pairs(2 * np.max(radii), p=2)
        
        if len(pairs) == 0:
            return True
            
        # Detailed distance check for actual overlaps
        for i, j in pairs:
            if i >= len(circles) or j >= len(circles):
                continue
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r1 + r2) * (r1 + r2)
            
            if dist_sq < min_dist_sq:
                return False
                
        return True
    
    # Multi-phase optimization with hierarchical approach
    def hierarchical_optimization(initial_circles):
        """
        Multi-phase optimization combining global and local search strategies.
        """
        current_circles = initial_circles.copy()
        
        # Phase 1: Global coarse search with large movements
        for phase in range(3):
            # Dynamic step sizes based on phase
            step_size = 0.15 - phase * 0.04
            
            for iter_step in range(50):
                # Randomly select circles to optimize
                selected_indices = random.sample(range(n_circles), max(3, n_circles // 5))
                
                for idx in selected_indices:
                    current_x, current_y, current_r = current_circles[idx]
                    
                    # Try several candidate moves
                    best_move = None
                    best_radius = current_r
                    best_position = [current_x, current_y]
                    best_valid = False
                    
                    # Try different radius changes
                    radius_changes = [0.0, -0.02, -0.01, 0.01, 0.02]
                    # Try different position changes
                    position_changes = [(-step_size, -step_size), (-step_size, 0), 
                                       (-step_size, step_size), (0, -step_size), 
                                       (0, 0), (0, step_size), (step_size, -step_size),
                                       (step_size, 0), (step_size, step_size)]
                    
                    # Try combinations of position and radius changes
                    for dr in radius_changes:
                        for dx, dy in position_changes:
                            trial_r = current_r + dr
                            trial_x = current_x + dx
                            trial_y = current_y + dy
                            
                            # Boundary checks
                            if (trial_x - trial_r < 0 or trial_x + trial_r > rect_width or
                                trial_y - trial_r < 0 or trial_y + trial_r > rect_height):
                                continue
                                
                            # Test if this would be valid
                            test_circles = current_circles.copy()
                            test_circles[idx] = [trial_x, trial_y, trial_r]
                            
                            if enhanced_collision_check(test_circles, rect_width, rect_height):
                                if trial_r > best_radius:
                                    best_radius = trial_r
                                    best_position = [trial_x, trial_y]
                                    best_valid = True
                    
                    # Apply best move if found
                    if best_valid:
                        current_circles[idx] = [best_position[0], best_position[1], best_radius]
        
        # Phase 2: Medium-scale refinement with tighter search
        for iter_step in range(100):
            selected_indices = random.sample(range(n_circles), max(5, n_circles // 4))
            
            for idx in selected_indices:
                current_x, current_y, current_r = current_circles[idx]
                
                # More precise search around current position
                best_move = None
                best_radius = current_r
                best_position = [current_x, current_y]
                best_valid = False
                
                # Fine grid search around current position
                step_size = 0.02
                
                # Try various radius adjustments
                radius_changes = np.linspace(-0.02, 0.02, 9)
                # Fine grid around current position
                position_changes = [(i*step_size, j*step_size) 
                                  for i in range(-3, 4) for j in range(-3, 4)]
                
                for dr in radius_changes:
                    for dx, dy in position_changes:
                        trial_r = current_r + dr
                        trial_x = current_x + dx
                        trial_y = current_y + dy
                        
                        # Boundary checks
                        if (trial_x - trial_r < 0 or trial_x + trial_r > rect_width or
                            trial_y - trial_r < 0 or trial_y + trial_r > rect_height):
                            continue
                            
                        # Test validity
                        test_circles = current_circles.copy()
                        test_circles[idx] = [trial_x, trial_y, trial_r]
                        
                        if enhanced_collision_check(test_circles, rect_width, rect_height):
                            if trial_r > best_radius:
                                best_radius = trial_r
                                best_position = [trial_x, trial_y]
                                best_valid = True
                
                # Apply best move if found
                if best_valid:
                    current_circles[idx] = [best_position[0], best_position[1], best_radius]
        
        # Phase 3: Fine local search with very tight parameters
        for iter_step in range(150):
            selected_indices = random.sample(range(n_circles), max(3, n_circles // 6))
            
            for idx in selected_indices:
                current_x, current_y, current_r = current_circles[idx]
                
                # Even finer grid search
                step_size = 0.005
                
                # Very narrow range for radius
                radius_changes = np.linspace(-0.01, 0.01, 11)
                # Very fine grid around current position
                position_changes = [(i*step_size, j*step_size) 
                                  for i in range(-5, 6) for j in range(-5, 6)]
                
                best_radius = current_r
                best_position = [current_x, current_y]
                best_valid = False
                
                for dr in radius_changes:
                    for dx, dy in position_changes:
                        trial_r = current_r + dr
                        trial_x = current_x + dx
                        trial_y = current_y + dy
                        
                        # Boundary checks
                        if (trial_x - trial_r < 0 or trial_x + trial_r > rect_width or
                            trial_y - trial_r < 0 or trial_y + trial_r > rect_height):
                            continue
                            
                        # Test validity
                        test_circles = current_circles.copy()
                        test_circles[idx] = [trial_x, trial_y, trial_r]
                        
                        if enhanced_collision_check(test_circles, rect_width, rect_height):
                            if trial_r > best_radius:
                                best_radius = trial_r
                                best_position = [trial_x, trial_y]
                                best_valid = True
                
                # Apply best move if found
                if best_valid:
                    current_circles[idx] = [best_position[0], best_position[1], best_radius]
        
        return current_circles
    
    # Apply the optimization
    # Start with hexagonal layout
    initial_circles = create_hexagonal_layout(n_circles, rect_width, rect_height)
    
    # Apply hierarchical optimization
    optimized_circles = hierarchical_optimization(initial_circles)
    
    # Post-processing cleanup: resolve any remaining overlaps
    def clean_up_overlaps(circles):
        """
        Final cleanup to ensure all constraints are satisfied.
        """
        # Try to slightly adjust each circle to resolve overlaps
        for attempt in range(50):
            any_improved = False
            
            # Iterate through circles in random order for better results
            indices = list(range(len(circles)))
            random.shuffle(indices)
            
            for i in indices:
                x, y, r = circles[i]
                
                # Check for overlaps with others
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dx = x - x2
                        dy = y - y2
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        # If overlapping
                        if dist < r + r2:
                            # Try to reduce radius to resolve overlap
                            new_r = max(0.001, (dist - 0.001) / 2)
                            if new_r < r:
                                circles[i, 2] = new_r
                                any_improved = True
                            break
                
                # Ensure boundary constraints
                if x - r < 0:
                    circles[i, 0] = r
                elif x + r > rect_width:
                    circles[i, 0] = rect_width - r
                    
                if y - r < 0:
                    circles[i, 1] = r
                elif y + r > rect_height:
                    circles[i, 1] = rect_height - r
            
            if not any_improved:
                break
                
        return circles
    
    # Apply cleanup
    final_circles = clean_up_overlaps(optimized_circles)
    
    # Final verification
    if not enhanced_collision_check(final_circles, rect_width, rect_height):
        # If final check fails, revert to best intermediate result
        return initial_circles
    
    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")