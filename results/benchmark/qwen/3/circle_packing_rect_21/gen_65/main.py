# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

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

    # Phase 1: Grid-based initialization with systematic approach
    n_circles = 21
    
    # Use multiple grid resolutions for better initial placement
    grid_sizes = [5, 7, 9]  # Different grid densities
    best_initial_config = None
    best_initial_sum = 0
    
    for grid_size in grid_sizes:
        # Generate grid points
        x_coords = np.linspace(0.05, width - 0.05, grid_size)
        y_coords = np.linspace(0.05, height - 0.05, grid_size)
        
        # Create all possible grid points
        grid_points = [(x, y) for x in x_coords for y in y_coords]
        
        # Shuffle points for randomness in selection
        np.random.shuffle(grid_points)
        
        # Greedy selection of points with max radii
        selected_points = []
        selected_radii = []
        
        for point in grid_points:
            cx, cy = point
            max_radius = min(cx, width - cx, cy, height - cy)
            
            # Check for conflicts with already selected circles
            valid = True
            for px, py, pr in zip(selected_points, selected_radii, selected_radii):
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < (pr + max_radius):
                    valid = False
                    break
            
            if valid:
                selected_points.append((cx, cy))
                selected_radii.append(max_radius)
            
            if len(selected_points) >= n_circles:
                break
        
        # If we have enough points, evaluate this configuration
        if len(selected_points) >= n_circles:
            config = np.array([[px, py, pr] for px, py, pr in zip(selected_points, selected_radii, selected_radii)])
            current_sum = np.sum(config[:, 2])
            
            if current_sum > best_initial_sum:
                best_initial_sum = current_sum
                best_initial_config = config.copy()
    
    # Fallback to simple random placement if needed
    if best_initial_config is None or len(best_initial_config) < n_circles:
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            cx = np.random.uniform(0.05, width - 0.05)
            cy = np.random.uniform(0.05, height - 0.05)
            max_radius = min(cx, width - cx, cy, height - cy)
            circles[i] = [cx, cy, max_radius]
    else:
        circles = best_initial_config[:n_circles]
    
    # Phase 2: Multi-scale local optimization with smart perturbations
    max_iterations = 1000
    tolerance = 1e-6
    best_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()
    
    # Scale ranges for multi-scale search
    scales = [0.15, 0.1, 0.05, 0.02]  # Different step sizes
    scale_probs = [0.2, 0.3, 0.3, 0.2]  # Probability distribution
    
    for iteration in range(max_iterations):
        # Copy current configuration
        current_circles = circles.copy()
        old_sum = np.sum(current_circles[:, 2])
        
        # Choose scale randomly based on probabilities
        scale_idx = np.random.choice(len(scales), p=scale_probs)
        move_range = scales[scale_idx]
        
        # Track improvements
        improved = False
        
        # Try to improve each circle
        for i in range(n_circles):
            orig_x, orig_y, orig_r = current_circles[i]
            
            # Try several moves in a structured pattern
            best_move_x, best_move_y, best_move_r = 0, 0, 0
            best_new_sum = old_sum
            
            # Sample moves in a structured way: center, then surrounding points
            move_offsets = [
                (0, 0),  # Stay put
                (-move_range, -move_range),
                (-move_range, 0),
                (-move_range, move_range),
                (0, -move_range),
                (0, move_range),
                (move_range, -move_range),
                (move_range, 0),
                (move_range, move_range)
            ]
            
            for dx, dy in move_offsets:
                # New position
                new_x = orig_x + dx
                new_y = orig_y + dy
                
                # Keep within bounds
                new_x = max(0.05, min(width - 0.05, new_x))
                new_y = max(0.05, min(height - 0.05, new_y))
                
                # Compute max radius at new location
                new_r = min(new_x, width - new_x, new_y, height - new_y)
                
                # Check for conflicts with all other circles
                valid = True
                for j in range(n_circles):
                    if i != j:
                        px, py, pr = current_circles[j]
                        dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                        if dist < (new_r + pr):
                            valid = False
                            break
                
                if valid:
                    # Calculate new sum
                    new_sum = old_sum - orig_r + new_r
                    
                    if new_sum > best_new_sum:
                        best_new_sum = new_sum
                        best_move_x, best_move_y, best_move_r = dx, dy, new_r
            
            # Apply improvement if found
            if best_new_sum > old_sum:
                current_circles[i] = [orig_x + best_move_x, orig_y + best_move_y, best_move_r]
                improved = True
        
        # Update global best if improved
        new_sum = np.sum(current_circles[:, 2])
        if new_sum > best_sum:
            best_sum = new_sum
            best_circles = current_circles.copy()
        elif not improved and abs(new_sum - best_sum) < tolerance:
            # If no improvement for several iterations, reduce learning rate
            break
        
        circles = current_circles.copy()
    
    # Phase 3: Fine-tuning with boundary and neighborhood optimization
    # Create a finer search grid around existing circles
    fine_grid_size = 12
    
    for _ in range(200):  # More iterations for fine-tuning
        # Select random subset of circles to potentially improve
        indices = np.random.choice(n_circles, size=min(8, n_circles), replace=False)
        
        for idx in indices:
            old_x, old_y, old_r = circles[idx]
            
            # Define fine search area around current circle
            search_bounds = [
                max(0.05, old_x - 0.15), 
                min(width - 0.05, old_x + 0.15),
                max(0.05, old_y - 0.15), 
                min(height - 0.05, old_y + 0.15)
            ]
            
            # Generate fine grid within search bounds
            x_fine = np.linspace(search_bounds[0], search_bounds[1], fine_grid_size)
            y_fine = np.linspace(search_bounds[2], search_bounds[3], fine_grid_size)
            
            best_x, best_y = old_x, old_y
            best_r = old_r
            best_sum = np.sum(circles[:, 2])
            
            for fx in x_fine:
                for fy in y_fine:
                    # Ensure within bounds
                    fx = max(0.05, min(width - 0.05, fx))
                    fy = max(0.05, min(height - 0.05, fy))
                    
                    # Compute max radius
                    new_r = min(fx, width - fx, fy, height - fy)
                    
                    # Check for conflicts
                    valid = True
                    for j in range(n_circles):
                        if j != idx:
                            px, py, pr = circles[j]
                            dist = np.sqrt((fx - px)**2 + (fy - py)**2)
                            if dist < (new_r + pr):
                                valid = False
                                break
                    
                    if valid:
                        # Evaluate improvement
                        new_sum = np.sum(circles[:, 2]) - old_r + new_r
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_x, best_y, best_r = fx, fy, new_r
            
            # Apply the best improvement found
            if best_sum > np.sum(circles[:, 2]):
                circles[idx] = [best_x, best_y, best_r]
    
    # Final validation and return
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")