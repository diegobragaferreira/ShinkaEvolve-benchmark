# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
from math import sqrt

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions - since perimeter = 4, width + height = 2
    # Using width=1.2, height=0.8 for optimal packing efficiency
    width, height = 1.2, 0.8
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize with improved hybrid strategy
    circles = initialize_hybrid_layout(width, height, 21)
    
    # Phase 1: Global optimization with adaptive parameters
    circles = optimize_global_phase(circles, width, height)
    
    # Phase 2: Local refinement with multi-resolution grid search
    circles = optimize_local_refinement(circles, width, height)
    
    # Phase 3: Final boundary and overlap correction
    circles = finalize_configuration(circles, width, height)
    
    return circles

def initialize_hybrid_layout(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circles using hybrid hexagonal and strategic seeding."""
    circles = np.zeros((n, 3))
    
    # Create hexagonal grid pattern
    rows = 5
    cols = 5
    
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    offset = spacing_x * 0.5
    
    hex_points = []
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            if i % 2 == 1:
                x += offset
            y = (i + 1) * spacing_y
            if 0.1 <= x <= width - 0.1 and 0.1 <= y <= height - 0.1:
                hex_points.append([x, y])
                idx += 1
    
    # Add strategic corner points  
    corner_points = [
        [0.1, 0.1], [width-0.1, 0.1], [0.1, height-0.1], [width-0.1, height-0.1],
        [width/2, 0.1], [width/2, height-0.1], [0.1, height/2], [width-0.1, height/2]
    ]
    
    # Combine and randomly select initial points
    all_points = hex_points + corner_points
    selected_points = random.sample(all_points, min(n, len(all_points)))
    
    # Initialize circles with computed max radii
    for i in range(n):
        if i < len(selected_points):
            x, y = selected_points[i]
        else:
            # Fallback to random placement
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            
        max_radius = compute_max_radius(x, y, width, height, circles[:i])
        circles[i] = [x, y, max_radius]
    
    return circles

def compute_max_radius(x: float, y: float, width: float, height: float, existing_circles: np.ndarray) -> float:
    """Compute maximum radius efficiently using vectorized operations."""
    # Boundary constraints
    min_dist_to_boundaries = min(x, width - x, y, height - y)
    
    if min_dist_to_boundaries <= 0:
        return 0
    
    # Vectorized overlap constraints
    if len(existing_circles) > 0:
        existing_array = np.array(existing_circles)
        centers = existing_array[:, :2]
        radii = existing_array[:, 2]
        
        # Compute distances to all existing circles
        diff = centers - np.array([x, y])
        distances = np.sqrt(np.sum(diff**2, axis=1))
        min_dist_to_others = np.min(distances - radii)
    else:
        min_dist_to_others = float('inf')
    
    max_radius = min(min_dist_to_boundaries, min_dist_to_others)
    return max(0.001, max_radius)

def optimize_global_phase(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Apply global optimization with adaptive parameters."""
    current_circles = circles.copy()
    max_iterations = 500
    improved_count = 0
    last_improved = 0
    best_sum = np.sum(current_circles[:, 2])
    
    for iteration in range(max_iterations):
        improved = False
        
        # Adaptive step size based on iteration progress
        if iteration < 150:
            step = 0.1
        elif iteration < 350:
            step = 0.05
        else:
            step = 0.02
        
        # Random shuffle for better exploration
        indices = list(range(len(current_circles)))
        random.shuffle(indices)
        
        # Process circles in random order
        for i in indices:
            # Try radius maximization first
            old_radius = current_circles[i, 2]
            new_radius = compute_max_radius(
                current_circles[i, 0], 
                current_circles[i, 1], 
                width, 
                height, 
                np.vstack([current_circles[:i], current_circles[i+1:]])
            )
            
            if new_radius > old_radius + 1e-6:
                current_circles[i, 2] = new_radius
                improved = True
                improved_count += 1
        
        # Position refinement with local grid search
        for i in indices:
            old_x, old_y, old_r = current_circles[i]
            
            # Grid search around current position
            best_x, best_y, best_r = old_x, old_y, old_r
            best_radius = old_r
            
            # Try various positions in adaptive grid
            step_sizes = [step, step/2, step/4] if iteration > 100 else [step]
            
            for step_size in step_sizes:
                for dx in [-step_size*2, -step_size, 0, step_size, step_size*2]:
                    for dy in [-step_size*2, -step_size, 0, step_size, step_size*2]:
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        if (0.01 <= new_x <= width - 0.01 and 
                            0.01 <= new_y <= height - 0.01):
                            
                            max_radius = compute_max_radius(
                                new_x, new_y, width, height,
                                np.vstack([current_circles[:i], current_circles[i+1:]])
                            )
                            
                            if max_radius > best_radius + 1e-6:
                                best_radius = max_radius
                                best_x, best_y = new_x, new_y
                                improved = True
                                improved_count += 1
            
            if best_radius > current_circles[i, 2] + 1e-6:
                current_circles[i] = [best_x, best_y, best_radius]
        
        # Early termination conditions
        current_sum = np.sum(current_circles[:, 2])
        if current_sum > best_sum + 1e-6:
            best_sum = current_sum
            last_improved = iteration
        elif iteration - last_improved > 50:
            break
            
    return current_circles

def optimize_local_refinement(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Apply fine-grained local optimization."""
    current_circles = circles.copy()
    
    # Multi-resolution grid search
    resolutions = [(0.05, 100), (0.02, 150), (0.01, 150)]
    
    for step_size, iterations in resolutions:
        for iteration in range(iterations):
            improved = False
            
            # Random ordering
            indices = list(range(len(current_circles)))
            random.shuffle(indices)
            
            for i in indices:
                old_x, old_y, old_r = current_circles[i]
                
                # Grid search around position
                best_x, best_y, best_r = old_x, old_y, old_r
                best_radius = old_r
                
                # Search grid
                for dx in np.arange(-step_size*2, step_size*2 + step_size/2, step_size):
                    for dy in np.arange(-step_size*2, step_size*2 + step_size/2, step_size):
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        if (0.01 <= new_x <= width - 0.01 and 
                            0.01 <= new_y <= height - 0.01):
                            
                            max_radius = compute_max_radius(
                                new_x, new_y, width, height,
                                np.vstack([current_circles[:i], current_circles[i+1:]])
                            )
                            
                            if max_radius > best_radius + 1e-6:
                                best_radius = max_radius
                                best_x, best_y = new_x, new_y
                                improved = True
                
                if best_radius > current_circles[i, 2] + 1e-6:
                    current_circles[i] = [best_x, best_y, best_radius]
            
            if not improved:
                break
    
    return current_circles

def finalize_configuration(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Perform final boundary and constraint validation."""
    current_circles = circles.copy()
    
    # Final safety adjustments
    for i in range(len(current_circles)):
        x, y, r = current_circles[i]
        
        # Ensure proper boundary constraints
        r = min(r, x - 0.01, width - x - 0.01, y - 0.01, height - y - 0.01)
        r = max(r, 0.001)
        current_circles[i] = [x, y, r]
    
    return current_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
