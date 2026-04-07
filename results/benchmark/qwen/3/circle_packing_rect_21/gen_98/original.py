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
    # Using a square container (1x1) for optimal packing efficiency
    rect_width = 1.0
    rect_height = 1.0
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize circles with strategic hexagonal grid placement
    circles = np.zeros((21, 3))
    
    # Create hexagonal grid pattern with optimal spacing
    rows = 5
    cols = 5
    
    # Calculate spacing for hexagonal packing
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)
    
    # Hexagonal offset for alternate rows
    offset = spacing_x * 0.5
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= 21:
                break
            x = (j + 1) * spacing_x
            if i % 2 == 1:
                x += offset
            y = (i + 1) * spacing_y
            circles[idx] = [x, y, 0.01]  # Small initial radius
            idx += 1
    
    # Ensure we have exactly 21 circles
    for i in range(idx, 21):
        circles[i] = [0.5, 0.5, 0.01]
    
    # Multi-stage optimization with progressive refinement
    best_sum = 0
    best_circles = None
    
    # Phase 1: Coarse optimization
    for stage in range(2):
        current_circles = circles.copy()
        
        # Coarse iteration with large steps
        for iteration in range(200):
            improved = False
            
            # Randomly shuffle circle order for better exploration
            indices = list(range(21))
            random.shuffle(indices)
            
            for i in indices:
                # Compute max radius for current position
                max_radius = compute_max_radius(current_circles, i, rect_width, rect_height)
                
                if max_radius > current_circles[i, 2] + 1e-6:
                    current_circles[i, 2] = max_radius
                    improved = True
                    
            if not improved:
                break
                
        # Phase 2: Fine-grained local search
        for iteration in range(300):
            improved = False
            
            # Try each circle in random order
            indices = list(range(21))
            random.shuffle(indices)
            
            for i in indices:
                old_x, old_y, old_r = current_circles[i]
                old_r = current_circles[i, 2]
                
                # Try to expand radius
                max_radius = compute_max_radius(current_circles, i, rect_width, rect_height)
                
                if max_radius > old_r + 1e-6:
                    current_circles[i, 2] = max_radius
                    improved = True
                    
            if not improved:
                break
                
        # Phase 3: Position refinement with spatial optimization
        for iteration in range(200):
            improved = False
            
            # Sample potential moves
            for i in range(21):
                old_x, old_y, old_r = current_circles[i]
                
                # Try several nearby positions
                best_pos = [old_x, old_y, old_r]
                best_radius = old_r
                
                # Grid search around current position
                step = 0.05 if iteration < 100 else 0.01
                for dx in [-step, -step/2, 0, step/2, step]:
                    for dy in [-step, -step/2, 0, step/2, step]:
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        # Ensure within bounds
                        if (0.01 <= new_x <= rect_width - 0.01 and 
                            0.01 <= new_y <= rect_height - 0.01):
                            
                            # Compute max radius at new position
                            max_radius = compute_max_radius_at_position(
                                current_circles, i, new_x, new_y, rect_width, rect_height
                            )
                            
                            if max_radius > best_radius + 1e-6:
                                best_radius = max_radius
                                best_pos = [new_x, new_y, max_radius]
                                
                if best_pos[2] > current_circles[i, 2] + 1e-6:
                    current_circles[i] = best_pos
                    improved = True
                    
            if not improved:
                break
                
        # Final validation and update best
        if is_valid_configuration(current_circles, rect_width, rect_height):
            current_sum = np.sum(current_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = current_circles.copy()
    
    # Final refinement with adaptive optimization
    if best_circles is None:
        best_circles = circles.copy()
        
    # Apply final optimization with careful boundary handling
    for iteration in range(100):
        improved = False
        
        # Shuffle indices for better exploration
        indices = list(range(21))
        random.shuffle(indices)
        
        for i in indices:
            # Move circle and try to maximize radius
            old_x, old_y, old_r = best_circles[i]
            
            # Try small random perturbations
            if random.random() < 0.7:
                dx = random.uniform(-0.02, 0.02)
                dy = random.uniform(-0.02, 0.02)
                new_x = old_x + dx
                new_y = old_y + dy
                
                # Clamp to valid range
                new_x = max(0.01, min(rect_width - 0.01, new_x))
                new_y = max(0.01, min(rect_height - 0.01, new_y))
                
                # Recalculate maximum radius
                max_radius = compute_max_radius_at_position(
                    best_circles, i, new_x, new_y, rect_width, rect_height
                )
                
                if max_radius > best_circles[i, 2] + 1e-6:
                    best_circles[i] = [new_x, new_y, max_radius]
                    improved = True
            else:
                # Adjust radius only
                max_radius = compute_max_radius(best_circles, i, rect_width, rect_height)
                if max_radius > best_circles[i, 2] + 1e-6:
                    best_circles[i, 2] = max_radius
                    improved = True
                    
        if not improved:
            break
    
    # Final boundary check and correction
    for i in range(21):
        x, y, r = best_circles[i]
        # Ensure circle stays within bounds
        r = min(r, x, rect_width - x, y, rect_height - y)
        r = max(r, 0.001)
        best_circles[i] = [x, y, r]
        
    return best_circles


def compute_max_radius(circles, index, width, height):
    """Compute maximum radius for circle at given index without overlapping others."""
    x, y, _ = circles[index]
    
    # Boundary constraints
    min_dist_to_boundaries = min(x, width - x, y, height - y)
    
    # Collision constraints with other circles
    min_dist_to_others = float('inf')
    
    for i in range(len(circles)):
        if i != index:
            cx, cy, cr = circles[i]
            distance = sqrt((x - cx)**2 + (y - cy)**2)
            # Must maintain minimum distance of (r + cr)
            min_dist_to_others = min(min_dist_to_others, distance - cr)
    
    # Return the minimum of all constraints
    max_radius = min(min_dist_to_boundaries, min_dist_to_others)
    return max(0.001, max_radius)


def compute_max_radius_at_position(circles, index, x, y, width, height):
    """Compute maximum radius for circle at given position without overlapping others."""
    # Boundary constraints
    min_dist_to_boundaries = min(x, width - x, y, height - y)
    
    # Collision constraints with other circles
    min_dist_to_others = float('inf')
    
    for i in range(len(circles)):
        if i != index:
            cx, cy, cr = circles[i]
            distance = sqrt((x - cx)**2 + (y - cy)**2)
            # Must maintain minimum distance of (r + cr)
            min_dist_to_others = min(min_dist_to_others, distance - cr)
    
    # Return the minimum of all constraints
    max_radius = min(min_dist_to_boundaries, min_dist_to_others)
    return max(0.001, max_radius)


def is_valid_configuration(circles, width, height):
    """Check if all circles satisfy constraints."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Check boundary conditions
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            return False
            
        # Check overlap with other circles
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            distance = sqrt((x - x2)**2 + (y - y2)**2)
            if distance < r + r2:
                return False
                
    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")