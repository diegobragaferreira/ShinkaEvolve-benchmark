# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimal aspect ratio chosen based on common circle packing studies
    width, height = 1.2, 0.8

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def compute_max_radius_at_position(x: float, y: float, circles: np.ndarray) -> float:
        """Compute maximum radius for a circle at given position without overlaps."""
        # Boundary constraints
        max_radius = min(x, width - x, y, height - y)
        
        # Overlap constraints
        for cx, cy, r in circles:
            if cx != x or cy != y:  # Skip self-comparison
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                max_radius = min(max_radius, dist - r)
        
        return max(max_radius, 0.001)  # Ensure positive radius

    def is_valid_configuration(circles: np.ndarray) -> bool:
        """Check if configuration is valid (no overlaps, within bounds)"""
        # Check boundary constraints
        for x, y, r in circles:
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False
        
        # Check overlap constraints using vectorized operations for efficiency
        if len(circles) < 2:
            return True
            
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute distance matrix
        dist_matrix = cdist(positions, positions)
        
        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(dist_matrix, float('inf'))
        
        # Minimum distances between circle centers
        min_distances = np.min(dist_matrix, axis=1)
        
        # Required minimum distances (sum of radii)
        required_distances = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Check if any overlap exists
        overlaps = min_distances < np.min(required_distances, axis=0)
        
        return not np.any(overlaps)

    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    # Phase 1: Intelligent initialization with strategic placement
    n_circles = 21
    circles = np.zeros((n_circles, 3))
    
    # Strategic corner placements
    corner_positions = [
        (0.1, 0.1),           # Bottom-left
        (width-0.1, 0.1),     # Bottom-right
        (0.1, height-0.1),    # Top-left
        (width-0.1, height-0.1), # Top-right
    ]
    
    # Edge midpoints
    edge_positions = [
        (width/2, 0.1),       # Bottom-middle
        (width/2, height-0.1), # Top-middle
        (0.1, height/2),      # Left-middle
        (width-0.1, height/2), # Right-middle
    ]
    
    # Center position
    center_position = (width/2, height/2)
    
    # Fill with strategic points first (8 points)
    positions_used = 0
    for pos in corner_positions + edge_positions:
        circles[positions_used] = [pos[0], pos[1], 0.02]
        positions_used += 1
    
    # Add center point
    circles[positions_used] = [center_position[0], center_position[1], 0.02]
    positions_used += 1
    
    # Fill remaining positions with systematic grid-based approach
    grid_size = 5
    x_coords = np.linspace(0.1, width - 0.1, grid_size)
    y_coords = np.linspace(0.1, height - 0.1, grid_size)
    
    grid_points = []
    for x in x_coords:
        for y in y_coords:
            grid_points.append((x, y))
    
    # Remove positions already used, shuffle remaining
    remaining_points = [p for p in grid_points if p not in [pos for pos in corner_positions + edge_positions + [center_position]]]
    random.shuffle(remaining_points)
    
    # Fill remaining circles
    for i in range(positions_used, n_circles):
        if i - positions_used < len(remaining_points):
            x, y = remaining_points[i - positions_used]
        else:
            # Fallback to random placement
            x = random.uniform(0.1, width - 0.1)
            y = random.uniform(0.1, height - 0.1)
        
        # Compute max radius at this position
        temp_circles = circles.copy()
        temp_circles[i] = [x, y, 0.01]  # Temporary small radius for computation
        
        max_radius = compute_max_radius_at_position(x, y, temp_circles)
        circles[i] = [x, y, max_radius]
    
    # Phase 2: Multi-stage optimization with adaptive parameters
    best_sum = calculate_radius_sum(circles)
    best_circles = circles.copy()
    
    # Stage 1: Coarse-grained optimization (exploration phase)
    print("Starting coarse optimization...")
    for stage in range(3):
        if stage == 0:
            iterations = 100
            step_size = 0.1
            max_radius_change = 0.1
        elif stage == 1:
            iterations = 150
            step_size = 0.05
            max_radius_change = 0.05
        else:
            iterations = 200
            step_size = 0.02
            max_radius_change = 0.02
            
        for iteration in range(iterations):
            improved = False
            current_sum = calculate_radius_sum(circles)
            
            # Randomly sample circles to optimize
            indices = list(range(n_circles))
            random.shuffle(indices)
            sample_size = max(5, n_circles // 4)
            sample_indices = indices[:sample_size]
            
            for idx in sample_indices:
                original_x, original_y, original_r = circles[idx]
                
                # Try several move strategies
                best_x, best_y, best_r = original_x, original_y, original_r
                best_delta = 0
                
                # Strategy 1: Gradient-like moves
                strategies = [
                    (0, 0),               # No move
                    (step_size, 0),       # Right
                    (-step_size, 0),      # Left
                    (0, step_size),       # Up
                    (0, -step_size),      # Down
                    (step_size, step_size), # Diagonal up-right
                    (-step_size, step_size), # Diagonal up-left
                    (step_size, -step_size), # Diagonal down-right
                    (-step_size, -step_size), # Diagonal down-left
                ]
                
                # Strategy 2: Larger moves in early stages
                if stage < 2:
                    large_strategies = [
                        (step_size*2, 0),
                        (-step_size*2, 0),
                        (0, step_size*2),
                        (0, -step_size*2),
                    ]
                    strategies.extend(large_strategies)
                
                # Strategy 3: Systematic grid sampling
                if stage == 2:  # Fine-tune stage
                    for dx in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                        for dy in [-step_size, -step_size/2, 0, step_size/2, step_size]:
                            if abs(dx) + abs(dy) > 0.001:
                                strategies.append((dx, dy))
                
                # Evaluate all strategies
                for dx, dy in strategies:
                    new_x = original_x + dx
                    new_y = original_y + dy
                    
                    # Keep within bounds
                    new_x = max(0.05, min(width - 0.05, new_x))
                    new_y = max(0.05, min(height - 0.05, new_y))
                    
                    # Compute new radius at this position
                    temp_circles = circles.copy()
                    temp_circles[idx] = [new_x, new_y, 0.01]  # Temporary small radius
                    
                    new_r = compute_max_radius_at_position(new_x, new_y, temp_circles)
                    
                    # Enforce maximum radius change constraint
                    if abs(new_r - original_r) <= max_radius_change:
                        # Check if this improves the configuration
                        delta = new_r - original_r
                        
                        # Consider overlap penalties
                        temp_circles_copy = circles.copy()
                        temp_circles_copy[idx] = [new_x, new_y, new_r]
                        
                        if is_valid_configuration(temp_circles_copy):
                            if delta > best_delta:
                                best_delta = delta
                                best_x, best_y, best_r = new_x, new_y, new_r
                
                # Apply the best move if found
                if best_delta > 0:
                    circles[idx] = [best_x, best_y, best_r]
                    improved = True
            
            # Check if there was improvement
            new_sum = calculate_radius_sum(circles)
            if new_sum > best_sum:
                best_sum = new_sum
                best_circles = circles.copy()
            
            # Early stopping condition
            if not improved and iteration > 10:
                break
    
    # Phase 3: Fine-grained local search
    print("Starting fine-grained optimization...")
    fine_iterations = 300
    for iteration in range(fine_iterations):
        improved = False
        
        # Focus on circle that has the most room to grow (largest current radius)
        radii = circles[:, 2]
        max_radius_idx = np.argmax(radii)
        
        # Try to improve this specific circle
        original_x, original_y, original_r = circles[max_radius_idx]
        
        # Find best improvement near the current position
        best_x, best_y, best_r = original_x, original_y, original_r
        best_delta = 0
        
        # Sample around the current position
        search_radius = 0.1
        samples_per_direction = 10
        
        # Generate samples in a more systematic way
        search_space = []
        for i in range(samples_per_direction):
            # Circular sampling
            angle = 2 * math.pi * i / samples_per_direction
            dist = search_radius * i / samples_per_direction
            dx = dist * math.cos(angle)
            dy = dist * math.sin(angle)
            search_space.append((dx, dy))
            
            # Also sample along axes
            if i < samples_per_direction // 2:
                search_space.append((dx, 0))
                search_space.append((0, dy))
        
        # Add some random samples for diversity
        for _ in range(10):
            dx = random.uniform(-search_radius, search_radius)
            dy = random.uniform(-search_radius, search_radius)
            search_space.append((dx, dy))
        
        # Evaluate all candidate positions
        for dx, dy in search_space:
            new_x = original_x + dx
            new_y = original_y + dy
            
            # Keep within bounds
            new_x = max(0.05, min(width - 0.05, new_x))
            new_y = max(0.05, min(height - 0.05, new_y))
            
            # Compute max radius at this position
            temp_circles = circles.copy()
            temp_circles[max_radius_idx] = [new_x, new_y, 0.01]  # Temporary small radius
            
            new_r = compute_max_radius_at_position(new_x, new_y, temp_circles)
            
            # Check if valid configuration
            temp_circles_copy = circles.copy()
            temp_circles_copy[max_radius_idx] = [new_x, new_y, new_r]
            
            if is_valid_configuration(temp_circles_copy):
                delta = new_r - original_r
                if delta > best_delta:
                    best_delta = delta
                    best_x, best_y, best_r = new_x, new_y, new_r
        
        # Apply improvement if found
        if best_delta > 0:
            circles[max_radius_idx] = [best_x, best_y, best_r]
            improved = True
        
        # Periodic validation
        if iteration % 50 == 0:
            if not is_valid_configuration(circles):
                # Restore best valid configuration if we went invalid
                circles = best_circles.copy()
        
        # Early stopping
        if not improved and iteration > 50:
            break
    
    # Phase 4: Validation and cleanup
    print("Final validation...")
    if not is_valid_configuration(circles):
        # Restore best valid configuration
        circles = best_circles.copy()
        
        # Ensure all circles are valid
        for i in range(n_circles):
            # Ensure minimum radius
            circles[i][2] = max(circles[i][2], 0.001)
            
            # Ensure within bounds
            circles[i][0] = max(0.001, min(width - 0.001, circles[i][0]))
            circles[i][1] = max(0.001, min(height - 0.001, circles[i][1]))
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")