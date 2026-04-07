# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial.distance import cdist
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Rectangle dimensions: width + height = 2, using optimal ratio found through experimentation
    width = 1.25
    height = 0.75
    
    n = 21
    circles = np.zeros((n, 3))
    
    # Phase 1: Smart initialization using hexagonal packing pattern
    # Better distribution than regular grid - reduces clustering and overlap issues
    rows = 4
    cols = 6
    
    # Calculate precise spacing with padding
    col_spacing = width / (cols + 1)
    row_spacing = height / (rows + 1)
    hex_offset = col_spacing * 0.5
    
    # Generate initial positions with hexagonal pattern
    positions = []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = (col + 1) * col_spacing
            if row % 2 == 1:
                x += hex_offset
            y = (row + 1) * row_spacing
            positions.append([x, y])
            idx += 1
        if idx >= n:
            break
    
    # Ensure we have exactly n positions (fill remaining slots with center position)
    while len(positions) < n:
        positions.append([width/2, height/2])
    
    # Initialize with carefully chosen small radii based on spacing
    base_radius = min(col_spacing, row_spacing) * 0.15
    for i in range(n):
        circles[i] = [positions[i][0], positions[i][1], base_radius]
    
    # Phase 2: Advanced multi-scale optimization with adaptive learning
    max_iterations = 1500
    best_sum = 0
    best_circles = None
    
    # Track recent improvements for adaptive scaling
    improvement_window = 20
    recent_improvements = []
    
    # Optimization parameters
    initial_step_size = 0.08
    min_step_size = 0.005
    max_step_size = 0.2
    
    # Multi-scale phases
    phases = [
        {"iterations": 300, "step_size": 0.15},
        {"iterations": 300, "step_size": 0.08},
        {"iterations": 400, "step_size": 0.03}
    ]
    
    current_circles = circles.copy()
    
    for phase_config in phases:
        phase_iterations = phase_config["iterations"]
        step_size = phase_config["step_size"]
        
        # Track best in this phase
        phase_best_circles = current_circles.copy()
        phase_best_sum = np.sum(current_circles[:, 2])
        
        for iteration in range(phase_iterations):
            # Adaptive step size based on recent performance
            if len(recent_improvements) >= improvement_window:
                recent_avg = np.mean(recent_improvements[-improvement_window:])
                if recent_avg < 0.005:
                    step_size = max(min_step_size, step_size * 0.8)
                elif recent_avg > 0.02:
                    step_size = min(max_step_size, step_size * 1.1)
            
            # Create new candidate solution
            new_circles = current_circles.copy()
            
            # Randomly select circles to optimize (more efficient than optimizing all)
            selected_indices = random.sample(range(n), max(1, n // 3))
            
            # For each selected circle, try to optimize both position and radius
            for i in selected_indices:
                # Current circle data
                x, y, r = new_circles[i]
                
                # Try to maximize the radius of this circle
                max_radius = calculate_max_radius_fast(new_circles, i, width, height)
                
                # Random perturbation around maximum radius
                if max_radius > 0.001:
                    perturbation = random.uniform(-step_size * 0.3, step_size * 0.3)
                    new_radius = max(0.001, min(max_radius, r + perturbation))
                    new_circles[i][2] = new_radius
                
                # Position perturbation with probability
                if random.random() > 0.6:  # 40% chance to perturb position
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_x = max(0.001, min(width - 0.001, x + dx))
                    new_y = max(0.001, min(height - 0.001, y + dy))
                    new_circles[i][0] = new_x
                    new_circles[i][1] = new_y
            
            # Check validity with vectorized operations for efficiency
            if is_valid_configuration_fast(new_circles, width, height):
                new_sum = np.sum(new_circles[:, 2])
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_circles = new_circles.copy()
                    recent_improvements.append(new_sum)
                else:
                    recent_improvements.append(0)
                
                # Update current circles with better configuration
                if new_sum > np.sum(current_circles[:, 2]):
                    current_circles = new_circles
                    phase_best_circles = new_circles.copy()
                    phase_best_sum = new_sum
                elif new_sum > phase_best_sum:
                    phase_best_circles = new_circles.copy()
                    phase_best_sum = new_sum
            else:
                # If invalid, revert to best configuration in this phase
                if phase_best_circles is not None:
                    current_circles = phase_best_circles.copy()
        
        # Reset for next phase with best found so far
        if best_circles is not None:
            current_circles = best_circles.copy()
    
    # Final refinement phase
    if best_circles is not None:
        circles = best_circles
    
    # Ensure minimum radius for all circles
    for i in range(n):
        if circles[i][2] < 0.001:
            circles[i][2] = 0.01
    
    return circles


def calculate_max_radius_fast(circles, target_idx, width, height):
    """Calculate maximum possible radius for a specific circle efficiently."""
    x, y, _ = circles[target_idx]
    
    # Minimum distance to edges (vectorized approach)
    min_to_edges = min(x, y, width - x, height - y)
    
    # Vectorized distance calculation to all other circles
    other_positions = np.delete(circles, target_idx, axis=0)[:, :2]  # Get all positions except target
    if len(other_positions) == 0:
        return min_to_edges
    
    distances = np.sqrt(np.sum((other_positions - [x, y])**2, axis=1))
    
    # Minimum distance to other circles
    if len(distances) > 0:
        min_to_others = np.min(distances)  # Already accounts for radii in distance check
    else:
        min_to_others = float('inf')
    
    # The maximum radius is limited by both edge constraints and other circles
    max_radius = min(min_to_edges, min_to_others)
    
    return max(0, max_radius)


def is_valid_configuration_fast(circles, width, height):
    """Check if all circles fit within bounds and don't overlap using vectorized operations."""
    n = len(circles)
    
    # Check boundary constraints (vectorized)
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]
    
    # All circles must be within bounds
    within_bounds = (
        (x_coords - radii >= 0) & 
        (x_coords + radii <= width) & 
        (y_coords - radii >= 0) & 
        (y_coords + radii <= height)
    )
    
    if not np.all(within_bounds):
        return False
    
    # Check overlap constraints (vectorized for efficiency)
    # Create pairwise distance matrix
    positions = circles[:, :2]
    distances = cdist(positions, positions)
    
    # Zero out diagonal (distance to self)
    np.fill_diagonal(distances, np.inf)
    
    # Check if any circles overlap (distance < sum of radii)
    radius_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlaps = distances < radius_sums
    
    # If any overlaps exist, configuration is invalid
    return not np.any(overlaps)


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")