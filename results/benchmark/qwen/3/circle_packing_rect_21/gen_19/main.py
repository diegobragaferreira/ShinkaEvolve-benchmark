# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set up container dimensions (width + height = 2)
    # We'll use a 1:1 ratio for simplicity, so width=height=1
    container_width, container_height = 1.0, 1.0
    
    # Number of circles
    n = 21
    
    # Initialize circles array
    circles = np.zeros((n, 3))
    
    # Pre-computed grid points for initialization
    # Using a multi-scale approach: coarse grid first, then fine
    coarse_grid_points = []
    fine_grid_points = []
    
    # Generate grid points for initialization
    grid_size_coarse = 8
    grid_size_fine = 16
    
    # Coarse grid points (for initial spread)
    for i in range(grid_size_coarse):
        for j in range(grid_size_coarse):
            x = (i + 0.5) / grid_size_coarse * container_width
            y = (j + 0.5) / grid_size_coarse * container_height
            coarse_grid_points.append((x, y))
    
    # Fine grid points (for refinement)
    for i in range(grid_size_fine):
        for j in range(grid_size_fine):
            x = (i + 0.5) / grid_size_fine * container_width
            y = (j + 0.5) / grid_size_fine * container_height
            fine_grid_points.append((x, y))
    
    # Strategy: Start with a diverse set of initial points
    # Place some circles at corners and edges, others in the interior
    initial_positions = []
    
    # Add corners
    initial_positions.extend([(0.05, 0.05), (container_width - 0.05, 0.05),
                             (0.05, container_height - 0.05), (container_width - 0.05, container_height - 0.05)])
    
    # Add edge centers
    initial_positions.extend([(container_width/2, 0.05), (container_width/2, container_height - 0.05),
                             (0.05, container_height/2), (container_width - 0.05, container_height/2)])
    
    # Fill remaining positions with grid points
    remaining_count = n - len(initial_positions)
    selected_grid_points = fine_grid_points[:remaining_count]
    initial_positions.extend(selected_grid_points)
    
    # Initialize circles with positions and small radii
    for i in range(n):
        x, y = initial_positions[i]
        circles[i] = [x, y, 0.02]
    
    # Optimization parameters
    max_iter = 1000
    tolerance = 1e-6
    learning_rate = 0.1
    
    # Constraint satisfaction loop
    for iteration in range(max_iter):
        # Compute pairwise distances between circles
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute distance matrix
        distances = cdist(positions, positions)
        
        # Calculate overlap violations
        overlap_violations = []
        total_overlap_penalty = 0
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    overlap_violations.append((i, j, overlap))
                    total_overlap_penalty += overlap ** 2
        
        # Calculate boundary violations
        boundary_violations = []
        boundary_penalty = 0
        
        for i in range(n):
            x, y, r = circles[i]
            # Check 4 boundaries (left, right, bottom, top)
            left_violation = max(0, r - x)
            right_violation = max(0, x + r - container_width)
            bottom_violation = max(0, r - y)
            top_violation = max(0, y + r - container_height)
            
            if left_violation > 0 or right_violation > 0 or bottom_violation > 0 or top_violation > 0:
                boundary_violations.append((i, left_violation, right_violation, bottom_violation, top_violation))
                boundary_penalty += (left_violation + right_violation + bottom_violation + top_violation) ** 2
        
        # If no violations, stop early
        if len(overlap_violations) == 0 and len(boundary_violations) == 0:
            break
            
        # Compute individual radius adjustments
        for i in range(n):
            # Calculate maximum possible radius without violating other circles
            max_radius = float('inf')
            
            # Check boundary constraints
            x, y, r = circles[i]
            boundary_radius = min(x, container_width - x, y, container_height - y)
            max_radius = min(max_radius, boundary_radius)
            
            # Check other circles
            for j in range(n):
                if i != j:
                    dist = distances[i, j]
                    # New radius cannot make circles overlap
                    # So new_radius <= dist - old_radius
                    max_radius = min(max_radius, dist - circles[j, 2])
            
            # Apply adaptive adjustment
            # Only adjust if we're not at the maximum possible radius
            if max_radius < 1.0:  # Prevent overly large radius
                # Adjust the radius towards the maximum possible
                circles[i, 2] = min(r + learning_rate * (max_radius - r), max_radius)
                
        # Perform local optimization for better arrangement
        # Use gradient-like descent for position updates considering overlap and boundary violations
        for i in range(n):
            # Compute gradients based on overlaps and boundaries
            x, y, r = circles[i]
            
            # Initial position update
            dx, dy = 0.0, 0.0
            
            # Overlap avoidance (repulsion forces)
            for j in range(n):
                if i != j:
                    dx_ij = circles[j, 0] - x
                    dy_ij = circles[j, 1] - y
                    dist = math.sqrt(dx_ij**2 + dy_ij**2)
                    
                    if dist < (r + circles[j, 2]) and dist > 0:
                        # Repulsive force
                        force = (r + circles[j, 2] - dist) / dist
                        dx -= force * dx_ij
                        dy -= force * dy_ij
            
            # Boundary forces (attract back to valid region)
            # Left boundary
            if x < r:
                dx += (r - x) * 0.5
            # Right boundary
            if x + r > container_width:
                dx -= (x + r - container_width) * 0.5
            # Bottom boundary
            if y < r:
                dy += (r - y) * 0.5
            # Top boundary
            if y + r > container_height:
                dy -= (y + r - container_height) * 0.5
                
            # Apply updates (with bounds checking)
            new_x = max(r, min(container_width - r, x + learning_rate * dx))
            new_y = max(r, min(container_height - r, y + learning_rate * dy))
            
            circles[i, 0] = new_x
            circles[i, 1] = new_y
        
        # Reduce learning rate over time for stability
        learning_rate *= 0.999
        
        # Occasionally perform a global refinement step
        if iteration % 50 == 0 and iteration > 0:
            # Reinitialize with better spread in case we got stuck
            for i in range(n):
                x, y, r = circles[i]
                # Try to find a better nearby position
                best_x, best_y = x, y
                best_radius = r
                best_score = 0
                
                # Sample nearby positions
                for _ in range(10):
                    # Random perturbation
                    delta_x = np.random.uniform(-0.05, 0.05)
                    delta_y = np.random.uniform(-0.05, 0.05)
                    new_x = max(r, min(container_width - r, x + delta_x))
                    new_y = max(r, min(container_height - r, y + delta_y))
                    
                    # Evaluate new configuration
                    test_circles = circles.copy()
                    test_circles[i, 0] = new_x
                    test_circles[i, 1] = new_y
                    
                    # Score based on overlap and boundary violations
                    test_positions = test_circles[:, :2]
                    test_radii = test_circles[:, 2]
                    test_distances = cdist(test_positions, test_positions)
                    score = 0
                    
                    for k in range(n):
                        for l in range(k+1, n):
                            dist = test_distances[k, l]
                            if dist < (test_radii[k] + test_radii[l]):
                                # Penalty for overlaps
                                penalty = (test_radii[k] + test_radii[l] - dist)**2
                                score -= penalty
                            
                    if score > best_score:
                        best_score = score
                        best_x, best_y = new_x, new_y
                        
                circles[i, 0] = best_x
                circles[i, 1] = best_y
    
    # Final cleanup: Ensure all circles respect constraints
    for i in range(n):
        # Bound positions properly
        x, y, r = circles[i]
        x = max(r, min(container_width - r, x))
        y = max(r, min(container_height - r, y))
        circles[i] = [x, y, r]

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
