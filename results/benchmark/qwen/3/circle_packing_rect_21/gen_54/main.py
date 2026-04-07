# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container setup (perimeter = 4, so width + height = 2)
    container_width, container_height = 1.0, 1.0
    
    # Number of circles
    n = 21
    
    # Initialize circles array
    circles = np.zeros((n, 3))
    
    # Phase 1: Hybrid Initialization
    # Get good starting positions using a combination of strategic placement and k-means clustering
    
    # Generate strategic anchor points (corners and edges)
    anchors = []
    # Corners
    anchors.extend([(0.05, 0.05), (container_width - 0.05, 0.05),
                   (0.05, container_height - 0.05), (container_width - 0.05, container_height - 0.05)])
    # Edge centers
    anchors.extend([(container_width/2, 0.05), (container_width/2, container_height - 0.05),
                   (0.05, container_height/2), (container_width - 0.05, container_height/2)])
    
    # Generate dense grid for remaining positions
    grid_points = []
    grid_density = 16
    for i in range(grid_density):
        for j in range(grid_density):
            x = (i + 0.5) / grid_density * container_width
            y = (j + 0.5) / grid_density * container_height
            grid_points.append((x, y))
    
    # Use k-means to cluster grid points to reduce redundancy
    if len(grid_points) >= n - len(anchors):
        kmeans = KMeans(n_clusters=n - len(anchors), random_state=42)
        grid_array = np.array(grid_points)
        kmeans.fit(grid_array)
        selected_grid = kmeans.cluster_centers_
        # Convert back to list of tuples
        selected_grid_points = [(x, y) for x, y in selected_grid]
    else:
        selected_grid_points = grid_points
    
    # Combine anchors and selected grid points
    initial_positions = anchors + selected_grid_points[:n-len(anchors)]
    
    # Initialize circles with positions and small radii
    for i in range(n):
        x, y = initial_positions[i]
        circles[i] = [x, y, 0.02]
    
    # Phase 2: Multi-phase Optimization
    max_iterations = 5000
    best_sum_radii = 0
    best_circles = None
    
    # Parameters that adapt during optimization
    learning_rate = 0.1
    penalty_weight = 1.0
    max_radius_limit = 0.4  # Prevent extremely large radii
    
    # Optimization phases - optimized parameters
    phase_1_iterations = 1000
    phase_2_iterations = 2000
    phase_3_iterations = 2000
    
    # Phase 1: Aggressive exploration with large steps
    for iteration in range(phase_1_iterations):
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Early termination if already optimal
        if iteration > 100 and iteration % 100 == 0:
            current_sum = np.sum(radii)
            if current_sum > best_sum_radii:
                best_sum_radii = current_sum
                best_circles = circles.copy()
        
        # Compute distance matrix efficiently
        distances = cdist(positions, positions)
        
        # Check for violations with early exit
        violation_count = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    violation_count += 1
                    if violation_count > 50:  # Early exit if too many violations
                        break
            if violation_count > 50:
                break
        
        # If few violations or later iterations, proceed with optimization
        if violation_count <= 10 or iteration > phase_1_iterations // 2:
            # Update radii based on available space
            for i in range(n):
                max_radius = float('inf')
                
                # Boundary constraints
                x, y, r = circles[i]
                boundary_radius = min(x, container_width - x, y, container_height - y)
                max_radius = min(max_radius, boundary_radius)
                
                # Circle-to-circle constraints
                for j in range(n):
                    if i != j:
                        dist = distances[i, j]
                        if dist > 0.001:  # Avoid division by zero
                            max_radius = min(max_radius, dist - circles[j, 2])
                
                # Limit maximum radius
                max_radius = min(max_radius, max_radius_limit)
                
                if max_radius > r and max_radius < float('inf'):
                    # Adaptive radius increase
                    circles[i, 2] = min(r + learning_rate * 0.5 * (max_radius - r), max_radius)
            
            # Position optimization with better repulsion
            for i in range(n):
                x, y, r = circles[i]
                
                # Compute forces from overlapping circles
                fx, fy = 0.0, 0.0
                
                for j in range(n):
                    if i != j:
                        dx = circles[j, 0] - x
                        dy = circles[j, 1] - y
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        if dist < (r + circles[j, 2]) and dist > 0.001:
                            # Repulsive force
                            force_magnitude = (r + circles[j, 2] - dist) / dist
                            fx -= force_magnitude * dx
                            fy -= force_magnitude * dy
                
                # Boundary forces
                if x < r:
                    fx += (r - x) * 2.0
                if x + r > container_width:
                    fx -= (x + r - container_width) * 2.0
                if y < r:
                    fy += (r - y) * 2.0
                if y + r > container_height:
                    fy -= (y + r - container_height) * 2.0
                
                # Apply movement with bounds checking
                new_x = max(r, min(container_width - r, x + learning_rate * fx))
                new_y = max(r, min(container_height - r, y + learning_rate * fy))
                
                circles[i, 0] = new_x
                circles[i, 1] = new_y
                
        # Decay learning rate
        learning_rate *= 0.9995
        
        # Global refinement occasionally
        if iteration % 200 == 0 and iteration > 0:
            # Simple restart with slightly perturbed positions
            for i in range(n):
                x, y, r = circles[i]
                circles[i] = [max(r, min(container_width - r, x + np.random.normal(0, 0.01))),
                              max(r, min(container_height - r, y + np.random.normal(0, 0.01))),
                              r]
    
    # Phase 2: Refinement with smaller steps
    learning_rate = 0.05
    for iteration in range(phase_2_iterations):
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute distance matrix efficiently
        distances = cdist(positions, positions)
        
        # Check for violations with early exit
        violation_count = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    violation_count += 1
                    if violation_count > 30:
                        break
            if violation_count > 30:
                break
        
        # Update radii
        for i in range(n):
            max_radius = float('inf')
            
            # Boundary constraints
            x, y, r = circles[i]
            boundary_radius = min(x, container_width - x, y, container_height - y)
            max_radius = min(max_radius, boundary_radius)
            
            # Circle-to-circle constraints
            for j in range(n):
                if i != j:
                    dist = distances[i, j]
                    if dist > 0.001:  # Avoid division by zero
                        max_radius = min(max_radius, dist - circles[j, 2])
            
            # Limit maximum radius
            max_radius = min(max_radius, max_radius_limit)
            
            if max_radius > r and max_radius < float('inf'):
                # More conservative radius increase
                circles[i, 2] = min(r + learning_rate * 0.3 * (max_radius - r), max_radius)
        
        # Position optimization
        for i in range(n):
            x, y, r = circles[i]
            
            # Compute forces
            fx, fy = 0.0, 0.0
            
            for j in range(n):
                if i != j:
                    dx = circles[j, 0] - x
                    dy = circles[j, 1] - y
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < (r + circles[j, 2]) and dist > 0.001:
                        # Repulsive force
                        force_magnitude = (r + circles[j, 2] - dist) / dist
                        fx -= force_magnitude * dx
                        fy -= force_magnitude * dy
            
            # Boundary forces
            if x < r:
                fx += (r - x) * 3.0
            if x + r > container_width:
                fx -= (x + r - container_width) * 3.0
            if y < r:
                fy += (r - y) * 3.0
            if y + r > container_height:
                fy -= (y + r - container_height) * 3.0
            
            # Apply movement with bounds checking
            new_x = max(r, min(container_width - r, x + learning_rate * fx))
            new_y = max(r, min(container_height - r, y + learning_rate * fy))
            
            circles[i, 0] = new_x
            circles[i, 1] = new_y
            
        # Decay learning rate
        learning_rate *= 0.9998
        
        # Save best configuration
        current_sum = np.sum(radii)
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_circles = circles.copy()
    
    # Phase 3: Fine-tuning
    learning_rate = 0.01
    for iteration in range(phase_3_iterations):
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        distances = cdist(positions, positions)
        
        # Update radii with fine adjustments
        for i in range(n):
            max_radius = float('inf')
            
            # Boundary constraints
            x, y, r = circles[i]
            boundary_radius = min(x, container_width - x, y, container_height - y)
            max_radius = min(max_radius, boundary_radius)
            
            # Circle-to-circle constraints
            for j in range(n):
                if i != j:
                    dist = distances[i, j]
                    if dist > 0.001:
                        max_radius = min(max_radius, dist - circles[j, 2])
            
            # Limit maximum radius
            max_radius = min(max_radius, max_radius_limit)
            
            if max_radius > r and max_radius < float('inf'):
                # Very slow radius increase
                circles[i, 2] = min(r + learning_rate * 0.1 * (max_radius - r), max_radius)
        
        # Position optimization with fine-grained movements
        for i in range(n):
            x, y, r = circles[i]
            
            # Compute forces
            fx, fy = 0.0, 0.0
            
            for j in range(n):
                if i != j:
                    dx = circles[j, 0] - x
                    dy = circles[j, 1] - y
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < (r + circles[j, 2]) and dist > 0.001:
                        # Repulsive force
                        force_magnitude = (r + circles[j, 2] - dist) / dist
                        fx -= force_magnitude * dx
                        fy -= force_magnitude * dy
            
            # Boundary forces
            if x < r:
                fx += (r - x) * 5.0
            if x + r > container_width:
                fx -= (x + r - container_width) * 5.0
            if y < r:
                fy += (r - y) * 5.0
            if y + r > container_height:
                fy -= (y + r - container_height) * 5.0
            
            # Apply small movements
            new_x = max(r, min(container_width - r, x + learning_rate * fx))
            new_y = max(r, min(container_height - r, y + learning_rate * fy))
            
            circles[i, 0] = new_x
            circles[i, 1] = new_y
            
        # Decay learning rate
        learning_rate *= 0.9999
        
        # Save best configuration
        current_sum = np.sum(radii)
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_circles = circles.copy()
    
    # Return the best configuration found
    if best_circles is not None:
        return best_circles
    
    # Final cleanup
    for i in range(n):
        x, y, r = circles[i]
        # Ensure circles stay within bounds
        x = max(r, min(container_width - r, x))
        y = max(r, min(container_height - r, y))
        circles[i] = [x, y, r]
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
