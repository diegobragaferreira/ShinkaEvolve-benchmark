# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses a physics-based simulation approach where circles repel each other and are attracted to boundaries.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container parameters (perimeter = 4 => width + height = 2)
    container_width = 1.0
    container_height = 1.0
    
    # Physics simulation parameters
    n_circles = 21
    max_iterations = 10000
    dt = 0.01
    repulsion_strength = 100.0
    boundary_strength = 50.0
    radius_adjustment_factor = 0.01
    
    # Initialize circles with random positions and small radii
    circles = np.zeros((n_circles, 3))
    
    # Random initialization within container bounds
    np.random.seed(42)  # For reproducibility
    circles[:, 0] = np.random.uniform(0.01, container_width - 0.01, n_circles)  # x coordinates
    circles[:, 1] = np.random.uniform(0.01, container_height - 0.01, n_circles)  # y coordinates  
    circles[:, 2] = np.random.uniform(0.01, 0.1, n_circles)  # Initial small radii
    
    # Normalize radii to get approximately right total area
    total_radius = np.sum(circles[:, 2])
    target_sum = 1.0  # Adjust this to make better use of space
    scaling_factor = target_sum / total_radius if total_radius > 0 else 1.0
    circles[:, 2] *= scaling_factor
    
    # Store previous positions for convergence check
    prev_positions = circles[:, :2].copy()
    
    # Physics simulation loop
    for iteration in range(max_iterations):
        # Compute pairwise distances between all circles
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Calculate distance matrix
        dist_matrix = cdist(positions, positions)
        
        # Initialize forces
        forces = np.zeros_like(positions)
        
        # Repulsion forces between overlapping circles
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                if i != j:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # If circles overlap
                    if distance < (radii[i] + radii[j]):
                        # Repulsive force magnitude
                        force_magnitude = repulsion_strength * (1.0 - distance/(radii[i] + radii[j]))
                        
                        # Direction of force (from j to i)
                        if distance > 1e-8:
                            fx = force_magnitude * dx / distance
                            fy = force_magnitude * dy / distance
                        else:
                            # Random direction if too close
                            angle = np.random.uniform(0, 2*np.pi)
                            fx = force_magnitude * np.cos(angle)
                            fy = force_magnitude * np.sin(angle)
                        
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
        
        # Boundary forces (attract circles back into container)
        for i in range(n_circles):
            x, y = positions[i]
            r = radii[i]
            
            # Left boundary
            if x - r < 0:
                forces[i, 0] += boundary_strength * (0 - (x - r))
            # Right boundary  
            if x + r > container_width:
                forces[i, 0] += boundary_strength * (container_width - (x + r))
            # Bottom boundary
            if y - r < 0:
                forces[i, 1] += boundary_strength * (0 - (y - r))
            # Top boundary
            if y + r > container_height:
                forces[i, 1] += boundary_strength * (container_height - (y + r))
        
        # Update positions
        for i in range(n_circles):
            # Apply forces to position
            positions[i, 0] += forces[i, 0] * dt
            positions[i, 1] += forces[i, 1] * dt
            
            # Keep within bounds
            positions[i, 0] = np.clip(positions[i, 0], radii[i], container_width - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], container_height - radii[i])
        
        # Gradually increase radii where there's room
        # Only modify radii if we're making progress
        if iteration % 100 == 0:
            # Check if we've made significant progress
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < 0.001 and iteration > 1000:
                # Try increasing radii slightly
                for i in range(n_circles):
                    # Check if we can safely increase radius
                    can_increase = True
                    current_radius = radii[i]
                    
                    # Check all other circles
                    for j in range(n_circles):
                        if i != j:
                            dx = positions[i, 0] - positions[j, 0]
                            dy = positions[i, 1] - positions[j, 1]
                            distance = np.sqrt(dx*dx + dy*dy)
                            
                            if distance < (current_radius + radii[j] + 0.001):
                                can_increase = False
                                break
                    
                    if can_increase:
                        # Increase radius safely
                        radii[i] = min(current_radius + radius_adjustment_factor, 
                                     container_width/4, container_height/4)
            
            prev_positions = positions.copy()
        
        # Check for convergence every 1000 iterations
        if iteration % 1000 == 0 and iteration > 0:
            pos_change = np.mean(np.linalg.norm(positions - prev_positions, axis=1))
            if pos_change < 0.0001:
                break
    
    # Final cleanup - ensure all circles are properly contained
    for i in range(n_circles):
        # Keep circle within container bounds
        circles[i, 0] = np.clip(positions[i, 0], circles[i, 2], container_width - circles[i, 2])
        circles[i, 1] = np.clip(positions[i, 1], circles[i, 2], container_height - circles[i, 2])
        circles[i, 2] = max(circles[i, 2], 0.001)  # Ensure positive radius
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
