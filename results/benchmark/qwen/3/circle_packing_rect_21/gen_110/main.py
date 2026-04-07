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
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    # Using width = 1.2, height = 0.8 for reasonable aspect ratio
    width, height = 1.2, 0.8
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Physics-based circle packing with force-directed approach
    # Parameters
    num_circles = 21
    max_iterations = 10000
    dt = 0.01  # time step
    friction = 0.9  # damping factor
    boundary_force = 100.0  # strength of boundary attraction
    repulsion_base = 1000.0  # base repulsion force
    repulsion_power = 2.0    # power law for repulsion decay
    
    # Initialize circles with random positions and small radii
    circles = np.zeros((num_circles, 3))
    
    # Initialize with random positions within bounds
    for i in range(num_circles):
        circles[i] = [
            np.random.uniform(0.05, width - 0.05),
            np.random.uniform(0.05, height - 0.05),
            0.02  # initial small radius
        ]
    
    # Main physics simulation loop
    for iteration in range(max_iterations):
        # Calculate forces on each circle
        forces = np.zeros((num_circles, 2))  # (dx, dy) for each circle
        
        # Boundary forces (attract circles toward center, push away from edges)
        for i in range(num_circles):
            x, y, r = circles[i]
            
            # Attraction to center (stronger for smaller circles)
            center_attraction = 0.1
            forces[i][0] += center_attraction * (width/2 - x)
            forces[i][1] += center_attraction * (height/2 - y)
            
            # Repulsion from boundaries
            boundary_repulsion = boundary_force * 0.1
            forces[i][0] += boundary_repulsion * max(0, 0.05 - x)  # left edge
            forces[i][0] -= boundary_repulsion * max(0, x - (width - 0.05))  # right edge
            forces[i][1] += boundary_repulsion * max(0, 0.05 - y)  # bottom edge
            forces[i][1] -= boundary_repulsion * max(0, y - (height - 0.05))  # top edge
        
        # Circle-to-circle repulsion forces
        positions = circles[:, :2]  # (x, y) coordinates only
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # For each pair of circles
        for i in range(num_circles):
            for j in range(i+1, num_circles):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Only apply force if circles are close enough or overlapping
                if distance > 0 and distance < (circles[i][2] + circles[j][2]) * 1.5:
                    # Repulsion force (inverse square law)
                    # Force decreases with distance squared
                    force_magnitude = repulsion_base / (distance ** repulsion_power + 1e-8)
                    
                    # Normalize direction and scale by force
                    if distance > 0:
                        fx = force_magnitude * dx / distance
                        fy = force_magnitude * dy / distance
                    else:
                        fx = fy = 0
                    
                    # Apply forces (opposite directions)
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
        
        # Update positions and radii
        for i in range(num_circles):
            x, y, r = circles[i]
            
            # Apply forces to velocity (implicit Euler integration)
            vx = forces[i][0]
            vy = forces[i][1]
            
            # Apply damping
            vx *= friction
            vy *= friction
            
            # Update position
            new_x = x + vx * dt
            new_y = y + vy * dt
            
            # Keep within bounds
            new_x = np.clip(new_x, 0.05, width - 0.05)
            new_y = np.clip(new_y, 0.05, height - 0.05)
            
            # Compute maximum possible radius at new position
            max_radius = min(
                new_x, 
                width - new_x, 
                new_y, 
                height - new_y
            )
            
            # Check for overlaps with all other circles and reduce radius accordingly
            for j in range(num_circles):
                if i != j:
                    dx = new_x - circles[j][0]
                    dy = new_y - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    min_distance = circles[i][2] + circles[j][2]
                    
                    if distance < min_distance:
                        # Reduce radius to prevent overlap
                        overlap_amount = min_distance - distance
                        max_radius = max(0.001, max_radius - overlap_amount * 0.5)
            
            # Limit maximum radius to reasonable values
            max_radius = min(max_radius, 0.3)
            
            # Update circle
            circles[i] = [new_x, new_y, max_radius]
        
        # Early termination: if nothing changed significantly, stop
        if iteration > 100 and iteration % 100 == 0:
            # Check for convergence by looking at total radius changes
            total_radius = np.sum(circles[:, 2])
            if np.std(circles[:, 2]) < 1e-6:
                break
    
    # Final refinement step: try to maximize individual radii
    # This helps improve the total sum of radii further
    for _ in range(500):
        improved = False
        # Try to increase each circle's radius individually
        for i in range(num_circles):
            x, y, r = circles[i]
            
            # Calculate maximum possible radius
            max_r = min(x, width - x, y, height - y)
            
            # Check for overlaps with other circles
            for j in range(num_circles):
                if i != j:
                    dx = x - circles[j][0]
                    dy = y - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    overlap = distance - (r + circles[j][2])
                    
                    if overlap < 0:
                        # Need to reduce radius to prevent overlap
                        max_r = min(max_r, distance - circles[j][2] - 1e-6)
            
            # Try to set radius to maximum allowed
            if max_r > r + 1e-6:
                circles[i][2] = max_r
                improved = True
        
        if not improved:
            break
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
