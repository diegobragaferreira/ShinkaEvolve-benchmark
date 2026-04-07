# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    
    Uses a physics-based simulation approach where circles are modeled as particles with repulsive forces
    and wall attractions, evolving until equilibrium.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Container parameters
    rect_width = 1.3
    rect_height = 0.7
    rect_perimeter = 4.0
    
    # Physics simulation parameters
    n_circles = 21
    dt = 0.001  # Time step
    max_steps = 20000  # Maximum simulation steps
    damping = 0.99  # Velocity damping factor
    repulsion_strength = 1000.0
    wall_attraction_strength = 500.0
    radius_min = 0.001
    radius_max = 0.25
    
    # Initialize circles with a hexagonal grid pattern for good starting configuration
    circles = np.zeros((n_circles, 3))
    
    # Create hexagonal grid pattern
    aspect_ratio = rect_width / rect_height
    cols = int(np.ceil(np.sqrt(n_circles * aspect_ratio * 1.2)))
    rows = int(np.ceil(n_circles / cols))
    
    # Ensure we have enough cells
    while cols * rows < n_circles:
        if aspect_ratio >= 1.2:
            cols += 1
        elif aspect_ratio <= 0.8:
            rows += 1
        else:
            cols += 1
            
    spacing_x = rect_width / (cols + 1.5) if cols > 0 else rect_width
    spacing_y = rect_height / (rows + 1.5) if rows > 0 else rect_height
    
    placed_count = 0
    for i in range(rows):
        for j in range(cols):
            if placed_count >= n_circles:
                break
            offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
            base_x = (j + 1) * spacing_x + offset_x
            base_y = (i + 1) * spacing_y
            
            # Add small random perturbation
            x = np.clip(base_x + np.random.uniform(-0.02, 0.02), 0.01, rect_width - 0.01)
            y = np.clip(base_y + np.random.uniform(-0.02, 0.02), 0.01, rect_height - 0.01)
            
            # Estimate initial radius
            max_r = min(x, rect_width - x, y, rect_height - y)
            r = min(0.1, max_r * 0.6)
            circles[placed_count] = [x, y, r]
            placed_count += 1
            
        if placed_count >= n_circles:
            break
    
    # Convert to arrays for vectorized operations
    positions = circles[:, :2].copy()  # (n_circles, 2)
    radii = circles[:, 2].copy()       # (n_circles,)
    velocities = np.zeros_like(positions)  # (n_circles, 2)
    
    # Simulation loop
    for step in range(max_steps):
        # Calculate forces between all pairs of circles
        forces = np.zeros_like(positions)
        
        # Compute pairwise distances and forces
        pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(pos_diff**2, axis=2))
        
        # Avoid division by zero and self-interactions
        mask = distances > 0.0001
        distances_masked = np.where(mask, distances, 1.0)
        
        # Repulsive forces (inverse square law)
        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap_distance = radii_sum - distances
        overlap_mask = overlap_distance > 0
        
        # Force magnitude inversely proportional to distance squared
        force_magnitude = np.where(overlap_mask, 
                                 repulsion_strength / (distances**2 + 1e-8),
                                 0)
        
        # Direction vectors
        direction = np.where(distances_masked[..., np.newaxis] != 0,
                           pos_diff / distances_masked[..., np.newaxis],
                           0)
        
        # Accumulate forces
        force_vectors = force_magnitude[..., np.newaxis] * direction
        forces = np.sum(force_vectors, axis=1)
        
        # Wall attraction forces
        wall_forces = np.zeros_like(positions)
        
        # Left and right walls
        left_wall_force = np.maximum(0, rect_width - positions[:, 0] - radii) * wall_attraction_strength
        right_wall_force = np.maximum(0, positions[:, 0] - radii) * wall_attraction_strength
        wall_forces[:, 0] = right_wall_force - left_wall_force
        
        # Top and bottom walls
        top_wall_force = np.maximum(0, rect_height - positions[:, 1] - radii) * wall_attraction_strength
        bottom_wall_force = np.maximum(0, positions[:, 1] - radii) * wall_attraction_strength
        wall_forces[:, 1] = top_wall_force - bottom_wall_force
        
        forces += wall_forces
        
        # Update velocities and positions using Verlet integration
        accelerations = forces  # In our model, force equals mass times acceleration (assuming unit mass)
        velocities = damping * velocities + dt * accelerations
        positions += dt * velocities
        
        # Update radii based on system energy (simple relaxation)
        # This helps maintain physically realistic sizes
        radius_changes = np.zeros(n_circles)
        for i in range(n_circles):
            # If there are significant overlaps, reduce radius
            overlap_sum = 0
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                    if dist < (radii[i] + radii[j]):
                        overlap_sum += (radii[i] + radii[j] - dist)
            
            # Apply a small reduction in radius when overlapping
            if overlap_sum > 0.001:
                radius_changes[i] = -0.0001 * overlap_sum
        
        # Apply radius changes
        radii += radius_changes
        radii = np.clip(radii, radius_min, radius_max)
        
        # Convergence check: if forces are small, stop early
        if np.all(np.abs(forces) < 1e-4):
            break
    
    # Final validation of constraints
    for i in range(n_circles):
        # Ensure circles are within bounds
        x, y = positions[i]
        r = radii[i]
        
        # Clamp to boundaries
        positions[i, 0] = np.clip(x, r, rect_width - r)
        positions[i, 1] = np.clip(y, r, rect_height - r)
        
        # Ensure minimum radius
        radii[i] = np.clip(r, radius_min, radius_max)
    
    # Return final configuration
    circles = np.column_stack([positions, radii])
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")