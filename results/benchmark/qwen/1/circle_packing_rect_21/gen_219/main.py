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
    # Using width = 1.5, height = 0.5 to improve packing efficiency
    rect_width = 1.5
    rect_height = 0.5
    
    n = 21
    
    # Initialize circles with random positions and small radii
    circles = np.zeros((n, 3))
    
    # Start with hexagonal-like arrangement for good initial packing
    rows = 4
    cols = 6
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)
    
    idx = 0
    for i in range(rows):
        offset = spacing_x * (i % 2) * 0.5
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x + offset
            y = (i + 1) * spacing_y
            
            # Ensure within bounds
            x = max(0.01, min(rect_width - 0.01, x))
            y = max(0.01, min(rect_height - 0.01, y))
            
            circles[idx] = [x, y, 0.05]
            idx += 1
    
    # Fill remaining circles
    while idx < n:
        x = np.random.uniform(0.01, rect_width - 0.01)
        y = np.random.uniform(0.01, rect_height - 0.01)
        circles[idx] = [x, y, 0.05]
        idx += 1
    
    # Physics-based optimization
    def compute_forces(circles):
        """Compute forces on each circle from all others."""
        n = len(circles)
        forces = np.zeros((n, 2))  # Only x,y components (no radius force)
        
        # Force magnitude function (inverse square law with cutoff)
        def force_magnitude(dist, r1, r2):
            if dist < 1e-8:
                return 1000  # Large repulsion when on top of each other
            # Avoid very strong forces at close range
            force = 1.0 / (dist * dist + 0.01 * (r1 + r2) * (r1 + r2))
            return force
        
        # Pairwise forces
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < (r1 + r2) and dist > 0:
                    # Overlapping circles - strong repulsion
                    force_mag = 10000 * (r1 + r2 - dist) / (r1 + r2 + 1e-8)
                else:
                    # Non-overlapping circles - inverse square force
                    force_mag = force_magnitude(dist, r1, r2)
                
                # Direction vector
                if dist > 0:
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Boundary forces (hard walls)
        for i in range(n):
            x, y, r = circles[i]
            # Left wall
            if x - r < 0:
                forces[i, 0] += 1000 * (r - x)
            # Right wall
            if x + r > rect_width:
                forces[i, 0] -= 1000 * (x + r - rect_width)
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += 1000 * (r - y)
            # Top wall
            if y + r > rect_height:
                forces[i, 1] -= 1000 * (y + r - rect_height)
        
        return forces
    
    def compute_radius_gradients(circles):
        """Compute gradients for radius optimization."""
        n = len(circles)
        grad_r = np.zeros(n)
        
        # Gradient based on overlap and boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            
            # Check overlap constraints
            overlap_penalty = 0
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x - x2
                    dy = y - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < (r + r2):
                        overlap_penalty += (r + r2 - dist) * 1000
                    
            # Check boundary constraints
            boundary_penalty = 0
            if x - r < 0:
                boundary_penalty += (r - x) * 100
            if x + r > rect_width:
                boundary_penalty += (x + r - rect_width) * 100
            if y - r < 0:
                boundary_penalty += (r - y) * 100
            if y + r > rect_height:
                boundary_penalty += (y + r - rect_height) * 100
                
            # Gradient: maximize radius when not constrained
            grad_r[i] = 1.0 - overlap_penalty - boundary_penalty
            
        return grad_r
    
    # Main optimization loop
    max_iter = 2000
    learning_rate = 0.001
    
    for iteration in range(max_iter):
        # Update positions using forces (velocity Verlet style)
        forces = compute_forces(circles)
        
        # Apply forces to update positions
        for i in range(n):
            fx, fy = forces[i]
            x, y, r = circles[i]
            
            # Update velocity and position
            # Simple integration with damping
            damping = 0.9
            circles[i, 0] += learning_rate * fx * damping
            circles[i, 1] += learning_rate * fy * damping
            
            # Keep within bounds
            circles[i, 0] = np.clip(circles[i, 0], r, rect_width - r)
            circles[i, 1] = np.clip(circles[i, 1], r, rect_height - r)
        
        # Update radii using gradients
        radius_grads = compute_radius_gradients(circles)
        
        # Apply radius updates
        for i in range(n):
            r = circles[i, 2]
            # Only increase radius if beneficial
            if radius_grads[i] > 0.1:
                # Small incremental growth
                delta_r = min(0.001, radius_grads[i] * 0.001)
                circles[i, 2] = min(0.5, r + delta_r)  # Cap maximum radius
        
        # Occasionally recenter to avoid drift
        if iteration % 100 == 0:
            # Keep circles near center to avoid getting stuck at boundaries
            center_x = rect_width / 2
            center_y = rect_height / 2
            for i in range(n):
                x, y, r = circles[i]
                # Pull towards center with smaller force
                force_x = (center_x - x) * 0.001
                force_y = (center_y - y) * 0.001
                circles[i, 0] += force_x
                circles[i, 1] += force_y
                # Keep within bounds again
                circles[i, 0] = np.clip(circles[i, 0], r, rect_width - r)
                circles[i, 1] = np.clip(circles[i, 1], r, rect_height - r)
    
    # Refinement phase - fine-grained optimization
    for _ in range(500):
        # More precise local optimization
        forces = compute_forces(circles)
        radius_grads = compute_radius_gradients(circles)
        
        # Update positions more carefully  
        for i in range(n):
            fx, fy = forces[i]
            x, y, r = circles[i]
            
            # Update with smaller steps
            circles[i, 0] += 0.0001 * fx
            circles[i, 1] += 0.0001 * fy
            
            # Keep within bounds
            circles[i, 0] = np.clip(circles[i, 0], r, rect_width - r)
            circles[i, 1] = np.clip(circles[i, 1], r, rect_height - r)
        
        # Update radii with more precision
        for i in range(n):
            r = circles[i, 2]
            if radius_grads[i] > 0.01:
                delta_r = min(0.0005, radius_grads[i] * 0.0005)
                circles[i, 2] = min(0.5, r + delta_r)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")