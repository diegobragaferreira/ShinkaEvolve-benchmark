# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a physics-based simulation approach.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Optimal rectangle ratio found through experimentation: 1.5:1 (width:height)
    rect_width = 1.5
    rect_height = 1.0
    
    # Number of circles
    n = 21
    
    # Physics parameters
    max_iterations = 5000
    dt = 0.01
    boundary_repulsion_strength = 10.0
    overlap_repulsion_strength = 50.0
    radius_growth_rate = 0.001
    min_radius = 0.001
    
    # Initialize circles with hexagonal packing
    def initialize_circles():
        circles = np.zeros((n, 3))
        
        # Hexagonal packing pattern
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Grid spacing
        spacing_x = rect_width / (cols + 1)
        spacing_y = rect_height / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Position with hexagonal offset
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Initial small radius
                r = min(spacing_x, spacing_y) * 0.2
                
                # Constrain to stay within bounds
                x = max(r, min(rect_width - r, x))
                y = max(r, min(rect_height - r, y))
                
                circles[idx] = [x, y, r]
                idx += 1
                if idx >= n:
                    break
        
        return circles
    
    # Physics simulation
    def simulate_physics(circles):
        for iteration in range(max_iterations):
            # Calculate forces
            forces = np.zeros_like(circles)
            
            # Boundary forces (repulsion from walls)
            for i in range(n):
                x, y, r = circles[i]
                
                # Left boundary
                if x - r < 0:
                    forces[i, 0] += boundary_repulsion_strength * (r - x)
                # Right boundary
                if x + r > rect_width:
                    forces[i, 0] -= boundary_repulsion_strength * (x + r - rect_width)
                # Bottom boundary
                if y - r < 0:
                    forces[i, 1] += boundary_repulsion_strength * (r - y)
                # Top boundary
                if y + r > rect_height:
                    forces[i, 1] -= boundary_repulsion_strength * (y + r - rect_height)
            
            # Overlap forces (repulsion from other circles)
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        # Only apply force if circles are overlapping or very close
                        overlap = (r1 + r2) - dist
                        if overlap > 0:
                            # Repulsion force
                            force_magnitude = overlap_repulsion_strength * overlap / dist
                            forces[i, 0] -= force_magnitude * dx
                            forces[i, 1] -= force_magnitude * dy
                            forces[j, 0] += force_magnitude * dx
                            forces[j, 1] += force_magnitude * dy
            
            # Update positions and velocities
            for i in range(n):
                # Simple integration
                circles[i, 0] += forces[i, 0] * dt
                circles[i, 1] += forces[i, 1] * dt
                
                # Constrain to boundaries
                x, y, r = circles[i]
                circles[i, 0] = max(r, min(rect_width - r, x))
                circles[i, 1] = max(r, min(rect_height - r, y))
        
        return circles
    
    # Maximization phase
    def maximize_radii(circles):
        # Greedy radius maximization
        improved = True
        max_iter = 1000
        
        for iteration in range(max_iter):
            if not improved:
                break
            improved = False
            
            # Process circles in random order for better exploration
            indices = list(range(n))
            np.random.shuffle(indices)
            
            for i in indices:
                x, y, r = circles[i]
                
                # Find maximum allowable radius considering all constraints
                max_radius = float('inf')
                
                # Boundary constraints
                max_radius = min(max_radius, x)
                max_radius = min(max_radius, rect_width - x)
                max_radius = min(max_radius, y)
                max_radius = min(max_radius, rect_height - y)
                
                # Overlap constraints with all other circles
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        max_allowed = dist - r2
                        if max_allowed > 0:
                            max_radius = min(max_radius, max_allowed)
                
                # Try to increase radius
                if max_radius > r and max_radius > min_radius:
                    new_radius = min(r + radius_growth_rate, max_radius)
                    # Check if this would cause overlaps
                    valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles[j]
                            dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                            if dist < new_radius + r2:
                                valid = False
                                break
                    
                    if valid:
                        circles[i, 2] = new_radius
                        improved = True
        
        return circles
    
    # Main algorithm
    circles = initialize_circles()
    
    # Phase 1: Physics simulation for initial arrangement
    circles = simulate_physics(circles)
    
    # Phase 2: Greedy radius maximization
    circles = maximize_radii(circles)
    
    # Phase 3: Final refinement with local optimization
    # Perform a few iterations of local optimization
    for _ in range(50):
        # Try to slightly adjust positions to allow for more radius
        for i in range(n):
            # Get neighbors
            neighbors = []
            for j in range(n):
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < (r1 + r2) * 1.5:  # Close enough to matter
                        neighbors.append(j)
            
            # Try small position adjustments for better packing
            if len(neighbors) > 0:
                # Perturb position slightly
                old_x, old_y, old_r = circles[i]
                perturbation = 0.01
                
                # Test small movements
                test_moves = [
                    (0, 0), (perturbation, 0), (-perturbation, 0),
                    (0, perturbation), (0, -perturbation),
                    (perturbation, perturbation), (-perturbation, perturbation),
                    (perturbation, -perturbation), (-perturbation, -perturbation)
                ]
                
                best_move = (0, 0)
                best_radius = old_r
                
                for dx, dy in test_moves:
                    test_x = old_x + dx
                    test_y = old_y + dy
                    
                    # Check if still within bounds
                    if test_x - old_r < 0 or test_x + old_r > rect_width:
                        continue
                    if test_y - old_r < 0 or test_y + old_r > rect_height:
                        continue
                    
                    # Test if this improves radius
                    max_rad = float('inf')
                    max_rad = min(max_rad, test_x)
                    max_rad = min(max_rad, rect_width - test_x)
                    max_rad = min(max_rad, test_y)
                    max_rad = min(max_rad, rect_height - test_y)
                    
                    # Check overlaps with neighbors
                    for neighbor_idx in neighbors:
                        nx, ny, nr = circles[neighbor_idx]
                        dist = np.sqrt((test_x - nx)**2 + (test_y - ny)**2)
                        max_allowed = dist - nr
                        if max_allowed > 0:
                            max_rad = min(max_rad, max_allowed)
                    
                    if max_rad > best_radius:
                        best_radius = max_rad
                        best_move = (dx, dy)
                
                # Apply best move if it improves radius
                if best_radius > old_r:
                    circles[i, 0] = old_x + best_move[0]
                    circles[i, 1] = old_y + best_move[1]
                    circles[i, 2] = best_radius
    
    # Final maximization pass
    circles = maximize_radii(circles)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
