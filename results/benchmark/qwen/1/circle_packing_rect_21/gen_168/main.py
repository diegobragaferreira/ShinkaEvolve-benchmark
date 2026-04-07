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
    # Set random seed for reproducibility
    np.random.seed(42)
    random_seed = 42
    np.random.seed(random_seed)
    
    # Rectangle dimensions (width + height = 2)
    rect_width = 1.0
    rect_height = 1.0
    
    # Number of circles
    n = 21
    
    # Physics simulation parameters
    dt = 0.001  # Time step
    max_steps = 20000  # Maximum simulation steps
    
    def is_valid_solution(circles, width, height):
        """Check if solution is valid - no overlaps and all within bounds"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False

        # Check overlap constraints - vectorized for efficiency
        if len(circles) > 1:
            coords = circles[:, :2]
            radii = circles[:, 2]
            distances = cdist(coords, coords)
            # Create mask for upper triangle (avoid double counting)
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            # Check overlaps
            overlap_distances = distances[mask]
            overlap_radii = (radii[:, None] + radii[None, :])[mask]
            if np.any(overlap_distances < overlap_radii):
                return False
        return True
    
    def compute_forces(circles, width, height):
        """Compute forces acting on each circle including repulsion, attraction to center, and boundary forces"""
        forces = np.zeros((len(circles), 2))
        k_repel = 100.0
        k_attract = 1.0
        k_boundary = 1000.0
        margin = 0.01
        
        # Repulsive forces between circles
        if len(circles) > 1:
            coords = circles[:, :2]
            radii = circles[:, 2]
            distances = cdist(coords, coords)
            
            # Create mask for upper triangle (avoid double counting)
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            
            # Compute repulsive forces
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    if mask[i, j]:
                        dx = coords[i, 0] - coords[j, 0]
                        dy = coords[i, 1] - coords[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        # Only apply force if circles are overlapping or very close
                        if dist < (radii[i] + radii[j]):
                            force_magnitude = k_repel * (radii[i] + radii[j] - dist) / (dist + 1e-8)
                            forces[i, 0] += force_magnitude * dx / (dist + 1e-8)
                            forces[i, 1] += force_magnitude * dy / (dist + 1e-8)
                            forces[j, 0] -= force_magnitude * dx / (dist + 1e-8)
                            forces[j, 1] -= force_magnitude * dy / (dist + 1e-8)
        
        # Attractive forces towards center and boundary repulsion
        center_x, center_y = width/2, height/2
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Attraction to center
            dx_center = center_x - x
            dy_center = center_y - y
            dist_center = np.sqrt(dx_center*dx_center + dy_center*dy_center)
            
            # Scale attraction based on distance from center (stronger pull when far from center)
            force_scale = k_attract * min(1.0, dist_center/0.5)
            forces[i, 0] += force_scale * dx_center / (dist_center + 1e-8)
            forces[i, 1] += force_scale * dy_center / (dist_center + 1e-8)
            
            # Boundary repulsion
            boundary_forces = np.array([0.0, 0.0])
            if x - r < margin:
                boundary_forces[0] += k_boundary * (margin - (x - r))
            elif x + r > width - margin:
                boundary_forces[0] += k_boundary * ((width - margin) - (x + r))
                
            if y - r < margin:
                boundary_forces[1] += k_boundary * (margin - (y - r))
            elif y + r > height - margin:
                boundary_forces[1] += k_boundary * ((height - margin) - (y + r))
                
            forces[i, 0] += boundary_forces[0]
            forces[i, 1] += boundary_forces[1]
            
        return forces
    
    def generate_initial_placement(width, height, n):
        """Generate initial placement using hexagonal packing with some randomness"""
        circles = np.zeros((n, 3))
        
        # Create hexagonal packing pattern
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))
        
        # Initial spacing based on circle density
        max_radius = min(width, height) * 0.08
        
        # Create hexagonal grid
        x_spacing = max_radius * 2.5
        y_spacing = max_radius * 2.165  # sqrt(3)/2 * 2
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.1 + j * x_spacing
                y = 0.1 + i * y_spacing

                if i % 2 == 1:
                    x += x_spacing / 2
                    
                # Random adjustment to avoid perfect symmetry
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                
                # Adjust for bounds
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))
                
                circles[idx] = [x, y, max_radius]
                idx += 1
                
        return circles
    
    def simulate_physics(circles, width, height, dt, max_steps):
        """Run physics simulation for circle packing"""
        # Initialize velocities
        velocities = np.zeros((len(circles), 2))
        
        # Store best solution so far
        best_circles = circles.copy()
        best_sum_radii = np.sum(circles[:, 2])
        
        # Simulate physics
        for step in range(max_steps):
            # Compute forces
            forces = compute_forces(circles, width, height)
            
            # Update velocities and positions
            for i in range(len(circles)):
                # Update velocity (v = v + F/m * dt, assuming unit mass)
                velocities[i] += forces[i] * dt
                
                # Apply damping (velocity decay)
                velocities[i] *= 0.99
                
                # Update position
                circles[i, 0] += velocities[i, 0] * dt
                circles[i, 1] += velocities[i, 1] * dt
                
                # Boundary constraints
                x, y, r = circles[i]
                if x - r < 0:
                    circles[i, 0] = r
                    velocities[i, 0] = -velocities[i, 0] * 0.5  # Bounce with damping
                elif x + r > width:
                    circles[i, 0] = width - r
                    velocities[i, 0] = -velocities[i, 0] * 0.5
                    
                if y - r < 0:
                    circles[i, 1] = r
                    velocities[i, 1] = -velocities[i, 1] * 0.5
                elif y + r > height:
                    circles[i, 1] = height - r
                    velocities[i, 1] = -velocities[i, 1] * 0.5
            
            # Periodically check and correct constraints
            if step % 100 == 0:
                # Validate and fix any constraint violations
                if not is_valid_solution(circles, width, height):
                    # Simple correction: push circles back into valid positions
                    for i in range(len(circles)):
                        x, y, r = circles[i]
                        if x - r < 0:
                            circles[i, 0] = r
                        elif x + r > width:
                            circles[i, 0] = width - r
                            
                        if y - r < 0:
                            circles[i, 1] = r
                        elif y + r > height:
                            circles[i, 1] = height - r
                
                # Update best solution
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_circles = circles.copy()
            
            # Gradually reduce time step for stability
            if step > 10000:
                dt *= 0.99995
            
            # Early stopping if improvement plateaus
            if step > 5000 and step % 1000 == 0:
                # Check if we haven't improved in last 1000 steps
                if abs(np.sum(best_circles[:, 2]) - best_sum_radii) < 1e-6:
                    break
        
        return best_circles
    
    def local_refinement(circles, width, height, iterations=50):
        """Refine solution with local optimization"""
        # Start with initial placement
        refined = circles.copy()
        
        # Gradient ascent approach with small perturbations
        for iter in range(iterations):
            # Get current objective
            current_obj = np.sum(refined[:, 2])
            
            # Try small random perturbations
            for i in range(len(refined)):
                # Save current state
                old_x, old_y, old_r = refined[i]
                
                # Try small random changes
                new_x = old_x + np.random.uniform(-0.001, 0.001)
                new_y = old_y + np.random.uniform(-0.001, 0.001)
                new_r = old_r + np.random.uniform(-0.0005, 0.0005)
                
                # Ensure radius remains positive
                new_r = max(0.001, new_r)
                
                # Test new configuration
                test_circles = refined.copy()
                test_circles[i] = [new_x, new_y, new_r]
                
                # Check if valid and improves the objective
                if is_valid_solution(test_circles, width, height):
                    test_obj = np.sum(test_circles[:, 2])
                    if test_obj > current_obj:
                        refined = test_circles.copy()
                        
        return refined
    
    # Generate initial placement
    circles = generate_initial_placement(rect_width, rect_height, n)
    
    # Run physics simulation
    circles = simulate_physics(circles, rect_width, rect_height, dt, max_steps)
    
    # Local refinement
    circles = local_refinement(circles, rect_width, rect_height, 100)
    
    # Final validation
    if not is_valid_solution(circles, rect_width, rect_height):
        # Fallback to better initial pattern if necessary
        circles = generate_initial_placement(rect_width, rect_height, n)
        circles = simulate_physics(circles, rect_width, rect_height, dt, max_steps//2)
        circles = local_refinement(circles, rect_width, rect_height, 50)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
