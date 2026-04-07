# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    
    # Container setup: perimeter = 4, so width + height = 2
    # Using width = 1.2, height = 0.8 for a reasonable aspect ratio
    width, height = 1.2, 0.8
    
    n_circles = 21
    
    # Initialize circles with random positions and small radii
    circles = np.zeros((n_circles, 3))
    
    # Initialize with random positions and small radii
    for i in range(n_circles):
        circles[i] = [
            np.random.uniform(0.1, width - 0.1),
            np.random.uniform(0.1, height - 0.1),
            0.05
        ]
    
    def get_distances(circles_array):
        """Compute pairwise distances between circle centers"""
        centers = circles_array[:, :2]
        return cdist(centers, centers)
    
    def check_overlap(circles_array):
        """Check if there are any overlaps"""
        distances = get_distances(circles_array)
        radii = circles_array[:, 2]
        
        # Create matrix of sum of radii
        sum_radii = radii[:, None] + radii[None, :]
        
        # Check if any distance is less than sum of radii
        overlap_matrix = distances < sum_radii
        
        # Set diagonal to False (circle doesn't overlap with itself)
        np.fill_diagonal(overlap_matrix, False)
        
        return np.any(overlap_matrix)
    
    def boundary_constraints(circles_array, w, h):
        """Penalty for being outside boundaries"""
        penalty = 0
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Penalty for being too close to boundaries
            penalty += max(0, 0.1 - x)**2  # left
            penalty += max(0, 0.1 - (w - x))**2  # right
            penalty += max(0, 0.1 - y)**2  # bottom
            penalty += max(0, 0.1 - (h - y))**2  # top
        return penalty
    
    def simulate_repulsion_step(circles_array, w, h, dt=0.01, repulsion_strength=1.0):
        """Perform one step of physics simulation with repulsion forces"""
        forces = np.zeros_like(circles_array[:, :2])
        
        # Compute pairwise repulsion forces
        for i in range(len(circles_array)):
            x_i, y_i, r_i = circles_array[i]
            fx, fy = 0.0, 0.0
            
            # Repulsion from other circles
            for j in range(len(circles_array)):
                if i != j:
                    x_j, y_j, r_j = circles_array[j]
                    dx, dy = x_i - x_j, y_i - y_j
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0 and dist < (r_i + r_j):
                        # Repulsion force (inverse square law)
                        force_magnitude = repulsion_strength / (dist * dist + 0.001)
                        fx += force_magnitude * dx / dist
                        fy += force_magnitude * dy / dist
                        
            # Boundary repulsion
            boundary_force = 10.0
            if x_i < 0.1: fx += boundary_force * (0.1 - x_i)
            if x_i > w - 0.1: fx -= boundary_force * (x_i - (w - 0.1))
            if y_i < 0.1: fy += boundary_force * (0.1 - y_i)
            if y_i > h - 0.1: fy -= boundary_force * (y_i - (h - 0.1))
            
            forces[i] = [fx, fy]
        
        # Update positions
        new_circles = circles_array.copy()
        for i in range(len(new_circles)):
            new_circles[i, :2] += dt * forces[i]
        
        return new_circles
    
    def grow_radii_safely(circles_array, w, h):
        """Safely increase radii while maintaining non-overlap"""
        # Create copies to avoid modifying during iteration
        new_circles = circles_array.copy()
        max_radius_increase = 0.05
        changed = True
        
        while changed:
            changed = False
            for i in range(len(new_circles)):
                old_radius = new_circles[i, 2]
                # Try to increase radius
                new_radius = min(old_radius + max_radius_increase, 
                               min(new_circles[i, 0], w - new_circles[i, 0]),
                               min(new_circles[i, 1], h - new_circles[i, 1]))
                
                # Test if this radius works
                test_circles = new_circles.copy()
                test_circles[i, 2] = new_radius
                
                if not check_overlap(test_circles):
                    new_circles[i, 2] = new_radius
                    changed = True
                else:
                    # If we can't increase, try a smaller increment
                    new_radius = old_radius + max_radius_increase * 0.1
                    test_circles[i, 2] = new_radius
                    if not check_overlap(test_circles) and new_radius <= old_radius + max_radius_increase:
                        new_circles[i, 2] = new_radius
                        changed = True
        
        return new_circles
    
    # Phase 1: Physics simulation to spread out circles
    for _ in range(1000):
        circles = simulate_repulsion_step(circles, width, height)
    
    # Phase 2: Refine positions and radii
    circles = grow_radii_safely(circles, width, height)
    
    # Phase 3: Local optimization using scipy minimize
    def objective(x_flat):
        # Reshape flat array back to circles
        circles_test = circles.copy()
        for i in range(n_circles):
            circles_test[i, 0] = x_flat[2*i]
            circles_test[i, 1] = x_flat[2*i+1]
            circles_test[i, 2] = x_flat[2*n_circles+i]
        
        # Negative because we want to maximize
        return -np.sum(circles_test[:, 2])
    
    def constraint_func(x_flat):
        circles_test = circles.copy()
        for i in range(n_circles):
            circles_test[i, 0] = x_flat[2*i]
            circles_test[i, 1] = x_flat[2*i+1]
            circles_test[i, 2] = x_flat[2*n_circles+i]
        
        # Constraint: no overlap
        if check_overlap(circles_test):
            return -1  # Violated
        # Constraint: all within boundaries
        for i in range(n_circles):
            x, y, r = circles_test[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return -1  # Violated
        return 1  # Satisfied
    
    # Flatten initial circles
    x0 = []
    for i in range(n_circles):
        x0.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
    
    # Use bounds for optimization
    bounds = []
    for i in range(n_circles):
        # x position
        bounds.append((0.01, width - 0.01))
        # y position
        bounds.append((0.01, height - 0.01))
        # radius (positive)
        bounds.append((0.001, min(width, height) / 2 - 0.01))
    
    # Optimize
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
        
        # Extract results
        if res.success:
            for i in range(n_circles):
                circles[i, 0] = res.x[2*i]
                circles[i, 1] = res.x[2*i+1]
                circles[i, 2] = res.x[2*n_circles+i]
    except:
        pass
    
    # Final refinement with safe radius growth
    circles = grow_radii_safely(circles, width, height)
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
