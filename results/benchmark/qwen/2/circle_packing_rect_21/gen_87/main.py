# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses an adaptive grid-based evolutionary approach for optimal packing.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    # Using optimal aspect ratio (based on benchmarks)
    width = 1.2
    height = 0.8
    
    def check_constraints(positions, radii):
        """Check if all circles are within bounds and non-overlapping"""
        # Check boundary constraints
        for i in range(len(positions)):
            x, y = positions[i]
            r = radii[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False
        
        # Check overlap constraints with optimized spatial indexing
        if len(positions) < 2:
            return True
            
        try:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(2 * max(radii) if radii else 1, output_type='ndarray')
            
            for i, j in pairs:
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < radii[i] + radii[j]:
                    return False
        except:
            # Fallback to brute force
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < radii[i] + radii[j]:
                        return False
                        
        return True

    def compute_radius_sum(positions, radii):
        """Compute sum of all radii"""
        return sum(radii)

    def generate_adaptive_grid(width, height, n_circles):
        """Generate initial circle placement using adaptive grid approach"""
        # Calculate optimal grid dimensions
        cols = int(math.ceil(math.sqrt(n_circles)))
        rows = int(math.ceil(n_circles / cols))
        
        # Adjust to ensure enough slots
        if cols * rows < n_circles:
            cols += 1
            
        # Calculate spacing with padding
        x_spacing = width / (cols + 1)
        y_spacing = height / (rows + 1)
        
        # Start with grid placement
        circles = np.zeros((n_circles, 3))
        idx = 0
        
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * x_spacing
                y = (i + 1) * y_spacing
                # Set initial radius proportional to available space
                max_radius = min(x_spacing, y_spacing) * 0.3
                circles[idx] = [x, y, max_radius]
                idx += 1
                
        # Fill remaining slots with random positions
        for i in range(idx, n_circles):
            circles[i] = [np.random.uniform(0.05, width - 0.05),
                         np.random.uniform(0.05, height - 0.05),
                         0.05]
        
        return circles

    def optimize_single_circle(index, circles, width, height):
        """Optimize a single circle by maximizing its radius while respecting constraints"""
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        def objective(r):
            # Create temporary array
            temp_positions = positions.copy()
            temp_radii = radii.copy()
            temp_radii[index] = r[0]
            
            # Check if this is valid
            temp_circles = np.column_stack([temp_positions, temp_radii])
            if not check_constraints(temp_circles[:, :2], temp_circles[:, 2]):
                return 1e10  # Large penalty for invalid configurations
            
            return -r[0]  # Negative because we want to maximize
        
        # Initial guess
        current_r = radii[index]
        
        # Bounds for radius (must be positive, and not cause overlaps)
        bounds = [(1e-6, min(width/2, height/2, current_r*3))]
        
        try:
            result = minimize(objective, [current_r], bounds=bounds, method='L-BFGS-B')
            if result.success:
                return max(1e-6, result.x[0])
        except:
            pass
        return current_r

    def fill_empty_spaces(circles, width, height):
        """Attempt to add more circles or increase existing ones in empty spaces"""
        # This is a simplified version - in practice could be more sophisticated
        return circles

    # Phase 1: Adaptive Grid Initialization
    circles = generate_adaptive_grid(width, height, 21)
    
    # Phase 2: Iterative Optimization with Multiple Rounds
    max_iterations = 100
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to optimize each circle individually
        for i in range(21):
            # Store current state
            old_rad = circles[i][2]
            
            # Try to increase radius
            new_rad = optimize_single_circle(i, circles, width, height)
            
            if new_rad > old_rad:
                circles[i][2] = new_rad
                improved = True
                
        # Occasionally refine all circles together
        if iteration % 10 == 0:
            # Do a more thorough optimization of all circles
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Reset all radii to a balanced value and recompute
            avg_radius = np.mean(radii)
            for i in range(21):
                circles[i][2] = avg_radius * 0.8
        
    # Phase 3: Final Constraint Enforcement and Refinement
    # Ensure all circles fit within rectangle and aren't overlapping
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Validate constraints
    if not check_constraints(positions, radii):
        # Try to resolve overlaps by reducing radii
        for i in range(21):
            for j in range(i+1, 21):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = radii[i] + radii[j]
                
                if distance < min_distance:
                    # Reduce radii to resolve overlap
                    reduction = (min_distance - distance) * 0.5
                    radii[i] = max(1e-6, radii[i] - reduction)
                    radii[j] = max(1e-6, radii[j] - reduction)
    
    # Boundary clamping
    for i in range(21):
        x, y, r = circles[i]
        circles[i] = [
            max(r, min(width - r, x)),
            max(r, min(height - r, y)),
            radii[i]
        ]
    
    # Final validation
    positions = circles[:, :2]
    radii = circles[:, 2]
    if not check_constraints(positions, radii):
        # Emergency repair
        for i in range(21):
            for j in range(i+1, 21):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = radii[i] + radii[j]
                
                if distance < min_distance:
                    # Very conservative reduction
                    reduction = (min_distance - distance) * 0.1
                    radii[i] = max(1e-6, radii[i] - reduction)
                    radii[j] = max(1e-6, radii[j] - reduction)
    
    # Final clamping
    for i in range(21):
        x, y, r = circles[i]
        circles[i] = [
            max(r, min(width - r, x)),
            max(r, min(height - r, y)),
            radii[i]
        ]
    
    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
