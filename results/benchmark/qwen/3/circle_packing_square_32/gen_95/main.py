# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Hexagonal grid initialization with better spacing
    def initialize_hexagonal_grid():
        # Calculate optimal grid dimensions for 32 circles
        # For hexagonal packing density ~0.9069
        # We need area for 32 circles: 32 * pi * r^2
        # Area of unit square = 1
        # Required radius: sqrt(32 * pi / (pi * 0.9069)) = sqrt(32/0.9069) ≈ 5.94
        # But this is for area filling, so let's use a more practical approach
        
        # Try different grid sizes and pick one that gives good coverage
        best_rows_cols = (5, 7)  # 35 positions = enough for 32 circles
        rows, cols = best_rows_cols
        
        # Calculate spacing based on number of rows/columns
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust to hexagonal packing
        hex_spacing_y = spacing_y * 2.0/np.sqrt(3)  # hexagonal vertical spacing
        
        # Compute actual grid dimensions
        grid_width = (cols - 1) * spacing_x
        grid_height = (rows - 1) * hex_spacing_y + hex_spacing_y/2
        
        # Center in unit square
        offset_x = (1 - grid_width) / 2.0
        offset_y = (1 - grid_height) / 2.0
        
        circles = []
        
        # Place in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                x = offset_x + j * spacing_x
                y = offset_y + i * hex_spacing_y
                
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2.0
                
                # Make sure it's inside unit square
                if x >= 0.05 and x <= 0.95 and y >= 0.05 and y <= 0.95:
                    circles.append([x, y, 0.02])  # Small initial radius
                    
                if len(circles) >= n:
                    break
            if len(circles) >= n:
                break
        
        # Fill remaining spots if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.01])
            
        return np.array(circles[:n])
    
    # Phase 2: Efficient overlap checking using KDTree
    def check_constraints(circles_array):
        """Check all constraints efficiently"""
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Check boundary constraints
        bound_violations = (
            np.sum(radii > positions[:, 0]) +           # Left boundary
            np.sum(radii > (1 - positions[:, 0])) +     # Right boundary  
            np.sum(radii > positions[:, 1]) +           # Bottom boundary
            np.sum(radii > (1 - positions[:, 1]))       # Top boundary
        )
        
        if bound_violations > 0:
            return False
            
        # Check overlap constraints using KDTree for efficiency
        tree = cKDTree(positions)
        # Find neighbors within distance of (r_i + r_j)
        pairs = tree.query_pairs(0.0001, output_type='ndarray')  # Very small threshold for safety
        if len(pairs) > 0:
            # Verify each pair actually violates constraints
            for i, j in pairs:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < (radii[i] + radii[j]):
                    return False
                    
        return True
    
    # Phase 3: Local optimization with radius expansion
    def optimize_circles(circles_array):
        """Perform local optimization to expand radii while maintaining constraints"""
        circles = circles_array.copy()
        max_iter = 200
        tolerance = 1e-6
        improved = True
        
        for iteration in range(max_iter):
            if not improved:
                break
                
            improved = False
            old_sum = np.sum(circles[:, 2])
            
            # Try to expand each circle
            for i in range(n):
                # Get current circle info
                x, y, r = circles[i]
                
                # Calculate maximum possible radius without violating boundary constraints
                max_radius = min(
                    x, 1-x,  # Distance to left/right boundaries
                    y, 1-y   # Distance to bottom/top boundaries
                )
                
                # Check overlap constraints with other circles
                for j in range(n):
                    if i != j:
                        xj, yj, rj = circles[j]
                        dx = x - xj
                        dy = y - yj
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # Maximum radius allowed so that this circle doesn't overlap with j
                        max_radius_with_j = distance - rj
                        max_radius = min(max_radius, max_radius_with_j)
                
                # Try to expand radius
                new_radius = min(r + 0.001, max_radius)
                
                if new_radius > r + tolerance:
                    circles[i, 2] = new_radius
                    improved = True
                    
            # Early stopping if improvement is minimal
            new_sum = np.sum(circles[:, 2])
            if abs(new_sum - old_sum) < tolerance:
                break
                
        return circles
    
    # Main execution
    # Initialize with hexagonal grid
    circles = initialize_hexagonal_grid()
    
    # Validate and optimize
    if check_constraints(circles):
        circles = optimize_circles(circles)
    
    # Final validation
    if not check_constraints(circles):
        # Reset to initial hexagonal grid if validation fails
        circles = initialize_hexagonal_grid()
        circles = optimize_circles(circles)
    
    # Ensure final solution is valid
    if not check_constraints(circles):
        # Last resort: revert to a basic good configuration
        circles = initialize_hexagonal_grid()
    
    return circles

# EVOLVE-BLOCK-END