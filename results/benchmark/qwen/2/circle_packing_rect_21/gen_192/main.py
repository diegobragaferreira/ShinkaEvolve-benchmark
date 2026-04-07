# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, using optimized aspect ratio
    width, height = 1.2, 0.8

    # Initialize parameters
    n_circles = 21
    random.seed(42)

    def initialize_circles():
        """Initialize circles with adaptive grid placement considering aspect ratio and area optimization"""
        circles = np.zeros((n_circles, 3))

        # Calculate ideal spacing based on area per circle
        area_per_circle = (width * height) / n_circles
        ideal_spacing = math.sqrt(area_per_circle)
        
        # Determine grid dimensions that match rectangle aspect ratio
        cols = max(1, int(width / ideal_spacing))
        rows = max(1, int(height / ideal_spacing))
        
        # Ensure sufficient grid cells
        while cols * rows < n_circles:
            if cols < rows:
                cols += 1
            else:
                rows += 1
                
        # Adjust for aspect ratio
        target_aspect = width / height
        if abs(cols / rows - target_aspect) > 0.2:
            if cols / rows > target_aspect:
                rows = max(1, int(cols / target_aspect))
            else:
                cols = max(1, int(rows * target_aspect))
                
        # Limit grid size if necessary
        actual_n = cols * rows
        if actual_n > n_circles:
            while cols * rows > n_circles:
                if cols > 1 and (cols * rows - n_circles) >= cols:
                    cols -= 1
                elif rows > 1:
                    rows -= 1
                else:
                    break

        # Recalculate spacing with final dimensions
        spacing_x = width / (cols + 1) if cols > 0 else width
        spacing_y = height / (rows + 1) if rows > 0 else height

        # Create grid with hexagonal offset for better packing
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                # Offset every other row for better packing
                offset = (j % 2) * spacing_x * 0.5
                x = (i + 1) * spacing_x + offset + random.uniform(-spacing_x*0.05, spacing_x*0.05)
                y = (j + 1) * spacing_y + random.uniform(-spacing_y*0.05, spacing_y*0.05)
                
                # Base radius calculation with better spatial awareness
                base_radius = min(spacing_x, spacing_y) * 0.15
                r = base_radius
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n_circles:
                break

        # Adjust initial radii to ensure they fit within bounds properly
        for i in range(n_circles):
            x, y, r = circles[i]
            min_dist = min(x, y, width - x, height - y)
            circles[i][2] = min(r, min_dist * 0.8)

        return circles

    def is_valid_position(circle):
        """Check if circle position is valid (within bounds)"""
        x, y, r = circle
        return (r <= x <= width - r and
                r <= y <= height - r)

    def calculate_constraint_penalty(circles_array):
        """Calculate total penalty for boundary and overlap violations"""
        penalty = 0.0

        # Check boundary violations with stronger penalty
        for circle in circles_array:
            x, y, r = circle
            boundary_dist = min(x, y, width - x, height - y)
            if boundary_dist < r:
                penalty += (r - boundary_dist) ** 3  # Cubic penalty for severe violations

        # Check overlap violations with more aggressive penalty
        for i in range(len(circles_array)):
            x1, y1, r1 = circles_array[i]
            for j in range(i):
                x2, y2, r2 = circles_array[j]
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)
                overlap = max(0, (r1 + r2) - distance)
                if overlap > 0:
                    penalty += overlap ** 3  # Cubic penalty for overlaps

        return penalty

    def get_spatial_index(circles_array):
        """Create spatial index for fast neighbor lookups with optimized parameters"""
        points = circles_array[:, :2]
        return cKDTree(points)

    def check_overlap(circle, existing_circles):
        """Check if circle overlaps with any existing circles"""
        x, y, r = circle
        for cx, cy, cr in existing_circles:
            dx = x - cx
            dy = y - cy
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < (r + cr):
                return True
        return False

    def apply_physics_update(circles_array, max_iter=1000):
        """Apply physics-based optimization to improve packing"""
        tree = get_spatial_index(circles_array)
        repulsion_strength = 3.0  # Increased repulsion strength
        attraction_strength = 0.15  # Increased attraction strength

        for iteration in range(max_iter):
            forces = np.zeros_like(circles_array)

            # Calculate repulsion forces with improved neighbor search
            for i in range(len(circles_array)):
                x1, y1, r1 = circles_array[i]

                # Find neighbors within reasonable range
                neighbors = tree.query_ball_point([x1, y1], 3 * (r1 + 0.01), p=2)
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = circles_array[j]

                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)

                        if distance > 0.001:
                            overlap_distance = (r1 + r2) - distance
                            if overlap_distance > 0:
                                force_magnitude = repulsion_strength * overlap_distance / (distance ** 2)

                                forces[i, 0] -= force_magnitude * dx / distance
                                forces[i, 1] -= force_magnitude * dy / distance

            # Apply boundary and center attraction forces
            for i in range(len(circles_array)):
                x, y, r = circles_array[i]

                # Attract to center of rectangle
                center_x, center_y = width/2, height/2
                dx = center_x - x
                dy = center_y - y
                forces[i, 0] += attraction_strength * dx
                forces[i, 1] += attraction_strength * dy

                # Boundary forces with higher strength
                boundary_force = 1.0

                if x - r < 0:
                    forces[i, 0] += boundary_force * (r - x)
                if x + r > width:
                    forces[i, 0] -= boundary_force * (x + r - width)
                if y - r < 0:
                    forces[i, 1] += boundary_force * (r - y)
                if y + r > height:
                    forces[i, 1] -= boundary_force * (y + r - height)

            # Update positions
            step_size = 0.02  # Increased step size for faster convergence
            for i in range(len(circles_array)):
                circles_array[i, 0] += forces[i, 0] * step_size
                circles_array[i, 1] += forces[i, 1] * step_size

                # Maintain positive radii
                if circles_array[i, 2] < 0.0001:
                    circles_array[i, 2] = 0.0001

                # Enforce valid positions
                if not is_valid_position(circles_array[i]):
                    x, y, r = circles_array[i]
                    x = max(r, min(width - r, x))
                    y = max(r, min(height - r, y))
                    circles_array[i] = [x, y, r]

            # Early termination check with stricter condition
            if iteration % 50 == 0:
                penalty = calculate_constraint_penalty(circles_array)
                if penalty < 1e-6:  # Stricter early termination
                    break

        return circles_array

    def refine_with_evolution(circles_array):
        """Use evolutionary approach to refine circle radii with aggressive improvements"""
        best_circles = circles_array.copy()
        best_sum = np.sum(best_circles[:, 2])

        # Generate variations and test with aggressive radius improvements
        for _ in range(300):  # More iterations for better search
            test_circles = best_circles.copy()

            # Randomly select one circle to modify
            idx = random.randint(0, len(test_circles) - 1)
            x, y, r = test_circles[idx]

            # Try to increase radius significantly 
            old_r = r
            new_r = min(old_r * 1.1, 0.2)  # More aggressive increase factor
            test_circles[idx, 2] = new_r

            # Verify constraint satisfaction
            if not is_valid_position(test_circles[idx]):
                continue

            # Remove this circle for overlap testing
            temp_circles = np.delete(test_circles, idx, axis=0)
            if not check_overlap(test_circles[idx], temp_circles):
                # Test the modified configuration
                new_sum = np.sum(test_circles[:, 2])
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_circles = test_circles.copy()

        return best_circles

    def maximize_individual_radii(circles_array):
        """Try to maximize individual radii after main optimization"""
        for _ in range(200):  # More iterations for better search
            improvement_made = False
            for i in range(len(circles_array)):
                original_circle = circles_array[i].copy()
                x, y, r = original_circle

                # Try to increase radius by larger amount
                new_r = min(r * 1.1, 0.2)  # More aggressive increase
                test_circle = [x, y, new_r]

                # Check if valid and causes no overlaps
                if is_valid_position(test_circle) and not check_overlap(test_circle, np.delete(circles_array, i, axis=0)):
                    circles_array[i] = test_circle
                    improvement_made = True

            if not improvement_made:
                break

        return circles_array

    def balance_and_refine(circles_array):
        """Additional balancing step to improve overall packing efficiency"""
        # Try to shrink too-large circles and increase smaller ones
        total_radius = np.sum(circles_array[:, 2])
        avg_radius = total_radius / n_circles
        
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            
            # If radius is significantly above average, try to reduce it slightly
            if r > avg_radius * 1.3:
                # Reduce by small amount if still valid
                new_r = r * 0.95
                if new_r > 0.01:  # Minimum radius threshold
                    test_circle = [x, y, new_r]
                    if is_valid_position(test_circle) and not check_overlap(test_circle, np.delete(circles_array, i, axis=0)):
                        circles_array[i] = test_circle
                        
        return circles_array

    # Execute multi-phase optimization
    # Phase 1: Initialization
    circles = initialize_circles()

    # Phase 2: Physics-based optimization with more iterations
    circles = apply_physics_update(circles, max_iter=2000)

    # Phase 3: Evolutionary refinement with more aggressive changes
    circles = refine_with_evolution(circles)

    # Phase 4: Additional balancing
    circles = balance_and_refine(circles)

    # Phase 5: Final polishing with less iterations
    circles = apply_physics_update(circles, max_iter=300)

    # Phase 6: Individual radius maximization
    circles = maximize_individual_radii(circles)

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")