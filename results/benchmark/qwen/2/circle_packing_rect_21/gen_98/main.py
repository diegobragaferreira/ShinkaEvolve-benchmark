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
    # Rectangle dimensions: width + height = 2, setting to 1x1 for simplicity (perimeter = 4)
    width, height = 1.0, 1.0

    # Initialize parameters
    n_circles = 21
    random.seed(42)

    def initialize_circles():
        """Initialize circles with adaptive grid placement based on optimal packing density"""
        circles = np.zeros((n_circles, 3))

        # More sophisticated grid sizing based on packing efficiency
        # For 21 circles, we want to use approximately the square root of n, but adjust for aspect ratio
        aspect_ratio = width / height
        grid_cols = max(1, int(math.ceil(math.sqrt(n_circles * aspect_ratio))))
        grid_rows = max(1, int(math.ceil(n_circles / grid_cols)))

        # Ensure we don't exceed the number of circles needed
        actual_n = grid_cols * grid_rows
        if actual_n > n_circles:
            # Reduce grid size if too many circles
            while grid_cols * grid_rows > n_circles:
                if grid_cols > grid_rows:
                    grid_cols = max(1, grid_cols - 1)
                else:
                    grid_rows = max(1, grid_rows - 1)
                if grid_cols * grid_rows == 0:
                    break
            if grid_cols * grid_rows == 0:
                grid_cols = 1
                grid_rows = n_circles

        # Calculate spacing that will allow for good packing
        spacing_x = width / (grid_cols + 1) * 0.8  # Slight padding
        spacing_y = height / (grid_rows + 1) * 0.8  # Slight padding

        # Create hexagonal-like grid with more intelligent spacing
        idx = 0
        for i in range(grid_cols):
            for j in range(grid_rows):
                if idx >= n_circles:
                    break
                # Offset every other row for better packing (hexagonal close packing)
                offset = (j % 2) * spacing_x * 0.5
                # Add more substantial random jitter for better diversity
                jitter_x = random.uniform(-spacing_x*0.15, spacing_x*0.15)
                jitter_y = random.uniform(-spacing_y*0.15, spacing_y*0.15)
                x = (i + 1) * spacing_x + offset + jitter_x
                y = (j + 1) * spacing_y + jitter_y
                # Use a more conservative initial radius
                r = min(spacing_x, spacing_y) * 0.12
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n_circles:
                break

        # Adjust initial radii based on position relative to center and boundaries
        for i in range(n_circles):
            x, y, r = circles[i]
            # Distance to nearest boundary
            min_dist = min(x, y, width - x, height - y)
            # Position-based adjustment - center positions get more room
            center_x, center_y = width/2, height/2
            dist_to_center = math.sqrt((x-center_x)**2 + (y-center_y)**2)
            max_dist_to_center = math.sqrt((width/2)**2 + (height/2)**2)
            if max_dist_to_center > 0:
                # Center positions get more room (higher adjustment factor)
                adjustment_factor = 0.6 + 0.4 * (1.0 - dist_to_center/max_dist_to_center)
            else:
                adjustment_factor = 0.6
            circles[i][2] = min(r, min_dist * adjustment_factor * 0.9)

        return circles

    def is_valid_position(circle):
        """Check if circle position is valid (within bounds)"""
        x, y, r = circle
        return (r <= x <= width - r and
                r <= y <= height - r)

    def calculate_constraint_penalty(circles_array):
        """Calculate total penalty for boundary and overlap violations"""
        penalty = 0.0

        # Check boundary violations
        for circle in circles_array:
            x, y, r = circle
            boundary_dist = min(x, y, width - x, height - y)
            if boundary_dist < r:
                penalty += (r - boundary_dist) ** 2

        # Check overlap violations
        for i in range(len(circles_array)):
            x1, y1, r1 = circles_array[i]
            for j in range(i):
                x2, y2, r2 = circles_array[j]
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)
                overlap = max(0, (r1 + r2) - distance)
                if overlap > 0:
                    penalty += overlap ** 2

        return penalty

    def get_spatial_index(circles_array):
        """Create spatial index for fast neighbor lookups"""
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

    def apply_physics_update(circles_array, max_iter=800):
        """Apply physics-based optimization to improve packing"""
        tree = get_spatial_index(circles_array)
        repulsion_strength = 2.0
        attraction_strength = 0.1

        for iteration in range(max_iter):
            forces = np.zeros_like(circles_array)

            # Calculate repulsion forces
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

                # Boundary forces
                boundary_force = 0.5

                if x - r < 0:
                    forces[i, 0] += boundary_force * (r - x)
                if x + r > width:
                    forces[i, 0] -= boundary_force * (x + r - width)
                if y - r < 0:
                    forces[i, 1] += boundary_force * (r - y)
                if y + r > height:
                    forces[i, 1] -= boundary_force * (y + r - height)

            # Update positions
            step_size = 0.01
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

            # Early termination check
            if iteration % 100 == 0:
                penalty = calculate_constraint_penalty(circles_array)
                if penalty < 1e-5:
                    break

        return circles_array

    def refine_with_evolution(circles_array):
        """Use evolutionary approach to refine circle radii"""
        best_circles = circles_array.copy()
        best_sum = np.sum(best_circles[:, 2])

        # Generate variations and test
        for _ in range(200):
            test_circles = best_circles.copy()

            # Randomly select one circle to modify
            idx = random.randint(0, len(test_circles) - 1)
            x, y, r = test_circles[idx]

            # Try to slightly increase radius
            old_r = r
            new_r = min(old_r * 1.05, 0.2)
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
        for _ in range(100):
            improvement_made = False
            for i in range(len(circles_array)):
                original_circle = circles_array[i].copy()
                x, y, r = original_circle

                # Try to increase radius by small amount
                new_r = min(r * 1.02, 0.2)
                test_circle = [x, y, new_r]

                # Check if valid and causes no overlaps
                if is_valid_position(test_circle) and not check_overlap(test_circle, np.delete(circles_array, i, axis=0)):
                    circles_array[i] = test_circle
                    improvement_made = True

            if not improvement_made:
                break

        return circles_array

    # Execute multi-phase optimization
    # Phase 1: Initialization
    circles = initialize_circles()

    # Phase 2: Physics-based optimization
    circles = apply_physics_update(circles, max_iter=1500)

    # Phase 3: Evolutionary refinement
    circles = refine_with_evolution(circles)

    # Phase 4: Final polishing
    circles = apply_physics_update(circles, max_iter=500)

    # Phase 5: Individual radius maximization
    circles = maximize_individual_radii(circles)

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")