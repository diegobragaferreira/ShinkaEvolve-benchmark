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
    width, height = 1.0, 1.0  # Optimal square configuration

    # Set seed for reproducibility
    np.random.seed(42)

    # Phase 1: Multi-resolution grid initialization
    n_circles = 21

    # Start with a coarse grid and refine
    coarse_grid_size = 4
    fine_grid_size = 8

    # Generate coarse grid points
    x_coarse = np.linspace(0.05, width - 0.05, coarse_grid_size)
    y_coarse = np.linspace(0.05, height - 0.05, coarse_grid_size)
    coarse_points = [(x, y) for x in x_coarse for y in y_coarse]

    # Initialize circles array
    circles = np.zeros((n_circles, 3))

    # Place circles using a greedy approach on coarse grid
    placed_points = []
    placed_radii = []

    # Select initial points greedily based on remaining space
    remaining_points = coarse_points.copy()
    np.random.shuffle(remaining_points)

    for i, (cx, cy) in enumerate(remaining_points[:n_circles]):
        # Compute max radius for this point considering boundaries
        max_radius = min(cx, width - cx, cy, height - cy)

        # Check for conflicts with already placed circles
        valid_placement = True
        for j in range(len(placed_points)):
            px, py = placed_points[j]
            r = placed_radii[j]
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < (r + max_radius):  # Conflict with existing circle
                valid_placement = False
                break

        if valid_placement:
            placed_points.append((cx, cy))
            placed_radii.append(max_radius)

        if len(placed_points) >= n_circles:
            break

    # Fill remaining circles if needed
    while len(placed_points) < n_circles:
        # Place remaining circles randomly but still respecting boundaries
        cx = np.random.uniform(0.05, width - 0.05)
        cy = np.random.uniform(0.05, height - 0.05)
        max_radius = min(cx, width - cx, cy, height - cy)

        # Check for conflicts
        valid_placement = True
        for j in range(len(placed_points)):
            px, py = placed_points[j]
            r = placed_radii[j]
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < (r + max_radius):
                valid_placement = False
                break

        if valid_placement:
            placed_points.append((cx, cy))
            placed_radii.append(max_radius)

    # Initialize circles array
    for i in range(n_circles):
        circles[i] = [placed_points[i][0], placed_points[i][1], placed_radii[i]]

    # Phase 2: Enhanced local refinement with gradient-guided search
    best_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Local search parameters
    max_iterations = 500
    tolerance = 1e-6
    step_size = 0.05

    for iteration in range(max_iterations):
        # Copy current configuration
        current_circles = circles.copy()
        old_sum = np.sum(current_circles[:, 2])

        # For each circle, try to improve its position using gradient approximation
        improvement_found = False

        for i in range(n_circles):
            # Store original position and radius
            orig_x, orig_y, orig_r = current_circles[i]

            # Compute current max radius at original position
            max_radius_current = min(orig_x, width - orig_x, orig_y, height - orig_y)

            # Check for conflicts to see how much we could possibly grow
            conflict_radius = max_radius_current
            for j in range(n_circles):
                if i != j:
                    px, py, pr = current_circles[j]
                    dist = np.sqrt((orig_x - px)**2 + (orig_y - py)**2)
                    if dist < (max_radius_current + pr):
                        # Calculate the maximum possible radius without overlapping
                        conflict_radius = min(conflict_radius, dist - pr)

            # If we couldn't even get the original radius, something is wrong
            if conflict_radius <= 0:
                continue

            # Estimate gradient directions for increasing radius
            grad_x, grad_y = 0.0, 0.0
            epsilon = 0.001

            # Compute numerical gradient of radius function
            test_points = [
                (orig_x + epsilon, orig_y),
                (orig_x - epsilon, orig_y),
                (orig_x, orig_y + epsilon),
                (orig_x, orig_y - epsilon)
            ]

            # Evaluate radius at each test point and compute approximate gradient
            test_radii = []
            for tx, ty in test_points:
                # Keep within bounds
                tx = max(0.05, min(width - 0.05, tx))
                ty = max(0.05, min(height - 0.05, ty))

                # Compute max radius at test point
                test_r = min(tx, width - tx, ty, height - ty)

                # Check conflicts with other circles
                for j in range(n_circles):
                    if i != j:
                        px, py, pr = current_circles[j]
                        dist = np.sqrt((tx - px)**2 + (ty - py)**2)
                        test_r = min(test_r, dist - pr)

                test_radii.append(max(test_r, 0.001))

            # Compute approximate gradient
            if len(test_radii) >= 4:
                grad_x = (test_radii[0] - test_radii[1]) / (2 * epsilon)
                grad_y = (test_radii[2] - test_radii[3]) / (2 * epsilon)

                # Normalize gradient vector
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                if grad_magnitude > 0:
                    grad_x /= grad_magnitude
                    grad_y /= grad_magnitude

                # Move along gradient direction
                new_x = orig_x + step_size * grad_x
                new_y = orig_y + step_size * grad_y

                # Keep within bounds
                new_x = max(0.05, min(width - 0.05, new_x))
                new_y = max(0.05, min(height - 0.05, new_y))

                # Compute max radius at new location
                new_r = min(new_x, width - new_x, new_y, height - new_y)

                # Check for conflicts with all other circles
                valid = True
                for j in range(n_circles):
                    if i != j:
                        px, py, pr = current_circles[j]
                        dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                        if dist < (new_r + pr):
                            valid = False
                            break

                if valid:
                    # Calculate change in sum
                    delta_sum = new_r - orig_r

                    # Prefer moves that increase radius significantly
                    if delta_sum > 0:
                        # Apply the move
                        current_circles[i] = [new_x, new_y, new_r]
                        improvement_found = True

        # Check for improvement
        new_sum = np.sum(current_circles[:, 2])
        if new_sum > best_sum:
            best_sum = new_sum
            best_circles = current_circles.copy()
            improvement_found = True
        elif not improvement_found and abs(new_sum - best_sum) < tolerance:
            break

        circles = current_circles.copy()

    # Final optimization using more sophisticated approach
    # Generate candidate positions using fine grid
    fine_x = np.linspace(0.05, width - 0.05, fine_grid_size)
    fine_y = np.linspace(0.05, height - 0.05, fine_grid_size)

    # Perform one final round of optimized placement
    for _ in range(50):
        # Randomly sample some circles to try to improve
        indices = np.random.choice(n_circles, size=min(10, n_circles), replace=False)

        for idx in indices:
            old_x, old_y, old_r = circles[idx]

            # Find best new position from nearby grid points
            best_x, best_y = old_x, old_y
            best_r = old_r
            best_sum = np.sum(circles[:, 2])

            # Sample around current position
            search_range = 0.1
            candidates = []

            # Sample nearby positions using fine grid
            for fx in fine_x:
                for fy in fine_y:
                    if abs(fx - old_x) < search_range and abs(fy - old_y) < search_range:
                        candidates.append((fx, fy))

            # Also consider boundary constraints
            candidates.extend([(0.05, fy) for fy in fine_y[1:-1]])
            candidates.extend([(width - 0.05, fy) for fy in fine_y[1:-1]])
            candidates.extend([(fx, 0.05) for fx in fine_x[1:-1]])
            candidates.extend([(fx, height - 0.05) for fx in fine_x[1:-1]])

            for new_x, new_y in candidates:
                # Ensure it's within bounds
                new_x = max(0.05, min(width - 0.05, new_x))
                new_y = max(0.05, min(height - 0.05, new_y))

                # Compute max radius
                new_r = min(new_x, width - new_x, new_y, height - new_y)

                # Check for conflicts
                valid = True
                for j in range(n_circles):
                    if j != idx:
                        px, py, pr = circles[j]
                        dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                        if dist < (new_r + pr):
                            valid = False
                            break

                if valid:
                    # See if this is better
                    new_sum = np.sum(circles[:, 2]) - old_r + new_r
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_x, best_y, best_r = new_x, new_y, new_r

            # Apply improvement if found
            if best_sum > np.sum(circles[:, 2]):
                circles[idx] = [best_x, best_y, best_r]

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")