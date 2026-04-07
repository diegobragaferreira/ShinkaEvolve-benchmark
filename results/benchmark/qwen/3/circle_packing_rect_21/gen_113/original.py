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

    # Phase 2: Multi-scale local refinement with boundary awareness
    best_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Local search parameters with multi-scale approach
    max_iterations = 500
    tolerance = 1e-6

    # Multi-scale perturbation ranges
    scale_ranges = [0.1, 0.05, 0.02]  # Large, medium, small steps
    scale_weights = [0.5, 0.3, 0.2]   # Probabilities of using each scale

    for iteration in range(max_iterations):
        # Copy current configuration
        current_circles = circles.copy()
        old_sum = np.sum(current_circles[:, 2])

        # Determine scale for this iteration (multi-scale approach)
        current_scale = np.random.choice(len(scale_ranges), p=scale_weights)
        move_range = scale_ranges[current_scale]

        # For each circle, try moves with current scale
        for i in range(n_circles):
            # Store original position and radius
            orig_x, orig_y, orig_r = current_circles[i]

            # Try perturbations with current scale
            best_move_x, best_move_y, best_move_r = 0, 0, 0
            best_new_sum = old_sum

            # Try various movements in x and y directions with current scale
            for dx in [-move_range, -move_range/2, 0, move_range/2, move_range]:
                for dy in [-move_range, -move_range/2, 0, move_range/2, move_range]:
                    # New position
                    new_x = orig_x + dx
                    new_y = orig_y + dy

                    # Keep within bounds
                    new_x = max(0.05, min(width - 0.05, new_x))
                    new_y = max(0.05, min(height - 0.05, new_y))

                    # Compute max radius at new location
                    new_r = min(new_x, width - new_x, new_y, height - new_y)

                    # Check for conflicts with all other circles
                    valid = True
                    overlap_penalty = 0.0
                    for j in range(n_circles):
                        if i != j:
                            px, py, pr = current_circles[j]
                            dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                            if dist < (new_r + pr):
                                valid = False
                                # Calculate overlap penalty (quadratic penalty for overlap depth)
                                overlap_depth = (new_r + pr) - dist
                                overlap_penalty += overlap_depth ** 2

                    if valid:
                        # Multi-objective evaluation: balance radius gain with positional stability
                        # Add penalty for overlap violation (which should not happen in valid moves)
                        new_sum = old_sum - orig_r + new_r

                        # Weighted evaluation considering both radius gain and stability
                        stability_weight = 0.01  # Small weight for stability consideration
                        # In case of overlap, we penalize heavily
                        if overlap_penalty > 0:
                            new_sum -= overlap_penalty * 1000  # Heavy penalty for constraint violations
                        else:
                            # Apply slight penalty for positional changes to maintain stability
                            pos_change = np.sqrt(dx**2 + dy**2)
                            stability_penalty = pos_change * stability_weight
                            new_sum -= stability_penalty

                        if new_sum > best_new_sum:
                            best_new_sum = new_sum
                            best_move_x, best_move_y, best_move_r = dx, dy, new_r

            # Apply the best move if it improves the sum
            if best_new_sum > old_sum:
                current_circles[i] = [orig_x + best_move_x, orig_y + best_move_y, best_move_r]
                circles = current_circles.copy()

        # Check for improvement
        new_sum = np.sum(circles[:, 2])
        if new_sum > best_sum:
            best_sum = new_sum
            best_circles = circles.copy()
        elif abs(new_sum - best_sum) < tolerance:
            break

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