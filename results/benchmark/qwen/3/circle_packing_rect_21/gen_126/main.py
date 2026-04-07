# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from collections import defaultdict
import math

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

    # Phase 1: Multi-resolution grid initialization with enhanced strategy
    n_circles = 21

    # Use a hybrid approach: strategic corner placement + hexagonal grid + random refinement
    circles = np.zeros((n_circles, 3))

    # Strategic corner and edge placement (more comprehensive approach)
    strategic_positions = [
        # Corners
        (0.05, 0.05), (width - 0.05, 0.05), (0.05, height - 0.05), (width - 0.05, height - 0.05),
        # Edges (middle points)
        (width/2, 0.05), (width/2, height - 0.05), (0.05, height/2), (width - 0.05, height/2),
        # Center and diagonals
        (width/2, height/2),
        (width*0.25, height*0.25), (width*0.75, height*0.75),
        (width*0.25, height*0.75), (width*0.75, height*0.25)
    ]

    # Fill with strategic positions first
    for i in range(min(len(strategic_positions), n_circles)):
        circles[i] = [strategic_positions[i][0], strategic_positions[i][1], 0.03]

    # Fill remaining with hexagonal grid pattern
    if n_circles > len(strategic_positions):
        remaining_slots = n_circles - len(strategic_positions)
        rows = int(np.ceil(np.sqrt(remaining_slots)))
        cols = int(np.ceil(remaining_slots / rows))

        # Generate hexagonal grid
        x_spacing = width / (cols + 1)
        y_spacing = height / (rows + 1)

        idx = len(strategic_positions)
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * x_spacing
                y = (i + 1) * y_spacing
                # Offset odd rows for hexagonal pattern
                if i % 2 == 1:
                    x += x_spacing * 0.5
                # Make sure placement is within bounds
                x = max(0.05, min(width - 0.05, x))
                y = max(0.05, min(height - 0.05, y))
                circles[idx] = [x, y, 0.02]  # Smaller initial radius
                idx += 1
                if idx >= n_circles:
                    break

    # Phase 2: Multi-scale optimization with adaptive learning rates
    best_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Multi-scale optimization with different resolution levels and learning rates
    scales = [1.0, 0.5, 0.25, 0.1]  # Resolution levels from coarse to fine
    max_iterations_per_scale = [500, 300, 200, 100]  # Iterations for each scale

    for scale_idx, (scale, max_iter) in enumerate(zip(scales, max_iterations_per_scale)):
        # Adaptive learning rate that decreases with scale
        initial_lr = 0.1 * scale
        min_lr = 0.001 * scale

        for iteration in range(max_iter):
            current_circles = circles.copy()
            old_sum = np.sum(current_circles[:, 2])

            # Adaptive learning rate based on iteration count
            # Decrease learning rate gradually for better convergence at finer scales
            current_lr = max(min_lr, initial_lr * (0.99 ** iteration))

            # Sort circles by radius (larger ones first) to prioritize improvements
            sorted_indices = np.argsort(current_circles[:, 2])[::-1]

            improved = False

            # Process circles in order of decreasing radius
            for i in sorted_indices:
                orig_x, orig_y, orig_r = current_circles[i]

                # Compute max possible radius at current position with boundary constraints
                max_possible_r = min(orig_x, width - orig_x, orig_y, height - orig_y)

                if max_possible_r <= orig_r:
                    continue  # Nothing to improve

                # Smart neighborhood search with variable step sizes
                best_move_x, best_move_y, best_move_r = 0, 0, orig_r
                best_new_sum = old_sum

                # Step sizes that adapt to current scale
                step_sizes = [max(0.01, 0.1 * scale), max(0.005, 0.05 * scale), max(0.001, 0.01 * scale)]

                # Try different search strategies based on current iteration
                if iteration < max_iter * 0.3:  # Early iterations: broader search
                    search_steps = []
                    for step in step_sizes:
                        # Try more exploratory directions
                        for dx in [-step*2, -step, 0, step, step*2]:
                            for dy in [-step*2, -step, 0, step, step*2]:
                                if abs(dx) + abs(dy) > 0:  # Skip center
                                    search_steps.append((dx, dy))
                else:  # Later iterations: focused search
                    search_steps = []
                    for step in step_sizes:
                        # Grid search around current position
                        for dx in [-step, 0, step]:
                            for dy in [-step, 0, step]:
                                if abs(dx) + abs(dy) > 0:
                                    search_steps.append((dx, dy))

                # Evaluate each possible move
                for dx, dy in search_steps:
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
                    for j in range(n_circles):
                        if i != j:
                            px, py, pr = current_circles[j]
                            dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                            if dist < (new_r + pr):
                                valid = False
                                break

                    if valid:
                        # Calculate new sum
                        new_sum = old_sum - orig_r + new_r

                        # Apply adaptive acceptance based on learning rate and improvement
                        if new_sum > best_new_sum:
                            best_new_sum = new_sum
                            best_move_x, best_move_y, best_move_r = dx, dy, new_r

                # Apply the best move if it improves the sum significantly
                if best_new_sum > old_sum + 1e-8:
                    current_circles[i] = [orig_x + best_move_x, orig_y + best_move_y, best_move_r]
                    circles = current_circles.copy()
                    improved = True

            # Update best solution
            new_sum = np.sum(circles[:, 2])
            if new_sum > best_sum:
                best_sum = new_sum
                best_circles = circles.copy()

            # Early stopping if no significant improvement
            if not improved and iteration > max_iter * 0.5:
                break

    # Phase 3: Final refinement with adaptive strategies
    # Focus more on small circles that can potentially gain more
    max_final_iter = 300
    no_improvement_count = 0
    max_no_improvement = 50

    for iteration in range(max_final_iter):
        current_circles = circles.copy()
        old_sum = np.sum(current_circles[:, 2])

        # Strategy: prioritize smaller circles for more aggressive refinement
        # Sort by radius ascending (smaller first) for this phase
        sorted_indices = np.argsort(current_circles[:, 2])

        # Sample subset of circles to work on each iteration
        sample_size = min(8, len(sorted_indices))
        indices_to_sample = np.random.choice(sorted_indices, size=sample_size, replace=False)

        improved = False

        for idx in indices_to_sample:
            old_x, old_y, old_r = current_circles[idx]

            # Smaller step sizes for fine tuning
            step_size = 0.02 if iteration < max_final_iter * 0.6 else 0.005

            # Try more focused neighborhood search
            search_radius = step_size * 4  # Larger search for early iterations
            if iteration > max_final_iter * 0.6:
                search_radius = step_size * 2  # Smaller search for later

            # Generate search points more intelligently
            candidate_positions = []

            # Grid search around current position
            steps = np.linspace(-search_radius, search_radius, 7)
            for dx in steps:
                for dy in steps:
                    candidate_positions.append((old_x + dx, old_y + dy))

            # Add boundary points for boundary exploitation
            boundary_points = [
                (0.05, old_y), (width - 0.05, old_y),  # Left/right boundaries
                (old_x, 0.05), (old_x, height - 0.05)   # Top/bottom boundaries
            ]
            candidate_positions.extend(boundary_points)

            # Filter and evaluate each candidate
            best_x, best_y = old_x, old_y
            best_r = old_r
            best_sum = old_sum

            for new_x, new_y in candidate_positions:
                # Ensure bounds
                new_x = max(0.05, min(width - 0.05, new_x))
                new_y = max(0.05, min(height - 0.05, new_y))

                # Compute max radius at new location
                new_r = min(new_x, width - new_x, new_y, height - new_y)

                # Check for conflicts
                valid = True
                for j in range(n_circles):
                    if j != idx:
                        px, py, pr = current_circles[j]
                        dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                        if dist < (new_r + pr):
                            valid = False
                            break

                if valid:
                    # Calculate new sum
                    new_sum = old_sum - old_r + new_r
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_x, best_y, best_r = new_x, new_y, new_r

            # Apply improvement if found
            if best_sum > old_sum:
                current_circles[idx] = [best_x, best_y, best_r]
                improved = True

        # Update circles and check for better solution
        new_sum = np.sum(current_circles[:, 2])
        if new_sum > best_sum:
            best_sum = new_sum
            best_circles = current_circles.copy()
            no_improvement_count = 0  # Reset counter
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= max_no_improvement:
            break

        circles = current_circles.copy()

    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")