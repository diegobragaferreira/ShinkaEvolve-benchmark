# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
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
    random.seed(42)

    # Phase 1: Advanced hybrid initialization
    n_circles = 21

    # Create initial positions using hexagonal packing pattern for better space utilization
    circles = np.zeros((n_circles, 3))

    # Hexagonal grid parameters
    rows = 5
    cols = 5
    x_spacing = width / (cols + 1)
    y_spacing = height / (rows + 1)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_circles:
                break
            x = (j + 1) * x_spacing
            y = (i + 1) * y_spacing
            if i % 2 == 1:
                x += x_spacing * 0.5
            # Add slight randomization to avoid perfect patterns
            x += random.uniform(-x_spacing*0.1, x_spacing*0.1)
            y += random.uniform(-y_spacing*0.1, y_spacing*0.1)
            circles[idx] = [x, y, 0.02]  # Initial small radius
            idx += 1

    # Fill any remaining spots with corner and edge seeds
    additional_seeds = [
        [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],  # corners
        [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]   # edges
    ]

    for i in range(len(additional_seeds)):
        if idx >= n_circles:
            break
        x, y = additional_seeds[i]
        circles[idx] = [x, y, 0.02]
        idx += 1

    # Phase 2: Multi-phase hybrid optimization with genetic algorithm component
    best_sum = -1
    best_circles = None

    # Multi-start approach with different strategies
    for start_iter in range(5):  # Increase from 3 to 5 for better exploration
        # Reset circles for this iteration
        current_circles = circles.copy()

        # Phase 2a: Genetic Algorithm-inspired global search (first 300 iterations)
        if start_iter < 2:  # Use GA for first 2 starts
            # Mutation-based search with adaptive parameters
            for ga_iter in range(300):
                # Mutate a few circles at a time
                mutation_count = min(5, n_circles // 2)
                mutated_indices = random.sample(range(n_circles), mutation_count)

                for i in mutated_indices:
                    # Save original
                    orig_x, orig_y, orig_r = current_circles[i]

                    # Apply random mutation to position
                    mutation_strength = 0.05 if ga_iter < 150 else 0.02
                    new_x = orig_x + random.uniform(-mutation_strength, mutation_strength)
                    new_y = orig_y + random.uniform(-mutation_strength, mutation_strength)

                    # Keep within bounds
                    new_x = max(0.05, min(width - 0.05, new_x))
                    new_y = max(0.05, min(height - 0.05, new_y))

                    # Compute new max radius
                    new_r = min(new_x, width - new_x, new_y, height - new_y)

                    # Check all conflicts and resolve
                    valid = True
                    for j in range(n_circles):
                        if i != j:
                            px, py, pr = current_circles[j]
                            dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                            if dist < (new_r + pr):
                                valid = False
                                break

                    if valid:
                        current_circles[i] = [new_x, new_y, new_r]

        # Phase 2b: Simulated Annealing with adaptive cooling (next 400 iterations)
        # Initialize cooling parameters
        temp = 1.0
        cooling_rate = 0.995
        min_temp = 0.001

        for sa_iter in range(400):
            # Adaptive cooling schedule
            if temp < min_temp:
                temp = min_temp

            # Select random circle to modify
            i = random.randint(0, n_circles - 1)

            # Save original
            orig_x, orig_y, orig_r = current_circles[i]

            # Generate neighbor through random displacement
            dx = random.uniform(-0.05, 0.05)
            dy = random.uniform(-0.05, 0.05)

            new_x = orig_x + dx
            new_y = orig_y + dy

            # Keep within bounds
            new_x = max(0.05, min(width - 0.05, new_x))
            new_y = max(0.05, min(height - 0.05, new_y))

            # Compute new max radius
            new_r = min(new_x, width - new_x, new_y, height - new_y)

            # Check conflicts
            valid = True
            conflict_count = 0
            for j in range(n_circles):
                if i != j:
                    px, py, pr = current_circles[j]
                    dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                    if dist < (new_r + pr):
                        valid = False
                        conflict_count += 1

            # Accept or reject based on simulated annealing criteria
            old_sum = np.sum(current_circles[:, 2])
            new_sum = old_sum - orig_r + new_r

            if valid:
                # Accept if better or with probability based on temperature
                if new_sum > old_sum or random.random() < math.exp((new_sum - old_sum) / (temp + 1e-8)):
                    current_circles[i] = [new_x, new_y, new_r]
            else:
                # If invalid, try to resolve conflicts by shrinking radius
                if conflict_count > 0:
                    # Shrink the radius to resolve conflicts, but keep it valid
                    shrink_factor = 0.9
                    new_r = max(0.001, new_r * shrink_factor)
                    current_circles[i] = [new_x, new_y, new_r]

            # Cooling schedule update
            temp *= cooling_rate

        # Phase 2c: Local refinement with overlap resolution
        for refine_iter in range(300):
            improved = False

            # Randomly shuffle circle order to avoid bias
            circle_order = list(range(n_circles))
            random.shuffle(circle_order)

            for i in circle_order:
                # Compute max radius for current position
                x, y, r = current_circles[i]
                max_radius = min(x, width - x, y, height - y)

                # Check conflicts with all other circles
                for j in range(n_circles):
                    if i != j:
                        px, py, pr = current_circles[j]
                        dist = np.sqrt((x - px)**2 + (y - py)**2)
                        if dist > 0:  # Avoid division by zero
                            max_radius = min(max_radius, dist - pr)

                # Enforce valid bounds
                max_radius = max(0.001, max_radius)

                if max_radius > r:
                    current_circles[i, 2] = max_radius
                    improved = True

            # If no improvement, break early
            if not improved and refine_iter > 100:
                break

        # Validate and track best solution
        if _validate_configuration(current_circles, width, height):
            current_sum = np.sum(current_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = current_circles.copy()

    # If no valid solution was found, use fallback
    if best_circles is None:
        # Fallback: simple optimization with better parameters
        circles = np.zeros((n_circles, 3))
        rows, cols = 4, 6
        x_spacing = width / (cols + 1)
        y_spacing = height / (rows + 1)

        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n_circles:
                    break
                x = (j + 1) * x_spacing + random.uniform(-x_spacing*0.1, x_spacing*0.1)
                y = (i + 1) * y_spacing + random.uniform(-y_spacing*0.1, y_spacing*0.1)
                circles[idx] = [x, y, 0.02]  # Small initial radius
                idx += 1

        # Local optimization with more iterations
        for _ in range(500):
            improved = False
            for i in range(n_circles):
                x, y, r = circles[i]
                max_radius = min(x, width - x, y, height - y)

                # Check conflicts with others
                for j in range(n_circles):
                    if i != j:
                        px, py, pr = circles[j]
                        dist = np.sqrt((x - px)**2 + (y - py)**2)
                        if dist > 0:
                            max_radius = min(max_radius, dist - pr)

                max_radius = max(0.001, max_radius)
                if max_radius > r:
                    circles[i, 2] = max_radius
                    improved = True

            if not improved:
                break

        best_circles = circles

    # Final correction to ensure valid constraints
    for i in range(n_circles):
        x, y, r = best_circles[i]
        # Ensure circles are within bounds and radius is reasonable
        r = min(r, x, width - x, y, height - y)
        if r <= 0.001:
            r = 0.01
        best_circles[i] = [x, y, r]

    return best_circles

def _validate_configuration(circles, width, height):
    """Validate that all circles are within bounds and non-overlapping."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Check boundary conditions
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            return False

        # Check overlap with other circles
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
            # Overlap occurs when distance < sum of radii
            if distance < r + r2 - 1e-8:
                return False

    return True

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")