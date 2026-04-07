# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Optimized width/height ratio for circle packing
    width, height = 1.2, 0.8

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Initialize circles array
    n = 21
    circles = np.zeros((n, 3))

    # Enhanced initialization with better spatial distribution
    # Strategy 1: Corner placements
    corner_positions = [
        (width * 0.1, height * 0.1),
        (width * 0.9, height * 0.1),
        (width * 0.1, height * 0.9),
        (width * 0.9, height * 0.9),
        (width * 0.5, height * 0.5)
    ]

    # Initialize with strategic corner positions
    for i in range(min(len(corner_positions), n)):
        x, y = corner_positions[i]
        circles[i] = [x, y, 0.02]

    # Strategy 2: Hexagonal grid for remaining positions
    remaining = n - len(corner_positions)
    if remaining > 0:
        # Create hexagonal grid pattern
        rows = int(np.ceil(np.sqrt(remaining)))
        cols = int(np.ceil(remaining / rows))

        cell_width = width / (cols + 1)
        cell_height = height / (rows + 1)

        idx = len(corner_positions)
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x_offset = 0.0 if i % 2 == 0 else 0.5
                x = (j + 1 + x_offset) * cell_width
                y = (i + 1) * cell_height
                # Ensure within bounds and not too close to edges
                x = max(0.05, min(width - 0.05, x))
                y = max(0.05, min(height - 0.05, y))
                circles[idx] = [x, y, 0.01]
                idx += 1
                if idx >= n:
                    break

    # Phase 1: Aggressive exploration with large steps
    max_radius_limit = 0.4
    learning_rate = 0.2
    for iteration in range(1000):
        # Update radii first
        for i in range(n):
            max_radius = calculate_max_radius(circles, i, width, height, max_radius_limit)
            if max_radius > circles[i][2]:
                circles[i][2] = max_radius

        # Position optimization with adaptive forces
        for i in range(n):
            x, y, r = circles[i]

            # Compute repulsive forces from overlapping circles
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i != j:
                    dx = circles[j, 0] - x
                    dy = circles[j, 1] - y
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < (r + circles[j, 2]) and dist > 0.001:
                        force_magnitude = (r + circles[j, 2] - dist) / dist
                        fx -= force_magnitude * dx
                        fy -= force_magnitude * dy

            # Boundary forces
            if x < r:
                fx += (r - x) * 2.0
            if x + r > width:
                fx -= (x + r - width) * 2.0
            if y < r:
                fy += (r - y) * 2.0
            if y + r > height:
                fy -= (y + r - height) * 2.0

            # Apply movement
            new_x = max(r, min(width - r, x + learning_rate * fx))
            new_y = max(r, min(height - r, y + learning_rate * fy))

            circles[i, 0] = new_x
            circles[i, 1] = new_y

        # Decrease learning rate gradually
        learning_rate *= 0.999

    # Phase 2: Refinement with moderate steps
    learning_rate = 0.05
    for iteration in range(1000):
        # Update radii
        for i in range(n):
            max_radius = calculate_max_radius(circles, i, width, height, max_radius_limit)
            if max_radius > circles[i][2]:
                circles[i][2] = max_radius

        # Position optimization
        for i in range(n):
            x, y, r = circles[i]

            # Compute repulsive forces
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i != j:
                    dx = circles[j, 0] - x
                    dy = circles[j, 1] - y
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < (r + circles[j, 2]) and dist > 0.001:
                        force_magnitude = (r + circles[j, 2] - dist) / dist
                        fx -= force_magnitude * dx
                        fy -= force_magnitude * dy

            # Boundary forces
            if x < r:
                fx += (r - x) * 3.0
            if x + r > width:
                fx -= (x + r - width) * 3.0
            if y < r:
                fy += (r - y) * 3.0
            if y + r > height:
                fy -= (y + r - height) * 3.0

            # Apply movement
            new_x = max(r, min(width - r, x + learning_rate * fx))
            new_y = max(r, min(height - r, y + learning_rate * fy))

            circles[i, 0] = new_x
            circles[i, 1] = new_y

        # Decrease learning rate
        learning_rate *= 0.9995

    # Phase 3: Fine-tuning with small steps
    learning_rate = 0.01
    for iteration in range(1000):
        # Update radii
        for i in range(n):
            max_radius = calculate_max_radius(circles, i, width, height, max_radius_limit)
            if max_radius > circles[i][2]:
                circles[i][2] = max_radius

        # Position optimization
        for i in range(n):
            x, y, r = circles[i]

            # Compute repulsive forces
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i != j:
                    dx = circles[j, 0] - x
                    dy = circles[j, 1] - y
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < (r + circles[j, 2]) and dist > 0.001:
                        force_magnitude = (r + circles[j, 2] - dist) / dist
                        fx -= force_magnitude * dx
                        fy -= force_magnitude * dy

            # Boundary forces
            if x < r:
                fx += (r - x) * 5.0
            if x + r > width:
                fx -= (x + r - width) * 5.0
            if y < r:
                fy += (r - y) * 5.0
            if y + r > height:
                fy -= (y + r - height) * 5.0

            # Apply movement
            new_x = max(r, min(width - r, x + learning_rate * fx))
            new_y = max(r, min(height - r, y + learning_rate * fy))

            circles[i, 0] = new_x
            circles[i, 1] = new_y

        # Decrease learning rate
        learning_rate *= 0.9999

    return circles


def calculate_max_radius(circles, index, width, height, max_radius_limit=0.4):
    """Calculate maximum radius for circle at given index without overlapping others."""
    x, y, current_radius = circles[index]

    # Maximum radius based on container boundaries
    max_radius_bound = min(x, y, width - x, height - y)

    # Maximum radius based on other circles
    max_radius_overlap = float('inf')

    for i, (cx, cy, cr) in enumerate(circles):
        if i != index:
            # Distance to other circle center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Max radius that avoids overlap
            max_radius_for_this_circle = dist - cr

            if max_radius_for_this_circle < max_radius_overlap:
                max_radius_overlap = max_radius_for_this_circle

    max_radius = min(max_radius_bound, max_radius_overlap, max_radius_limit)
    return max(max_radius, 0.001)  # Ensure minimum radius


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")