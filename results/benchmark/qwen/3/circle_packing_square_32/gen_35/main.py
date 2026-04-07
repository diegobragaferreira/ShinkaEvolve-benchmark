# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    # Set seed for reproducibility
    np.random.seed(42)

    # Hexagonal grid initialization
    circles = initialize_hexagonal_grid(n)

    # Optimize radii iteratively
    circles = optimize_radii(circles)

    return circles

def initialize_hexagonal_grid(n):
    """Initialize circles in a hexagonal grid pattern"""
    # Create hexagonal grid
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    # Adjust grid size to accommodate exactly n circles
    if rows * cols < n:
        cols += 1

    # Calculate spacing for hexagonal packing
    spacing_x = 0.95  # Leave some margin
    spacing_y = spacing_x * np.sqrt(3) / 2

    # Calculate grid dimensions
    grid_width = cols * spacing_x
    grid_height = rows * spacing_y

    # Center the grid
    offset_x = (1 - grid_width) / 2
    offset_y = (1 - grid_height) / 2

    circles = np.zeros((n, 3))

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break

            # Offset every other row for hexagonal packing
            x_offset = spacing_x * j
            if i % 2 == 1:
                x_offset += spacing_x / 2

            x = offset_x + x_offset
            y = offset_y + spacing_y * i

            # Ensure positions are within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Initial radius based on spacing (smaller than spacing to allow for optimization)
                circles[idx] = [x, y, min(spacing_x, spacing_y) / 4]
                idx += 1
        if idx >= n:
            break

    # Fill remaining positions with zeros or adjust if needed
    if idx < n:
        # Fill with zeros or reuse last valid circle (this shouldn't happen in normal execution)
        for i in range(idx, n):
            circles[i] = circles[idx-1] if idx > 0 else [0.5, 0.5, 0.05]

    return circles

def check_constraints(circle_data):
    """Check if all circles satisfy containment and overlap constraints"""
    n = len(circle_data)

    # Check containment constraints
    for i in range(n):
        x, y, r = circle_data[i]
        if not (r <= x <= 1 - r and r <= y <= 1 - r):
            return False

    # Check overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circle_data[i]
            x2, y2, r2 = circle_data[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False

    return True

def calculate_sum_radii(circle_data):
    """Calculate the sum of all radii"""
    return np.sum(circle_data[:, 2])

def expand_radii(circle_data, max_iterations=100):
    """Increase radii while maintaining constraints"""
    n = len(circle_data)
    circles = circle_data.copy()

    # Create a distance matrix for efficient overlap checking
    def get_distances():
        centers = circles[:, :2]
        distances = cdist(centers, centers)
        return distances

    for iteration in range(max_iterations):
        improved = False

        # Try to increase radii one by one
        for i in range(n):
            original_radius = circles[i, 2]
            x, y, r = circles[i]

            # Find minimum allowed radius
            min_radius = max(r, 0.0001)  # Minimum reasonable radius

            # Calculate maximum possible radius
            max_radius = min(x, 1 - x, y, 1 - y)

            # Get distances to other circles
            distances = np.sqrt(np.sum((circles[:, :2] - np.array([x, y]))**2, axis=1))

            # Find the minimum distance to other circles
            min_distance_to_others = np.min(distances[distances > 0])

            # Maximum radius allowed by overlap constraint
            max_radius_overlap = min_distance_to_others - 0.0001

            # Take the most restrictive constraint
            max_possible_radius = min(max_radius, max_radius_overlap)

            if max_possible_radius > r:
                # Increase radius up to limit
                new_radius = min(max_possible_radius, 1.0)
                circles[i, 2] = new_radius
                improved = True

        if not improved:
            break

    return circles

def optimize_radii(circle_data):
    """Main optimization routine"""
    circles = circle_data.copy()

    # Initialize with some optimization steps
    for _ in range(50):  # Multiple rounds of improvement
        # Expand radii to maximum feasible values
        circles = expand_radii(circles, 50)

        # Check if we've reached a good configuration
        if check_constraints(circles):
            break

    # Final constraint validation
    if not check_constraints(circles):
        # Try a more conservative approach if constraints violated
        circles = circles.copy()
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Ensure containment
            r = min(r, x, 1 - x, y, 1 - y)
            # Reduce radius slightly to avoid numerical issues
            r = max(r * 0.99, 0.001)
            circles[i] = [x, y, r]

    return circles


# EVOLVE-BLOCK-END