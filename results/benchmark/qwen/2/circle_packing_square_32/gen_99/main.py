# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def compute_local_density(circles, point, k=5):
    """Compute local density around a point using k-nearest neighbors."""
    if len(circles) < 2:
        return 0.0

    # Convert to numpy array for efficient processing
    pts = np.array(circles)[:, :2]
    tree = cKDTree(pts)

    # Query k nearest neighbors (excluding the point itself if it exists)
    distances, indices = tree.query(point, k=min(k+1, len(pts)), p=2)

    # Average distance to neighbors (excluding self if present)
    if len(distances) > 1:
        avg_distance = np.mean(distances[1:])  # Skip the first (distance to itself)
    else:
        avg_distance = distances[0]

    # Density is inversely proportional to average distance
    if avg_distance > 0:
        return 1.0 / avg_distance
    else:
        return float('inf')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Hexagonal grid initialization with adaptive radii
    # Arrange in a hexagonal pattern for better initial packing
    rows = 6
    cols = 6

    # Ensure we have exactly 32 circles
    actual_rows = rows
    actual_cols = cols
    actual_count = actual_rows * actual_cols

    if actual_count < n:
        # Try to get closer to 32 by adjusting dimensions
        while actual_count < n:
            actual_rows += 1
            actual_count = actual_rows * actual_cols
    elif actual_count > n:
        # Reduce rows if necessary
        while actual_count > n and actual_rows > 1:
            actual_rows -= 1
            actual_count = actual_rows * actual_cols

    # Create hexagonal grid
    # Calculate spacing based on desired circle density
    spacing_x = 1.0 / (actual_cols + 1)
    spacing_y = 1.0 / (actual_rows + 1)

    # Hexagonal offset for alternating rows
    hex_offset = spacing_x * 0.5

    idx = 0
    points = []  # Keep track of placed points for density calculation

    # First pass: place circles in hexagonal grid
    for i in range(actual_rows):
        for j in range(actual_cols):
            if idx >= n:
                break
            # Even rows are aligned, odd rows offset
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y

            if i % 2 == 1:
                x += hex_offset

            # Ensure we're within bounds (with margin for radii)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))

            points.append([x, y])
            idx += 1
            if idx >= n:
                break

    # Fill remaining circles if needed
    while idx < n:
        points.append([0.5, 0.5])
        idx += 1

    # Second pass: assign adaptive radii based on local density
    # We maintain the hexagonal structure but adjust radii to be density-adaptive
    for i, (x, y) in enumerate(points[:n]):
        # Estimate local density by looking at neighbors
        density = compute_local_density(points[:i] + points[i+1:], [x, y], k=5)

        # Base radius calculation - inversely proportional to density
        # Higher density means smaller circles to allow more neighbors
        base_radius = min(0.15, 0.3 * (1.0 / (1.0 + density * 0.5)))

        # Ensure the radius respects boundaries
        max_radius = min(x, y, 1-x, 1-y)
        radius = min(base_radius, max_radius * 0.8)  # Leave some margin

        # Ensure minimum positive radius
        radius = max(0.005, radius)

        circles[i, 0] = x
        circles[i, 1] = y
        circles[i, 2] = radius

    return circles


# EVOLVE-BLOCK-END