# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def is_valid_placement(circles, x, y, r):
    """Check if placing a circle at (x,y) with radius r is valid."""
    # Check boundary constraints
    if r > x or r > y or r > (1-x) or r > (1-y):
        return False

    # Check overlap with existing circles
    for i in range(len(circles)):
        cx, cy, cr = circles[i]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        if distance < (r + cr):
            return False

    return True

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

def initialize_circles_heuristic(n=32):
    """Initialize circle positions using a density-adaptive heuristic approach."""
    circles = []

    # Start with a coarse grid and then refine
    # Try to place circles in a hexagonal-like pattern
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)

    # Create initial placements with decreasing radii based on density
    for i in range(grid_size):
        for j in range(grid_size):
            if len(circles) >= n:
                break
            x = (i + 1) * spacing
            y = (j + 1) * spacing

            # Initial radius estimate based on available space
            r_min = min(x, y, 1-x, 1-y)
            r = min(r_min * 0.3, 0.15)

            # Compute local density at this potential location
            density = compute_local_density(circles, [x, y], k=5)

            # Scale radius inversely with density (higher density = smaller radius)
            radius_adjustment = 1.0 / (1.0 + 0.5 * density)
            r = min(r * radius_adjustment, r * 0.8)

            # Only add if valid
            if is_valid_placement(circles, x, y, r):
                circles.append([x, y, r])

    # Fill remaining spots with density-aware approach
    while len(circles) < n:
        best_r = 0
        best_x, best_y = 0, 0

        # Sample potential positions
        for _ in range(1000):  # Sample many points
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)

            # Estimate max radius at this location
            r_max = min(x, y, 1-x, 1-y)
            if r_max <= 0:
                continue

            # Compute local density at this point for adaptive sizing
            density = compute_local_density(circles, [x, y], k=5)
            radius_adjustment = 1.0 / (1.0 + 0.3 * density)
            r_adjusted = min(r_max * 0.4 * radius_adjustment, r_max * 0.4)

            # Try different radii
            test_radii = np.linspace(0.01, r_adjusted, 10)
            for r in test_radii:
                if is_valid_placement(circles, x, y, r):
                    if r > best_r:
                        best_r = r
                        best_x, best_y = x, y
                        break

        if best_r > 0:
            circles.append([best_x, best_y, best_r])

    return np.array(circles)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = initialize_circles_heuristic(n)

    return circles


# EVOLVE-BLOCK-END