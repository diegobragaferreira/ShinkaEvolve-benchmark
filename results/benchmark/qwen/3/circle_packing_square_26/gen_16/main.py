# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def voronoi_initialization(n_circles: int, max_attempts: int = 100) -> np.ndarray:
    """Initialize circle positions using Voronoi tessellation for better spatial distribution."""
    # Generate random points for Voronoi
    np.random.seed(42)  # Fixed seed for reproducibility
    points = np.random.rand(max_attempts * 10, 2)

    # Filter points that are far enough from boundaries
    valid_points = []
    for point in points:
        if 0.1 <= point[0] <= 0.9 and 0.1 <= point[1] <= 0.9:
            valid_points.append(point)

    # Use first n_circles valid points
    valid_points = np.array(valid_points[:n_circles])

    # If we don't have enough points, fall back to random initialization
    if len(valid_points) < n_circles:
        return np.random.rand(n_circles, 2) * 0.8 + 0.1

    # Create Voronoi diagram
    try:
        vor = Voronoi(valid_points)
        centroids = vor.points

        # Ensure all centroids are within bounds
        centroids[:, 0] = np.clip(centroids[:, 0], 0.05, 0.95)
        centroids[:, 1] = np.clip(centroids[:, 1], 0.05, 0.95)

        return centroids
    except:
        # Fallback to random initialization if Voronoi fails
        return np.random.rand(n_circles, 2) * 0.8 + 0.1

def validate_and_adjust_circles(circles: np.ndarray) -> np.ndarray:
    """Validate circle placements and adjust radii to meet containment constraints."""
    adjusted = circles.copy()

    # Adjust radii to ensure containment
    for i in range(len(adjusted)):
        x, y, r = adjusted[i]
        # Ensure radius doesn't exceed boundary constraints
        max_radius = min(x, y, 1-x, 1-y)
        adjusted[i, 2] = min(r, max_radius)

    return adjusted

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = np.zeros((n, 3))

    # Initialize using Voronoi-based approach
    init_positions = voronoi_initialization(n)

    # Set initial positions and estimate radii
    for i in range(n):
        circles[i, 0] = init_positions[i, 0]
        circles[i, 1] = init_positions[i, 1]
        # Initial radius guess based on distance to nearest neighbors
        distances = cdist([init_positions[i]], init_positions)[0]
        distances = distances[distances > 0]  # Exclude self-distance
        if len(distances) > 0:
            # Set radius to half the minimum distance to neighbors minus some buffer
            circles[i, 2] = min(0.1, 0.5 * np.min(distances) - 0.01)
        else:
            circles[i, 2] = 0.05

    # Ensure all circles are within bounds and adjust radii
    circles = validate_and_adjust_circles(circles)

    return circles


# EVOLVE-BLOCK-END