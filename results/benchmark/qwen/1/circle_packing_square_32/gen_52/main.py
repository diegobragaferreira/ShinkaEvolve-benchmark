# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _generate_voronoi_points(n_points, seed=42):
    """Generate initial points using Voronoi-based sampling"""
    np.random.seed(seed)
    # Generate points slightly inside the unit square to avoid boundary issues
    points = np.random.rand(n_points, 2) * 0.8 + 0.1
    return points

def _compute_max_radius(x, y, existing_circles, min_radius=0.001):
    """Compute maximum possible radius for a circle at (x,y) given existing circles"""
    max_r = min(x, 1-x, y, 1-y)  # Boundary constraints

    if len(existing_circles) == 0:
        return max_r

    # Check overlap constraints with existing circles
    for ex_x, ex_y, ex_r in existing_circles:
        dist = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
        max_r = min(max_r, dist - ex_r)

    return max(max_r, min_radius)

def _initialize_with_voronoi(n_circles=32, seed=42):
    """Initialize circles using Voronoi-based approach"""
    # Generate Voronoi points
    voronoi_points = _generate_voronoi_points(n_circles, seed=seed)

    # Create Voronoi diagram
    vor = Voronoi(voronoi_points)

    # Get Voronoi vertices as potential circle centers
    vertices = vor.vertices

    # Filter vertices to be within unit square
    valid_vertices = vertices[
        (vertices[:, 0] >= 0) & (vertices[:, 0] <= 1) &
        (vertices[:, 1] >= 0) & (vertices[:, 1] <= 1)
    ]

    # If we don't have enough vertices, use original points
    if len(valid_vertices) < n_circles:
        # Use original Voronoi points in the unit square
        valid_points = voronoi_points[
            (voronoi_points[:, 0] >= 0) & (voronoi_points[:, 0] <= 1) &
            (voronoi_points[:, 1] >= 0) & (voronoi_points[:, 1] <= 1)
        ]

        # Fill remaining circles with points near corners and center
        if len(valid_points) < n_circles:
            additional_points = []

            # Add corner points
            corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]
            additional_points.extend(corners[:min(4, n_circles-len(valid_points))])

            # Add center points
            if len(valid_points) + 4 < n_circles:
                center_points = np.random.rand(n_circles - len(valid_points) - 4, 2) * 0.4 + 0.3
                additional_points.extend(center_points)

            valid_points = np.vstack([valid_points, additional_points])

        centers = valid_points[:n_circles]
    else:
        centers = valid_vertices[:n_circles]

    # Initialize circles with computed maximum radii
    circles = []
    for x, y in centers:
        max_r = _compute_max_radius(x, y, circles)
        circles.append([x, y, max_r])

    return np.array(circles)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Initialize using Voronoi-based approach
    circles = _initialize_with_voronoi(n_circles=32, seed=42)

    # Ensure no negative or zero radii
    circles[:, 2] = np.maximum(circles[:, 2], 1e-6)

    return circles


# EVOLVE-BLOCK-END