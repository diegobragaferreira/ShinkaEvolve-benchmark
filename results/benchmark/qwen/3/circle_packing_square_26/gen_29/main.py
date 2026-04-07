# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.spatial import distance

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _compute_optimal_voronoi_circles(n_points, boundary_margin=0.05):
    """Compute optimal circle positions and radii using Voronoi-based approach."""
    # Generate random points within a margin to avoid edge constraints
    np.random.seed(42)
    points = np.random.rand(n_points, 2) * (1 - 2*boundary_margin) + boundary_margin

    # Create Voronoi diagram
    vor = Voronoi(points)

    # Extract Voronoi vertices and regions
    # For each original point, compute its Voronoi cell
    circles = []

    for i in range(len(points)):
        # Get the vertices of the Voronoi cell for point i
        region = vor.regions[vor.point_region[i]]

        if len(region) > 0 and -1 not in region:
            # Compute centroid of Voronoi cell
            vertices = np.array([vor.vertices[j] for j in region])

            # Ensure vertices are within bounds
            vertices = vertices[
                (vertices[:, 0] >= 0) &
                (vertices[:, 0] <= 1) &
                (vertices[:, 1] >= 0) &
                (vertices[:, 1] <= 1)
            ]

            if len(vertices) >= 3:
                # Use the centroid of the bounded Voronoi cell as circle center
                centroid = np.mean(vertices, axis=0)
            else:
                # Fall back to original point if not enough vertices
                centroid = points[i]
        else:
            # Fall back to original point if cell is unbounded
            centroid = points[i]

        # Compute maximum possible radius for this location
        # Distance to closest boundary
        min_boundary_dist = min(
            centroid[0], 1-centroid[0],
            centroid[1], 1-centroid[1]
        )

        # Distance to nearest neighbor points
        distances_to_others = cdist([centroid], np.delete(points, i, axis=0))[0]
        min_neighbor_dist = np.min(distances_to_others) if len(distances_to_others) > 0 else float('inf')

        # Maximum radius is limited by either boundary or neighbor distance
        max_radius = min(min_boundary_dist, min_neighbor_dist / 2.0)
        max_radius = max(max_radius, 0.001)  # Minimum radius

        circles.append([centroid[0], centroid[1], max_radius])

    return np.array(circles)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = _compute_optimal_voronoi_circles(n)

    # Ensure all circles are within bounds
    for i in range(n):
        x, y, r = circles[i]
        # Adjust position if needed to keep circle within unit square
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]

    return circles


# EVOLVE-BLOCK-END