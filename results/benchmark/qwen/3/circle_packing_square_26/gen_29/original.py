# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _generate_voronoi_seeds(n_points, boundary_margin=0.1):
    """Generate well-distributed seed points using Voronoi diagram."""
    # Generate random points within a margin to avoid edge constraints
    np.random.seed(42)
    points = np.random.rand(n_points, 2) * (1 - 2*boundary_margin) + boundary_margin
    return points

def _compute_voronoi_radii(points, boundary_margin=0.1):
    """Compute radii based on Voronoi cell areas."""
    # Create Voronoi diagram
    vor = Voronoi(points)

    # Compute area of each Voronoi cell
    # We'll use a simple approximation based on the distance to nearest neighbors
    # and ensure circles don't exceed boundary constraints
    radii = []
    for i, point in enumerate(points):
        # Find the minimum distance to any other point to estimate cell size
        distances = cdist([point], np.delete(points, i, axis=0))[0]
        min_distance = np.min(distances)

        # Set radius to be proportional to cell size but constrained by boundaries
        max_radius = min(point[0], 1-point[0], point[1], 1-point[1])
        estimated_radius = min(min_distance/2.0, max_radius)
        radii.append(max(estimated_radius, 0.001))  # Ensure minimum radius

    return np.array(radii)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = np.zeros((n, 3))

    # Generate Voronoi-based seed points
    seed_points = _generate_voronoi_seeds(n)

    # Assign radii based on Voronoi cells
    radii = _compute_voronoi_radii(seed_points)

    # Set initial circle positions and radii
    for i in range(n):
        circles[i] = [seed_points[i][0], seed_points[i][1], radii[i]]

    return circles


# EVOLVE-BLOCK-END