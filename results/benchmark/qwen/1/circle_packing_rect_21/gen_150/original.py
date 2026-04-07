# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import time

def calculate_voronoi_areas(points):
    """Calculate approximate Voronoi cell areas for a set of points."""
    try:
        vor = Voronoi(points)
        areas = []
        for i in range(len(points)):
            # Get the vertices of the Voronoi cell for point i
            region = vor.regions[vor.point_region[i]]
            if -1 in region:
                # Infinite region, skip
                areas.append(0)
            else:
                # Calculate area of polygon using shoelace formula
                vertices = [vor.vertices[j] for j in region]
                if len(vertices) < 3:
                    areas.append(0)
                else:
                    # Convert to numpy array
                    poly = np.array(vertices)
                    # Shoelace formula for polygon area
                    x = poly[:, 0]
                    y = poly[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    areas.append(area)
        return np.array(areas)
    except:
        # Fallback in case Voronoi computation fails
        return np.ones(len(points)) * 100.0  # Assume moderate areas

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    # Rectangle with perimeter = 4, so width + height = 2
    # Optimal rectangle: 1.5 x 0.5 (width x height) for better packing efficiency
    rect_width = 1.5
    rect_height = 0.5

    # Phase 1: Initial placement using hexagonal lattice with adaptive spacing
    circles = np.zeros((n, 3))

    # Hexagonal packing arrangement adapted for rectangular container
    rows = 4
    cols = 6

    # Calculate spacing based on rectangle dimensions
    spacing_x = rect_width / (cols + 1)
    spacing_y = rect_height / (rows + 1)

    # Place circles in hexagonal pattern
    idx = 0
    for i in range(rows):
        offset = spacing_x * (i % 2) * 0.5  # Offset every other row
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x + offset
            y = (i + 1) * spacing_y

            # Ensure position is within bounds
            x = max(0.01, min(rect_width - 0.01, x))
            y = max(0.01, min(rect_height - 0.01, y))

            # Set initial radius to a small value
            circles[idx] = [x, y, 0.05]
            idx += 1

        if idx >= n:
            break

    # Fill remaining circles if needed
    while idx < n:
        x = np.random.uniform(0.01, rect_width - 0.01)
        y = np.random.uniform(0.01, rect_height - 0.01)
        circles[idx] = [x, y, 0.05]
        idx += 1

    # Phase 2: Voronoi-based optimization with adaptive mutation
    max_iterations = 200
    for iteration in range(max_iterations):
        # Compute Voronoi diagram for current configuration
        points = circles[:, :2]  # x, y coordinates only
        voronoi_areas = calculate_voronoi_areas(points)

        # Phase 3: Optimized expansion based on Voronoi analysis and adaptive mutation
        improved = False

        # Sort circles by Voronoi area (descending - largest areas first)
        # Large Voronoi cells = sparse regions = more room to expand
        sorted_indices = np.argsort(voronoi_areas)[::-1]

        # Try to expand circles that have room to grow
        for i in sorted_indices:
            # Calculate maximum allowable radius for this circle
            max_radius = min(
                circles[i][0],  # Distance to left edge
                rect_width - circles[i][0],  # Distance to right edge
                circles[i][1],  # Distance to bottom edge
                rect_height - circles[i][1]   # Distance to top edge
            ) - 0.001

            # Consider collision constraints with neighbors
            for j in range(n):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - circles[j][2] - 0.001
                    if collision_radius > 0:
                        max_radius = min(max_radius, collision_radius)

            # If we can expand and there's significant improvement
            if max_radius > circles[i][2] and max_radius > 0.001:
                # Adaptive expansion: scale based on Voronoi cell area
                # Smaller cells (denser regions) = smaller mutations
                # Larger cells (sparse regions) = larger mutations
                base_delta = min(0.02, max_radius - circles[i][2])

                # Normalize Voronoi area to [0,1] scale for mutation factor
                normalized_area = voronoi_areas[i] / (rect_width * rect_height) if voronoi_areas[i] > 0 else 0.01
                # Map area to mutation factor: 0.01 to 1.0
                mutation_factor = 0.01 + 0.99 * (1.0 - np.exp(-normalized_area * 5.0))

                # Apply adaptive mutation with base_delta scaled by Voronoi factor
                delta = base_delta * mutation_factor

                if delta > 0.001:
                    circles[i][2] += delta
                    improved = True

        # Early stopping if no improvement
        if not improved:
            break

    # Phase 4: Final refinement using iterative local optimization
    for _ in range(100):
        improved = False
        for i in range(n):
            # Calculate maximum allowable radius for this circle
            max_radius = min(
                circles[i][0],  # Distance to left edge
                rect_width - circles[i][0],  # Distance to right edge
                circles[i][1],  # Distance to bottom edge
                rect_height - circles[i][1]   # Distance to top edge
            ) - 0.001

            # Consider collision constraints with neighbors
            for j in range(n):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    collision_radius = distance - circles[j][2] - 0.001
                    if collision_radius > 0:
                        max_radius = min(max_radius, collision_radius)

            # Increase radius if beneficial
            if max_radius > circles[i][2] and max_radius > 0.001:
                # Try to increase by a small amount
                new_radius = min(max_radius, circles[i][2] + 0.01)
                if new_radius > circles[i][2]:
                    circles[i][2] = new_radius
                    improved = True

        # Stop if no improvement made
        if not improved:
            break

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")