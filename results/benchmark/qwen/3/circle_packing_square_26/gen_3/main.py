# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = np.zeros((n, 3))

    # Initialize with heuristic approach
    # Place circles in a structured way to get a good starting configuration
    # Start with a few larger circles in corners and edges

    # Create initial placement using a combination of grid-based and corner placement
    # We'll use a simpler approach with fixed placement pattern for initial configuration

    # For first few circles, place them strategically
    # Circle 0: Center of square (good for large radius)
    circles[0] = [0.5, 0.5, 0.3]  # Large center circle

    # Place 4 corner circles
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i in range(4):
        circles[i+1] = [corners[i][0], corners[i][1], 0.05]

    # Place 8 edge circles
    edges = [
        (0.5, 0.1), (0.5, 0.9),  # top and bottom center
        (0.1, 0.5), (0.9, 0.5),  # left and right center
        (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)  # near corners
    ]
    for i in range(8):
        circles[i+5] = [edges[i][0], edges[i][1], 0.04]

    # Fill remaining circles with small radii
    for i in range(13, 26):
        circles[i] = [0.5, 0.5, 0.01]

    # Ensure initial placement is valid
    # Adjust radii to prevent overlap and ensure containment
    for i in range(n):
        # Ensure containment constraint
        x, y, r = circles[i]
        r = min(r, x, y, 1-x, 1-y)
        circles[i] = [x, y, r]

    # Fine-tune the configuration slightly
    # Increase some radii where possible without overlap
    for i in range(1, n):  # Skip first circle which is central
        x, y, r = circles[i]
        # Try to increase radius while maintaining non-overlap with others
        max_radius = min(x, y, 1-x, 1-y)
        for j in range(i):  # Check against previous circles
            if i != j:
                x_prev, y_prev, r_prev = circles[j]
                dist = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                max_radius = min(max_radius, dist - r_prev)
        circles[i] = [x, y, max_radius]

    return circles


# EVOLVE-BLOCK-END