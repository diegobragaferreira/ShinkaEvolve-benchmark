# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26

    # Generate initial points using a Voronoi-based approach for better spatial distribution
    initial_points = generate_voronoi_initial_points(n)

    # Initialize circles with positions and set initial radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [initial_points[i][0], initial_points[i][1], 0.0]

    # Set initial radii using a simple heuristic based on Voronoi cell areas
    for i in range(n):
        circles[i][2] = calculate_initial_radius(circles, i)

    # Optimize using local search to maximize sum of radii
    optimize_circles(circles)

    return circles

def generate_voronoi_initial_points(n):
    """Generate initial points using a Voronoi-based approach"""
    # Create a regular grid of candidate points
    grid_size = int(np.ceil(np.sqrt(n)))
    x = np.linspace(0.05, 0.95, grid_size)
    y = np.linspace(0.05, 0.95, grid_size)

    # Generate candidate points
    candidates = []
    for i in range(len(x)):
        for j in range(len(y)):
            candidates.append([x[i], y[j]])

    # Ensure we have enough candidates
    if len(candidates) < n:
        # Add more points randomly
        additional_points = n - len(candidates)
        for _ in range(additional_points):
            candidates.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])

    # Use Voronoi to select n points that are well distributed
    candidates = np.array(candidates[:n])

    # Simple approach: choose points that are furthest apart
    selected_points = []
    remaining_points = candidates.tolist()

    # Start with a random point
    start_idx = np.random.randint(len(remaining_points))
    selected_points.append(remaining_points.pop(start_idx))

    # Greedily select points that are farthest from existing selected points
    while len(selected_points) < n and remaining_points:
        max_dist = -1
        max_idx = -1

        for i, point in enumerate(remaining_points):
            min_dist = float('inf')
            for selected_point in selected_points:
                dist = np.sqrt((point[0] - selected_point[0])**2 + (point[1] - selected_point[1])**2)
                min_dist = min(min_dist, dist)

            if min_dist > max_dist:
                max_dist = min_dist
                max_idx = i

        if max_idx != -1:
            selected_points.append(remaining_points.pop(max_idx))

    return np.array(selected_points)

def calculate_initial_radius(circles, idx):
    """Calculate initial radius for circle at index idx based on neighbors"""
    # Get distance to all other circles
    distances = []
    for i in range(len(circles)):
        if i != idx:
            dist = np.sqrt((circles[idx][0] - circles[i][0])**2 + (circles[idx][1] - circles[i][1])**2)
            distances.append(dist)

    # Radius is limited by the minimum distance to any neighbor divided by 2
    if len(distances) == 0:
        # If no neighbors, set a reasonable default radius
        return 0.05

    # Minimum distance to other circles (minus some buffer to prevent overlap)
    min_dist = min(distances)
    max_radius = min_dist / 2.0 - 0.001

    # Also constrain by boundary conditions (circle must fit in unit square)
    boundary_radius = min(
        circles[idx][0],
        circles[idx][1],
        1 - circles[idx][0],
        1 - circles[idx][1]
    ) - 0.001

    radius = min(max_radius, boundary_radius)

    # Ensure minimum positive radius
    return max(0.001, radius)

def optimize_circles(circles):
    """Simple local optimization to maximize sum of radii"""
    # Gradient ascent approach - modify radii to increase the sum
    learning_rate = 0.01
    iterations = 1000

    for iter in range(iterations):
        # Calculate current sum
        old_sum = np.sum(circles[:, 2])

        # For each circle, try to increase its radius
        for i in range(len(circles)):
            # Try increasing the radius slightly
            test_radius = circles[i][2] + learning_rate

            # Check boundary constraints
            boundary_radius = min(
                circles[i][0],
                circles[i][1],
                1 - circles[i][0],
                1 - circles[i][1]
            )

            test_radius = min(test_radius, boundary_radius - 0.001)

            if test_radius <= 0.001:
                continue

            # Check overlap constraints with other circles
            valid = True
            old_radius = circles[i][2]
            circles[i][2] = test_radius

            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                    required_dist = circles[i][2] + circles[j][2]
                    if dist < required_dist:
                        valid = False
                        break

            if valid:
                # Accept the change
                pass
            else:
                # Revert the change
                circles[i][2] = old_radius


# EVOLVE-BLOCK-END