# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_hexagonal_grid(n_circles):
    """Initialize circles in a hexagonal grid pattern"""
    # Create hexagonal grid points
    rows = int(np.ceil(np.sqrt(n_circles)))
    cols = int(np.ceil(n_circles / rows))

    # Adjust dimensions to fit exactly n_circles
    while rows * cols < n_circles:
        rows += 1

    # Hexagonal grid spacing
    spacing_x = 1.0 / cols
    spacing_y = spacing_x * np.sqrt(3) / 2

    # Generate grid points
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n_circles:
                break
            x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            points.append([x, y])

    # Trim to exact number of circles
    points = points[:n_circles]

    # Initialize radii to small values
    radii = [0.02] * n_circles
    return np.array(points), radii

def get_circle_constraints(circles):
    """Get constraints for circle packing problem"""
    n = len(circles)
    constraints = []

    # Boundary constraints: each circle must fit entirely in the unit square
    for i in range(n):
        x, y, r = circles[i]
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - c[i*3]})  # r >= x
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - (1 - c[i*3])})  # r >= 1-x
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - c[i*3+1]})  # r >= y
        constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[i*3+2] - (1 - c[i*3+1])})  # r >= 1-y

    # Overlap constraints: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(c, i=i, j=j):
                x1, y1, r1 = c[i*3], c[i*3+1], c[i*3+2]
                x2, y2, r2 = c[j*3], c[j*3+1], c[j*3+2]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                return dist - (r1 + r2)
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})

    return constraints

def objective_function(circles_flat):
    """Objective function to maximize sum of radii"""
    # Sum of all radii
    return -np.sum(circles_flat[2::3])  # Negative because we want to maximize

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32

    # Initialize with hexagonal grid
    points, radii = initialize_hexagonal_grid(n)

    # Combine into flat array [x0, y0, r0, x1, y1, r1, ...]
    circles_flat = np.array([coord for point, rad in zip(points, radii) for coord in [*point, rad]])

    # Define bounds for variables (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])  # x, y, r ranges

    # Optimize using scipy minimize
    try:
        result = minimize(
            objective_function,
            circles_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'gtol': 1e-9}
        )

        if result.success:
            circles_flat = result.x
        else:
            print("Optimization failed:", result.message)

    except Exception as e:
        print(f"Optimization error: {e}")

    # Convert back to final format
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [circles_flat[i*3], circles_flat[i*3+1], circles_flat[i*3+2]]

    return circles


# EVOLVE-BLOCK-END