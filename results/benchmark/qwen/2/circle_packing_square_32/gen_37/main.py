# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _initialize_hexagonal_grid(n):
    """Initialize circles using a hexagonal grid pattern with density-adaptive radii"""
    # Calculate grid dimensions for n circles
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n / rows))

    # Adjust to ensure we have enough slots
    while rows * cols < n:
        rows += 1

    # Create hexagonal grid points
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows

    # Hexagonal offset
    offset = spacing_x * 0.5

    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            x = (j + (i % 2) * 0.5) * spacing_x
            y = i * spacing_y
            points.append([x, y])

    # Ensure we have exactly n points
    points = points[:n]

    # Compute density-adaptive radii
    points_array = np.array(points)
    radii = []

    # For each point, compute a radius based on minimum distance to neighbors
    for i in range(len(points)):
        current_point = points_array[i]
        distances = np.linalg.norm(points_array - current_point, axis=1)

        # Exclude self-distance
        distances[i] = np.inf

        # Use minimum distance to nearest neighbor as basis for radius
        min_distance = np.min(distances)

        # Set radius as fraction of minimum neighbor distance, but cap at reasonable value
        base_radius = min_distance * 0.4  # Allow circles to be close but not too tight
        max_radius = min(0.1, 0.5)  # Cap at reasonable maximum
        radius = max(0.001, min(base_radius, max_radius))

        radii.append(radius)

    return np.array(points), radii

def _get_radius_bounds(circles, idx):
    """Get bounds for radius of circle at index idx"""
    x, y, r = circles[idx]

    # Minimum radius is 0 (though practically will be very small)
    min_r = 0.0001

    # Maximum radius is constrained by boundaries and neighbors
    max_r = min(x, y, 1-x, 1-y)

    return min_r, max_r

def _compute_penalty_distance(circles, i, j):
    """Compute the squared distance between two circles"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    dx = x1 - x2
    dy = y1 - y2
    dist_sq = dx*dx + dy*dy
    return dist_sq

def _compute_violations(circles):
    """Compute total penalty for constraint violations with adaptive penalties"""
    n = len(circles)
    penalty = 0.0

    # Check containment constraints with adaptive penalties
    for i in range(n):
        x, y, r = circles[i]
        # Calculate boundary violations (negative values mean violation)
        boundary_violations = [
            x - r,          # left boundary
            1 - (x + r),    # right boundary
            y - r,          # bottom boundary
            1 - (y + r)     # top boundary
        ]

        # Apply adaptive exponential penalty for each boundary violation
        for violation in boundary_violations:
            if violation < 0:
                # Penalty scales exponentially with violation severity
                penalty += math.exp(-violation * 100)  # Larger violations get higher penalties

    # Check overlap constraints with adaptive penalties
    tree = cKDTree(circles[:, :2])

    # Query pairs within a reasonable distance
    pairs = tree.query_pairs(0.01)

    for i, j in pairs:
        # Only consider actual overlap violations
        dist_sq = _compute_penalty_distance(circles, i, j)
        r1, r2 = circles[i, 2], circles[j, 2]
        distance = math.sqrt(dist_sq)

        if distance < r1 + r2:
            # Exponential penalty for overlaps scaled by violation amount
            overlap_violation = (r1 + r2) - distance
            penalty += math.exp(overlap_violation * 100)

    return penalty

def _objective_function(params, circles, n):
    """Objective function to maximize sum of radii (negative for minimization)"""
    # Update circles array with current parameters
    for i in range(n):
        circles[i, 0] = params[3*i]
        circles[i, 1] = params[3*i+1]
        circles[i, 2] = params[3*i+2]

    # Negative sum of radii (since we want to maximize)
    objective = -np.sum(circles[:, 2])

    # Add penalty for constraint violations
    penalty = _compute_violations(circles)
    objective += penalty

    return objective

def _constraint_function(params, circles, n):
    """Constraint function that returns 0 when all constraints satisfied"""
    # Update circles array with current parameters
    for i in range(n):
        circles[i, 0] = params[3*i]
        circles[i, 1] = params[3*i+1]
        circles[i, 2] = params[3*i+2]

    # Check containment constraints
    violations = []
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0:
            violations.append(-(x - r))
        if x + r > 1:
            violations.append(x + r - 1)
        if y - r < 0:
            violations.append(-(y - r))
        if y + r > 1:
            violations.append(y + r - 1)

    # Check overlap constraints
    tree = cKDTree(circles[:, :2])
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = _compute_penalty_distance(circles, i, j)
            r1, r2 = circles[i, 2], circles[j, 2]
            if dist_sq < (r1 + r2)**2:
                violations.append((r1 + r2 - math.sqrt(dist_sq)))

    return np.array(violations)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # Initialize using hexagonal grid
    points, radii = _initialize_hexagonal_grid(n)

    # Set initial positions and radii
    for i in range(n):
        circles[i, 0] = points[i, 0]
        circles[i, 1] = points[i, 1]
        circles[i, 2] = radii[i]

    # Flatten for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])

    # Optimization with bounds
    bounds = []
    for i in range(n):
        # Bounds for x and y coordinates
        bounds.append((0.001, 0.999))  # Positions
        bounds.append((0.001, 0.999))  # Positions
        bounds.append((0.001, 0.49))   # Radii (max radius limited to avoid overflow)

    # Run optimization
    try:
        # First pass with a coarse optimization
        result = minimize(
            _objective_function,
            initial_params,
            args=(circles, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-5}
        )

        # Update circles with optimized results
        for i in range(n):
            circles[i, 0] = result.x[3*i]
            circles[i, 1] = result.x[3*i+1]
            circles[i, 2] = result.x[3*i+2]

        # Second pass with refined optimization
        result2 = minimize(
            _objective_function,
            result.x,
            args=(circles, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )

        # Final update
        for i in range(n):
            circles[i, 0] = result2.x[3*i]
            circles[i, 1] = result2.x[3*i+1]
            circles[i, 2] = result2.x[3*i+2]

    except Exception as e:
        # Fallback to hexagonal grid if optimization fails
        print(f"Optimization failed with error: {e}")

    # Ensure final constraints are met
    for i in range(n):
        x, y, r = circles[i]
        # Clamp to valid ranges
        circles[i, 0] = np.clip(x, r, 1-r)
        circles[i, 1] = np.clip(y, r, 1-r)
        circles[i, 2] = np.clip(r, 0.001, min(1-x, 1-y, x, y))

    return circles


# EVOLVE-BLOCK-END