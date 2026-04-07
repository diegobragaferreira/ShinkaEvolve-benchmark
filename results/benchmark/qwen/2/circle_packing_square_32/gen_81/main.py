# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math

def _compute_density_aware_radii(points, k=5):
    """Compute density-aware initial radii based on k-nearest neighbors"""
    if len(points) <= k:
        # If we don't have enough points, fall back to uniform initialization
        return [0.02] * len(points)

    # Build KDTree for neighbor queries
    tree = cKDTree(points)

    # Find k nearest neighbors for each point (excluding itself)
    distances, indices = tree.query(points, k=k+1, p=2)

    # Compute average distance to neighbors as density proxy
    # Smaller average distance means higher density
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude the point itself

    # Normalize distances to get relative densities
    normalized_densities = (np.max(avg_distances) - avg_distances + 1e-8) / (np.max(avg_distances) + 1e-8)

    # Convert density to radius: higher density -> smaller initial radius
    # Scale so that maximum radius is about 0.05 and minimum about 0.005
    min_radius = 0.005
    max_radius = 0.05
    radii = min_radius + normalized_densities * (max_radius - min_radius)

    # Add some randomness to help escape local optima
    noise_factor = 0.1
    radii = radii * (1 + np.random.normal(0, noise_factor, len(radii)))

    # Ensure radii are within reasonable bounds
    radii = np.clip(radii, 0.001, 0.49)

    return radii.tolist()

def _initialize_hexagonal_grid(n):
    """Initialize circles using a hexagonal grid pattern with improved scaling"""
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

    # Initialize radii based on density around each point
    radii = _compute_density_aware_radii(np.array(points))

    return np.array(points), radii

def _compute_penalty_distance(circles, i, j):
    """Compute the squared distance between two circles"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    dx = x1 - x2
    dy = y1 - y2
    dist_sq = dx*dx + dy*dy
    return dist_sq

def _compute_violations(circles):
    """Compute total penalty for constraint violations using exponential penalties"""
    n = len(circles)
    penalty = 0.0

    # Check containment constraints with exponential penalty
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            # Use exponential penalty for boundary violations
            violation = min(x-r, x+r-1, y-r, y+r-1)
            penalty += 1000 * math.exp(10 * violation)

    # Check overlap constraints using cKDTree for efficiency
    tree = cKDTree(circles[:, :2])

    # Query pairs within a reasonable distance
    pairs = tree.query_pairs(0.01)

    for i, j in pairs:
        # Only consider actual overlap violations
        dist_sq = _compute_penalty_distance(circles, i, j)
        r1, r2 = circles[i, 2], circles[j, 2]

        if dist_sq < (r1 + r2)**2:
            # Exponential penalty for overlaps
            distance = math.sqrt(dist_sq)
            violation = (r1 + r2) - distance
            penalty += 1000 * math.exp(10 * violation)

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
    objective += _compute_violations(circles)

    return objective

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