# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings

def _initialize_voronoi_placement(n_circles: int, seed: int = 42) -> np.ndarray:
    """Initialize circle positions using a Voronoi-based approach with improved distribution."""
    np.random.seed(seed)

    # Create a more structured grid with better spacing
    grid_size = max(5, int(np.ceil(np.sqrt(n_circles)) + 2))
    x_coords = np.linspace(0.05, 0.95, grid_size)
    y_coords = np.linspace(0.05, 0.95, grid_size)
    grid_points = np.array([[x, y] for x in x_coords for y in y_coords])

    # Take the first n_circles points (or generate more if needed)
    if len(grid_points) >= n_circles:
        initial_points = grid_points[:n_circles]
    else:
        # If we don't have enough grid points, add random points
        extra_points = n_circles - len(grid_points)
        random_points = np.random.uniform(0.05, 0.95, (extra_points, 2))
        initial_points = np.vstack([grid_points, random_points])

    # Create Voronoi diagram and use centroids as circle centers
    try:
        vor = Voronoi(initial_points)
        # Use Voronoi vertices as circle centers
        centroids = vor.points
        # Filter out points that are too close to edges or invalid
        valid_centroids = []
        for point in centroids:
            if 0.05 <= point[0] <= 0.95 and 0.05 <= point[1] <= 0.95:
                valid_centroids.append(point)

        if len(valid_centroids) < n_circles:
            # Fill in missing points randomly
            remaining = n_circles - len(valid_centroids)
            random_points = np.random.uniform(0.05, 0.95, (remaining, 2))
            final_points = np.vstack([valid_centroids, random_points])
        else:
            final_points = np.array(valid_centroids[:n_circles])

    except Exception:
        # Fallback to random initialization if Voronoi fails
        final_points = np.random.uniform(0.05, 0.95, (n_circles, 2))

    return final_points

def _get_max_radius_at_position(x: float, y: float, circles: np.ndarray) -> float:
    """Calculate maximum possible radius at given position without overlapping existing circles."""
    # Maximum radius to stay within unit square
    max_radius = min(x, y, 1-x, 1-y)

    # Check overlaps with existing circles
    for i in range(len(circles)):
        cx, cy, r = circles[i]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        # If circle would overlap, reduce radius accordingly
        if distance < r + 0.0001:  # Small tolerance
            max_radius = min(max_radius, distance - 0.0001)

    return max(max_radius, 0.001)

def _objective_and_constraints(circles_flat: np.ndarray, n_circles: int) -> tuple:
    """Return objective value and constraint violations for optimization."""
    # Reshape flat array back to circles format
    circles = circles_flat.reshape((n_circles, 3))

    # Calculate total radius (negative because we want to maximize)
    total_radius = -np.sum(circles[:, 2])

    # Constraint violations (penalty for overlaps and boundary violations)
    penalty = 0.0

    # Boundary constraint penalties
    for i in range(n_circles):
        x, y, r = circles[i]
        # Penalty for being too close to boundaries
        boundary_penalty = 0.0
        if x - r < 0:
            boundary_penalty += (0 - (x - r))**2
        if y - r < 0:
            boundary_penalty += (0 - (y - r))**2
        if x + r > 1:
            boundary_penalty += ((x + r) - 1)**2
        if y + r > 1:
            boundary_penalty += ((y + r) - 1)**2
        penalty += boundary_penalty * 1000000

    # Overlap constraint penalties
    for i in range(n_circles):
        x1, y1, r1 = circles[i]
        for j in range(i+1, n_circles):
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < r1 + r2:
                overlap = (r1 + r2) - dist
                penalty += overlap**2 * 100000

    return total_radius + penalty, penalty

def _project_to_feasible(circles: np.ndarray) -> np.ndarray:
    """Project circles to feasible region with proper boundary handling."""
    # First pass: enforce boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Ensure all circles are within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]

    # Second pass: resolve overlaps
    n = len(circles)
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i+1, n):
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < r1 + r2:
                # Calculate overlap amount
                overlap = (r1 + r2) - dist
                # Reduce both radii proportionally
                reduction = overlap / 2.0
                circles[i, 2] = max(0.001, r1 - reduction)
                circles[j, 2] = max(0.001, r2 - reduction)

                # Recheck this circle against others
                x1, y1, r1 = circles[i]

    return circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = np.zeros((n, 3))

    # Initialize with Voronoi-based approach
    initial_centers = _initialize_voronoi_placement(n, seed=42)

    # Set initial radii to small positive values
    for i in range(n):
        circles[i, :2] = initial_centers[i]
        # Initial radius estimation based on Voronoi cell size
        circles[i, 2] = 0.03

    # Refine using optimization approach
    # Flatten circles for optimization
    initial_flat = circles.flatten()

    # Use scipy minimize with SLSQP method for constrained optimization
    try:
        result = minimize(
            lambda x: _objective_and_constraints(x, n)[0],
            initial_flat,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-6},
            callback=lambda x: print(f"Current sum of radii: {-_objective_and_constraints(x, n)[0]}")
        )
        circles_optimized = result.x.reshape((n, 3))
    except Exception:
        # Fallback to simple projection if optimization fails
        circles_optimized = circles.copy()

    # Final projection to feasible region
    circles_optimized = _project_to_feasible(circles_optimized)

    # Additional local refinement steps
    for _ in range(20):
        # Try to increase individual radii where possible
        for i in range(n):
            x, y, r = circles_optimized[i]
            max_radius = _get_max_radius_at_position(x, y, circles_optimized)
            # Gradually allow increasing radius up to max allowed
            if r < max_radius:
                circles_optimized[i, 2] = min(max_radius, r + 0.002)

        # Project again to maintain feasibility
        circles_optimized = _project_to_feasible(circles_optimized)

        # Apply boundary constraints again
        for i in range(n):
            x, y, r = circles_optimized[i]
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            circles_optimized[i] = [x, y, r]

    # Final cleanup
    circles_optimized = _project_to_feasible(circles_optimized)

    return circles_optimized

# EVOLVE-BLOCK-END