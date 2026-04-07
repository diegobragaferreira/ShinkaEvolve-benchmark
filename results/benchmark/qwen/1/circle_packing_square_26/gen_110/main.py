# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings

def _initialize_voronoi_placement(n_circles: int, seed: int = 42) -> np.ndarray:
    """Initialize circle positions using a Voronoi-based approach with adaptive radius estimation."""
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

def _calculate_voronoi_radii(circles: np.ndarray, voronoi_points: np.ndarray) -> np.ndarray:
    """Estimate initial radii based on Voronoi cell areas."""
    n = len(circles)
    radii = np.zeros(n)

    # Calculate Voronoi cell areas for each point
    if len(voronoi_points) < n:
        # Not enough Voronoi points, use fallback
        return np.full(n, 0.03)

    # Simple area estimation based on distances to neighbors
    # For each circle, estimate its area by looking at surrounding points
    for i in range(n):
        x, y = circles[i, 0], circles[i, 1]

        # Find distances to nearby points and estimate area
        distances = np.sqrt(np.sum((voronoi_points - [x, y])**2, axis=1))
        sorted_indices = np.argsort(distances)

        # Get distances to first few neighbors
        neighbor_distances = distances[sorted_indices[1:5]]  # First 4 neighbors

        if len(neighbor_distances) > 0:
            # Estimate area based on average neighbor distance
            avg_dist = np.mean(neighbor_distances)
            # Convert to radius (simplified approach)
            estimated_radius = max(0.01, avg_dist * 0.4)
            radii[i] = estimated_radius
        else:
            radii[i] = 0.03  # Default radius

    # Normalize radii to avoid extremely large values
    max_radius = np.max(radii)
    if max_radius > 0.1:
        radii = radii * 0.1 / max_radius

    return radii

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

def _simple_local_optimization(circles: np.ndarray) -> np.ndarray:
    """Simple but effective local search to improve solution."""
    n = len(circles)
    max_iter = 30

    for iteration in range(max_iter):
        # Try to improve by adjusting positions slightly
        for i in range(n):
            x, y, r = circles[i]

            # Try small random movements to see if we can improve
            best_pos = [x, y, r]
            best_radius = r

            # Try several small moves
            for _ in range(5):
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)

                new_x = x + dx
                new_y = y + dy
                new_x = np.clip(new_x, r, 1-r)
                new_y = np.clip(new_y, r, 1-r)

                # Check if this move helps with constraints
                valid = True
                for j in range(n):
                    if i != j:
                        cx, cy, cr = circles[j]
                        dist = np.sqrt((new_x - cx)**2 + (new_y - cy)**2)
                        if dist < r + cr:
                            valid = False
                            break

                if valid:
                    # Keep this better position for now
                    best_pos = [new_x, new_y, r]

        # Also try to increase radius slightly if possible
        for i in range(n):
            x, y, r = circles[i]
            max_radius = min(x, y, 1-x, 1-y)

            # Try to increase radius a bit if there's room
            if r < max_radius - 0.001:
                new_r = min(max_radius, r + 0.002)
                # Check if this works with neighbors
                valid = True
                for j in range(n):
                    if i != j:
                        cx, cy, cr = circles[j]
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if dist < new_r + cr:
                            valid = False
                            break

                if valid:
                    circles[i, 2] = new_r

    return circles

def _project_to_feasible(circles: np.ndarray) -> np.ndarray:
    """Project circles to feasible region with proper boundary handling and thorough overlap resolution."""
    # First pass: enforce boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Ensure all circles are within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]

    # Multiple passes of overlap resolution for better convergence
    n = len(circles)
    max_iterations = 3

    for iteration in range(max_iterations):
        changed = False
        # Pass 1: resolve overlaps more aggressively
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if dist < r1 + r2:
                    # Calculate overlap amount
                    overlap = (r1 + r2) - dist
                    # More aggressive reduction - reduce both by larger amounts
                    reduction = overlap * 0.7  # More aggressive than before
                    new_r1 = max(0.001, r1 - reduction/2)
                    new_r2 = max(0.001, r2 - reduction/2)

                    if new_r1 < r1 or new_r2 < r2:
                        circles[i, 2] = new_r1
                        circles[j, 2] = new_r2
                        changed = True

        # Break early if no changes made
        if not changed:
            break

    return circles

def _local_search_refinement(circles: np.ndarray) -> np.ndarray:
    """Perform local search to further improve the configuration."""
    n = len(circles)
    max_attempts = 100

    for attempt in range(max_attempts):
        improved = False

        # Try to increase radii safely
        for i in range(n):
            x, y, r = circles[i]

            # Calculate maximum possible radius at this position
            max_radius = min(x, y, 1-x, 1-y)

            # If we can increase radius, try to do so
            if r < max_radius - 0.001:  # Leave some margin
                # Check if we can safely increase radius
                can_increase = True
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                        # Check if increasing would cause overlap
                        if dist < r + 0.001 + r1:  # Added buffer
                            can_increase = False
                            break

                if can_increase:
                    # Try to increase radius
                    new_r = min(max_radius, r + 0.005)
                    circles[i, 2] = new_r
                    improved = True

        if not improved:
            break

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

    # Set initial radii using Voronoi-based estimation
    initial_radii = _calculate_voronoi_radii(initial_centers, initial_centers)

    # Set initial positions and radii
    for i in range(n):
        circles[i, :2] = initial_centers[i]
        circles[i, 2] = initial_radii[i]

    # Apply initial feasibility check
    circles = _project_to_feasible(circles)

    # Iterative refinement loop
    best_sum = 0.0
    no_improvement_count = 0
    max_no_improvement = 50

    for iteration in range(200):
        # Store current sum for monitoring
        current_sum = np.sum(circles[:, 2])

        # Try to improve radii locally
        circles = _local_search_refinement(circles)

        # Apply boundary constraints and resolve overlaps
        circles = _project_to_feasible(circles)

        # Try additional local optimization
        circles = _simple_local_optimization(circles)

        # Apply boundary constraints again
        circles = _project_to_feasible(circles)

        # Check for improvement
        new_sum = np.sum(circles[:, 2])
        if new_sum > best_sum:
            best_sum = new_sum
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early termination if no improvement for a while
        if no_improvement_count > max_no_improvement:
            break

    # Final cleanup
    circles = _project_to_feasible(circles)
    circles = _local_search_refinement(circles)
    circles = _project_to_feasible(circles)

    return circles

# EVOLVE-BLOCK-END