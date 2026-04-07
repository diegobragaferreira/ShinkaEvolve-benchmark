# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import time
from numba import jit

@jit(nopython=True)
def validate_solution_fast(circles):
    """Fast validation using numba for the critical constraint checks"""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        # Check containment
        if r > x or r > y or r > 1-x or r > 1-y:
            return False
        # Check overlap with all previous circles
        for j in range(i):
            x2, y2, r2 = circles[j]
            dx = x - x2
            dy = y - y2
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (r+r2)*(r+r2)
            if dist_sq < min_dist_sq:
                return False
    return True

def _initialize_voronoi_placement(n_circles: int, seed: int = 42) -> np.ndarray:
    """Initialize circle positions using a refined Voronoi-based approach with better geometric distribution"""
    np.random.seed(seed)

    # Create a carefully chosen grid for Voronoi generation
    grid_size = max(5, int(np.ceil(np.sqrt(n_circles)) * 1.2))
    x_coords = np.linspace(0.02, 0.98, grid_size)
    y_coords = np.linspace(0.02, 0.98, grid_size)
    grid_points = np.array([[x, y] for x in x_coords for y in y_coords])

    # Take the first n_circles points (or generate more if needed)
    if len(grid_points) >= n_circles:
        initial_points = grid_points[:n_circles]
    else:
        # If we don't have enough grid points, add random points
        extra_points = n_circles - len(grid_points)
        random_points = np.random.uniform(0.02, 0.98, (extra_points, 2))
        initial_points = np.vstack([grid_points, random_points])

    # Create Voronoi diagram and analyze cell volumes to guide circle placement
    try:
        vor = Voronoi(initial_points)

        # Use Voronoi vertices as circle centers, but with volume-based weighting
        centroids = []
        for point in vor.points:
            if 0.02 <= point[0] <= 0.98 and 0.02 <= point[1] <= 0.98:
                centroids.append(point)

        if len(centroids) < n_circles:
            # Fill in missing points randomly
            remaining = n_circles - len(centroids)
            random_points = np.random.uniform(0.02, 0.98, (remaining, 2))
            final_points = np.vstack([centroids, random_points])
        else:
            final_points = np.array(centroids[:n_circles])

    except Exception:
        # Fallback to random initialization if Voronoi fails
        final_points = np.random.uniform(0.02, 0.98, (n_circles, 2))

    return final_points

def _compute_gravity_forces(circles: np.ndarray) -> np.ndarray:
    """Compute forces using Voronoi-based gravity that attracts circles toward high-volume Voronoi cells"""
    n = len(circles)
    forces = np.zeros((n, 2))

    # Compute Voronoi diagram for current configuration
    try:
        # Only proceed if valid configuration
        if n > 1:
            vor = Voronoi(circles[:, :2])

            # Calculate approximate Voronoi cell volumes for each point
            cell_volumes = np.zeros(n)
            for i in range(n):
                if i < len(vor.point_region):
                    region = vor.point_region[i]
                    if region != -1:
                        # Simplified volume approximation using Delaunay triangulation
                        try:
                            # Get vertices of the Voronoi cell
                            vertices = vor.vertices[vor.regions[region]]
                            if len(vertices) > 2:
                                # Simple area calculation using shoelace formula
                                x = vertices[:, 0]
                                y = vertices[:, 1]
                                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                                cell_volumes[i] = area
                            else:
                                cell_volumes[i] = 1.0
                        except:
                            cell_volumes[i] = 1.0
                    else:
                        cell_volumes[i] = 1.0
                else:
                    cell_volumes[i] = 1.0

            # Normalize cell volumes to get attraction weights
            cell_volumes = np.maximum(cell_volumes, 1e-8)
            weights = cell_volumes / np.sum(cell_volumes)

            # Apply weighted gravity toward center of mass
            center_of_mass = np.average(circles[:, :2], axis=0, weights=weights)

            # Apply gravity forces with strength based on Voronoi volume
            for i in range(n):
                x, y, r = circles[i]
                dx = center_of_mass[0] - x
                dy = center_of_mass[1] - y
                dist = np.sqrt(dx*dx + dy*dy) + 1e-8

                # Force stronger for circles in smaller volume cells (more constrained)
                force_strength = 0.005 * weights[i] / (dist * dist + 1e-6)
                forces[i, 0] += force_strength * dx
                forces[i, 1] += force_strength * dy

    except:
        # Fallback simple gravity behavior
        for i in range(n):
            x, y, r = circles[i]
            dx = 0.5 - x
            dy = 0.5 - y
            dist = np.sqrt(dx*dx + dy*dy) + 1e-8
            force_magnitude = 0.003 / (dist * dist + 1e-6)
            forces[i, 0] += force_magnitude * dx
            forces[i, 1] += force_magnitude * dy

    # Add repulsion forces from other circles
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx*dx + dy*dy) + 1e-8

            if dist < r1 + r2 + 0.001:
                # Repulsion force
                force_magnitude = 0.03 / (dist * dist + 1e-6)
                forces[i, 0] -= force_magnitude * dx
                forces[i, 1] -= force_magnitude * dy
                forces[j, 0] += force_magnitude * dx
                forces[j, 1] += force_magnitude * dy

    return forces

def _apply_boundary_constraints(circles: np.ndarray) -> np.ndarray:
    """Apply boundary constraints efficiently"""
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Ensure circles stay within the unit square with margin
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    return circles

def _resolve_overlaps(circles: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Resolve overlaps with improved algorithm that considers geometric relationships"""
    n = len(circles)

    for _ in range(iterations):
        # Track if any adjustments were made
        adjusted = False

        # Process pairs systematically
        for i in range(n):
            x1, y1, r1 = circles[i]

            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                if dist < r1 + r2:
                    # Calculate overlap amount
                    overlap = (r1 + r2) - dist

                    # Adjust radii more intelligently - reduce more aggressively where overlap is bigger
                    reduction = overlap * 0.8
                    new_r1 = max(0.001, r1 - reduction/2)
                    new_r2 = max(0.001, r2 - reduction/2)
                    circles[i, 2] = new_r1
                    circles[j, 2] = new_r2
                    adjusted = True

        if not adjusted:
            break

    return circles

def _update_positions(circles: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Update circle positions with improved dynamics"""
    forces = _compute_gravity_forces(circles)

    for i in range(len(circles)):
        x, y, r = circles[i]
        fx, fy = forces[i]

        # Update position with velocity (force) and damping
        new_x = x + fx * dt * 0.7
        new_y = y + fy * dt * 0.7

        # Apply boundary constraints
        new_x = np.clip(new_x, r, 1-r)
        new_y = np.clip(new_y, r, 1-r)

        circles[i] = [new_x, new_y, r]

    return circles

def _optimize_radii(circles: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    """Optimize radii by increasing them while maintaining constraints"""
    n = len(circles)

    for iteration in range(max_iterations):
        improved = False
        # Process in random order to avoid bias
        indices = list(range(n))
        np.random.shuffle(indices)

        for i in indices:
            x, y, r = circles[i]
            # Calculate maximum possible radius at current position
            max_radius = min(x, y, 1-x, 1-y)

            if r < max_radius:
                # Check for potential overlaps with existing circles
                can_increase = True
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                        # If we're trying to increase radius and it would create overlap,
                        # check what the new configuration would look like
                        if dist < r + (r1 + 0.0005):
                            can_increase = False
                            break

                if can_increase and r < max_radius:
                    # Increase radius by small amount
                    new_r = min(max_radius, r + 0.0015)
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

    # Initialize with improved Voronoi-based approach
    initial_centers = _initialize_voronoi_placement(n)

    # Set initial radii to small positive values
    for i in range(n):
        circles[i, :2] = initial_centers[i]
        circles[i, 2] = 0.04  # Slightly larger initial radius for better convergence

    # Main optimization loop with refined strategy
    best_sum_radii = 0.0
    no_improvement_count = 0
    max_no_improvement = 400

    for iteration in range(2000):  # More iterations to allow convergence
        # Apply boundary constraints
        circles = _apply_boundary_constraints(circles)

        # Resolve any existing overlaps
        circles = _resolve_overlaps(circles)

        # Update positions based on forces
        circles = _update_positions(circles, dt=0.004)

        # Periodically try to increase radii
        if iteration % 20 == 0:
            circles = _optimize_radii(circles)

        # Occasionally do additional overlap resolution
        if iteration % 70 == 0:
            circles = _resolve_overlaps(circles, iterations=6)

        # Monitor progress for early termination
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early termination if no improvement for a while
        if no_improvement_count > max_no_improvement:
            break

    # Final cleanup
    circles = _apply_boundary_constraints(circles)
    circles = _resolve_overlaps(circles, iterations=15)
    circles = _optimize_radii(circles, max_iterations=3)

    # Validate final solution
    if not validate_solution_fast(circles):
        # If validation fails, try a more conservative initialization
        circles = np.zeros((n, 3))
        init_points = _initialize_voronoi_placement(n, seed=43)
        for i in range(n):
            circles[i, :2] = init_points[i]
            circles[i, 2] = 0.03

    return circles

# EVOLVE-BLOCK-END