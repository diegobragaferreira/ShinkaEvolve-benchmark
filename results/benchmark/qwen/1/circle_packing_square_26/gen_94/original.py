# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

def _initialize_voronoi_placement(n_circles: int, seed: int = 42) -> np.ndarray:
    """Initialize circle positions using a Voronoi-based approach."""
    np.random.seed(seed)

    # Start with a regular grid of points
    grid_size = int(np.ceil(np.sqrt(n_circles)))
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
        # Use Voronoi vertices as circle centers (but ensure they're within bounds)
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

def _compute_forces(circles: np.ndarray) -> np.ndarray:
    """Compute net forces on each circle based on gravity and repulsion."""
    n = len(circles)
    forces = np.zeros((n, 2))
    
    # Gravity force towards center (0.5, 0.5)
    for i in range(n):
        x, y, r = circles[i]
        dx = 0.5 - x
        dy = 0.5 - y
        # Normalize and scale by inverse of distance squared (stronger pull when farther)
        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
        force_magnitude = 0.001 / (dist * dist)
        forces[i, 0] += force_magnitude * dx
        forces[i, 1] += force_magnitude * dy
    
    # Repulsion forces from other circles
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx*dx + dy*dy) + 1e-8
            
            # Only repel if circles are overlapping or very close
            if dist < r1 + r2 + 0.001:
                # Repulsion force magnitude inversely proportional to distance
                force_magnitude = 0.01 / (dist * dist + 1e-6)
                forces[i, 0] -= force_magnitude * dx
                forces[i, 1] -= force_magnitude * dy
                forces[j, 0] += force_magnitude * dx
                forces[j, 1] += force_magnitude * dy
    
    return forces

def _apply_boundary_constraints(circles: np.ndarray) -> np.ndarray:
    """Apply boundary constraints to circles."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Ensure circles stay within the unit square with margin
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    return circles

def _resolve_overlaps(circles: np.ndarray) -> np.ndarray:
    """Resolve overlaps by adjusting radii."""
    n = len(circles)
    for i in range(n):
        x1, y1, r1 = circles[i]
        for j in range(i+1, n):
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if dist < r1 + r2:
                # Calculate overlap amount
                overlap = (r1 + r2) - dist
                # Reduce both radii by half the overlap amount
                reduction = overlap / 2.0
                circles[i, 2] = max(0.001, r1 - reduction)
                circles[j, 2] = max(0.001, r2 - reduction)
    return circles

def _update_positions(circles: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Update circle positions based on computed forces."""
    forces = _compute_forces(circles)
    
    for i in range(len(circles)):
        x, y, r = circles[i]
        fx, fy = forces[i]
        
        # Update position with velocity (force) and damping
        new_x = x + fx * dt
        new_y = y + fy * dt
        
        # Apply boundary constraints
        new_x = np.clip(new_x, r, 1-r)
        new_y = np.clip(new_y, r, 1-r)
        
        circles[i] = [new_x, new_y, r]
    
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
    initial_centers = _initialize_voronoi_placement(n)
    
    # Set initial radii to small positive values
    for i in range(n):
        circles[i, :2] = initial_centers[i]
        circles[i, 2] = 0.05  # Small initial radius
    
    # Physics-based optimization
    for iteration in range(1000):
        # Apply boundary constraints
        circles = _apply_boundary_constraints(circles)
        
        # Resolve any existing overlaps
        circles = _resolve_overlaps(circles)
        
        # Update positions based on forces
        circles = _update_positions(circles, dt=0.005)
        
        # Occasionally try to increase radii slightly
        if iteration % 50 == 0:
            for i in range(n):
                x, y, r = circles[i]
                # Try to increase radius while maintaining constraints
                max_radius = min(x, y, 1-x, 1-y)
                if r < max_radius and iteration % 200 == 0:
                    circles[i, 2] = min(max_radius, r + 0.001)
    
    # Final cleanup and validation
    circles = _apply_boundary_constraints(circles)
    circles = _resolve_overlaps(circles)
    
    return circles

# EVOLVE-BLOCK-END