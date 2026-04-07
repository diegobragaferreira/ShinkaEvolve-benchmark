# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

def _initialize_voronoi_placement(n_circles: int, seed: int = 42) -> np.ndarray:
    """Initialize circle positions using a Voronoi-based approach with better distribution."""
    np.random.seed(seed)
    
    # Generate a more refined grid for better spread
    grid_size = max(4, int(np.ceil(np.sqrt(n_circles)) + 1))
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
    """Compute net forces on each circle based on gravity and repulsion with improved scaling."""
    n = len(circles)
    forces = np.zeros((n, 2))
    
    # Gravity force towards center (0.5, 0.5) - stronger pull for distant circles
    for i in range(n):
        x, y, r = circles[i]
        dx = 0.5 - x
        dy = 0.5 - y
        dist = np.sqrt(dx*dx + dy*dy) + 1e-8
        # Scale force inversely with distance to center (stronger pull when farther)
        force_magnitude = 0.002 / (dist * dist + 1e-6)
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
                force_magnitude = 0.02 / (dist * dist + 1e-6)
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

def _resolve_overlaps(circles: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Resolve overlaps by adjusting radii with iterative refinement."""
    n = len(circles)
    
    for _ in range(iterations):
        # Track if any adjustments were made
        adjusted = False
        
        # Process pairs of circles in a consistent order
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if dist < r1 + r2:
                    # Calculate overlap amount
                    overlap = (r1 + r2) - dist
                    
                    # Reduce both radii proportionally (more aggressive adjustment)
                    reduction = overlap * 0.7  # More aggressive than before
                    circles[i, 2] = max(0.001, r1 - reduction/2)
                    circles[j, 2] = max(0.001, r2 - reduction/2)
                    adjusted = True
        
        # Early termination if no adjustments made
        if not adjusted:
            break
    
    return circles

def _update_positions(circles: np.ndarray, dt: float = 0.01) -> np.ndarray:
    """Update circle positions based on computed forces with momentum."""
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

def _try_increase_radii(circles: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    """Try to increase radii safely without causing overlaps."""
    n = len(circles)
    
    # Try multiple times to increase radii
    for iteration in range(max_iterations):
        improved = False
        for i in range(n):
            x, y, r = circles[i]
            # Calculate maximum possible radius at current position
            max_radius = min(x, y, 1-x, 1-y)
            
            # Only attempt increase if there's room and it won't cause immediate overlap
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
                        if dist < r + (r1 + 0.001):  # Add small buffer
                            can_increase = False
                            break
                
                if can_increase:
                    # Increase radius by small amount
                    new_r = min(max_radius, r + 0.001)
                    circles[i, 2] = new_r
                    improved = True
        
        # Stop early if no improvements made
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
    initial_centers = _initialize_voronoi_placement(n)
    
    # Set initial radii to small positive values
    for i in range(n):
        circles[i, :2] = initial_centers[i]
        circles[i, 2] = 0.05  # Small initial radius
    
    # Physics-based optimization with adaptive parameters
    best_sum_radii = 0.0
    no_improvement_count = 0
    
    for iteration in range(1500):  # Increased iterations
        # Apply boundary constraints
        circles = _apply_boundary_constraints(circles)
        
        # Resolve any existing overlaps
        circles = _resolve_overlaps(circles)
        
        # Update positions based on forces
        circles = _update_positions(circles, dt=0.005)
        
        # Periodically try to increase radii
        if iteration % 30 == 0:
            circles = _try_increase_radii(circles)
        
        # Occasionally do additional overlap resolution
        if iteration % 100 == 0:
            circles = _resolve_overlaps(circles, iterations=5)
        
        # Monitor progress for early termination
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Early termination if no improvement for a while
        if no_improvement_count > 200:
            break
    
    # Final cleanup and validation
    circles = _apply_boundary_constraints(circles)
    circles = _resolve_overlaps(circles, iterations=10)
    
    return circles

# EVOLVE-BLOCK-END