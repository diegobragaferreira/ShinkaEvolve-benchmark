# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

def _initialize_voronoi_placement(n_circles: int, seed: int = 42) -> np.ndarray:
    """Initialize circle positions using Voronoi-based approach with better distribution."""
    np.random.seed(seed)
    
    # Create a more sophisticated grid pattern
    grid_size = max(5, int(np.ceil(np.sqrt(n_circles)) * 1.2))
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
        # Use the original points as circle centers (more reliable than vertices)
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

def _compute_energy_gradient(circles: np.ndarray) -> np.ndarray:
    """Compute gradient of energy function for optimization."""
    n = len(circles)
    gradients = np.zeros((n, 2))
    
    # Energy components:
    # 1. Repulsion energy between overlapping circles
    # 2. Boundary penalty for circles outside unit square
    # 3. Center attraction energy
    
    # Repulsion forces for overlaps
    for i in range(n):
        x1, y1, r1 = circles[i]
        
        # Attraction to center
        dx_center = 0.5 - x1
        dy_center = 0.5 - y1
        gradients[i, 0] += dx_center * 0.01
        gradients[i, 1] += dy_center * 0.01
        
        # Repulsion from other circles
        for j in range(n):
            if i != j:
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy) + 1e-8
                
                if dist < r1 + r2:
                    # Repulsion force when overlapping
                    force_magnitude = 1.0 / (dist * dist + 1e-6)
                    gradients[i, 0] -= force_magnitude * dx * 0.1
                    gradients[i, 1] -= force_magnitude * dy * 0.1
    
    # Boundary penalty forces
    for i in range(n):
        x, y, r = circles[i]
        # Boundary forces - push away from edges
        boundary_force_x = 0.0
        boundary_force_y = 0.0
        
        if x < r:
            boundary_force_x += (r - x) * 20.0
        elif x > 1 - r:
            boundary_force_x -= (x - (1 - r)) * 20.0
            
        if y < r:
            boundary_force_y += (r - y) * 20.0
        elif y > 1 - r:
            boundary_force_y -= (y - (1 - r)) * 20.0
            
        gradients[i, 0] += boundary_force_x
        gradients[i, 1] += boundary_force_y
    
    return gradients

def _project_to_feasible_region(circles: np.ndarray) -> np.ndarray:
    """Project circles to feasible region while preserving relative positions."""
    n = len(circles)
    # Create a working copy
    result = circles.copy()
    
    # Project each circle to stay within bounds
    for i in range(n):
        x, y, r = result[i]
        # Ensure minimum distance from boundaries
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        result[i] = [x, y, r]
    
    return result

def _resolve_overlaps_projective(circles: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Resolve overlaps using a projective approach that moves circles to resolve conflicts."""
    n = len(circles)
    result = circles.copy()
    
    # Create KDTree for efficient neighbor queries
    tree = cKDTree(result[:, :2])
    
    for _ in range(iterations):
        # Find all overlapping pairs
        pairs = tree.query_pairs(r=0.001)  # Very small threshold for nearby points
        adjusted = False
        
        # Process each pair to resolve overlap
        for i, j in pairs:
            if i >= n or j >= n:
                continue
            x1, y1, r1 = result[i]
            x2, y2, r2 = result[j]
            
            # Calculate separation distance
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx*dx + dy*dy) + 1e-8
            
            if dist < r1 + r2:
                # Calculate overlap amount
                overlap = (r1 + r2) - dist
                
                # Move circles apart along the line connecting their centers
                if dist > 1e-8:
                    move_factor = overlap / (dist * 2.0)
                    dx_norm = dx / dist
                    dy_norm = dy / dist
                    
                    # Move them apart
                    result[i, 0] -= dx_norm * move_factor
                    result[i, 1] -= dy_norm * move_factor
                    result[j, 0] += dx_norm * move_factor
                    result[j, 1] += dy_norm * move_factor
                    
                    # Reduce radii to compensate for overlap
                    result[i, 2] = max(0.001, r1 - overlap * 0.3)
                    result[j, 2] = max(0.001, r2 - overlap * 0.3)
                    adjusted = True
        
        if not adjusted:
            break
    
    return result

def _energy_based_optimization_step(circles: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Perform one optimization step based on energy gradient."""
    gradients = _compute_energy_gradient(circles)
    
    # Update positions
    result = circles.copy()
    for i in range(len(result)):
        result[i, 0] -= gradients[i, 0] * alpha
        result[i, 1] -= gradients[i, 1] * alpha
    
    # Project to feasible region
    result = _project_to_feasible_region(result)
    
    return result

def _adaptive_radius_enlargement(circles: np.ndarray, max_iterations: int = 10) -> np.ndarray:
    """Attempt to increase radii while maintaining feasibility."""
    result = circles.copy()
    n = len(result)
    
    # Try to increase radii iteratively
    for iteration in range(max_iterations):
        improved = False
        # Process in random order for better exploration
        indices = list(range(n))
        np.random.shuffle(indices)
        
        for i in indices:
            x, y, r = result[i]
            # Calculate maximum possible radius at current position
            max_radius = min(x, y, 1-x, 1-y)
            
            if r < max_radius and r < 0.5:  # Don't allow too large radii
                # Check if we can safely increase radius
                can_increase = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = result[j]
                        dist = np.sqrt((x2-x)**2 + (y2-y)**2)
                        
                        # Check if increasing radius would create conflict
                        if dist < r + r2 + 0.001:
                            can_increase = False
                            break
                
                if can_increase:
                    # Increase radius by small amount
                    new_r = min(max_radius, r + 0.0015)
                    result[i, 2] = new_r
                    improved = True
        
        if not improved:
            break
    
    return result

def _validate_solution(circles: np.ndarray) -> bool:
    """Validate solution constraints."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        # Check containment
        if r > x or r > y or r > 1-x or r > 1-y:
            return False
        # Check overlaps with previous circles
        for j in range(i):
            x2, y2, r2 = circles[j]
            dist_sq = (x-x2)**2 + (y-y2)**2
            min_dist_sq = (r+r2)**2
            if dist_sq < min_dist_sq:
                return False
    return True

def _compute_total_radius(circles: np.ndarray) -> float:
    """Compute sum of all radii."""
    return np.sum(circles[:, 2])

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = np.zeros((n, 3))
    
    # Phase 1: Initialize with Voronoi-based approach
    initial_centers = _initialize_voronoi_placement(n)
    
    # Set initial radii to small positive values
    for i in range(n):
        circles[i, :2] = initial_centers[i]
        circles[i, 2] = 0.03  # Initial small radius
    
    # Phase 2: Multi-stage optimization 
    best_sum = 0.0
    best_circles = circles.copy()
    
    # Coarse optimization phase
    for phase in range(3):
        lr = 0.01 if phase == 0 else 0.005 if phase == 1 else 0.001
        iterations = 200 if phase == 0 else 150 if phase == 1 else 100
        
        for iteration in range(iterations):
            # Optimization step
            circles = _energy_based_optimization_step(circles, lr)
            
            # Resolve overlaps
            circles = _resolve_overlaps_projective(circles, 3)
            
            # Adaptive radius enlargement
            circles = _adaptive_radius_enlargement(circles, 2)
            
            # Project to feasible region
            circles = _project_to_feasible_region(circles)
            
            # Track best solution
            current_sum = _compute_total_radius(circles)
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
    
    # Phase 3: Fine-tuning with iterative improvement
    # Try several refinement passes
    for _ in range(5):
        # Apply several rounds of energy optimization
        for _ in range(50):
            circles = _energy_based_optimization_step(circles, 0.001)
            circles = _resolve_overlaps_projective(circles, 2)
            circles = _adaptive_radius_enlargement(circles, 1)
            circles = _project_to_feasible_region(circles)
        
        # Evaluate and keep best
        current_sum = _compute_total_radius(circles)
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    # Final validation and refinement
    final_circles = _project_to_feasible_region(best_circles)
    final_circles = _resolve_overlaps_projective(final_circles, 10)
    final_circles = _adaptive_radius_enlargement(final_circles, 5)
    
    # Double-check constraints
    if not _validate_solution(final_circles):
        # Fallback to best known solution
        final_circles = best_circles.copy()
    
    return final_circles

# EVOLVE-BLOCK-END