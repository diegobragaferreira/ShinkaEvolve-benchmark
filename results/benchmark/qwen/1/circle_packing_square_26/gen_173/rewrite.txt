# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a physics-inspired Voronoi gravity optimization approach that simulates forces between
    circles to find optimal packings, avoiding expensive genetic operations.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)
    
    N_CIRCLES = 26
    MAX_ITERATIONS = 1000
    DAMPING_FACTOR = 0.1
    FORCE_CONSTANT = 1.0
    BOUNDARY_STIFFNESS = 100.0
    MIN_RADIUS = 0.001
    MAX_RADIUS = 0.5
    
    def create_voronoi_initialization():
        """Create initial configuration using Voronoi-based approach"""
        # Generate points in a hexagonal grid pattern
        grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
        points = []
        
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < N_CIRCLES:
                    offset = (j % 2) * spacing_x / 2
                    x = (i + 1) * spacing_x + offset
                    y = (j + 1) * spacing_y
                    
                    # Add jitter for better distribution
                    x += np.random.uniform(-spacing_x/6, spacing_x/6)
                    y += np.random.uniform(-spacing_y/6, spacing_y/6)
                    
                    # Ensure within bounds
                    x = np.clip(x, 0.01, 0.99)
                    y = np.clip(y, 0.01, 0.99)
                    
                    points.append([x, y])
        
        # If we don't have enough points, add random ones
        while len(points) < N_CIRCLES:
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            points.append([x, y])
        
        points = np.array(points[:N_CIRCLES])
        
        try:
            # Compute Voronoi diagram
            vor = Voronoi(points)
            
            # Get Voronoi cell centroids as initial positions
            centroids = []
            for i, (x, y) in enumerate(vor.points):
                if i < len(vor.point_region) and vor.point_region[i] >= 0:
                    region = vor.regions[vor.point_region[i]]
                    if len(region) > 0 and all(r >= 0 for r in region):
                        vertices = np.array([vor.vertices[r] for r in region])
                        if len(vertices) > 0:
                            centroid = np.mean(vertices, axis=0)
                            centroid[0] = np.clip(centroid[0], 0.01, 0.99)
                            centroid[1] = np.clip(centroid[1], 0.01, 0.99)
                            centroids.append(centroid)
            
            if len(centroids) < N_CIRCLES:
                centroids = points[:N_CIRCLES].tolist()
            else:
                centroids = centroids[:N_CIRCLES]
                
            # Calculate initial radii based on Voronoi cell properties
            circles = np.zeros((N_CIRCLES, 3))
            for i, (cx, cy) in enumerate(centroids):
                # Calculate minimum distance to other points
                min_dist = float('inf')
                for j, (px, py) in enumerate(centroids):
                    if i != j:
                        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                        min_dist = min(min_dist, dist)
                
                # Set radius based on Voronoi cell size
                if min_dist > 0:
                    r = min(0.15, min_dist/3.0)
                else:
                    r = np.random.uniform(0.01, 0.05)
                
                # Enforce bounds
                r = max(MIN_RADIUS, min(MAX_RADIUS, r))
                circles[i] = [cx, cy, r]
                
        except Exception:
            # Fallback to simple initialization
            circles = np.zeros((N_CIRCLES, 3))
            for i in range(N_CIRCLES):
                x = np.random.uniform(0.01, 0.99)
                y = np.random.uniform(0.01, 0.99)
                r = np.random.uniform(0.01, 0.1)
                circles[i] = [x, y, r]
        
        return circles
    
    def is_valid_solution(circles):
        """Check if solution satisfies all constraints efficiently"""
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check non-overlap using KDTree
        try:
            points = circles[:, :2]
            tree = KDTree(points)
            pairs = tree.query_pairs(0, return_distance=False)
            
            for i, j in pairs:
                if i < j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        except Exception:
            # Fallback to brute force
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i+1, len(circles)):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        
        return True
    
    def calculate_forces(circles):
        """Calculate net forces on each circle from all other circles and boundaries"""
        forces = np.zeros_like(circles[:, :2])
        
        # Repulsion forces between circles
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0 and dist < (r1 + r2):
                        # Overlapping - strong repulsion
                        force_magnitude = FORCE_CONSTANT * (r1 + r2 - dist) / dist
                    elif dist > 0:
                        # Non-overlapping - inverse distance repulsion
                        force_magnitude = FORCE_CONSTANT / (dist * dist)
                    else:
                        force_magnitude = 0.0
                    
                    forces[i, 0] += force_magnitude * dx
                    forces[i, 1] += force_magnitude * dy
        
        # Attraction to Voronoi cell centers (if Voronoi can be computed)
        try:
            points = circles[:, :2]
            vor = Voronoi(points)
            
            # For each circle, add attraction towards its Voronoi cell center
            for i in range(len(circles)):
                if i < len(vor.point_region) and vor.point_region[i] >= 0:
                    region = vor.regions[vor.point_region[i]]
                    if len(region) > 0 and all(r >= 0 for r in region):
                        vertices = np.array([vor.vertices[r] for r in region])
                        if len(vertices) > 0:
                            # Calculate centroid of Voronoi cell (simplified)
                            voronoi_center = np.mean(vertices, axis=0)
                            
                            # Attract to Voronoi center
                            dx = voronoi_center[0] - circles[i, 0]
                            dy = voronoi_center[1] - circles[i, 1]
                            
                            # Apply attractive force (weaker than repulsion)
                            attraction_force = 0.1 * np.sqrt(dx*dx + dy*dy)
                            forces[i, 0] -= attraction_force * dx / (np.sqrt(dx*dx + dy*dy) + 1e-10)
                            forces[i, 1] -= attraction_force * dy / (np.sqrt(dx*dx + dy*dy) + 1e-10)
        except:
            # If Voronoi fails, skip attraction forces
            pass
        
        # Boundary forces - push circles back into valid region
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Left boundary
            if x - r < 0:
                forces[i, 0] += BOUNDARY_STIFFNESS * (r - x)
            
            # Right boundary
            if x + r > 1:
                forces[i, 0] += BOUNDARY_STIFFNESS * (1 - r - x)
            
            # Bottom boundary  
            if y - r < 0:
                forces[i, 1] += BOUNDARY_STIFFNESS * (r - y)
            
            # Top boundary
            if y + r > 1:
                forces[i, 1] += BOUNDARY_STIFFNESS * (1 - r - y)
        
        return forces
    
    def update_positions_and_radii(circles, forces, dt=0.01):
        """Update positions and radii based on forces"""
        # Apply forces to positions
        for i in range(len(circles)):
            dx = forces[i, 0] * dt
            dy = forces[i, 1] * dt
            
            # Apply damping
            circles[i, 0] += dx * DAMPING_FACTOR
            circles[i, 1] += dy * DAMPING_FACTOR
            
            # Keep within bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        return circles
    
    def maximize_radii(circles):
        """Try to maximize radii while maintaining constraints"""
        improved = False
        
        for i in range(len(circles)):
            orig_x, orig_y, orig_r = circles[i]
            
            # Find minimum distance to other circles
            min_dist_to_others = float('inf')
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                    min_dist_to_others = min(min_dist_to_others, dist)
            
            # Calculate maximum possible radius
            max_new_radius = min_dist_to_others - 0.001 if min_dist_to_others > 0.001 else orig_r
            
            if max_new_radius > orig_r:
                # Binary search for optimal radius improvement
                low, high = orig_r, max_new_radius
                best_radius = orig_r
                
                # Binary search for best radius
                for _ in range(10):
                    test_r = (low + high) / 2
                    
                    # Test validity of configuration with this radius
                    temp_circles = circles.copy()
                    temp_circles[i, 2] = test_r
                    
                    if is_valid_solution(temp_circles):
                        best_radius = test_r
                        low = test_r
                    else:
                        high = test_r
                
                if best_radius > orig_r:
                    circles[i, 2] = best_radius
                    improved = True
        
        return circles, improved
    
    def local_search_improvement(circles):
        """Perform local search to improve radii and positions"""
        improved = False
        
        # Try to increase radii
        for i in range(len(circles)):
            orig_x, orig_y, orig_r = circles[i]
            
            # Find minimum distance to other circles
            min_dist_to_others = float('inf')
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((orig_x - x2)**2 + (orig_y - y2)**2)
                    min_dist_to_others = min(min_dist_to_others, dist)
            
            # Calculate maximum possible radius
            max_new_radius = min_dist_to_others - 0.001 if min_dist_to_others > 0.001 else orig_r
            
            if max_new_radius > orig_r:
                # Try to increase radius
                target_r = min(max_new_radius, orig_r + 0.01)
                target_r = max(MIN_RADIUS, min(MAX_RADIUS, target_r))
                
                # Check if valid
                temp_circles = circles.copy()
                temp_circles[i, 2] = target_r
                
                if is_valid_solution(temp_circles):
                    circles[i, 2] = target_r
                    improved = True
        
        # Try small position adjustments
        for i in range(len(circles)):
            orig_x, orig_y, orig_r = circles[i]
            
            # Try several small moves
            for _ in range(5):
                new_x = orig_x + np.random.uniform(-0.005, 0.005)
                new_y = orig_y + np.random.uniform(-0.005, 0.005)
                
                # Clip to bounds
                new_x = np.clip(new_x, orig_r, 1 - orig_r)
                new_y = np.clip(new_y, orig_r, 1 - orig_r)
                
                # Check validity
                temp_circles = circles.copy()
                temp_circles[i, 0] = new_x
                temp_circles[i, 1] = new_y
                
                if is_valid_solution(temp_circles):
                    circles[i, 0] = new_x
                    circles[i, 1] = new_y
                    improved = True
                    break
        
        return circles, improved
    
    # Initialize solution
    circles = create_voronoi_initialization()
    
    # Main optimization loop
    for iteration in range(MAX_ITERATIONS):
        # Calculate forces
        forces = calculate_forces(circles)
        
        # Update positions
        circles = update_positions_and_radii(circles, forces, dt=0.01)
        
        # Occasionally try to maximize radii
        if iteration % 50 == 0:
            circles, improved = maximize_radii(circles)
            if not improved and iteration > 200:
                # Try local search if no improvement for a while
                circles, improved = local_search_improvement(circles)
                if not improved:
                    # Early termination if no improvement
                    break
    
    # Final refinement
    for _ in range(100):
        circles, improved = maximize_radii(circles)
        if not improved:
            break
    
    # Final local search
    for _ in range(50):
        circles, improved = local_search_improvement(circles)
        if not improved:
            break
    
    # Final validation
    if not is_valid_solution(circles):
        # Try one more optimization pass if needed
        for _ in range(20):
            forces = calculate_forces(circles)
            circles = update_positions_and_radii(circles, forces, dt=0.005)
    
    return circles

# EVOLVE-BLOCK-END