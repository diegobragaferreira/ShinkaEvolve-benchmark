# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import random
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    N_CIRCLES = 26
    MAX_ITERATIONS = 1000
    TOLERANCE = 1e-6
    INITIAL_RADIUS_SCALE = 0.1
    
    def create_voronoi_initialization(n_circles):
        """Create initial configuration using Voronoi-based distribution with better spatial awareness."""
        # Generate initial points using a structured approach
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        points = []
        
        # Create regular grid points
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n_circles:
                    # Add hexagonal offset for better distribution
                    offset = (j % 2) * spacing_x / 2
                    x = (j + 1) * spacing_x + offset
                    y = (i + 1) * spacing_y
                    points.append([x, y])
        
        # Add additional boundary points for better edge coverage
        boundary_points = []
        for _ in range(15):
            # Random points near edges
            side = np.random.randint(0, 4)
            if side == 0:  # Top
                boundary_points.append([np.random.uniform(0.1, 0.9), 0.99])
            elif side == 1:  # Bottom
                boundary_points.append([np.random.uniform(0.1, 0.9), 0.01])
            elif side == 2:  # Left
                boundary_points.append([0.01, np.random.uniform(0.1, 0.9)])
            else:  # Right
                boundary_points.append([0.99, np.random.uniform(0.1, 0.9)])
        
        points.extend(boundary_points)
        points = points[:n_circles]
        
        # Convert to numpy array and apply small random perturbations
        points = np.array(points)
        points += np.random.normal(0, 0.02, points.shape) * 0.5
        
        # Clip to valid range
        points = np.clip(points, 0.01, 0.99)
        
        # Create Voronoi diagram to analyze spatial relationships
        try:
            vor = Voronoi(points)
            # Use Voronoi cell centers as primary positions
            centroids = vor.points[vor.point_region[:-1]]
            centroids = centroids[:n_circles]
        except:
            # Fallback to direct points if Voronoi fails
            centroids = points[:n_circles]
        
        # Create initial circle configuration with radii based on Voronoi properties
        circles = np.zeros((n_circles, 3))
        
        # Compute radii based on Voronoi cell sizes and local density
        for i in range(n_circles):
            x, y = centroids[i]
            
            # Calculate distances to nearest neighbors
            distances = []
            for j in range(n_circles):
                if i != j:
                    dist = np.sqrt((x - centroids[j][0])**2 + (y - centroids[j][1])**2)
                    distances.append(dist)
            
            # Estimate appropriate initial radius
            if distances:
                # Use minimum distance to neighbors for better spacing
                min_distance = np.min(distances)
                # Radius proportional to local density
                radius = min(min_distance * 0.3, 0.2)  # Avoid very large radii
            else:
                radius = 0.1
            
            # Respect square boundaries
            boundary_radius = min(x, 1-x, y, 1-y)
            radius = min(radius, boundary_radius * 0.8)
            radius = max(0.005, min(0.15, radius))
            
            circles[i] = [x, y, radius]
            
        return circles
    
    def is_valid_configuration(circles):
        """Check if configuration is valid (no overlaps, fully contained)."""
        n = len(circles)
        
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if (r <= 0 or x < r or x > 1-r or y < r or y > 1-r):
                return False
        
        # Check overlaps with early termination
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    return False
                    
        return True
    
    def calculate_total_radius(circles):
        """Calculate sum of all radii."""
        return np.sum(circles[:, 2])
    
    def compute_forces(circles):
        """Compute net forces on each circle based on repulsion and boundary constraints."""
        n = len(circles)
        forces = np.zeros((n, 2))
        
        # Repulsion forces between circles
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        # Minimum distance without overlap
                        min_dist = r1 + r2
                        
                        # Only apply force if circles are overlapping or close
                        if dist < min_dist * 1.5:
                            if dist < min_dist * 0.1:
                                # Strong repulsion when very close
                                force_magnitude = 1000 / (dist + 1e-8)
                            else:
                                # Normal repulsion
                                force_magnitude = 10 / (dist + 1e-8)
                            
                            # Normalize force direction
                            dx_norm = dx / (dist + 1e-8)
                            dy_norm = dy / (dist + 1e-8)
                            
                            forces[i, 0] += dx_norm * force_magnitude
                            forces[i, 1] += dy_norm * force_magnitude
        
        # Boundary forces (spring-like constraints)
        for i in range(n):
            x, y, r = circles[i]
            
            # Left boundary
            if x < r:
                forces[i, 0] += (r - x) * 100
            # Right boundary
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 100
            # Bottom boundary
            if y < r:
                forces[i, 1] += (r - y) * 100
            # Top boundary
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 100
        
        return forces
    
    def update_positions(circles, forces, dt=0.01):
        """Update circle positions using computed forces."""
        updated = circles.copy()
        
        for i in range(len(updated)):
            x, y, r = updated[i]
            
            # Apply force to velocity (simplified physics)
            new_x = x + forces[i, 0] * dt
            new_y = y + forces[i, 1] * dt
            
            # Ensure positions remain within valid range
            new_x = np.clip(new_x, r, 1 - r)
            new_y = np.clip(new_y, r, 1 - r)
            
            updated[i] = [new_x, new_y, r]
            
        return updated
    
    def maximize_radii(circles):
        """Increase radii as much as possible while maintaining constraints."""
        updated = circles.copy()
        
        for iteration in range(100):
            improved = False
            
            # For each circle, try to increase its radius
            for i in range(len(updated)):
                x, y, r = updated[i]
                
                # Calculate maximum possible radius
                max_radius = min(x, 1-x, y, 1-y)
                
                # Find minimum distance to other centers
                min_dist = float('inf')
                for j in range(len(updated)):
                    if i != j:
                        x2, y2, _ = updated[j]
                        dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                        min_dist = min(min_dist, dist)
                
                # Safe maximum radius (accounting for overlap)
                if min_dist < float('inf') and min_dist > 0:
                    # Maximum radius without overlap
                    safe_radius = min_dist * 0.5
                    max_radius = min(max_radius, safe_radius)
                
                # Try to increase radius using binary search
                if max_radius > r:
                    # Binary search for maximum safe radius
                    low, high = r, max_radius
                    best_radius = r
                    
                    # Binary search iterations
                    for _ in range(15):
                        test_radius = (low + high) / 2
                        
                        # Check if this radius works
                        valid = True
                        for j in range(len(updated)):
                            if i != j:
                                x2, y2, r2 = updated[j]
                                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                                if dist < (test_radius + r2):
                                    valid = False
                                    break
                        
                        if valid:
                            best_radius = test_radius
                            low = test_radius
                        else:
                            high = test_radius
                    
                    if best_radius > r:
                        updated[i, 2] = best_radius
                        improved = True
            
            if not improved:
                break
                
        return updated
    
    def optimize_with_gravity(circles):
        """Optimize using gravity-inspired physics-based approach."""
        # Initial optimization
        optimized = circles.copy()
        
        # Apply force relaxation
        for iteration in range(500):
            forces = compute_forces(optimized)
            optimized = update_positions(optimized, forces)
            
            # Periodic radius maximization
            if iteration % 20 == 0:
                optimized = maximize_radii(optimized)
            
            # Check for convergence
            if iteration > 50:
                # Check if movement is minimal
                total_movement = np.sum(np.abs(forces))
                if total_movement < TOLERANCE:
                    break
                
        return optimized
    
    # Main optimization procedure
    
    # Step 1: Create initial configuration using Voronoi-based approach
    initial_config = create_voronoi_initialization(N_CIRCLES)
    
    # Step 2: Validate and fix initial configuration
    if not is_valid_configuration(initial_config):
        # Fallback to grid if initial is invalid
        grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        
        initial_config = np.zeros((N_CIRCLES, 3))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < N_CIRCLES:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    # Adjust for boundary constraints
                    x = np.clip(x, r, 1 - r)
                    y = np.clip(y, r, 1 - r)
                    initial_config[count] = [x, y, r]
                    count += 1
    
    # Step 3: Apply physics-based optimization
    final_config = optimize_with_gravity(initial_config)
    
    # Step 4: Final refinement with local optimization
    final_config = maximize_radii(final_config)
    
    # Final validation
    if not is_valid_configuration(final_config):
        # Last resort fallback
        grid_size = int(np.ceil(np.sqrt(N_CIRCLES)))
        spacing = 1.0 / grid_size
        r = spacing * 0.3
        
        final_config = np.zeros((N_CIRCLES, 3))
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count < N_CIRCLES:
                    x = (j + 0.5) * spacing
                    y = (i + 0.5) * spacing
                    x = np.clip(x, r, 1 - r)
                    y = np.clip(y, r, 1 - r)
                    final_config[count] = [x, y, r]
                    count += 1
    
    return final_config

# EVOLVE-BLOCK-END