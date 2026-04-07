# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import time
from sklearn.cluster import KMeans

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses enhanced Voronoi-based initial placement followed by physics simulation and local optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    n_circles = 26
    max_radius = 0.5
    min_radius = 0.001

    # Generate initial Voronoi-based configuration with better distribution
    def generate_voronoi_initial():
        # More sophisticated Voronoi seed point generation
        np.random.seed(42)
        
        # Method 1: Multiple clustering attempts to find best distribution
        best_config = None
        best_min_dist = 0
        
        for attempt in range(10):
            # Generate candidate points
            candidate_points = []
            for _ in range(100):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                candidate_points.append([x, y])
            
            candidate_points = np.array(candidate_points)
            
            # Use KMeans clustering to find good seed points
            try:
                kmeans = KMeans(n_clusters=n_circles, init='k-means++', n_init=5,
                              random_state=42, max_iter=100)
                kmeans.fit(candidate_points)
                clustered_points = kmeans.cluster_centers_
                
                # Calculate minimum distance between clustered points
                if len(clustered_points) >= 2:
                    distances = cdist(clustered_points, clustered_points)
                    np.fill_diagonal(distances, np.inf)
                    min_dist = np.min(distances)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_config = clustered_points.copy()
            except:
                continue
        
        # Fallback to simple grid if k-means fails
        if best_config is None:
            grid_size = 6
            spacing = 1.0 / (grid_size + 1)
            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    x = (i + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                    y = (j + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                    points.append([x, y])
            best_config = np.array(points[:n_circles])
        
        # Create Voronoi diagram from best configuration
        vor = Voronoi(best_config)
        
        # Generate circles with better radius estimation based on Voronoi cells
        circles = []
        for i in range(n_circles):
            if i < len(vor.points):
                x, y = vor.points[i]
                
                # Better radius calculation using Voronoi cell properties
                # Find nearest neighbor to estimate cell size
                min_dist = float('inf')
                for j in range(n_circles):
                    if i != j:
                        dist = distance.euclidean(vor.points[i], vor.points[j])
                        min_dist = min(min_dist, dist)
                
                # Estimate radius based on Voronoi cell area or neighbor spacing
                estimated_radius = min(0.2, min_dist / 3.0) if min_dist < float('inf') else 0.1
                
                # Ensure reasonable bounds
                radius = max(min_radius, min(max_radius, estimated_radius))
                
                # Ensure within bounds
                x = max(radius, min(1-radius, x))
                y = max(radius, min(1-radius, y))
                
                circles.append([x, y, radius])
            else:
                # Fallback to random
                x = np.random.uniform(min_radius, 1-min_radius)
                y = np.random.uniform(min_radius, 1-min_radius)
                r = np.random.uniform(min_radius, max_radius)
                circles.append([x, y, r])
        
        return np.array(circles)

    # Physics simulation with improved convergence
    def optimize_with_physics(circles, max_iterations=1000, dt=0.01):
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        # Physics constants
        repulsion_strength = 50.0
        boundary_strength = 100.0
        damping = 0.95
        max_velocity = 0.05
        
        for iteration in range(max_iterations):
            # Compute forces between circles
            forces = np.zeros_like(positions)
            total_energy = 0
            
            # Circle-circle repulsion
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    pos_i = positions[i]
                    pos_j = positions[j]
                    r_i = radii[i]
                    r_j = radii[j]
                    
                    # Distance vector
                    diff = pos_i - pos_j
                    dist = np.linalg.norm(diff)
                    
                    # Only repel if overlapping or nearly touching
                    if dist < (r_i + r_j):
                        # Repulsion force (inverse square law)
                        force_mag = repulsion_strength / (dist * dist + 0.001)
                        
                        # Normalize and apply
                        if dist > 0:
                            force = force_mag * diff / dist
                        else:
                            force = np.array([0, 0])
                        
                        forces[i] += force
                        forces[j] -= force
                        
                        # Energy calculation for convergence detection
                        total_energy += abs(force_mag)
            
            # Boundary repulsion (stronger force near boundaries)
            for i in range(len(positions)):
                pos = positions[i]
                r = radii[i]
                
                # Forces from four walls
                left_force = max(0, (r - pos[0])) * boundary_strength
                right_force = max(0, (pos[0] + r - 1)) * boundary_strength
                bottom_force = max(0, (r - pos[1])) * boundary_strength
                top_force = max(0, (pos[1] + r - 1)) * boundary_strength
                
                forces[i][0] += left_force - right_force
                forces[i][1] += bottom_force - top_force
                
                # Add to total energy
                total_energy += abs(left_force) + abs(right_force) + abs(bottom_force) + abs(top_force)

            # Apply forces and update positions
            for i in range(len(positions)):
                # Limit maximum velocity
                velocity = forces[i] * dt
                vel_norm = np.linalg.norm(velocity)
                if vel_norm > max_velocity:
                    velocity = velocity * max_velocity / vel_norm
                
                positions[i] += velocity
                positions[i] *= damping  # Apply damping
                
                # Ensure positions stay within bounds
                positions[i][0] = np.clip(positions[i][0], radii[i], 1 - radii[i])
                positions[i][1] = np.clip(positions[i][1], radii[i], 1 - radii[i])

            # Check for convergence using energy threshold
            if total_energy < 1e-6:
                break

        # Update circles with optimized positions
        optimized_circles = circles.copy()
        optimized_circles[:, :2] = positions
        return optimized_circles

    # Local optimization to maximize sum of radii
    def refine_radii(circles, max_iter=100):
        circles = circles.copy()
        
        # Binary search for optimal radius for each circle
        for iter_num in range(max_iter):
            improved = False
            for i in range(len(circles)):
                x, y, r = circles[i]
                
                # Find maximum possible radius while maintaining constraints
                # Binary search approach
                left = r
                right = min(0.5, 1-x, x, 1-y, y)  # Upper bound based on boundaries
                best_radius = r
                
                # Check if we can increase the radius
                if right > r:
                    # Test various radius values to find maximum
                    test_values = np.linspace(r, right, 20)
                    for test_r in test_values:
                        # Temporarily update radius
                        temp_circles = circles.copy()
                        temp_circles[i, 2] = test_r
                        
                        # Check if this would violate constraints
                        valid = True
                        for j in range(len(temp_circles)):
                            if i != j:
                                x1, y1, r1 = temp_circles[i]
                                x2, y2, r2 = temp_circles[j]
                                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                                if dist < r1 + r2:
                                    valid = False
                                    break
                        
                        if valid:
                            best_radius = test_r
                            improved = True
                        
                        # Early termination if we have good solution already
                        if test_r > best_radius * 1.01:
                            break
                
                # Apply the best radius if it improves the total
                if best_radius > r:
                    circles[i, 2] = best_radius
                    improved = True
            
            if not improved:
                break
        
        return circles

    # Constraint checking function
    def is_valid_configuration(circles_array):
        # Check containment constraints
        for x, y, r in circles_array:
            if not (r >= min_radius and
                   r <= x <= 1 - r and
                   r <= y <= 1 - r):
                return False

        # Check overlap constraints using efficient pairwise checking
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]

        n = len(circles_array)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1, r2 = radii[i], radii[j]

                dist_squared = (x1 - x2)**2 + (y1 - y2)**2
                radius_sum = r1 + r2

                if dist_squared < radius_sum**2:  # Overlapping
                    return False

        return True

    # Main algorithm
    # Step 1: Generate initial configuration using enhanced Voronoi method
    initial_circles = generate_voronoi_initial()

    # Step 2: Optimize using physics simulation with proper convergence detection
    optimized_circles = optimize_with_physics(initial_circles)

    # Step 3: Refine radii to maximize sum through local optimization
    final_circles = refine_radii(optimized_circles)

    # Final validation and cleanup
    if not is_valid_configuration(final_circles):
        # Fall back to a more robust initialization
        final_circles = np.zeros((n_circles, 3))
        # Use a more systematic approach
        grid_size = 6
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Set radius to ensure some margin for overlap
                r = min(spacing_x, spacing_y) * 0.25
                final_circles[idx] = [x, y, r]
                idx += 1
        
        # Add extra circles with random placement
        for i in range(idx, n_circles):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.005, 0.05)
            final_circles[i] = [x, y, r]

    return final_circles

# EVOLVE-BLOCK-END