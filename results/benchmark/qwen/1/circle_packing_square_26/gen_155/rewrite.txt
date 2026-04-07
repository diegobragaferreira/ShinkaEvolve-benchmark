# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses Voronoi-based initial placement followed by gravity-field optimization.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    n_circles = 26
    max_radius = 0.5
    min_radius = 0.001

    # Generate initial Voronoi-based configuration
    def generate_voronoi_initial():
        # Create seed points using hexagonal grid pattern for better coverage
        np.random.seed(42)
        points = []
        grid_size = 6
        spacing = 1.0 / (grid_size + 1)
        hex_spacing = spacing * 0.866  # sqrt(3)/2 for hexagonal packing

        for i in range(grid_size):
            for j in range(grid_size):
                # Create hexagonal offset pattern
                x = (i + 1) * spacing + np.random.uniform(-spacing/6, spacing/6)
                y = (j + 1) * spacing + np.random.uniform(-spacing/6, spacing/6)
                # Offset every other row for hexagonal pattern
                if i % 2 == 1:
                    y += spacing/2
                points.append([x, y])

        # Ensure we have enough points
        while len(points) < n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])

        points = np.array(points[:n_circles])

        # Create Voronoi diagram
        vor = Voronoi(points)

        # For each Voronoi cell, place a circle with better radius estimation
        circles = []
        for i in range(n_circles):
            if i < len(vor.points):
                x, y = vor.points[i]

                # Estimate radius based on minimum neighbor distance
                min_dist = float('inf')
                for j in range(n_circles):
                    if i != j:
                        dist = distance.euclidean(vor.points[i], vor.points[j])
                        min_dist = min(min_dist, dist)

                # Conservative estimate: use 1/3 of minimum neighbor distance
                estimated_radius = min(0.25, min_dist / 3.0) if min_dist < float('inf') else 0.1

                # Ensure it's within bounds and reasonable
                radius = max(min_radius, min(max_radius, estimated_radius))

                # Make sure it's within unit square
                x = max(radius, min(1-radius, x))
                y = max(radius, min(1-radius, y))

                circles.append([x, y, radius])
            else:
                # Fallback for edge cases
                x = np.random.uniform(min_radius, 1-min_radius)
                y = np.random.uniform(min_radius, 1-min_radius)
                r = np.random.uniform(min_radius, max_radius)
                circles.append([x, y, r])

        return np.array(circles)

    # Gravity-based optimization approach
    def optimize_with_gravity(circles, max_iterations=500):
        # Convert to arrays for easier computation
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        # Optimization parameters
        learning_rate = 0.05
        damping = 0.95
        boundary_repulsion = 100.0
        circle_repulsion = 10.0
        attraction_strength = 5.0
        max_velocity = 0.02
        
        for iteration in range(max_iterations):
            # Compute net forces on each circle
            forces = np.zeros_like(positions)
            
            # Circle-circle repulsion and attraction forces
            for i in range(len(positions)):
                for j in range(len(positions)):
                    if i != j:
                        pos_i = positions[i]
                        pos_j = positions[j]
                        r_i = radii[i]
                        r_j = radii[j]
                        
                        # Distance vector
                        diff = pos_i - pos_j
                        dist = np.linalg.norm(diff)
                        
                        if dist > 0:
                            # Repulsion force (when circles are too close)
                            if dist < (r_i + r_j) * 1.2:  # Slightly overlapping
                                force_mag = circle_repulsion / (dist * dist + 0.001)
                                force = force_mag * diff / dist
                                forces[i] += force
                            
                            # Attraction force (toward nearby circles to encourage packing)
                            if dist < (r_i + r_j) * 3.0:  # Within attraction range
                                force_mag = -attraction_strength / (dist * dist + 0.001)
                                force = force_mag * diff / dist
                                forces[i] += force
            
            # Boundary repulsion
            for i in range(len(positions)):
                pos = positions[i]
                r = radii[i]
                
                # Forces from four walls (stronger near boundaries)
                left_force = max(0, (r - pos[0])) * boundary_repulsion
                right_force = max(0, (pos[0] + r - 1)) * boundary_repulsion
                bottom_force = max(0, (r - pos[1])) * boundary_repulsion
                top_force = max(0, (pos[1] + r - 1)) * boundary_repulsion
                
                forces[i][0] += left_force - right_force
                forces[i][1] += bottom_force - top_force
            
            # Apply forces and update positions
            for i in range(len(positions)):
                # Limit maximum velocity
                velocity = forces[i] * learning_rate
                vel_norm = np.linalg.norm(velocity)
                if vel_norm > max_velocity:
                    velocity = velocity * max_velocity / vel_norm
                
                positions[i] += velocity
                positions[i] *= damping  # Apply damping
                
                # Ensure positions stay within bounds
                positions[i][0] = np.clip(positions[i][0], radii[i], 1 - radii[i])
                positions[i][1] = np.clip(positions[i][1], radii[i], 1 - radii[i])
            
            # Check for convergence (small changes in positions)
            if np.all(np.abs(forces) < 1e-5):
                break
                
        # Update circles with optimized positions
        optimized_circles = circles.copy()
        optimized_circles[:, :2] = positions
        return optimized_circles

    # Constraint checking function
    def is_valid_configuration(circles_array):
        # Check containment constraints
        for x, y, r in circles_array:
            if not (r >= min_radius and
                   r <= x <= 1 - r and
                   r <= y <= 1 - r):
                return False

        # Check overlap constraints
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]

        # Use pair-wise distance checking
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

    # Refinement function to maximize radii while maintaining constraints
    def refine_radii(circles, max_iter=30):
        # Simple greedy refinement: try to increase radii one by one
        circles = circles.copy()

        for iter_num in range(max_iter):
            improved = False
            for i in range(len(circles)):
                # Try to increase this circle's radius
                old_radius = circles[i, 2]
                new_radius = min(old_radius * 1.02, 0.5)  # 2% increase max

                # Check if this radius works
                c_temp = circles.copy()
                c_temp[i, 2] = new_radius

                # Check if this still satisfies constraints
                if is_valid_configuration(c_temp):
                    circles = c_temp
                    improved = True

            # If no improvements were made, stop
            if not improved:
                break

        return circles

    # Main algorithm
    # Step 1: Generate initial Voronoi-based configuration
    initial_circles = generate_voronoi_initial()

    # Step 2: Optimize using gravity-based approach
    optimized_circles = optimize_with_gravity(initial_circles)

    # Step 3: Refine radii to maximize sum
    final_circles = refine_radii(optimized_circles)

    # Final validation
    if not is_valid_configuration(final_circles):
        # Fallback to grid method if something went wrong
        final_circles = np.zeros((n_circles, 3))
        grid_size = int(np.ceil(np.sqrt(n_circles)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n_circles:
                    break
                x = (i + 1) * spacing_x
                y = (j + 1) * spacing_y
                r = min(spacing_x, spacing_y) * 0.3
                final_circles[idx] = [x, y, r]
                idx += 1

        # Add remaining circles
        for i in range(idx, n_circles):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            final_circles[i] = [x, y, r]

    return final_circles

# EVOLVE-BLOCK-END