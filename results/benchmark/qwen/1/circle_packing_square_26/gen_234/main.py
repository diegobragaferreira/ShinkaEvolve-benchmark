# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import time
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses Voronoi-based initial placement followed by physics simulation and refinement.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates
                 of the i-th circle of radius r.
    """
    n_circles = 26
    max_radius = 0.5
    min_radius = 0.001

    # Generate improved Voronoi-based configuration
    def generate_improved_voronoi_initial():
        np.random.seed(42)
        points = []

        # Better approach: Use iterative refinement to improve initial distribution
        best_points = None
        best_min_distance = 0

        # Try several configurations to find a good distribution
        for attempt in range(10):
            # Create a mixture of structured and random points
            candidate_points = []

            # Grid-based points for better coverage
            grid_size = 7
            spacing = 1.0 / (grid_size + 1)
            for i in range(grid_size):
                for j in range(grid_size):
                    x = (i + 1) * spacing + np.random.uniform(-spacing/6, spacing/6)
                    y = (j + 1) * spacing + np.random.uniform(-spacing/6, spacing/6)
                    candidate_points.append([x, y])

            # Add random points for diversity
            for _ in range(20):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                candidate_points.append([x, y])

            candidate_points = np.array(candidate_points)

            # Cluster to get evenly distributed points
            kmeans = KMeans(n_clusters=n_circles, init='k-means++', n_init=5,
                          random_state=42, max_iter=300)
            kmeans.fit(candidate_points)
            clustered_points = kmeans.cluster_centers_

            # Calculate minimum distance between clustered points
            if len(clustered_points) >= 2:
                distances = cdist(clustered_points, clustered_points)
                np.fill_diagonal(distances, np.inf)
                min_dist = np.min(distances)
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_points = clustered_points.copy()

        # If we couldn't find a good configuration, fall back to simple approach
        if best_points is None:
            # Create a grid of seed points with some randomness
            grid_size = 6
            spacing = 1.0 / (grid_size + 1)

            for i in range(grid_size):
                for j in range(grid_size):
                    x = (i + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                    y = (j + 1) * spacing + np.random.uniform(-spacing/4, spacing/4)
                    points.append([x, y])

            points = np.array(points[:n_circles])
        else:
            points = best_points

        # Create Voronoi diagram and compute areas for better circle placement
        try:
            vor = Voronoi(points)

            # For each Voronoi cell, place a circle at the centroid with radius based on cell area
            circles = []
            for i in range(n_circles):
                if i < len(vor.points):
                    # Get the Voronoi cell vertices for area calculation
                    # For simplicity, use the distance to nearest neighbors as proxy for area
                    x, y = vor.points[i]

                    # Estimate radius based on minimum distance to neighbors
                    min_dist = float('inf')
                    for j in range(n_circles):
                        if i != j:
                            dist = distance.euclidean(vor.points[i], vor.points[j])
                            min_dist = min(min_dist, dist)

                    # Use a more sophisticated formula for radius estimation
                    # Larger cells (more distant neighbors) should have larger radii
                    estimated_radius = min(0.25, min_dist / 3.0) if min_dist < float('inf') else 0.1

                    # Also consider local density - if neighbors are close, use smaller radius
                    if min_dist < 0.2:
                        estimated_radius *= 0.7

                    # Ensure it's within bounds and reasonable
                    radius = max(min_radius, min(max_radius, estimated_radius))

                    # Make sure it's within unit square
                    x = max(radius, min(1-radius, x))
                    y = max(radius, min(1-radius, y))

                    circles.append([x, y, radius])

            return np.array(circles)

        except Exception:
            # Fallback to simpler approach if Voronoi fails
            circles = []
            for i in range(n_circles):
                x = np.random.uniform(0.1, 0.9)
                y = np.random.uniform(0.1, 0.9)
                r = np.random.uniform(0.05, 0.2)
                circles.append([x, y, r])
            return np.array(circles)

    # Enhanced physics simulation to optimize circle packing
    def optimize_with_enhanced_physics(circles, max_iterations=500, dt=0.01):
        # Convert to arrays for easier computation
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()

        # Physics constants
        repulsion_strength = 20.0
        boundary_strength = 100.0
        damping = 0.97
        max_velocity = 0.05

        for iteration in range(max_iterations):
            # Compute forces between circles
            forces = np.zeros_like(positions)

            # Circle-circle repulsion (inverse distance squared)
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
                    if dist < (r_i + r_j) and dist > 0.0001:
                        # Repulsion force (inverse square law, scaled)
                        force_mag = repulsion_strength / (dist * dist + 0.001)

                        # Normalize and apply
                        force = force_mag * diff / dist
                        forces[i] += force
                        forces[j] -= force  # Newton's third law

            # Boundary repulsion (stronger force near boundaries)
            for i in range(len(positions)):
                pos = positions[i]
                r = radii[i]

                # Forces from four walls (with stronger repulsion for very close)
                left_force = boundary_strength * max(0, (r - pos[0]))**2
                right_force = boundary_strength * max(0, (pos[0] + r - 1))**2
                bottom_force = boundary_strength * max(0, (r - pos[1]))**2
                top_force = boundary_strength * max(0, (pos[1] + r - 1))**2

                forces[i][0] += left_force - right_force
                forces[i][1] += bottom_force - top_force

            # Apply forces and update positions
            for i in range(len(positions)):
                # Limit maximum velocity (avoid oscillation)
                velocity = forces[i] * dt
                vel_norm = np.linalg.norm(velocity)
                if vel_norm > max_velocity:
                    velocity = velocity * max_velocity / vel_norm

                positions[i] += velocity
                positions[i] *= damping  # Apply damping

                # Ensure positions stay within bounds
                positions[i][0] = np.clip(positions[i][0], radii[i], 1 - radii[i])
                positions[i][1] = np.clip(positions[i][1], radii[i], 1 - radii[i])

            # Check for convergence with stricter criteria
            if iteration > 20 and np.all(np.abs(forces) < 1e-5):
                break

        # Update circles with optimized positions
        optimized_circles = circles.copy()
        optimized_circles[:, :2] = positions
        return optimized_circles

    # Improved refinement function to maximize radii while maintaining constraints
    def refine_radii_improved(circles, max_iter=100):
        # More sophisticated refinement approach:
        # Try to increase all radii simultaneously while respecting constraints
        circles = circles.copy()

        # Sort circles by their current radius (largest first) to prioritize increasing larger radii
        sort_indices = np.argsort(circles[:, 2])[::-1]

        for iter_num in range(max_iter):
            improved = False

            # Try to increase each circle's radius in order of current size
            for i in sort_indices:
                # Save original values
                old_pos = circles[i, :2].copy()
                old_radius = circles[i, 2]

                # Try to increase radius by a larger amount for better improvement
                new_radius = min(old_radius * 1.1, 0.45)  # 10% increase max

                # Temporarily adjust the radius
                circles[i, 2] = new_radius

                # Check if this configuration is still valid
                if is_valid_configuration(circles):
                    improved = True
                else:
                    # Restore original values if invalid
                    circles[i, 2] = old_radius
                    circles[i, :2] = old_pos

            # If no improvements were made, stop
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

        # Check overlap constraints with early termination
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]

        # Use efficient pairwise comparison with early exit
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
    # Step 1: Generate improved Voronoi-based configuration
    initial_circles = generate_improved_voronoi_initial()

    # Step 2: Optimize using enhanced physics simulation
    optimized_circles = optimize_with_enhanced_physics(initial_circles)

    # Step 3: Refine radii to maximize sum
    final_circles = refine_radii_improved(optimized_circles)

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