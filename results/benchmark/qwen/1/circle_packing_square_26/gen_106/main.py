# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    n = 26
    circles = np.zeros((n, 3))

    # Step 1: Initialize with Voronoi-inspired placement
    # Generate points using a hexagonal grid pattern for better coverage
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)

    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n:
                # Add slight perturbation for better distribution
                x = (i + 1) * spacing + random.uniform(-spacing/4, spacing/4)
                y = (j + 1) * spacing + random.uniform(-spacing/4, spacing/4)
                points.append([x, y])

    # Fill remaining positions if needed
    while len(points) < n:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        points.append([x, y])

    # Step 2: Assign initial radii based on available space
    for i in range(n):
        x, y = points[i]
        # Compute max possible radius at this position
        max_r = min(x, y, 1-x, 1-y)

        # Find minimum distance to existing circles
        min_dist = float('inf')
        for j in range(i):
            x2, y2 = points[j]
            dist = np.sqrt((x - x2)**2 + (y - y2)**2)
            min_dist = min(min_dist, dist)

        # Set radius based on proximity to others and boundary constraints
        if min_dist == float('inf'):
            r = max_r * 0.3  # No neighbors, use boundary constraint
        else:
            r = min(max_r, min_dist * 0.3)  # Balance between boundary and neighbor constraints

        # Ensure reasonable minimum radius
        r = max(0.01, min(0.2, r))
        circles[i] = [x, y, r]

    # Step 3: Local optimization using gradient-based approach
    def is_valid(circs):
        """Check if all circles are valid (containment and non-overlap)"""
        for i in range(len(circs)):
            x, y, r = circs[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False

        # Check non-overlap with KDTree for efficiency
        try:
            points = circs[:, :2]
            tree = cKDTree(points)
            pairs = tree.query_pairs(0, return_distance=False)
            for i, j in pairs:
                if i < j:
                    x1, y1, r1 = circs[i]
                    x2, y2, r2 = circs[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False
        except Exception:
            # Fallback to brute force if KDTree fails
            for i in range(len(circs)):
                x1, y1, r1 = circs[i]
                for j in range(i+1, len(circs)):
                    x2, y2, r2 = circs[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return False

        return True

    def calculate_forces(circs):
        """Calculate repulsive forces between overlapping circles"""
        forces = np.zeros_like(circs[:, :2])

        for i in range(len(circs)):
            x1, y1, r1 = circs[i]
            for j in range(len(circs)):
                if i != j:
                    x2, y2, r2 = circs[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < r1 + r2:
                        # Repel force
                        if dist > 0.001:
                            force_magnitude = (r1 + r2 - dist) * 0.1
                            forces[i, 0] += dx / dist * force_magnitude
                            forces[i, 1] += dy / dist * force_magnitude
                    elif dist < (r1 + r2 + 0.02):  # Near-contact force
                        if dist > 0.001:
                            force_magnitude = (r1 + r2 + 0.02 - dist) * 0.01
                            forces[i, 0] -= dx / dist * force_magnitude
                            forces[i, 1] -= dy / dist * force_magnitude

        return forces

    def optimize_step(circs, learning_rate=0.01, max_iterations=100):
        """Perform gradient-based optimization to maximize radii sum"""
        for _ in range(max_iterations):
            # Calculate forces
            forces = calculate_forces(circs)

            # Apply forces with clipping to maintain validity
            improved = False
            for i in range(len(circs)):
                x, y, r = circs[i]
                fx, fy = forces[i]

                # Move circle
                new_x = x + fx * learning_rate
                new_y = y + fy * learning_rate

                # Clip to valid positions
                new_x = np.clip(new_x, r, 1 - r)
                new_y = np.clip(new_y, r, 1 - r)

                # Accept change if it improves validity
                temp_circs = circs.copy()
                temp_circs[i] = [new_x, new_y, r]

                if is_valid(temp_circs):
                    circs[i] = [new_x, new_y, r]
                    improved = True

            # Break early if no improvement
            if not improved:
                break

        return circs

    # Apply optimization
    circles = optimize_step(circles)

    # Final validation and repair
    if not is_valid(circles):
        # Simple repair - adjust positions to satisfy constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Fix containment
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            circles[i] = [x, y, r]

        # Ensure no overlaps
        for iter_count in range(10):
            changed = False
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(i):
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    if dist < r1 + r2:
                        # Separate circles
                        dx = x2 - x1
                        dy = y2 - y1
                        if dx == 0 and dy == 0:
                            dx = 1
                            dy = 1
                        length = np.sqrt(dx*dx + dy*dy)
                        dx /= length
                        dy /= length

                        separation = (r1 + r2) - dist
                        circles[i][0] -= dx * separation * 0.5
                        circles[i][1] -= dy * separation * 0.5
                        circles[j][0] += dx * separation * 0.5
                        circles[j][1] += dy * separation * 0.5
                        changed = True

            if not changed:
                break

    return circles


# EVOLVE-BLOCK-END