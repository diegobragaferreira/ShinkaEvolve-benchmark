# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility

    n = 32
    circles = np.zeros((n, 3))

    # Step 1: Initialize positions using improved hexagonal grid pattern
    # Better hexagonal packing approach for 32 circles in unit square
    rows = 6
    cols = 6

    # Calculate optimal hexagonal spacing for better packing efficiency
    # Estimate radius based on area requirement
    total_area_needed = 32 * (1.0/32)**2  # Each circle needs area for 1/32 of square
    estimated_radius = np.sqrt(total_area_needed / (np.pi * 32))  # Rough estimate

    # Refine spacing based on hexagonal packing theory (optimal packing density ~0.9069)
    # For n circles in unit square, spacing should be such that we maximize utilization
    spacing_x = 2 * estimated_radius
    spacing_y = spacing_x * np.sqrt(3) / 2

    # Create hexagonal grid that fits well within unit square
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Hexagonal offset pattern
            x = (j + (i % 2) * 0.5) * spacing_x
            y = i * spacing_y

            # Only include positions that fit within unit square with margin for radius
            if x + estimated_radius <= 1.0 and y + estimated_radius <= 1.0:
                positions.append([x, y])
        if len(positions) >= n:
            break

    # Fill remaining positions with carefully placed random points
    # Ensure they don't overlap with existing hexagonal positions
    if len(positions) < n:
        # Add random positions with better spatial distribution
        for i in range(n - len(positions)):
            # Place randomly but avoid clustering near edges initially
            x = np.random.uniform(estimated_radius, 1.0 - estimated_radius)
            y = np.random.uniform(estimated_radius, 1.0 - estimated_radius)
            positions.append([x, y])

    # Adjust to exactly n positions
    positions = positions[:n]

    # Initialize with equal small radii
    for i in range(n):
        circles[i, 0] = positions[i][0]  # x
        circles[i, 1] = positions[i][1]  # y
        circles[i, 2] = 0.02  # initial small radius

    # Step 2: Crowd-based optimization with progressive refinement
    max_iterations = 500
    for iteration in range(max_iterations):
        # Update pairwise distances
        positions = circles[:, :2]  # (n, 2) array of positions
        radii = circles[:, 2]       # (n,) array of radii

        # Compute distance matrix
        dist_matrix = cdist(positions, positions)

        # Apply repulsion forces between overlapping circles
        force_magnitudes = np.zeros((n, 2))

        for i in range(n):
            for j in range(i+1, n):
                if dist_matrix[i, j] < (radii[i] + radii[j]):  # Overlapping
                    # Calculate normalized vector from i to j
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist_ij = max(np.sqrt(dx*dx + dy*dy), 1e-8)  # Prevent division by zero

                    # Repulsion force magnitude (stronger when more overlapping)
                    overlap = radii[i] + radii[j] - dist_ij
                    force_mag = overlap * 0.01

                    # Normalize and apply force
                    force_magnitudes[i, 0] -= force_mag * dx / dist_ij
                    force_magnitudes[i, 1] -= force_mag * dy / dist_ij
                    force_magnitudes[j, 0] += force_mag * dx / dist_ij
                    force_magnitudes[j, 1] += force_mag * dy / dist_ij

        # Apply forces to positions
        step_size = 0.01
        for i in range(n):
            new_x = positions[i, 0] + force_magnitudes[i, 0] * step_size
            new_y = positions[i, 1] + force_magnitudes[i, 1] * step_size

            # Project back to valid region
            min_r = radii[i]
            new_x = np.clip(new_x, min_r, 1.0 - min_r)
            new_y = np.clip(new_y, min_r, 1.0 - min_r)

            circles[i, 0] = new_x
            circles[i, 1] = new_y

        # Step 3: Try to increase radii while maintaining constraints
        # Precompute distances for efficiency
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Attempt to grow radii gradually
        growth_factor = 0.005
        for attempt in range(50):  # Multiple attempts per iteration
            # Try to increase all radii slightly
            new_radii = radii.copy()
            for i in range(n):
                new_radius = radii[i] + growth_factor

                # Check containment
                if (new_radius > circles[i, 0] or
                    new_radius > 1.0 - circles[i, 0] or
                    new_radius > circles[i, 1] or
                    new_radius > 1.0 - circles[i, 1]):
                    continue  # Cannot grow further in this direction

                # Check overlap with all others
                valid_growth = True
                for j in range(n):
                    if i == j:
                        continue

                    # Distance between centers
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist_ij = np.sqrt(dx*dx + dy*dy)

                    # If too close to other circles
                    if dist_ij < (new_radius + circles[j, 2]):
                        valid_growth = False
                        break

                if valid_growth:
                    new_radii[i] = new_radius

            # Update if improvement was found
            if np.any(new_radii > radii):
                radii = new_radii
                circles[:, 2] = radii
            else:
                break  # No more improvements possible

        # Occasionally reduce step size for better convergence
        if iteration % 100 == 0:
            step_size *= 0.8

    # Final refinement to make sure we're not missing anything
    # Try aggressive local optimization
    for _ in range(20):
        # Try to shrink radii and see if it helps with overlaps
        for i in range(n):
            old_r = circles[i, 2]
            old_x, old_y = circles[i, 0], circles[i, 1]

            # Try to shrink radius slightly and reposition to improve others
            test_r = old_r * 0.99
            if test_r > 0.001:  # Minimum radius threshold
                # Check if shrinking helps
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = old_x - circles[j, 0]
                    dy = old_y - circles[j, 1]
                    dist_ij = np.sqrt(dx*dx + dy*dy)
                    if dist_ij < (test_r + circles[j, 2]):
                        valid = False
                        break

                if valid:
                    circles[i, 2] = test_r

    return circles

# EVOLVE-BLOCK-END