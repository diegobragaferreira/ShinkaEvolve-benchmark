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

    # Step 1: Enhanced hexagonal grid initialization for better space filling
    # Using a more systematic approach to create hexagonal packing
    # Calculate optimal parameters for hexagonal packing
    sqrt_3 = np.sqrt(3)

    # Estimate number of rows and columns needed for 32 circles
    # In hexagonal packing, we need roughly sqrt(n/sqrt(3)) in each dimension
    target_density = sqrt_3 / 2  # Hexagonal packing density
    estimated_rows_cols = int(np.ceil(np.sqrt(n / target_density)))

    # Create a denser hexagonal grid
    rows = estimated_rows_cols + 2  # Extra rows to ensure we have enough points
    cols = estimated_rows_cols + 2

    # Calculate proper hexagonal spacing based on circle radius
    # For optimal hexagonal packing, circles touch each other when spacing = 2*radius
    # But we need to account for square boundaries, so let's start with a reasonable radius
    max_radius_est = 0.15  # Initial guess for maximum radius that might work
    spacing_x = max_radius_est * 2.0
    spacing_y = max_radius_est * sqrt_3

    # Adjust spacing to fit in unit square
    # We want to fit at least the right number of circles
    actual_rows = int(np.floor(1.0 / spacing_y))
    actual_cols = int(np.floor(1.0 / spacing_x))

    # If we don't have enough circles, we expand spacing
    if actual_rows * actual_cols < n:
        # Increase spacing to get fewer but properly distributed circles
        spacing_x = 1.0 / (np.sqrt(n) * 0.8)
        spacing_y = spacing_x * sqrt_3
        actual_rows = int(np.floor(1.0 / spacing_y))
        actual_cols = int(np.floor(1.0 / spacing_x))

    # Generate hexagonal grid with proper offsetting
    positions = []
    for i in range(actual_rows):
        for j in range(actual_cols):
            # Offset every other row
            x_offset = (i % 2) * (spacing_x / 2)
            x = x_offset + j * spacing_x
            y = i * spacing_y

            # Only include points that are inside the unit square
            if x <= 1.0 and y <= 1.0:
                # Also check that the circle fits completely within the square
                if x >= max_radius_est and x <= 1.0 - max_radius_est and \
                   y >= max_radius_est and y <= 1.0 - max_radius_est:
                    positions.append([x, y])
                    if len(positions) >= n:
                        break
        if len(positions) >= n:
            break

    # Adjust to exactly n positions
    positions = positions[:n]

    # Initialize with better starting radii - use a more adaptive approach
    # Start with a radius that allows for good expansion potential
    initial_radius = min(0.08, 0.5 / np.sqrt(n))  # Scale based on problem size
    for i in range(n):
        circles[i, 0] = positions[i][0]  # x
        circles[i, 1] = positions[i][1]  # y
        circles[i, 2] = initial_radius   # initial radius

    # Step 2: Enhanced optimization with improved algorithms
    max_iterations = 1000

    # Precompute indices for faster access
    indices = np.arange(n)

    for iteration in range(max_iterations):
        # Update positions and radii
        positions = circles[:, :2]  # (n, 2) array of positions
        radii = circles[:, 2]       # (n,) array of radii

        # Apply repulsion forces using vectorized operations for better performance
        force_magnitudes = np.zeros((n, 2))

        # Vectorized computation of distance matrix
        dist_matrix = cdist(positions, positions)

        # Create masks for overlapping pairs
        radii_sum = np.add.outer(radii, radii)
        np.fill_diagonal(radii_sum, np.inf)

        # Find overlapping pairs
        overlap_mask = dist_matrix < radii_sum

        # Vectorized force calculation
        for i in range(n):
            # Get indices of overlapping circles
            overlap_indices = np.where(overlap_mask[i])[0]
            if len(overlap_indices) > 0:
                # Calculate forces from all overlapping circles
                for j in overlap_indices:
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist_ij = max(np.sqrt(dx*dx + dy*dy), 1e-10)

                        # Repulsion force (stronger when more overlapping)
                        overlap = radii[i] + radii[j] - dist_ij
                        if overlap > 0:  # Only apply force when overlapping
                            force_mag = overlap * 0.05

                            # Normalize and apply force
                            force_magnitudes[i, 0] -= force_mag * dx / dist_ij
                            force_magnitudes[i, 1] -= force_mag * dy / dist_ij

        # Apply forces to positions with adaptive step size
        step_size = 0.02
        if iteration > 500:
            step_size = 0.01
        elif iteration > 200:
            step_size = 0.015

        for i in range(n):
            new_x = positions[i, 0] + force_magnitudes[i, 0] * step_size
            new_y = positions[i, 1] + force_magnitudes[i, 1] * step_size

            # Project back to valid region
            min_r = radii[i]
            new_x = np.clip(new_x, min_r, 1.0 - min_r)
            new_y = np.clip(new_y, min_r, 1.0 - min_r)

            circles[i, 0] = new_x
            circles[i, 1] = new_y

        # Step 3: Improved radius maximization with better heuristics
        # Sort circles by radius (smallest first) to prioritize expanding smaller circles
        sorted_indices = np.argsort(circles[:, 2])

        # Attempt to increase radii with better strategy
        growth_factor = 0.003
        improved = True
        attempts = 0
        max_attempts = 100

        while improved and attempts < max_attempts:
            improved = False
            attempts += 1

            # Try to grow each circle in order of increasing radius
            for i in sorted_indices:
                current_radius = circles[i, 2]
                new_radius = current_radius + growth_factor

                # Check if new radius would violate containment
                if (new_radius > circles[i, 0] or
                    new_radius > 1.0 - circles[i, 0] or
                    new_radius > circles[i, 1] or
                    new_radius > 1.0 - circles[i, 1]):
                    continue  # Cannot grow further in this direction

                # Check overlap with all others using vectorized approach
                # Create a mask to avoid self-comparison
                valid_overlap = True
                test_positions = circles[:, :2]
                test_radii = circles[:, 2]

                # Compute distances to other circles
                dx = test_positions[:, 0] - circles[i, 0]
                dy = test_positions[:, 1] - circles[i, 1]
                distances = np.sqrt(dx*dx + dy*dy)

                # Check for overlaps with all other circles
                for j in range(n):
                    if i == j:
                        continue
                    if distances[j] < (new_radius + test_radii[j]):
                        valid_overlap = False
                        break

                if valid_overlap:
                    circles[i, 2] = new_radius
                    improved = True

        # Occasionally re-scale step size for better convergence
        if iteration % 200 == 0 and iteration > 0:
            growth_factor *= 0.95

    # Final validation and refinement
    # Try to squeeze radii back if there are minor overlaps
    positions = circles[:, :2]
    radii = circles[:, 2]

    # Perform one final optimization pass
    for _ in range(50):
        # Try to adjust all circles slightly
        for i in range(n):
            old_r = circles[i, 2]
            old_x, old_y = circles[i, 0], circles[i, 1]

            # Try very small adjustments
            test_r = old_r * 0.9999  # Slight shrinkage
            if test_r > 0.001:
                # Check if shrinking helps with constraints
                valid = True
                test_positions = circles[:, :2]
                test_radii = circles[:, 2]

                dx = test_positions[:, 0] - old_x
                dy = test_positions[:, 1] - old_y
                distances = np.sqrt(dx*dx + dy*dy)

                for j in range(n):
                    if i == j:
                        continue
                    if distances[j] < (test_r + test_radii[j]):
                        valid = False
                        break

                if valid:
                    circles[i, 2] = test_r

    return circles

# EVOLVE-BLOCK-END