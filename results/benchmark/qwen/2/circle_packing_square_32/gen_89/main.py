# EVOLVE-BLOCK-START
import numpy as np

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))

    # More sophisticated hexagonal grid initialization
    # For 32 circles, we want to create a hexagonal close packing arrangement

    # Calculate optimal hexagonal packing parameters
    # In hexagonal packing, the packing density is π/(2√3) ≈ 0.9069
    # For n circles in unit square, the total area covered by circles should be <= 1
    # So π*r²*n <= 1, thus r_max <= sqrt(1/(π*n))

    # Estimate initial radius based on optimal packing
    estimated_radius = np.sqrt(1.0 / (np.pi * n))

    # For hexagonal arrangement, let's use a more systematic approach
    # We'll arrange in a triangular lattice pattern
    sqrt_n = int(np.ceil(np.sqrt(n)))

    # Use a hexagonal pattern that minimizes wasted space
    # Try to make a roughly square-ish hexagonal packing
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    # Ensure we don't exceed 32 circles
    if rows * cols > n:
        cols = int(np.ceil(n / rows))
        if rows * cols > n:
            rows = int(np.ceil(n / cols))

    # Create hexagonal grid with proper spacing for circle packing
    # Hexagonal packing spacing (center-to-center distance)
    spacing = 2 * estimated_radius  # Distance between centers in hexagonal packing

    # Adjust spacing to fit within unit square
    max_width = 1.0
    max_height = 1.0

    # Calculate how many circles we can fit in each direction
    actual_cols = int(max_width / spacing)
    actual_rows = int(max_height / (spacing * np.sqrt(3)/2))

    # Make sure we don't exceed our required count
    actual_cols = min(actual_cols, cols)
    actual_rows = min(actual_rows, rows)

    # If we don't have enough circles, increase spacing slightly to accommodate
    if actual_cols * actual_rows < n:
        # Recalculate spacing to fit exactly n circles
        total_area_needed = n * np.pi * estimated_radius**2
        avg_spacing = np.sqrt(total_area_needed / (0.9069 * 1.0))  # Using packing density

        # But keep it reasonable and based on the actual number of rows/columns
        spacing = 1.0 / max(actual_cols, actual_rows)

    # Generate positions in hexagonal pattern
    idx = 0
    hex_offset = spacing * 0.5

    for i in range(actual_rows):
        for j in range(actual_cols):
            if idx >= n:
                break

            # Calculate position
            x = (j + 0.5) * spacing
            y = (i + 0.5) * spacing * np.sqrt(3)/2

            # Apply hexagonal offset for odd rows
            if i % 2 == 1:
                x += hex_offset

            # Adjust to fit in unit square
            x = max(estimated_radius, min(1.0 - estimated_radius, x))
            y = max(estimated_radius, min(1.0 - estimated_radius, y))

            # Store position and initial radius
            circles[idx, 0] = x
            circles[idx, 1] = y
            circles[idx, 2] = estimated_radius  # Use estimated radius as initial value

            idx += 1

        if idx >= n:
            break

    # Fill remaining positions if needed (shouldn't be necessary with correct logic above)
    for i in range(idx, n):
        circles[i, 0] = 0.5
        circles[i, 1] = 0.5
        circles[i, 2] = estimated_radius

    return circles


# EVOLVE-BLOCK-END