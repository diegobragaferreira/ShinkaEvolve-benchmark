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

    # Improved hexagonal grid initialization
    # For 32 circles, we'll use a more efficient hexagonal packing approach

    # Calculate ideal hexagonal packing parameters
    # In a hexagonal lattice, the packing density is ~0.9069
    # For n circles in unit square, we want to maximize total area covered

    # Estimate optimal radius based on hexagonal packing
    # Area of n circles = n * π * r²
    # For hexagonal packing in unit square:
    # n * π * r² ≈ 0.9069 * 1  =>  r ≈ sqrt(0.9069/(n*π))
    estimated_radius = np.sqrt(0.9069 / (n * np.pi))

    # Better approach: place circles in a triangular lattice pattern
    # Calculate rows and columns for hexagonal arrangement
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    # Ensure we don't exceed 32 circles
    while rows * cols < n:
        rows += 1

    # Adjust to exactly 32 circles
    actual_count = rows * cols
    if actual_count > n:
        # Reduce rows if necessary
        while rows * cols > n:
            rows -= 1
            cols = int(np.ceil(n / rows))

    # Use proper hexagonal grid spacing
    # For hexagonal packing, spacing should be 2*r between centers
    spacing_x = 2 * estimated_radius
    spacing_y = 2 * estimated_radius * np.sqrt(3) / 2  # Height of equilateral triangle

    # Make sure spacing is reasonable (not too large)
    max_spacing = 0.3
    spacing_x = min(spacing_x, max_spacing)
    spacing_y = min(spacing_y, max_spacing)

    # Create hexagonal grid with proper offset
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break

            # Hexagonal offset for alternating rows
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y

            # Offset every other row
            if i % 2 == 1:
                x += spacing_x / 2.0

            # Adjust to fit within unit square with proper margins
            margin = estimated_radius
            x = margin + (1 - 2*margin) * (x / (cols * spacing_x))
            y = margin + (1 - 2*margin) * (y / (rows * spacing_y))

            # Ensure we're within bounds
            x = max(margin, min(1-margin, x))
            y = max(margin, min(1-margin, y))

            # Set initial radius - this will be optimized later
            # Start with a reasonable initial radius that satisfies containment constraints
            circles[idx, 0] = x
            circles[idx, 1] = y
            circles[idx, 2] = estimated_radius * 0.8  # Slightly reduced for safety

            idx += 1
            if idx >= n:
                break

    # Fill remaining circles if needed (shouldn't happen with correct calculation)
    for i in range(idx, n):
        circles[i, 0] = 0.5
        circles[i, 1] = 0.5
        circles[i, 2] = estimated_radius * 0.5

    return circles


# EVOLVE-BLOCK-END