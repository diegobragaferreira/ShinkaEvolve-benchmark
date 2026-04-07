# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_hexagonal_grid(n_circles):
    """Initialize circle positions using a hexagonal grid pattern."""
    # Estimate how many rows/columns we need
    sqrt_n = int(np.ceil(np.sqrt(n_circles)))

    # Create hexagonal grid with appropriate spacing
    # For a hexagonal lattice, the distance between centers is 2*r
    # We'll use a slightly smaller spacing to allow room for optimization

    # Try to fit circles in a roughly square arrangement
    rows = int(np.ceil(n_circles / np.ceil(np.sqrt(n_circles))))
    cols = int(np.ceil(n_circles / rows))

    # Create positions for rows and columns
    positions = []

    # Hexagonal grid with staggered rows
    for i in range(rows):
        for j in range(cols):
            # Offset every other row
            x_offset = 0.5 * (i % 2)
            x = (j + x_offset) * 0.15  # Adjust spacing here
            y = i * 0.15
            positions.append([x, y])

    # Trim to exact number of circles needed
    positions = positions[:n_circles]

    # Ensure positions are within bounds
    for pos in positions:
        pos[0] = max(0.05, min(0.95, pos[0]))
        pos[1] = max(0.05, min(0.95, pos[1]))

    return np.array(positions)

def calculate_penalty(circles, penalty_weight=1000.0):
    """Calculate penalty for constraint violations."""
    n = len(circles)
    penalty = 0.0

    # Boundary penalties
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += penalty_weight

    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < r1 + r2:
                penalty += penalty_weight * (r1 + r2 - dist)

    return penalty

def objective_function(circles_flat):
    """Objective function to maximize the sum of radii."""
    n = len(circles_flat) // 3
    circles = circles_flat.reshape((n, 3))

    # Sum of radii (negative because we're minimizing)
    sum_radii = -np.sum(circles[:, 2])

    # Add penalty for constraint violations
    penalty = calculate_penalty(circles)

    return sum_radii + penalty

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32

    # Initialize using hexagonal grid
    initial_positions = initialize_hexagonal_grid(n)

    # Initialize with equal small radii
    initial_radii = np.full(n, 0.05)

    # Combine into single array [x1, y1, r1, x2, y2, r2, ...]
    initial_circles = np.column_stack([initial_positions, initial_radii]).flatten()

    # Use scipy optimization to improve the packing
    result = minimize(
        objective_function,
        initial_circles,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    # Extract final circles
    final_circles = result.x.reshape((n, 3))

    # Ensure all circles are valid
    for i in range(n):
        x, y, r = final_circles[i]
        # Bound radii to be positive and within square
        r = max(0.001, min(0.45, r))
        # Bound positions to be within square
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        final_circles[i] = [x, y, r]

    return final_circles


# EVOLVE-BLOCK-END