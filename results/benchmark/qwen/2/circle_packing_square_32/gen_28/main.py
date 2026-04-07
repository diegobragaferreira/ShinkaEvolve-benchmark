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

def initialize_density_adaptive_radii(positions, n_circles):
    """Initialize radii based on local density - circles in dense regions start smaller."""
    # Calculate density by finding average distance to nearest neighbors
    if len(positions) < 2:
        return np.full(n_circles, 0.05)

    # Use cKDTree for efficient neighbor searching
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)

    # Find 4 nearest neighbors for each point to estimate local density
    distances, indices = tree.query(positions, k=min(5, len(positions)), p=2)

    # Average distance to nearest neighbors (excluding self-distance)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # Normalize average distances
    avg_distances = avg_distances / np.max(avg_distances)

    # Assign radii inversely proportional to density (lower density = larger initial radius)
    # Scale to make radii reasonable (around 0.05 to 0.15)
    radii = 0.05 + (0.10 * (1 - avg_distances))

    # Ensure minimum radius
    radii = np.maximum(radii, 0.01)

    # Trim or extend to exact number of circles
    if len(radii) > n_circles:
        radii = radii[:n_circles]
    elif len(radii) < n_circles:
        # Fill remaining with average radius
        avg_radius = np.mean(radii)
        radii = np.pad(radii, (0, n_circles - len(radii)), constant_values=avg_radius)

    return radii

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
    best_sum_radii = -np.inf
    best_circles = None

    # Multi-start optimization with different initial configurations
    for start_iter in range(5):  # Try 5 different starting points
        # Initialize using hexagonal grid
        initial_positions = initialize_hexagonal_grid(n)

        # Apply slight random perturbations to break symmetry
        if start_iter > 0:
            np.random.seed(start_iter)
            # Perturb positions slightly
            initial_positions += np.random.uniform(-0.02, 0.02, initial_positions.shape)

            # Ensure positions remain within bounds
            initial_positions[:, 0] = np.clip(initial_positions[:, 0], 0.05, 0.95)
            initial_positions[:, 1] = np.clip(initial_positions[:, 1], 0.05, 0.95)

        # Initialize with density-adaptive radii
        initial_radii = initialize_density_adaptive_radii(initial_positions, n)

        # Combine into single array [x1, y1, r1, x2, y2, r2, ...]
        initial_circles = np.column_stack([initial_positions, initial_radii]).flatten()

        # Use scipy optimization to improve the packing
        try:
            result = minimize(
                objective_function,
                initial_circles,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )

            if result.success:
                # Extract final circles
                final_circles = result.x.reshape((n, 3))

                # Calculate sum of radii for this solution
                sum_radii = np.sum(final_circles[:, 2])

                # Check if this is our best solution so far
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_circles = final_circles.copy()
            else:
                # Even if optimization fails, try to salvage a good configuration
                final_circles = initial_circles.reshape((n, 3))
                sum_radii = np.sum(final_circles[:, 2])
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_circles = final_circles.copy()

        except Exception as e:
            # If optimization fails completely, use the initial configuration
            print(f"Optimization failed on start iteration {start_iter}: {e}")
            final_circles = initial_circles.reshape((n, 3))
            sum_radii = np.sum(final_circles[:, 2])
            if sum_radii > best_sum_radii:
                best_sum_radii = sum_radii
                best_circles = final_circles.copy()

    # Ensure all circles are valid
    if best_circles is not None:
        for i in range(n):
            x, y, r = best_circles[i]
            # Bound radii to be positive and within square
            r = max(0.001, min(0.45, r))
            # Bound positions to be within square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            best_circles[i] = [x, y, r]

    return best_circles if best_circles is not None else np.zeros((n, 3))


# EVOLVE-BLOCK-END