# EVOLVE-BLOCK-START
import numpy as np

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses hexagonal lattice arrangement optimized for point dispersion.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """

    # Known optimal configuration for 16 points in a square - hexagonal lattice arrangement
    # Arrange points in a 4x4 hexagonal pattern scaled to fit in [0,1]x[0,1]

    # Create a hexagonal lattice with appropriate spacing
    # For 16 points, we can arrange them in 4 rows of 4 points each in a hexagonal pattern

    # Hexagonal lattice parameters
    rows = 4
    cols = 4

    # Calculate spacing to fill unit square effectively
    spacing_x = 1.0 / (cols - 1) if cols > 1 else 1.0
    spacing_y = 1.0 / (rows - 1) if rows > 1 else 1.0

    # Adjust spacing to get good dispersion
    # Using golden ratio-inspired approach for better distribution
    points = []

    # Create hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            # Offset every other row for hexagonal packing
            x = j * spacing_x
            y = i * spacing_y

            # Apply slight offset to create hexagonal structure
            if i % 2 == 1:
                x += spacing_x * 0.5

            points.append([x, y])

    # Convert to numpy array and adjust to fit within [0,1]x[0,1]
    points = np.array(points)

    # Normalize to unit square
    # Find bounding box and scale appropriately
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    if x_max > x_min:
        points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
    if y_max > y_min:
        points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)

    # Scale to fit nicely in [0,1]x[0,1] with some margin
    points = np.clip(points, 0, 1)

    # Fine-tune positions using simple gradient ascent approach
    # This is a much simpler but effective way to improve the configuration
    np.random.seed(42)

    # Perform a few iterations of local optimization
    for _ in range(100):
        # Calculate distances
        distances = []
        for i in range(16):
            for j in range(i+1, 16):
                dist = np.sqrt((points[i, 0] - points[j, 0])**2 + (points[i, 1] - points[j, 1])**2)
                distances.append(dist)

        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)

            if max_dist > 0:
                # Simple heuristic: move points away from each other
                step_size = 0.001
                for i in range(16):
                    # Compute forces from nearby points
                    force_x, force_y = 0.0, 0.0
                    for j in range(16):
                        if i != j:
                            dx = points[i, 0] - points[j, 0]
                            dy = points[i, 1] - points[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > 0:
                                # Repulsive force
                                force_x += dx / (dist * dist + 0.001)
                                force_y += dy / (dist * dist + 0.001)

                    # Move point
                    points[i, 0] += step_size * force_x
                    points[i, 1] += step_size * force_y

                    # Keep within bounds
                    points[i, 0] = np.clip(points[i, 0], 0, 1)
                    points[i, 1] = np.clip(points[i, 1], 0, 1)

    # Ensure we're returning a clean array
    points = np.array(points)

    return points

# EVOLVE-BLOCK-END