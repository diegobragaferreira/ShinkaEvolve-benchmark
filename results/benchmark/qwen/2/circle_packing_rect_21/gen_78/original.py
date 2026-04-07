# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    # We'll use width = 1.2 and height = 0.8 for a good aspect ratio
    width = 1.2
    height = 0.8

    def check_constraints(positions, radii):
        """Check if all circles are within bounds and non-overlapping"""
        # Check boundary constraints
        for i in range(len(positions)):
            x, y = positions[i]
            r = radii[i]
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False

        # Check overlap constraints
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < radii[i] + radii[j]:
                    return False
        return True

    def compute_radius_sum(positions, radii):
        """Compute sum of all radii"""
        return sum(radii)

    def optimize_single_circle(index, positions, radii, width, height):
        """Optimize a single circle's position and radius"""
        def objective(r):
            # Create temporary arrays
            temp_positions = positions.copy()
            temp_radii = radii.copy()
            temp_radii[index] = r[0]

            # Check if this is valid
            if not check_constraints(temp_positions, temp_radii):
                return 1e10  # Large penalty for invalid configurations

            return -r[0]  # Negative because we want to maximize

        # Initial guess
        current_r = radii[index]

        # Bounds for radius (must be positive, and not cause overlaps)
        bounds = [(1e-6, min(width/2, height/2, current_r*2))]

        try:
            result = minimize(objective, [current_r], bounds=bounds, method='L-BFGS-B')
            if result.success:
                return max(1e-6, result.x[0])
        except:
            pass
        return current_r

    # Initialize with a structured grid-like approach
    n = 21
    circles = np.zeros((n, 3))

    # Create a grid arrangement (approximate)
    rows = 4
    cols = 6
    if rows * cols < n:
        rows = 5
        cols = 5

    # Calculate spacing
    x_spacing = width / (cols + 1)
    y_spacing = height / (rows + 1)

    # Place initial circles
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * x_spacing
            y = (i + 1) * y_spacing
            circles[idx] = [x, y, min(x_spacing, y_spacing) * 0.3]
            idx += 1

    # Ensure we have exactly 21 circles
    if idx < n:
        # Fill remaining slots with random positions
        for i in range(idx, n):
            circles[i] = [np.random.uniform(0.1, width-0.1),
                         np.random.uniform(0.1, height-0.1),
                         0.05]

    # Iterative improvement loop
    max_iterations = 100
    for iteration in range(max_iterations):
        improved = False

        # Try to optimize each circle individually
        for i in range(n):
            # Store current state
            old_pos = circles[i][:2].copy()
            old_rad = circles[i][2]

            # Try to increase radius
            new_rad = optimize_single_circle(i, circles[:,:2], circles[:,2], width, height)

            if new_rad > old_rad:
                circles[i][2] = new_rad
                improved = True

        if not improved:
            break

    # Final constraint validation and adjustment
    positions = circles[:,:2]
    radii = circles[:,2]

    # If constraints violated, try to fix by shrinking overlapping circles
    if not check_constraints(positions, radii):
        # Simple greedy approach: reduce radii of overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = radii[i] + radii[j]

                if distance < min_distance:
                    # Reduce radii to resolve overlap
                    reduction = (min_distance - distance) * 0.5
                    radii[i] = max(1e-6, radii[i] - reduction)
                    radii[j] = max(1e-6, radii[j] - reduction)

    # Ensure all circles fit within the rectangle
    for i in range(n):
        x, y, r = circles[i]
        # Clamp to boundaries
        circles[i] = [
            max(r, min(width - r, x)),
            max(r, min(height - r, y)),
            radii[i]
        ]

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")