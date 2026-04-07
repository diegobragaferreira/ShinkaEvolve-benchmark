# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, let's use 1.5 x 0.5 for good aspect ratio
    width, height = 1.5, 0.5

    # Number of circles
    n = 21

    # Initialize circles with adaptive hexagonal packing pattern
    def initialize_hexagonal_layout():
        circles = []

        # Try to estimate optimal radius based on area
        rectangle_area = width * height
        # Estimate max possible area occupied by circles
        total_circle_area = 0.9 * rectangle_area  # Leave some margin
        avg_circle_area = total_circle_area / n
        estimated_radius = np.sqrt(avg_circle_area / np.pi)

        # Start with a more refined hexagonal grid
        # Calculate spacing based on the estimated radius
        radius_guess = max(estimated_radius, 0.01)  # Minimum radius
        spacing = 2 * radius_guess * 1.05  # Slight overlap to allow for optimization

        # Determine grid dimensions that would fit well
        cols = max(1, int(width / spacing))
        rows = max(1, int(height / (spacing * np.sqrt(3)/2)))

        # Ensure we have enough circles
        if cols * rows < n:
            # Increase spacing to get more circles
            spacing = np.sqrt((width * height) / (n * np.pi)) * 2.0  # More conservative estimate
            cols = max(1, int(width / spacing))
            rows = max(1, int(height / (spacing * np.sqrt(3)/2)))

        # Create the hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                offset = spacing * 0.5 if i % 2 == 1 else 0
                x = offset + j * spacing + spacing/2
                y = i * spacing * np.sqrt(3)/2 + spacing/2

                # Check if within bounds (with some margin)
                if x - radius_guess >= 0 and x + radius_guess <= width and \
                   y - radius_guess >= 0 and y + radius_guess <= height:
                    circles.append([x, y, radius_guess])

        # If we don't have enough circles, fill with random positions
        while len(circles) < n:
            # Distribute more evenly in the available space
            x = np.random.uniform(radius_guess, width - radius_guess)
            y = np.random.uniform(radius_guess, height - radius_guess)
            circles.append([x, y, radius_guess])

        return np.array(circles)

    # Get initial configuration
    circles = initialize_hexagonal_layout()
    circles[:, 2] = 0.05  # Set all to same radius initially

    # Physics-based optimization
    def compute_forces(circles):
        """Compute forces between circles"""
        n_circles = len(circles)
        forces = np.zeros_like(circles)

        # Distance matrix
        positions = circles[:, :2]
        distances = cdist(positions, positions)

        # Force calculation (repulsion between all pairs)
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)

                if dist > 0 and dist < (circles[i, 2] + circles[j, 2]):
                    # Overlapping, strong repulsion
                    force_magnitude = 1000.0 / (dist * dist)
                elif dist > 0:
                    # Not overlapping, weak repulsion
                    force_magnitude = 0.01 / (dist * dist)
                else:
                    force_magnitude = 0

                if dist > 0:
                    fx = force_magnitude * dx / dist
                    fy = force_magnitude * dy / dist

                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Boundary forces (repulsion from walls)
        for i in range(n_circles):
            x, y = positions[i]
            r = circles[i, 2]

            # Left boundary
            if x - r < 0:
                forces[i, 0] += 1000 * (0 - (x - r))
            # Right boundary
            if x + r > width:
                forces[i, 0] -= 1000 * ((x + r) - width)
            # Bottom boundary
            if y - r < 0:
                forces[i, 1] += 1000 * (0 - (y - r))
            # Top boundary
            if y + r > height:
                forces[i, 1] -= 1000 * ((y + r) - height)

        return forces

    # Optimization loop
    max_iterations = 500
    dt = 0.01

    for iteration in range(max_iterations):
        # Compute forces
        forces = compute_forces(circles)

        # Update positions
        for i in range(len(circles)):
            # Limit velocity
            max_velocity = 0.1
            fx = np.clip(forces[i, 0], -max_velocity, max_velocity)
            fy = np.clip(forces[i, 1], -max_velocity, max_velocity)

            circles[i, 0] += fx * dt
            circles[i, 1] += fy * dt

        # Apply boundary constraints
        for i in range(len(circles)):
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, width - r)
            circles[i, 1] = np.clip(circles[i, 1], r, height - r)

        # Occasionally adjust radii based on spacing
        if iteration % 20 == 0:
            # Simple heuristic to increase radii when space allows
            positions = circles[:, :2]
            distances = cdist(positions, positions)

            # For each circle, see what's the minimum distance to neighbors
            min_distances = []
            for i in range(len(circles)):
                dists = distances[i]
                dists[i] = np.inf  # Exclude self
                min_dist = np.min(dists[dists > 0])
                min_distances.append(min_dist)

            # Increase radii slightly where there's more space
            for i in range(len(circles)):
                min_dist = min_distances[i]
                if min_dist > 2 * circles[i, 2]:
                    # Add small increment to radius
                    circles[i, 2] = min(circles[i, 2] + 0.001, 0.2)

    # Final optimization of radii to maximize sum
    def objective(radii):
        # Calculate total area of circles (for comparison, but we care about sum)
        return -np.sum(radii)  # Negative because we want to maximize

    def constraint_func(circles):
        # Constraint: no overlaps, all circles fit in rectangle
        positions = circles[:, :2]
        distances = cdist(positions, positions)

        # Check overlap constraints
        violations = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_distance = circles[i, 2] + circles[j, 2]

                if dist < min_distance:
                    violations.append(dist - min_distance)  # Negative value if violated

        # Bound constraints
        for i in range(len(circles)):
            r = circles[i, 2]
            if r <= 0:
                violations.append(-r)  # Radius too small
            if circles[i, 0] - r < 0 or circles[i, 0] + r > width:
                violations.append(-0.01)  # Out of bounds
            if circles[i, 1] - r < 0 or circles[i, 1] + r > height:
                violations.append(-0.01)  # Out of bounds

        return violations

    # Simple final refinement with bounds checking
    old_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()
    best_sum = old_sum

    # Final iterative refinement
    for _ in range(50):
        # Try to increase radii slightly
        test_circles = circles.copy()
        for i in range(len(test_circles)):
            test_circles[i, 2] = min(test_circles[i, 2] + 0.005, 0.2)

        # Check constraints
        positions = test_circles[:, :2]
        distances = cdist(positions, positions)

        valid = True
        for i in range(len(test_circles)):
            for j in range(i+1, len(test_circles)):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_distance = test_circles[i, 2] + test_circles[j, 2]

                if dist < min_distance * 0.99:
                    valid = False
                    break
            if not valid:
                break

        # If valid config, use it
        if valid:
            new_sum = np.sum(test_circles[:, 2])
            if new_sum > best_sum:
                best_circles = test_circles.copy()
                best_sum = new_sum

    # Final check and return
    final_sum = np.sum(best_circles[:, 2])

    # Return optimized circles array
    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")