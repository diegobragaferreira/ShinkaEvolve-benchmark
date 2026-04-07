# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)

    n = 32
    circles = np.zeros((n, 3))

    # Initialize using enhanced hexagonal packing pattern with strategic corner placement
    def initialize_enhanced_hexagonal():
        # Start with strategic corner and edge placements for larger initial radii
        positions = []

        # Place some circles near corners and edges where large radii are possible
        corner_positions = [
            [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
            [0.05, 0.5], [0.5, 0.05], [0.5, 0.95], [0.95, 0.5]
        ]

        # Add the corner positions
        positions.extend(corner_positions)

        # Fill remaining positions with hexagonal grid
        if len(positions) < n:
            rows = int(math.sqrt(n - len(positions))) + 2
            cols = int((n - len(positions)) / rows) + 2

            # Hexagon parameters - slightly smaller to allow room for growth
            side_length = 0.12
            height = side_length * math.sqrt(3) / 2

            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= n:
                        break
                    x = (j + (i % 2) * 0.5) * side_length * 2
                    y = i * height
                    if x <= 1 and y <= 1:
                        positions.append([x, y])

        # Trim to exactly n positions
        positions = positions[:n]

        # Add small random offsets to avoid perfect regularity
        for i, pos in enumerate(positions):
            # Add larger random offset for better exploration
            pos[0] += np.random.uniform(-0.015, 0.015)
            pos[1] += np.random.uniform(-0.015, 0.015)
            # Clamp to valid range to ensure circles stay within bounds
            pos[0] = np.clip(pos[0], 0.01, 0.99)
            pos[1] = np.clip(pos[1], 0.01, 0.99)

        return positions

    # Initialize positions
    positions = initialize_enhanced_hexagonal()

    # Set initial radii - start with small values that can grow
    radii = np.full(n, 0.02)

    # Constraint checking functions
    def is_valid_position(x, y, r):
        """Check if circle at (x,y) with radius r is fully contained"""
        return (r <= x <= 1-r) and (r <= y <= 1-r)

    def is_overlapping(i, j, x_i, y_i, r_i, x_j, y_j, r_j):
        """Check if two circles overlap"""
        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
        return dist_sq < (r_i + r_j)**2

    def check_constraints(circles_array):
        """Check all constraints for current configuration"""
        # Check containment
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if not is_valid_position(x, y, r):
                return False

        # Check overlaps
        for i in range(len(circles_array)):
            for j in range(i+1, len(circles_array)):
                x_i, y_i, r_i = circles_array[i]
                x_j, y_j, r_j = circles_array[j]
                if is_overlapping(i, j, x_i, y_i, r_i, x_j, y_j, r_j):
                    return False
        return True

    # Calculate total sum of radii
    def total_radius_sum(circles_array):
        return np.sum(circles_array[:, 2])

    # Simulated Annealing optimization
    def optimize_with_sa():
        # Start with initial configuration
        current_circles = np.column_stack([positions, radii])
        current_score = total_radius_sum(current_circles)

        # Parameters for SA
        temperature = 0.1
        min_temp = 1e-8
        cooling_rate = 0.999
        max_iterations = 100000
        iteration = 0

        best_circles = current_circles.copy()
        best_score = current_score

        # Keep track of recent improvements to decide when to stop
        recent_improvements = []
        patience = 1000
        patience_counter = 0

        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor solution by perturbing one circle
            idx = np.random.randint(0, n)

            # Save current state
            old_x, old_y, old_r = current_circles[idx]

            # Perturb the circle
            new_x = old_x + np.random.uniform(-0.01, 0.01)
            new_y = old_y + np.random.uniform(-0.01, 0.01)
            new_r = old_r + np.random.uniform(-0.005, 0.005)

            # Ensure new_r is positive
            new_r = max(0.001, new_r)

            # Create new circles array
            new_circles = current_circles.copy()
            new_circles[idx] = [new_x, new_y, new_r]

            # Check if new configuration is valid
            if check_constraints(new_circles):
                new_score = total_radius_sum(new_circles)

                # Accept or reject based on simulated annealing criteria
                delta = new_score - current_score
                if delta > 0 or np.exp(delta / temperature) > np.random.random():
                    current_circles = new_circles
                    current_score = new_score

                    # Update best solution
                    if current_score > best_score:
                        best_circles = current_circles.copy()
                        best_score = current_score
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Early stopping if no improvement for a while
                    if patience_counter > patience:
                        break

            iteration += 1
            # Cool down
            temperature *= cooling_rate

            # Track recent improvements for early stopping
            recent_improvements.append(current_score)
            if len(recent_improvements) > 100:
                recent_improvements.pop(0)

        return best_circles

    # Run optimization
    try:
        final_circles = optimize_with_sa()

        # Final validation
        if not check_constraints(final_circles):
            # If invalid, fallback to simple initialization
            final_circles = np.column_stack([positions, [0.02]*n])

        circles = final_circles
    except Exception as e:
        # Fallback to simple uniform distribution if optimization fails
        positions = [[i*0.1 + 0.05, j*0.1 + 0.05] for i in range(6) for j in range(6)][:n]
        circles = np.column_stack([positions, [0.02]*n])

    return circles

# EVOLVE-BLOCK-END