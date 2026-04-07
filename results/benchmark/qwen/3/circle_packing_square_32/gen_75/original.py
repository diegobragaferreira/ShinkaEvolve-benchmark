# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32

    # Hexagonal grid initialization
    def initialize_hexagonal_grid():
        # Create hexagonal grid pattern that fits in unit square
        # Using hexagonal packing density approximation
        rows = int(math.sqrt(n))
        cols = int(math.ceil(n / rows))

        # Adjust dimensions to fit in unit square
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows

        # Hexagon radius (circumradius) based on spacing
        hex_radius = min(spacing_x, spacing_y) * 0.4

        circles = []

        # Place circles in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break

                # Offset every other row
                x_offset = (i % 2) * (spacing_x / 2)
                x = (j * spacing_x) + x_offset + hex_radius
                y = (i * spacing_y) + hex_radius + hex_radius

                # Ensure it's within bounds
                if x <= 1 - hex_radius and y <= 1 - hex_radius:
                    circles.append([x, y, hex_radius])

        # If we don't have enough circles, fill remaining with smaller ones
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.01])

        return np.array(circles[:n])

    # Check constraints efficiently
    def check_constraints(circles):
        """Check if all constraints are satisfied"""
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Check boundary constraints
        left_ok = np.all(radii <= positions[:, 0])
        right_ok = np.all(radii <= 1 - positions[:, 0])
        bottom_ok = np.all(radii <= positions[:, 1])
        top_ok = np.all(radii <= 1 - positions[:, 1])

        if not (left_ok and right_ok and bottom_ok and top_ok):
            return False

        # Check overlap constraints with efficient matrix operation
        if len(circles) >= 2:
            dist_matrix = cdist(positions, positions)
            overlap_matrix = dist_matrix < (radii.reshape(-1, 1) + radii.reshape(1, -1))
            np.fill_diagonal(overlap_matrix, False)
            if np.any(overlap_matrix):
                return False

        return True

    # Initialize with hexagonal grid
    circles = initialize_hexagonal_grid()

    # Iterative refinement to increase radii while maintaining constraints
    max_iterations = 100
    tolerance = 1e-6
    step_size = 0.01

    for iteration in range(max_iterations):
        improved = False
        current_sum = np.sum(circles[:, 2])

        # Try to increase each radius slightly
        new_circles = circles.copy()

        # Try to increase all radii by small amounts
        for i in range(n):
            test_circles = new_circles.copy()
            test_circles[i, 2] += step_size

            # Check if this change violates constraints
            if check_constraints(test_circles):
                new_circles = test_circles
                improved = True

        # If we made progress, update circles and continue
        if improved:
            circles = new_circles
        else:
            # No improvement, reduce step size
            step_size *= 0.5
            if step_size < tolerance:
                break

    # Final check of constraints
    if not check_constraints(circles):
        # If constraints violated, revert to hexagonal grid
        circles = initialize_hexagonal_grid()

    # If we still have issues, try optimization with better bounds
    if not check_constraints(circles):
        return initialize_hexagonal_grid()

    return circles

# EVOLVE-BLOCK-END