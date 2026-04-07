# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For deterministic behavior
    random.seed(42)

    n = 32

    def initialize_hexagonal_grid():
        """Create initial configuration using hexagonal packing"""
        # Calculate grid dimensions
        rows = int(math.sqrt(n))
        cols = int(math.ceil(n / rows))

        # Adjust spacing to fit within unit square
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows

        # Use hexagon radius based on grid spacing - slightly more conservative
        hex_radius = min(spacing_x, spacing_y) * 0.35

        circles = []

        # Fill grid with circles in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break

                # Offset every other row for hexagonal pattern
                x_offset = (i % 2) * (spacing_x / 2)
                x = (j * spacing_x) + x_offset + hex_radius
                y = (i * spacing_y) + hex_radius

                # Ensure circle stays within bounds
                if x <= 1 - hex_radius and y <= 1 - hex_radius:
                    circles.append([x, y, hex_radius])

        # Fill remaining circles with small radii if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.01])

        return np.array(circles[:n])

    def check_boundary_constraints(circles):
        """Check if all circles are within the unit square"""
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Check all constraints: left, right, bottom, top
        left_valid = np.all(radii <= positions[:, 0])
        right_valid = np.all(radii <= 1 - positions[:, 0])
        bottom_valid = np.all(radii <= positions[:, 1])
        top_valid = np.all(radii <= 1 - positions[:, 1])

        return left_valid and right_valid and bottom_valid and top_valid

    def check_overlap_constraints(circles):
        """Efficiently check for circle overlaps using KDTree"""
        if len(circles) < 2:
            return True

        positions = circles[:, :2]
        radii = circles[:, 2]

        # Build KDTree for fast neighbor search
        tree = cKDTree(positions)

        # For each circle, find neighbors within sum of radii
        total_overlaps = 0

        for i, (pos, rad) in enumerate(zip(positions, radii)):
            # Find nearby points
            neighbors = tree.query_ball_point(pos, 2 * rad)

            # Check actual distance to neighbors (excluding self)
            for j in neighbors:
                if i != j:
                    distance = np.linalg.norm(pos - positions[j])
                    if distance < (rad + radii[j]):
                        total_overlaps += 1

        return total_overlaps == 0

    def refine_radius_expansion(circles, max_iterations=1000):
        """Iteratively expand radii while maintaining constraints"""
        current_circles = circles.copy()
        improved = True
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # Try to increase each radius slightly
            for i in range(len(current_circles)):
                # Store original state
                original_radius = current_circles[i, 2]
                original_pos = current_circles[i, :2].copy()

                # Try to increase radius by small amount
                test_radius = min(0.49, original_radius + 0.001)

                # Temporarily update radius
                current_circles[i, 2] = test_radius

                # Check if constraints are still satisfied
                if (check_boundary_constraints(current_circles) and
                    check_overlap_constraints(current_circles)):
                    # Accept the increase
                    improved = True
                else:
                    # Revert to original
                    current_circles[i, 2] = original_radius

        return current_circles

    # Initialize with hexagonal grid
    initial_circles = initialize_hexagonal_grid()

    # Apply iterative refinement to expand radii
    refined_circles = refine_radius_expansion(initial_circles)

    # Final validation
    if (check_boundary_constraints(refined_circles) and
        check_overlap_constraints(refined_circles)):
        return refined_circles
    else:
        # Fallback to initial configuration
        return initial_circles

# EVOLVE-BLOCK-END