# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from itertools import combinations
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Using width=1.2, height=0.8 for a reasonable aspect ratio
    rect_width = 1.2
    rect_height = 0.8

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    def calculate_max_radius_at_position(point, points, rect_width, rect_height):
        """Calculate maximum possible radius for a circle at given position"""
        center_x, center_y = point
        # Distance to rectangle edges
        dist_to_edges = [
            center_x,                    # distance to left edge
            rect_width - center_x,       # distance to right edge
            center_y,                    # distance to bottom edge
            rect_height - center_y       # distance to top edge
        ]

        # Find minimum distance to other circles (excluding self)
        min_dist_to_others = float('inf')
        for i, other_point in enumerate(points):
            if not (abs(other_point[0] - center_x) < 1e-10 and abs(other_point[1] - center_y) < 1e-10):
                dist = distance.euclidean(point, other_point)
                min_dist_to_others = min(min_dist_to_others, dist)

        # Maximum radius is limited by both edges and other circles
        max_radius = min(min(dist_to_edges), min_dist_to_others/2.0)
        return max(0.001, max_radius)

    def evaluate_configuration(points, rect_width, rect_height):
        """Evaluate a configuration by computing sum of radii"""
        total_radius = 0
        circles = []
        for point in points:
            radius = calculate_max_radius_at_position(point, points, rect_width, rect_height)
            circles.append([point[0], point[1], radius])
            total_radius += radius
        return total_radius, np.array(circles)

    def generate_initial_config():
        """Generate a better initial configuration using hybrid approach"""
        # Start with corner and center positions
        points = []

        # Corner positions with slight variation
        corner_positions = [
            (rect_width * 0.1, rect_height * 0.1),
            (rect_width * 0.9, rect_height * 0.1),
            (rect_width * 0.1, rect_height * 0.9),
            (rect_width * 0.9, rect_height * 0.9),
            (rect_width / 2, rect_height / 2)
        ]

        for x, y in corner_positions:
            points.append([x + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02)])

        # Fill remaining positions using a more structured approach
        # Use grid points with noise
        grid_size = 4
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < 21:
                    x = rect_width * (0.15 + i * 0.2)
                    y = rect_height * (0.15 + j * 0.2)
                    points.append([x + np.random.normal(0, 0.03), y + np.random.normal(0, 0.03)])

        # Fill remaining slots randomly
        while len(points) < 21:
            x = np.random.uniform(0.05, rect_width - 0.05)
            y = np.random.uniform(0.05, rect_height - 0.05)
            points.append([x, y])

        return np.array(points[:21])

    def voronoi_based_refinement(points, rect_width, rect_height):
        """Use Voronoi-based refinement to improve point distribution"""
        # Create Voronoi diagram to understand spatial relationships
        try:
            vor = Voronoi(points)
            # Get Voronoi vertices and regions for better understanding of space
        except:
            pass

        # Create a more aggressive refinement using neighborhood information
        refined_points = points.copy()

        # For each point, try to improve its position more aggressively
        for i in range(len(points)):
            current_point = points[i].copy()
            best_point = current_point.copy()
            best_radius = calculate_max_radius_at_position(current_point, points, rect_width, rect_height)

            # Try multiple directions and distances
            directions = [(0, 0), (-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1)]
            step_sizes = [0.05, 0.025, 0.01]

            for dx, dy in directions:
                for step in step_sizes:
                    test_x = current_point[0] + dx * step
                    test_y = current_point[1] + dy * step

                    # Keep within bounds
                    if (0.05 <= test_x <= rect_width - 0.05 and
                        0.05 <= test_y <= rect_height - 0.05):

                        test_point = np.array([test_x, test_y])
                        test_radius = calculate_max_radius_at_position(test_point, points, rect_width, rect_height)

                        if test_radius > best_radius:
                            best_radius = test_radius
                            best_point = test_point

            refined_points[i] = best_point

        return refined_points

    def constrained_optimization_step(points, rect_width, rect_height):
        """Perform constrained optimization step with better constraint handling"""
        # Get current configuration
        _, current_circles = evaluate_configuration(points, rect_width, rect_height)

        # For each point, try to find a better position that increases the sum
        best_points = points.copy()
        best_sum = np.sum(current_circles[:, 2])

        # Multi-scale local search
        for k in range(1000):  # More iterations for better search
            # Select a random point to optimize
            idx = np.random.randint(0, len(points))

            # Try multiple perturbations
            current_point = points[idx].copy()
            current_radius = calculate_max_radius_at_position(current_point, points, rect_width, rect_height)

            # Sample in various directions
            best_new_point = current_point.copy()
            best_new_radius = current_radius

            # Sample around the current point with different step sizes
            step_sizes = [0.005, 0.01, 0.02, 0.05]
            directions = [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

            for step in step_sizes:
                for dx, dy in directions:
                    new_x = current_point[0] + dx * step
                    new_y = current_point[1] + dy * step

                    # Check bounds
                    if (0.05 <= new_x <= rect_width - 0.05 and
                        0.05 <= new_y <= rect_height - 0.05):

                        test_point = np.array([new_x, new_y])
                        test_radius = calculate_max_radius_at_position(test_point, points, rect_width, rect_height)

                        if test_radius > best_new_radius:
                            best_new_radius = test_radius
                            best_new_point = test_point

            # If we found an improvement, update the point
            if best_new_radius > current_radius:
                updated_points = points.copy()
                updated_points[idx] = best_new_point
                # Re-evaluate the whole configuration
                new_sum, _ = evaluate_configuration(updated_points, rect_width, rect_height)

                if new_sum > best_sum:
                    best_points = updated_points
                    best_sum = new_sum

        return best_points

    # Generate initial configuration
    points = generate_initial_config()

    # Main optimization loop with multiple refinement stages
    best_sum = 0
    best_points = None

    # Stage 1: Basic iterative optimization
    for iteration in range(300):
        # Voronoi-based refinement
        points = voronoi_based_refinement(points, rect_width, rect_height)

        # Constrained optimization step
        points = constrained_optimization_step(points, rect_width, rect_height)

        # Evaluate current configuration
        current_sum, _ = evaluate_configuration(points, rect_width, rect_height)

        # Keep track of best configuration
        if current_sum > best_sum:
            best_sum = current_sum
            best_points = points.copy()

    # Stage 2: Final intensive refinement with adaptive step sizes
    if best_points is not None:
        final_points = best_points.copy()

        # Use even more aggressive local search
        for _ in range(500):  # More iterations for better convergence
            # Try to improve each point individually
            for i in range(21):
                current_point = final_points[i].copy()
                current_radius = calculate_max_radius_at_position(current_point, final_points, rect_width, rect_height)

                # Aggressive search with adaptive steps
                best_new_point = current_point.copy()
                best_new_radius = current_radius

                # Multiple search strategies
                strategies = [
                    # Fine grid around current point
                    [(dx, dy) for dx in [-0.05, -0.025, 0, 0.025, 0.05] for dy in [-0.05, -0.025, 0, 0.025, 0.05]],
                    # Diagonal search
                    [(dx, dy) for dx in [-0.1, 0, 0.1] for dy in [-0.1, 0, 0.1] if dx != 0 or dy != 0],
                    # Random small perturbations
                    [(np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)) for _ in range(10)]
                ]

                for strategy in strategies:
                    for dx, dy in strategy:
                        test_x = current_point[0] + dx
                        test_y = current_point[1] + dy

                        # Keep within bounds
                        if (0.05 <= test_x <= rect_width - 0.05 and
                            0.05 <= test_y <= rect_height - 0.05):

                            test_point = np.array([test_x, test_y])
                            test_radius = calculate_max_radius_at_position(test_point, final_points, rect_width, rect_height)

                            if test_radius > best_new_radius:
                                best_new_radius = test_radius
                                best_new_point = test_point

                # Update if improvement
                if best_new_radius > final_points[i, 2]:
                    final_points[i] = best_new_point

        # Final evaluation
        _, circles = evaluate_configuration(final_points, rect_width, rect_height)
        return circles

    # Fallback to simple grid pattern
    circles = np.zeros((21, 3))
    row_size = int(np.ceil(np.sqrt(21)))
    col_size = int(np.ceil(21 / row_size))

    spacing_x = rect_width / (col_size + 1)
    spacing_y = rect_height / (row_size + 1)

    count = 0
    for i in range(row_size):
        for j in range(col_size):
            if count < 21:
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                # Set radius to be proportional to available space
                radius = min(x, rect_width - x, y, rect_height - y) * 0.4
                circles[count] = [x, y, max(radius, 0.001)]
                count += 1

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")