# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
import time
from collections import defaultdict

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2, optimized ratio
    rect_width = 1.2
    rect_height = 0.8

    n = 21

    # Improved initial grid generation using a more strategic approach
    def generate_improved_initial_placement(num_circles, width, height):
        circles = []

        # For 21 circles, we'll try to arrange them in a more systematic way:
        # 1. Use a triangular lattice pattern for better packing (approximate hexagonal)
        # 2. Adjust the grid to fit the rectangular container optimally

        # Estimate optimal density and grid dimensions
        # For 21 circles, we'll try to use a 5x5 grid pattern that can be scaled appropriately
        rows = 5
        cols = 5

        # Calculate cell size that will give us good density
        cell_width = width / cols * 0.85
        cell_height = height / rows * 0.85

        # Adjust for hexagonal packing effect
        spacing_x = cell_width
        spacing_y = cell_height * np.sqrt(3) / 2

        # Make sure we don't exceed the container size
        actual_rows = int(height / spacing_y) + 1
        actual_cols = int(width / spacing_x) + 1

        # If we need fewer than 21 circles, reduce grid
        if actual_rows * actual_cols < num_circles:
            actual_rows = max(1, int(np.sqrt(num_circles)))
            actual_cols = max(1, int(num_circles / actual_rows))

        # Generate grid
        circle_count = 0
        y_offset = 0.1
        x_offset = 0.1

        for i in range(actual_rows):
            y = y_offset + i * spacing_y
            x_start = x_offset + (i % 2) * spacing_x / 2  # Offset every other row for hex pattern
            for j in range(actual_cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x
                if x < width - 0.1 and y < height - 0.1:
                    # Start with adaptive radius that scales with spacing
                    r = min(0.04, spacing_x / 4, spacing_y / 4)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break

        # Fill remaining spots randomly if needed
        while len(circles) < num_circles:
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            r = np.random.uniform(0.01, 0.05)
            circles.append([x, y, r])

        return np.array(circles)

    # Efficient constraint validation with spatial indexing and better penalty calculation
    def calculate_fitness_with_spatial_indexing(circles_array):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0

        # Boundary penalties with stronger penalty near edges
        for i in range(n):
            cx, cy, r = circles_array[i]
            if cx - r < 0.01:
                penalty += 100000 * (r - cx)**2
            if cx + r > rect_width - 0.01:
                penalty += 100000 * (cx + r - rect_width)**2
            if cy - r < 0.01:
                penalty += 100000 * (r - cy)**2
            if cy + r > rect_height - 0.01:
                penalty += 100000 * (cy + r - rect_height)**2

        # Overlap penalties using spatial indexing for efficiency
        points = circles_array[:, :2]
        tree = KDTree(points)

        # Query for all possible overlapping pairs efficiently
        for i in range(n):
            cx, cy, r = circles_array[i]

            # Find neighbors within 2*(r + safety_margin) distance
            neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

            for j in neighbor_indices:
                if i != j:
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist

                    if overlap > 0:
                        penalty += 500000 * overlap**2

        return total_radius - penalty

    # Enhanced local refinement with multiple strategies
    def enhanced_refinement(circles_array, max_iter=100):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness_with_spatial_indexing(best_circles)

        # Track improvement history to detect when to stop early
        improvement_history = []

        for iteration in range(max_iter):
            improved = False

            # Strategy 1: Try to increase radii of circles that have room
            for i in range(n):
                cx, cy, r = best_circles[i]

                # Compute max allowable radius
                max_radius = float('inf')

                # Boundary constraints
                max_radius = min(max_radius, cx - 0.01)
                max_radius = min(max_radius, rect_width - cx - 0.01)
                max_radius = min(max_radius, cy - 0.01)
                max_radius = min(max_radius, rect_height - cy - 0.01)

                # Overlap constraints using spatial indexing on current neighbors
                points = best_circles[:, :2]
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)

                # Try to increase radius if beneficial
                if max_radius > r and max_radius > 0.001:
                    new_r = min(r + 0.01, max_radius)
                    if new_r > r + 0.001:  # Significant improvement needed
                        # Check if this actually improves fitness
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r

                        # Use spatial indexing for validation
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([cx, cy], 2*(new_r + 0.01) + 0.001)

                        for k in temp_neighbor_indices:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    valid = False
                                    break

                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 2] = new_r
                            test_fitness = calculate_fitness_with_spatial_indexing(test_circles)

                            if test_fitness > best_fitness:
                                best_fitness = test_fitness
                                best_circles = test_circles
                                improved = True

            # Strategy 2: Try small position perturbations
            if not improved and iteration % 5 == 0:
                # Perturb a few circles to escape local minima
                for _ in range(3):  # Try up to 3 perturbations
                    i = random.randint(0, n-1)
                    x_old, y_old, r = best_circles[i]

                    # Small positional perturbations
                    dx = np.random.uniform(-0.02, 0.02)
                    dy = np.random.uniform(-0.02, 0.02)

                    new_x = x_old + dx
                    new_y = y_old + dy

                    # Check if new position is valid
                    if (0.01 + r <= new_x <= rect_width - 0.01 - r and
                        0.01 + r <= new_y <= rect_height - 0.01 - r):

                        # Validate with spatial indexing
                        temp_circles = best_circles.copy()
                        temp_circles[i, 0] = new_x
                        temp_circles[i, 1] = new_y

                        # Check overlap
                        valid = True
                        temp_points = temp_circles[:, :2]
                        temp_tree = KDTree(temp_points)
                        temp_neighbor_indices = temp_tree.query_ball_point([new_x, new_y], 2*(r + 0.01) + 0.001)

                        for k in temp_neighbor_indices:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((new_x - other_cx)**2 + (new_y - other_cy)**2)
                                if dist < r + other_r:
                                    valid = False
                                    break

                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 0] = new_x
                            test_circles[i, 1] = new_y
                            test_fitness = calculate_fitness_with_spatial_indexing(test_circles)

                            if test_fitness > best_fitness:
                                best_fitness = test_fitness
                                best_circles = test_circles
                                improved = True

            # Track improvement
            improvement_history.append(best_fitness)
            if len(improvement_history) > 10:
                improvement_history.pop(0)

            # Stop early if no meaningful improvement for several iterations
            if len(improvement_history) >= 10:
                recent_improvements = [improvement_history[i] - improvement_history[i-1]
                                     for i in range(1, len(improvement_history))]
                if all(improvement < 1e-6 for improvement in recent_improvements):
                    break

        return best_circles

    # Multi-start optimization to avoid local minima
    def multi_start_optimization(initial_solution):
        best_solution = initial_solution.copy()
        best_fitness = calculate_fitness_with_spatial_indexing(best_solution)

        # Try multiple random restarts to find better solutions
        num_restarts = 5
        for restart in range(num_restarts):
            # Create a new random initialization
            random_solution = generate_improved_initial_placement(n, rect_width, rect_height)

            # Apply refinement to this random start
            refined_solution = enhanced_refinement(random_solution, max_iter=50)
            refined_fitness = calculate_fitness_with_spatial_indexing(refined_solution)

            if refined_fitness > best_fitness:
                best_fitness = refined_fitness
                best_solution = refined_solution.copy()

        return best_solution

    # Main optimization pipeline
    # Phase 1: Generate initial solution with better placement
    initial_circles = generate_improved_initial_placement(n, rect_width, rect_height)

    # Phase 2: Multi-start local optimization to improve starting point
    improved_initial = multi_start_optimization(initial_circles)

    # Phase 3: Enhanced local refinement
    refined_solution = enhanced_refinement(improved_initial, max_iter=80)

    # Final validation and cleanup
    final_fitness = calculate_fitness_with_spatial_indexing(refined_solution)

    # Ensure all constraints are satisfied
    points = refined_solution[:, :2]
    tree = KDTree(points)

    # Final validation pass
    valid = True
    for i in range(n):
        cx, cy, r = refined_solution[i]

        # Check boundary constraints
        if cx - r < 0.01 or cx + r > rect_width - 0.01 or \
           cy - r < 0.01 or cy + r > rect_height - 0.01:
            valid = False
            break

        # Check overlap constraints
        neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)
        for j in neighbor_indices:
            if i != j:
                other_cx, other_cy, other_r = refined_solution[j]
                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                if dist < r + other_r:
                    valid = False
                    break

    # If validation fails, do one last adaptive refinement
    if not valid:
        refined_solution = enhanced_refinement(refined_solution, max_iter=30)

    return refined_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")