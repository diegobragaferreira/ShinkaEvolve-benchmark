# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2, using 1.2 x 0.8 ratio as it's often effective
    rect_width = 1.2
    rect_height = 0.8

    # Number of circles
    n = 21

    # Generate grid-based initial placement using hexagonal packing pattern
    def generate_hexagonal_grid(num_circles, width, height):
        # Hexagonal grid parameters
        rows = int(np.ceil(np.sqrt(num_circles)))
        cols = int(np.ceil(num_circles / rows))

        # Calculate spacing to fit within container
        cell_size = min(width / cols, height / rows) * 0.9  # Safety margin
        spacing_x = cell_size * 1.1  # Slightly offset to make hexagonal pattern
        spacing_y = cell_size * np.sqrt(3) / 2 * 1.1

        circles = []
        y_offset = 0.1  # Margin from top
        x_offset = 0.1  # Margin from left

        circle_count = 0
        for i in range(rows):
            y = y_offset + i * spacing_y
            # Offset every other row for hexagonal pattern
            x_start = x_offset + (i % 2) * spacing_x / 2
            for j in range(cols):
                if circle_count >= num_circles:
                    break
                x = x_start + j * spacing_x
                if x < width - 0.1 and y < height - 0.1:  # Within bounds
                    # Start with small radius
                    r = min(0.05, spacing_x / 4, spacing_y / 4)
                    circles.append([x, y, r])
                    circle_count += 1
            if circle_count >= num_circles:
                break

        # If we didn't place enough circles, fill remaining space randomly
        while len(circles) < num_circles:
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            r = np.random.uniform(0.01, 0.05)
            circles.append([x, y, r])

        return np.array(circles)

    # Initial grid-based configuration
    circles = generate_hexagonal_grid(n, rect_width, rect_height)

    # Constraint validation and penalty calculation with spatial indexing for efficiency
    def calculate_fitness(circles_array):
        total_radius = np.sum(circles_array[:, 2])

        penalty = 0

        # Boundary penalties (quadratic for smooth gradient)
        for i in range(n):
            cx, cy, r = circles_array[i]
            if cx - r < 0.01:
                penalty += 10000 * (r - cx)**2
            if cx + r > rect_width - 0.01:
                penalty += 10000 * (cx + r - rect_width)**2
            if cy - r < 0.01:
                penalty += 10000 * (r - cy)**2
            if cy + r > rect_height - 0.01:
                penalty += 10000 * (cy + r - rect_height)**2

        # Overlap penalties using spatial indexing for efficiency (O(n) instead of O(n^2))
        # Build KDTree for efficient neighbor search
        points = circles_array[:, :2]  # Only x,y coordinates
        tree = KDTree(points)

        # Query for neighbors within sum of radii distance
        for i in range(n):
            cx, cy, r = circles_array[i]

            # Find neighbors that could possibly overlap
            neighbor_indices = tree.query_ball_point([cx, cy], 2*r + 0.001)

            # Check actual overlaps with neighbors
            for j in neighbor_indices:
                if i != j:  # Don't compare with self
                    other_cx, other_cy, other_r = circles_array[j]
                    dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                    overlap = (r + other_r) - dist

                    if overlap > 0:
                        penalty += 100000 * overlap**2

        return total_radius - penalty

    # Enhanced local refinement with greedy improvement and spatial indexing
    def refine_circles(circles_array, max_iter=100):
        best_circles = circles_array.copy()
        best_fitness = calculate_fitness(best_circles)

        for iteration in range(max_iter):
            improved = False

            # Try to increase each circle's radius
            for i in range(n):
                cx, cy, r = best_circles[i]

                # Compute max allowable radius using spatial indexing for efficiency
                max_radius = float('inf')

                # Boundary constraints
                max_radius = min(max_radius, cx - 0.01)
                max_radius = min(max_radius, rect_width - cx - 0.01)
                max_radius = min(max_radius, cy - 0.01)
                max_radius = min(max_radius, rect_height - cy - 0.01)

                # Overlap constraints with nearby circles using spatial indexing
                points = best_circles[:, :2]  # Only x,y coordinates
                tree = KDTree(points)
                neighbor_indices = tree.query_ball_point([cx, cy], 2*(r + 0.01) + 0.001)

                for j in neighbor_indices:
                    if i != j:
                        other_cx, other_cy, other_r = best_circles[j]
                        dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                        max_radius = min(max_radius, dist - other_r - 0.001)  # Small safety margin

                # Try to increase radius
                if max_radius > r and max_radius > 0.001:
                    # Test several increments
                    test_increments = [0.005, 0.01, 0.02]
                    for incr in test_increments:
                        new_r = min(r + incr, max_radius)
                        if new_r <= r:
                            continue

                        # Check validity of new configuration using spatial indexing
                        valid = True
                        temp_circles = best_circles.copy()
                        temp_circles[i, 2] = new_r

                        # Quick neighbor check using spatial index before full validation
                        points_new = temp_circles[:, :2]
                        tree_new = KDTree(points_new)
                        neighbor_indices_new = tree_new.query_ball_point([cx, cy], 2*(new_r + 0.01) + 0.001)

                        for k in neighbor_indices_new:
                            if k != i:
                                other_cx, other_cy, other_r = temp_circles[k]
                                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    valid = False
                                    break

                        if valid:
                            test_circles = best_circles.copy()
                            test_circles[i, 2] = new_r
                            test_fitness = calculate_fitness(test_circles)

                            if test_fitness > best_fitness:
                                best_circles = test_circles
                                best_fitness = test_fitness
                                improved = True
                                break

            if not improved:
                break

        return best_circles

    # Refine initial grid placement
    refined_circles = refine_circles(circles)
    best_solution = refined_circles.copy()
    best_fitness = calculate_fitness(refined_circles)

    # Optimization using scipy minimize with gradient-free method
    def objective(x):
        circles = x.reshape(-1, 3)
        # Calculate sum of radii (we want to maximize this)
        total_radius = np.sum(circles[:, 2])
        # Return negative because we're minimizing in scipy
        return -total_radius

    def penalty_function(x):
        """Calculate penalty for constraint violations"""
        circles = x.reshape(-1, 3)
        
        penalty = 0
        
        # Boundary penalties (penalize if circles go outside)
        for i in range(n):
            cx, cy, r = circles[i]
            # Penalty for going outside
            if cx - r < 0:
                penalty += 1000 * (r - cx)**2
            if cx + r > rect_width:
                penalty += 1000 * (cx + r - rect_width)**2
            if cy - r < 0:
                penalty += 1000 * (r - cy)**2
            if cy + r > rect_height:
                penalty += 1000 * (cy + r - rect_height)**2
        
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1, r1 = circles[i]
                cx2, cy2, r2 = circles[j]
                
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                overlap = (r1 + r2) - dist
                
                if overlap > 0:  # Overlapping
                    penalty += 10000 * overlap**2
        
        return penalty

    # Run scipy optimization as preliminary step
    try:
        # Combine objective and penalty function
        combined_objective = lambda x: objective(x) + penalty_function(x)
        
        result = minimize(
            combined_objective,
            best_solution.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'disp': False, 'maxfev': 2000}
        )
        
        if result.success:
            scipy_solution = result.x.reshape(-1, 3)
            scipy_fitness = calculate_fitness(scipy_solution)
            if scipy_fitness > best_fitness:
                best_solution = scipy_solution
                best_fitness = scipy_fitness
                
    except Exception as e:
        pass

    # Final refinement to maximize sum of radii
    final_solution = refine_circles(best_solution, max_iter=100)

    return final_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
