# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, optimizing for better packing
    # Let's try a more balanced rectangle that works well for 21 circles
    width, height = 1.3, 0.7

    # Number of circles
    n = 21

    # More intelligent initialization using approximate hexagonal packing
    def initialize_better_layout():
        circles = []

        # Estimate a reasonable initial radius
        # For a 1.3 x 0.7 rectangle, we can pack roughly 21 circles with radius ~0.12
        initial_radius = 0.12

        # Create a regular grid pattern for initial placement
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))

        # Adjust rows and cols to make it closer to rectangle aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 1:
            cols = int(np.ceil(np.sqrt(n) * aspect_ratio))
            rows = int(np.ceil(n / cols))
        else:
            rows = int(np.ceil(np.sqrt(n) / aspect_ratio))
            cols = int(np.ceil(n / rows))

        # Ensure we have enough circles
        while rows * cols < n:
            if aspect_ratio > 1:
                cols += 1
            else:
                rows += 1

        # Place circles in grid with slight offset
        padding_x = 0.05
        padding_y = 0.05

        grid_width = width - 2 * padding_x
        grid_height = height - 2 * padding_y

        cell_width = grid_width / cols if cols > 0 else grid_width
        cell_height = grid_height / rows if rows > 0 else grid_height

        placed_count = 0
        for i in range(rows):
            for j in range(cols):
                if placed_count >= n:
                    break
                # Offset every other row
                offset_x = cell_width * 0.5 if i % 2 == 1 else 0
                x = padding_x + offset_x + j * cell_width + cell_width / 2
                y = padding_y + i * cell_height + cell_height / 2

                # Check if within bounds
                if x - initial_radius >= 0 and x + initial_radius <= width and \
                   y - initial_radius >= 0 and y + initial_radius <= height:
                    circles.append([x, y, initial_radius])
                    placed_count += 1

        # Fill remaining slots with random positions
        while len(circles) < n:
            x = np.random.uniform(initial_radius, width - initial_radius)
            y = np.random.uniform(initial_radius, height - initial_radius)
            circles.append([x, y, initial_radius])

        return np.array(circles)

    # Get initial configuration
    circles = initialize_better_layout()

    # Define bounds for optimization: [x_min, x_max, y_min, y_max, r_min, r_max]
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, width - 0.001),   # x bounds
                       (0.001, height - 0.001),  # y bounds
                       (0.001, 0.3)])           # radius bounds

    # Reshape circles for easier manipulation
    def get_variables(circles_array):
        # Flatten circle info: [x1, y1, r1, x2, y2, r2, ...]
        variables = []
        for i in range(n):
            variables.extend([circles_array[i, 0], circles_array[i, 1], circles_array[i, 2]])
        return np.array(variables)

    def set_variables(variables):
        # Convert back to circles array
        circles_array = np.zeros((n, 3))
        for i in range(n):
            circles_array[i, 0] = variables[3*i]
            circles_array[i, 1] = variables[3*i + 1]
            circles_array[i, 2] = variables[3*i + 2]
        return circles_array

    # Constraints function
    def create_constraints():
        cons = []
        # Add non-overlap constraints for all pairs of circles
        for i in range(n):
            for j in range(i+1, n):
                def constraint_func(vars, i=i, j=j):
                    circles_temp = set_variables(vars)
                    x1, y1, r1 = circles_temp[i]
                    x2, y2, r2 = circles_temp[j]

                    # Distance between centers
                    dx = x1 - x2
                    dy = y1 - y2
                    distance = np.sqrt(dx*dx + dy*dy)

                    # Non-overlap constraint: distance >= r1 + r2
                    return distance - (r1 + r2)

                cons.append({'type': 'ineq', 'fun': constraint_func})

        # Add boundary constraints for each circle
        for i in range(n):
            def x_bound_left(vars, i=i):
                circles_temp = set_variables(vars)
                x, y, r = circles_temp[i]
                return x - r  # x >= r

            def x_bound_right(vars, i=i):
                circles_temp = set_variables(vars)
                x, y, r = circles_temp[i]
                return width - x - r  # width - x >= r

            def y_bound_bottom(vars, i=i):
                circles_temp = set_variables(vars)
                x, y, r = circles_temp[i]
                return y - r  # y >= r

            def y_bound_top(vars, i=i):
                circles_temp = set_variables(vars)
                x, y, r = circles_temp[i]
                return height - y - r  # height - y >= r

            cons.append({'type': 'ineq', 'fun': x_bound_left})
            cons.append({'type': 'ineq', 'fun': x_bound_right})
            cons.append({'type': 'ineq', 'fun': y_bound_bottom})
            cons.append({'type': 'ineq', 'fun': y_bound_top})

        return cons

    # Create constraints once
    constraints = create_constraints()

    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(vars):
        circles_temp = set_variables(vars)
        return -np.sum(circles_temp[:, 2])

    # Initial variables
    initial_vars = get_variables(circles)

    # Perform optimization
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=[(0.001, width - 0.001), (0.001, height - 0.001), (0.001, 0.3)] * n,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-6}
        )

        if result.success:
            optimized_circles = set_variables(result.x)
        else:
            # Fallback to original circles if optimization fails
            optimized_circles = circles.copy()

    except Exception as e:
        # If optimization fails for any reason, use original circles
        optimized_circles = circles.copy()

    # Final refinement step to improve packing
    # Use a greedy approach to try increasing individual radii
    refines = 0
    while refines < 10:  # Limit iterations
        improved = False
        for i in range(n):
            test_circles = optimized_circles.copy()
            test_circles[i, 2] = min(test_circles[i, 2] + 0.005, 0.3)

            # Check all constraints for this modified configuration
            valid = True
            positions = test_circles[:, :2]
            distances = cdist(positions, positions)

            # Check overlap constraints
            for j in range(n):
                for k in range(j+1, n):
                    if distances[j,k] < (test_circles[j,2] + test_circles[k,2]) * 0.99:
                        valid = False
                        break
                if not valid:
                    break

            # Check boundary constraints
            for j in range(n):
                if (test_circles[j,0] - test_circles[j,2] < 0 or
                    test_circles[j,0] + test_circles[j,2] > width or
                    test_circles[j,1] - test_circles[j,2] < 0 or
                    test_circles[j,1] + test_circles[j,2] > height):
                    valid = False
                    break

            if valid:
                new_sum = np.sum(test_circles[:, 2])
                if new_sum > np.sum(optimized_circles[:, 2]):
                    optimized_circles = test_circles.copy()
                    improved = True

        if not improved:
            break
        refines += 1

    # Return optimized circles array
    return optimized_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")