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

    # Enhanced hexagonal initialization based on rectangle dimensions and circle count
    def initialize_hexagonal_layout():
        circles = []

        # Calculate optimal spacing using theoretical hexagonal packing principles
        # For n circles in area A, optimal radius ~ sqrt(A/(pi*n)) * 0.9
        rect_area = width * height
        estimated_radius = np.sqrt(rect_area / (np.pi * n)) * 0.9

        # Determine grid dimensions respecting aspect ratio
        aspect_ratio = width / height

        # Use more sophisticated approach for grid sizing
        if aspect_ratio >= 1.2:  # Landscape orientation
            cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
            rows = int(np.ceil(n / cols))
            # Ensure balance
            if cols * rows < n:
                cols += 1
        elif aspect_ratio <= 0.8:  # Portrait orientation
            rows = int(np.ceil(np.sqrt(n / aspect_ratio)))
            cols = int(np.ceil(n / rows))
            # Ensure balance
            if cols * rows < n:
                rows += 1
        else:  # Balanced
            cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
            rows = int(np.ceil(n / cols))
            # Ensure balance
            if cols * rows < n:
                cols += 1

        # Ensure sufficient grid size
        while cols * rows < n:
            if aspect_ratio >= 1.2:
                cols += 1
            elif aspect_ratio <= 0.8:
                rows += 1
            else:
                cols += 1

        # Calculate actual spacing
        spacing_x = width / (cols + 1) if cols > 0 else width
        spacing_y = height / (rows + 1) if rows > 0 else height

        # Create hexagonal grid with proper spacing
        placed_count = 0

        # For hexagonal packing, we want to use the natural spacing of circles
        # In ideal hexagonal packing, circles are arranged with horizontal spacing = 2*r
        # and vertical spacing = sqrt(3)*r

        # Adjust spacing based on actual geometry
        ideal_spacing_x = 2 * estimated_radius
        ideal_spacing_y = np.sqrt(3) * estimated_radius

        # Determine how many rows/columns we actually need for ideal packing
        actual_cols = int(np.floor(width / ideal_spacing_x)) if ideal_spacing_x > 0 else 1
        actual_rows = int(np.floor(height / ideal_spacing_y)) if ideal_spacing_y > 0 else 1

        # If this gives too few circles, make it larger
        if actual_cols * actual_rows < n:
            # Scale up to accommodate more circles
            scale_factor = np.sqrt(n / (actual_cols * actual_rows)) if (actual_cols * actual_rows) > 0 else 1
            actual_cols = max(1, int(actual_cols * scale_factor))
            actual_rows = max(1, int(actual_rows * scale_factor))

        # Use adjusted spacing
        final_spacing_x = width / (actual_cols + 1) if actual_cols > 0 else width
        final_spacing_y = height / (actual_rows + 1) if actual_rows > 0 else height

        # Create hexagonal-like arrangement
        for i in range(actual_rows):
            for j in range(actual_cols):
                if placed_count >= n:
                    break

                # Hexagonal offset for alternate rows
                offset_x = final_spacing_x * 0.5 if i % 2 == 1 else 0
                base_x = (j + 1) * final_spacing_x + offset_x
                base_y = (i + 1) * final_spacing_y

                # Add more substantial random perturbation for better diversity
                perturbation_x = np.random.uniform(-0.2 * final_spacing_x, 0.2 * final_spacing_x)
                perturbation_y = np.random.uniform(-0.2 * final_spacing_y, 0.2 * final_spacing_y)

                x = np.clip(base_x + perturbation_x, estimated_radius, width - estimated_radius)
                y = np.clip(base_y + perturbation_y, estimated_radius, height - estimated_radius)

                # Estimate radius based on available space
                max_radius = min(x, width - x, y, height - y)
                # Use more informed initial radius
                r = min(estimated_radius * np.random.uniform(0.8, 1.2), max_radius * 0.8)

                circles.append([x, y, r])
                placed_count += 1

        # Fill remaining slots with more strategic random placements
        while len(circles) < n:
            x = np.random.uniform(estimated_radius, width - estimated_radius)
            y = np.random.uniform(estimated_radius, height - estimated_radius)

            # Check if this position conflicts with existing circles
            valid = True
            for cx, cy, cr in circles:
                if np.sqrt((x - cx)**2 + (y - cy)**2) < cr + estimated_radius * 0.3:
                    valid = False
                    break

            if valid:
                max_radius = min(x, width - x, y, height - y)
                r = np.random.uniform(estimated_radius * 0.5, min(0.15, max_radius * 0.7))
                circles.append([x, y, r])

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