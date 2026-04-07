# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, optimizing for better packing
    # Test several aspect ratios to find optimal
    test_ratios = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    best_sum = 0
    best_circles = None

    for ratio in test_ratios:
        width = 2 * ratio / (1 + ratio)  # Ensure perimeter = 4
        height = 2 * 1 / (1 + ratio)

        # Number of circles
        n = 21

        # Enhanced grid initialization with better aspect ratio adaptation
        def initialize_better_layout():
            circles = []

            # Calculate optimal grid dimensions based on rectangle aspect ratio
            aspect_ratio = width / height
            if aspect_ratio >= 1.2:  # Landscape
                cols = int(np.ceil(np.sqrt(n * aspect_ratio * 1.3)))
                rows = int(np.ceil(n / cols))
            elif aspect_ratio <= 0.8:  # Portrait
                rows = int(np.ceil(np.sqrt(n / aspect_ratio * 1.3)))
                cols = int(np.ceil(n / rows))
            else:  # Balanced
                cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
                rows = int(np.ceil(n / cols))

            # Ensure we have enough cells
            while cols * rows < n:
                if aspect_ratio >= 1.2:
                    cols += 1
                elif aspect_ratio <= 0.8:
                    rows += 1
                else:
                    cols += 1

            # Calculate spacing with better consideration
            spacing_x = width / (cols + 1) if cols > 0 else width
            spacing_y = height / (rows + 1) if rows > 0 else height

            # Create more efficient hexagonal-like packing with improved spacing
            placed_count = 0
            for i in range(rows):
                for j in range(cols):
                    if placed_count >= n:
                        break

                    # Offset every other row for hexagonal packing
                    offset_x = spacing_x * 0.5 if i % 2 == 1 else 0
                    base_x = (j + 1) * spacing_x + offset_x
                    base_y = (i + 1) * spacing_y

                    # Add small random perturbation for diversity
                    perturbation_factor = min(0.15, 0.2 * min(spacing_x, spacing_y))
                    x = np.clip(base_x + np.random.uniform(-perturbation_factor, perturbation_factor),
                               0.01, width - 0.01)
                    y = np.clip(base_y + np.random.uniform(-perturbation_factor, perturbation_factor),
                               0.01, height - 0.01)

                    # Initial radius estimation with better heuristics
                    max_r = min(x, width - x, y, height - y)
                    estimated_radius = min(0.15, max_r * 0.7)
                    r = np.random.uniform(estimated_radius * 0.6, estimated_radius * 1.0)

                    circles.append([x, y, r])
                    placed_count += 1

                if placed_count >= n:
                    break

            # Fill remaining slots if needed
            while len(circles) < n:
                x = np.random.uniform(0.01, width - 0.01)
                y = np.random.uniform(0.01, height - 0.01)
                # Ensure reasonable initial radius
                r = min(0.1, width/10, height/10)
                circles.append([x, y, r])

            return np.array(circles)

        # Get initial configuration
        circles = initialize_better_layout()

        # Define bounds for optimization
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

        # Optimized constraint validation with spatial indexing
        def validate_constraints(circles_array):
            """Validate constraints more efficiently using spatial indexing"""
            # Boundary check
            for i in range(n):
                x, y, r = circles_array[i]
                if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                    return False

            # Overlap check using spatial indexing for better performance
            try:
                # Build tree for efficient neighbor search
                points = circles_array[:, :2]
                tree = cKDTree(points)

                # Find near neighbors to check for overlaps
                max_radius = np.max(circles_array[:, 2])
                pairs = tree.query_pairs(2 * max_radius, output_type='ndarray')

                for i, j in pairs:
                    if i < j:  # Avoid checking same pair twice
                        x1, y1, r1 = circles_array[i]
                        x2, y2, r2 = circles_array[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        distance_sq = dx*dx + dy*dy
                        radius_sum_sq = (r1 + r2) * (r1 + r2)

                        if distance_sq < radius_sum_sq:
                            return False
            except:
                # Fallback to direct checking if spatial index fails
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1, r1 = circles_array[i]
                        x2, y2, r2 = circles_array[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        distance = np.sqrt(dx*dx + dy*dy)

                        if distance < (r1 + r2):
                            return False

            return True

        # Constraints function using optimized validation
        def create_constraints():
            cons = []
            # Add simple boundary constraints that are easier for SLSQP to handle
            for i in range(n):
                def bound_constraint(vars, i=i):
                    circles_temp = set_variables(vars)
                    x, y, r = circles_temp[i]
                    return min(x - r, width - x - r, y - r, height - y - r)
                cons.append({'type': 'ineq', 'fun': bound_constraint})
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
                options={'maxiter': 200, 'ftol': 1e-6}
            )

            if result.success:
                optimized_circles = set_variables(result.x)
            else:
                # Fallback to original circles if optimization fails
                optimized_circles = circles.copy()

        except Exception as e:
            # If optimization fails for any reason, use original circles
            optimized_circles = circles.copy()

        # Final refinement step with probabilistic reinitialization
        def advanced_refinement(circles_array):
            # First, validate current solution
            if not validate_constraints(circles_array):
                # Try to fix by reinitializing some circles
                ref_count = int(0.1 * n)  # Reinitialize 10% of circles
                indices = random.sample(range(n), ref_count)
                for idx in indices:
                    x = np.random.uniform(0.01, width - 0.01)
                    y = np.random.uniform(0.01, height - 0.01)
                    max_r = min(x, width - x, y, height - y)
                    r = min(0.1, max_r * 0.7)
                    circles_array[idx] = [x, y, r]

            # Greedy refinement with smarter approach
            improved = True
            iter_count = 0
            while improved and iter_count < 50:
                improved = False
                iter_count += 1

                # Try to increase each circle's radius if possible
                for i in range(n):
                    test_circles = circles_array.copy()
                    old_radius = test_circles[i, 2]

                    # Try to increase radius with small increments
                    delta = min(0.005, 0.1 - old_radius)
                    if delta > 0:
                        test_circles[i, 2] = min(old_radius + delta, 0.3)

                        # Validate the new configuration
                        if validate_constraints(test_circles):
                            new_sum = np.sum(test_circles[:, 2])
                            old_sum = np.sum(circles_array[:, 2])

                            if new_sum > old_sum:
                                circles_array = test_circles.copy()
                                improved = True

            return circles_array

        # Apply advanced refinement
        refined_circles = advanced_refinement(optimized_circles)

        # Check if this configuration is better
        current_sum = np.sum(refined_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = refined_circles.copy()

    return best_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")