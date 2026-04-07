# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Rectangle dimensions: width + height = 2
    # Use optimized aspect ratio (2:1) for better packing efficiency
    rect_width = 1.3333333333333333  # 2/3
    rect_height = 0.6666666666666666  # 1/3

    # Number of circles
    n = 21

    def objective(x):
        # x contains [cx1, cy1, r1, cx2, cy2, r2, ..., cxn, cyn, rn]
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
            # Penalty for going outside with stronger penalties
            if cx - r < 0:
                penalty += 10000 * (r - cx)**2
            if cx + r > rect_width:
                penalty += 10000 * (cx + r - rect_width)**2
            if cy - r < 0:
                penalty += 10000 * (r - cy)**2
            if cy + r > rect_height:
                penalty += 10000 * (cy + r - rect_height)**2

        # Overlap penalties with higher penalty weights
        for i in range(n):
            for j in range(i+1, n):
                cx1, cy1, r1 = circles[i]
                cx2, cy2, r2 = circles[j]

                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                overlap = (r1 + r2) - dist

                if overlap > 0:  # Overlapping
                    penalty += 100000 * overlap**2

        return penalty

    # Generate initial guess with improved hexagonal arrangement and adaptive grid
    def generate_initial_guess():
        circles = np.zeros((n, 3))

        # Calculate grid dimensions adapted to the 2:1 aspect ratio
        # For 21 circles, a 5x5 grid works well (25 positions)
        grid_rows = 5
        grid_cols = 5

        # Calculate spacing with safety margins
        spacing_x = rect_width / (grid_cols + 1)
        spacing_y = rect_height / (grid_rows + 1)

        # Hexagonal packing factor (more compact arrangement)
        hex_spacing_x = spacing_x * 0.75
        hex_spacing_y = spacing_y * 0.866  # sqrt(3)/2

        idx = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                if idx >= n:
                    break
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * (hex_spacing_x / 2)
                x = hex_spacing_x * j + x_offset + hex_spacing_x
                y = hex_spacing_y * i + hex_spacing_y

                # Ensure within bounds with safety margin
                x = max(hex_spacing_x, min(rect_width - hex_spacing_x, x))
                y = max(hex_spacing_y, min(rect_height - hex_spacing_y, y))

                # Initialize with radius based on spacing, but smaller for optimization
                r = min(hex_spacing_x, hex_spacing_y) * 0.3

                circles[idx] = [x, y, r]
                idx += 1
                if idx >= n:
                    break

        # Fill remaining slots if needed
        if idx < n:
            for i in range(idx, n):
                # Place randomly within bounds
                x = np.random.uniform(hex_spacing_x, rect_width - hex_spacing_x)
                y = np.random.uniform(hex_spacing_y, rect_height - hex_spacing_y)
                r = min(hex_spacing_x, hex_spacing_y) * 0.3
                circles[i] = [x, y, r]

        return circles.flatten()

    # Fast overlap check using spatial indexing
    def fast_overlap_check(circles, tree, i, new_radius):
        """Fast overlap check using spatial indexing for efficiency"""
        cx, cy, _ = circles[i]
        # Query nearby circles within 2*new_radius distance (more efficient than full check)
        nearby_indices = tree.query_ball_point([cx, cy], 2 * new_radius)

        for j in nearby_indices:
            if i != j:
                other_cx, other_cy, other_r = circles[j]
                dist = np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2)
                if dist < new_radius + other_r:
                    return False
        return True

    # Multi-stage optimization approach
    try:
        # Generate initial guess
        initial_guess = generate_initial_guess()

        # Stage 1: Coarse optimization with COBYLA for better constraint handling
        print("Running coarse optimization...")
        combined_objective_stage1 = lambda x: objective(x) + penalty_function(x)

        result1 = minimize(
            combined_objective_stage1,
            initial_guess,
            method='COBYLA',
            options={'maxiter': 1000, 'disp': False, 'catol': 1e-6, 'rhobeg': 0.1}
        )

        if result1.success:
            stage1_solution = result1.x.reshape(-1, 3)
            print(f"Stage 1 success, objective: {-result1.fun}")
        else:
            stage1_solution = initial_guess.reshape(-1, 3)
            print("Stage 1 failed, using initial guess")

        # Stage 2: Fine-tuned optimization with L-BFGS-B
        print("Running fine-tuning optimization...")
        combined_objective_stage2 = lambda x: objective(x) + penalty_function(x)

        result2 = minimize(
            combined_objective_stage2,
            stage1_solution.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False, 'ftol': 1e-9, 'gtol': 1e-6}
        )

        if result2.success:
            final_circles = result2.x.reshape(-1, 3)
            print(f"Stage 2 success, objective: {-result2.fun}")
        else:
            final_circles = stage1_solution
            print("Stage 2 failed, using stage 1 solution")

    except Exception as e:
        print(f"Optimization failed with exception: {e}")
        # Fallback to initial guess if anything goes wrong
        final_circles = generate_initial_guess().reshape(-1, 3)

    # Final adjustment using improved greedy refinement with spatial indexing
    refined_circles = final_circles.copy()

    # Build spatial index for efficient overlap checking
    tree = cKDTree(refined_circles[:, :2])

    # Advanced refinement with better constraint validation
    max_iter = 1000
    improvement_threshold = 1e-6

    print("Performing final refinement...")
    for iteration in range(max_iter):
        improved = False
        total_improvement = 0

        # Shuffle circle indices for better exploration
        indices = list(range(n))
        np.random.shuffle(indices)

        for i in indices:
            current_cx, current_cy, current_r = refined_circles[i]

            # Find maximum allowable radius efficiently
            max_radius = float('inf')

            # Check boundary constraints
            boundary_radius = min([
                current_cx,  # left
                rect_width - current_cx,  # right
                current_cy,  # bottom
                rect_height - current_cy   # top
            ])
            max_radius = min(max_radius, boundary_radius)

            # Check overlap constraints with nearby circles only (with spatial indexing)
            nearby_indices = tree.query_ball_point([current_cx, current_cy], 3 * max_radius)
            for j in nearby_indices:
                if i != j:
                    other_cx, other_cy, other_r = refined_circles[j]
                    dist = np.sqrt((current_cx - other_cx)**2 + (current_cy - other_cy)**2)
                    # Max radius to prevent overlap
                    max_allowed_radius = dist - other_r
                    if max_allowed_radius > 0:
                        max_radius = min(max_radius, max_allowed_radius)

            # Try to increase radius with adaptive step sizes
            if max_radius > current_r and max_radius > 0:
                # Try several step sizes for better adaptation
                step_sizes = [0.005, 0.01, 0.02, 0.03]
                best_new_r = current_r
                best_valid = False

                for step in step_sizes:
                    new_r = min(current_r + step, max_radius)
                    if new_r <= current_r:
                        continue

                    # Quick check first using spatial index
                    valid = fast_overlap_check(refined_circles, tree, i, new_r)

                    if valid:
                        # Full validation to be extra sure
                        full_valid = True
                        for k in range(n):
                            if i != k:
                                other_cx, other_cy, other_r = refined_circles[k]
                                dist = np.sqrt((current_cx - other_cx)**2 + (current_cy - other_cy)**2)
                                if dist < new_r + other_r:
                                    full_valid = False
                                    break

                        if full_valid:
                            best_new_r = new_r
                            best_valid = True
                            break

                if best_valid:
                    refined_circles[i, 2] = best_new_r
                    improved = True
                    total_improvement += (best_new_r - current_r)

        # Rebuild spatial index after updates for accuracy
        tree = cKDTree(refined_circles[:, :2])

        # Stop if no significant improvement
        if not improved or total_improvement < improvement_threshold:
            break

    print(f"Final radii sum: {np.sum(refined_circles[:,-1])}")
    return refined_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")