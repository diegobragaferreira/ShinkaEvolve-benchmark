# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from collections import defaultdict
from scipy.spatial import cKDTree
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width, height = 1.0, 1.0  # Optimal square configuration

    # Set seed for reproducibility
    np.random.seed(42)

    # Phase 1: Multi-resolution grid initialization with improved strategy
    n_circles = 21

    # Start with a coarse grid and refine
    coarse_grid_size = 4
    fine_grid_size = 8

    # Generate coarse grid points
    x_coarse = np.linspace(0.05, width - 0.05, coarse_grid_size)
    y_coarse = np.linspace(0.05, height - 0.05, coarse_grid_size)
    coarse_points = [(x, y) for x in x_coarse for y in y_coarse]

    # Initialize circles array
    circles = np.zeros((n_circles, 3))

    # Place circles using a greedy approach on coarse grid with better selection strategy
    placed_points = []
    placed_radii = []

    # Shuffle points for better randomness
    remaining_points = coarse_points.copy()
    np.random.shuffle(remaining_points)

    for i, (cx, cy) in enumerate(remaining_points[:n_circles]):
        # Compute max radius for this point considering boundaries
        max_radius = min(cx, width - cx, cy, height - cy)

        # Check for conflicts with already placed circles using efficient spatial search
        valid_placement = True
        if len(placed_points) > 0:
            # Use KDTree for efficient neighbor search
            placed_array = np.array(placed_points)
            tree = cKDTree(placed_array)
            distances, indices = tree.query([(cx, cy)], k=len(placed_points))

            for j, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if dist < (placed_radii[idx] + max_radius):  # Conflict with existing circle
                    valid_placement = False
                    break

        if valid_placement:
            placed_points.append((cx, cy))
            placed_radii.append(max_radius)

        if len(placed_points) >= n_circles:
            break

    # Fill remaining circles if needed with improved placement strategy
    while len(placed_points) < n_circles:
        # Place remaining circles using a combination of strategic placement and random sampling
        cx = np.random.uniform(0.05, width - 0.05)
        cy = np.random.uniform(0.05, height - 0.05)
        max_radius = min(cx, width - cx, cy, height - cy)

        # Check for conflicts using spatial indexing for better performance
        valid_placement = True
        if len(placed_points) > 0:
            placed_array = np.array(placed_points)
            tree = cKDTree(placed_array)
            distances, indices = tree.query([(cx, cy)], k=len(placed_points))

            for j, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if dist < (placed_radii[idx] + max_radius):
                    valid_placement = False
                    break

        if valid_placement:
            placed_points.append((cx, cy))
            placed_radii.append(max_radius)

    # Initialize circles array
    for i in range(n_circles):
        circles[i] = [placed_points[i][0], placed_points[i][1], placed_radii[i]]

    # Phase 2: Enhanced hybrid optimization with simulated annealing
    best_sum = np.sum(circles[:, 2])
    best_circles = circles.copy()

    # Local search parameters
    max_iterations = 2000  # Increased iterations for better exploration
    tolerance = 1e-6
    max_no_improvement = 100  # Stop if no improvement for N iterations
    convergence_threshold = 0.0001  # Threshold to detect convergence

    # Track the number of iterations without improvement and convergence history
    no_improvement_count = 0
    improvement_history = []
    last_sum = best_sum

    # Simulated Annealing parameters
    initial_temp = 0.5
    cooling_rate = 0.995
    min_temp = 1e-6

    # Pre-compute distances for faster neighbor checking
    def get_valid_neighbors(current_circles, i, step_size, search_density):
        """Get valid neighbor positions for a circle."""
        neighbors = []

        # Generate search grid with adaptive density and step size
        x_range = np.linspace(-step_size * 2, step_size * 2, search_density)
        y_range = np.linspace(-step_size * 2, step_size * 2, search_density)

        # Add boundary points
        orig_x, orig_y, orig_r = current_circles[i]
        boundary_points = [
            (0.05, orig_y), (width - 0.05, orig_y),
            (orig_x, 0.05), (orig_x, height - 0.05)
        ]

        # Add regular grid points
        for dx in x_range:
            for dy in y_range:
                neighbors.append((orig_x + dx, orig_y + dy))

        # Add boundary points
        for bx, by in boundary_points:
            neighbors.append((bx, by))

        # Add center point for local refinement
        neighbors.append((orig_x, orig_y))

        return neighbors

    def is_valid_position(pos_x, pos_y, radius, current_circles, i):
        """Check if position is valid without overlapping other circles."""
        # Check boundary constraints
        if pos_x - radius < 0.05 or pos_x + radius > width - 0.05 or \
           pos_y - radius < 0.05 or pos_y + radius > height - 0.05:
            return False

        # Check overlap constraints
        for j in range(len(current_circles)):
            if i != j:
                px, py, pr = current_circles[j]
                dist = np.sqrt((pos_x - px)**2 + (pos_y - py)**2)
                if dist < (radius + pr):
                    return False

        return True

    def compute_max_radius_at_position(pos_x, pos_y, current_circles, i):
        """Compute the maximum possible radius at a given position."""
        # Boundary constraints
        max_radius = min(pos_x - 0.05, width - 0.05 - pos_x,
                         pos_y - 0.05, height - 0.05 - pos_y)

        # Overlap constraints with existing circles
        for j in range(len(current_circles)):
            if i != j:
                px, py, pr = current_circles[j]
                dist = np.sqrt((pos_x - px)**2 + (pos_y - py)**2)
                if dist > 0:
                    max_radius_for_circle = dist - pr
                    if max_radius_for_circle < max_radius:
                        max_radius = max_radius_for_circle

        return max(max_radius, 0.001)

    # Main optimization loop with hybrid approach
    temp = initial_temp
    for iteration in range(max_iterations):
        # Copy current configuration
        current_circles = circles.copy()
        old_sum = np.sum(current_circles[:, 2])

        # Determine phase based on iteration count
        phase = "exploration"
        if iteration > max_iterations * 0.3:
            phase = "balance"
        if iteration > max_iterations * 0.7:
            phase = "exploitation"

        # Adaptive step size based on iteration and phase
        if phase == "exploration":
            step_size = 0.15
            search_density = 5
        elif phase == "balance":
            step_size = 0.08
            search_density = 7
        else:  # exploitation
            step_size = 0.03
            search_density = 9

        # For each circle, try to improve its position and radius
        improved = False

        # Instead of random ordering, sort by current radius (larger first) to prioritize large circles
        sorted_indices = np.argsort(current_circles[:, 2])[::-1]

        # Hybrid approach: alternate between different optimization strategies
        strategy = "hybrid" if iteration % 3 == 0 else "local" if iteration % 3 == 1 else "global"

        # Choose circle indices to optimize based on strategy and iteration
        if strategy == "global":
            # Optimize all circles in random order
            indices_to_optimize = list(range(n_circles))
            np.random.shuffle(indices_to_optimize)
        elif strategy == "hybrid":
            # Optimize circles in descending order of size but with some randomness
            indices_to_optimize = sorted_indices.copy()
            # Mix in some randomness for diversity
            for i in range(0, len(indices_to_optimize), 3):
                if i + 2 < len(indices_to_optimize):
                    np.random.shuffle(indices_to_optimize[i:i+3])
        else:  # local
            # Select a subset for focused optimization
            indices_to_optimize = []
            for i in sorted_indices[:min(10, len(sorted_indices))]:
                if np.random.random() < 0.7:  # 70% chance to optimize
                    indices_to_optimize.append(i)

        # Process circles to optimize
        for i_idx, i in enumerate(indices_to_optimize):
            # Store original position and radius
            orig_x, orig_y, orig_r = current_circles[i]

            # Compute max radius at current position
            max_possible_r = min(orig_x, width - orig_x, orig_y, height - orig_y)

            # Only attempt change if we can potentially increase radius
            if max_possible_r > orig_r:
                # Use adaptive neighborhood search with multi-scale approach
                best_move_x, best_move_y, best_move_r = 0, 0, orig_r
                best_new_sum = old_sum

                # Generate neighbors and test them
                neighbors = get_valid_neighbors(current_circles, i, step_size, search_density)

                # Evaluate all candidate positions
                for new_x, new_y in neighbors:
                    # Keep within bounds
                    new_x = max(0.05, min(width - 0.05, new_x))
                    new_y = max(0.05, min(height - 0.05, new_y))

                    # Compute max radius at new location
                    new_r = compute_max_radius_at_position(new_x, new_y, current_circles, i)

                    # Check if the new configuration is valid
                    if is_valid_position(new_x, new_y, new_r, current_circles, i):
                        # Calculate new sum
                        new_sum = old_sum - orig_r + new_r

                        # Simulated Annealing probability check
                        if new_sum > best_new_sum:
                            best_new_sum = new_sum
                            best_move_x, best_move_y, best_move_r = new_x - orig_x, new_y - orig_y, new_r
                        elif temp > min_temp:
                            # Accept worse solutions with probability based on temperature
                            delta = new_sum - best_new_sum
                            acceptance_prob = math.exp(delta / temp)
                            if np.random.random() < acceptance_prob:
                                best_new_sum = new_sum
                                best_move_x, best_move_y, best_move_r = new_x - orig_x, new_y - orig_y, new_r

                # Apply the best move if it improves the sum or is accepted via SA
                if best_new_sum > old_sum:
                    current_circles[i] = [orig_x + best_move_x, orig_y + best_move_y, best_move_r]
                    circles = current_circles.copy()
                    improved = True

        # Check for improvement and update history
        new_sum = np.sum(circles[:, 2])
        improvement = new_sum - last_sum
        improvement_history.append(improvement)
        last_sum = new_sum

        if new_sum > best_sum:
            best_sum = new_sum
            best_circles = circles.copy()
            no_improvement_count = 0  # Reset counter on improvement
        else:
            no_improvement_count += 1

        # Apply cooling schedule for simulated annealing
        temp *= cooling_rate

        # Early stopping condition
        if no_improvement_count >= max_no_improvement:
            break

        # Convergence check
        if len(improvement_history) > 20:
            recent_avg = np.mean(improvement_history[-20:])
            if abs(recent_avg) < convergence_threshold:
                break

    # Phase 3: Enhanced final optimization with specialized strategies
    # Use a more efficient search for final refinement with fewer iterations
    max_final_iter = 500  # Increased for better final optimization

    # Further enhance the final optimization with a more diverse search space
    for iteration in range(max_final_iter):
        # Perform different types of optimization based on iteration cycle
        optimization_type = iteration % 4

        # Randomly select circles to optimize
        if optimization_type == 0:  # Full optimization
            indices = list(range(n_circles))
        elif optimization_type == 1:  # Subset optimization
            indices = np.random.choice(n_circles, size=min(12, n_circles), replace=False)
        elif optimization_type == 2:  # Small subset, focus on smaller circles
            # Focus on smaller circles which are more likely to gain
            sorted_radii = np.argsort(circles[:, 2])
            indices = sorted_radii[:min(8, n_circles)]
        else:  # Focus on largest circles
            sorted_radii = np.argsort(circles[:, 2])[::-1]
            indices = sorted_radii[:min(8, n_circles)]

        improved = False
        for idx in indices:
            old_x, old_y, old_r = circles[idx]

            # Adaptive search - start with larger steps, get more precise near end
            step_size = 0.1 if iteration < max_final_iter * 0.5 else 0.03

            # Try multiple configurations around current position
            best_x, best_y = old_x, old_y
            best_r = old_r
            best_sum = np.sum(circles[:, 2])

            # Sample neighborhood more intelligently with variable density
            if iteration < max_final_iter * 0.3:
                # Coarse search early on
                search_density = 5
            elif iteration < max_final_iter * 0.7:
                # Medium search
                search_density = 7
            else:
                # Fine search at the end
                search_density = 9

            # Try multiple candidate positions
            candidate_positions = []
            search_radius = step_size * 3  # Search area

            # Regular grid around current position
            x_vals = np.linspace(-search_radius, search_radius, search_density)
            y_vals = np.linspace(-search_radius, search_radius, search_density)

            for dx in x_vals:
                for dy in y_vals:
                    candidate_positions.append((old_x + dx, old_y + dy))

            # Add boundary points to encourage boundary exploitation
            candidate_positions.extend([(0.05, old_y), (width-0.05, old_y)])  # Left/right boundaries
            candidate_positions.extend([(old_x, 0.05), (old_x, height-0.05)])  # Top/bottom boundaries

            # Filter out invalid positions and evaluate each
            for new_x, new_y in candidate_positions:
                new_x = max(0.05, min(width - 0.05, new_x))
                new_y = max(0.05, min(height - 0.05, new_y))

                # Compute max radius at new location
                # Special handling for boundary cases
                new_r = min(new_x - 0.05, width - 0.05 - new_x,
                           new_y - 0.05, height - 0.05 - new_y)

                # Check for conflicts
                valid = True
                for j in range(n_circles):
                    if j != idx:
                        px, py, pr = circles[j]
                        dist = np.sqrt((new_x - px)**2 + (new_y - py)**2)
                        if dist < (new_r + pr):
                            valid = False
                            break

                if valid:
                    # See if this is better
                    new_sum = np.sum(circles[:, 2]) - old_r + new_r
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_x, best_y, best_r = new_x, new_y, new_r

            # Apply improvement if found
            if best_sum > np.sum(circles[:, 2]):
                circles[idx] = [best_x, best_y, best_r]
                improved = True

    return circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")