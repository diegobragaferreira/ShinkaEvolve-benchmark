# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import math
from typing import Tuple, List
from scipy.spatial import Voronoi
import warnings

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2, using 1.5 x 0.5 for good aspect ratio
    rect_width, rect_height = 1.5, 0.5

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    def create_voronoi_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using Voronoi-based seeding for even distribution"""
        circles = np.zeros((n_circles, 3))

        # Generate initial seed points using a modified grid pattern to avoid clustering
        seed_points = []

        # Add corner and edge points for boundary coverage
        corner_positions = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
        ]

        seed_points.extend(corner_positions)

        # Add additional points in a grid pattern
        grid_rows, grid_cols = 4, 4
        for i in range(grid_rows):
            for j in range(grid_cols):
                if len(seed_points) >= n_circles:
                    break
                x = width * (0.1 + 0.8 * j / (grid_cols - 1) if grid_cols > 1 else 0.5)
                y = height * (0.1 + 0.8 * i / (grid_rows - 1) if grid_rows > 1 else 0.5)
                seed_points.append((x, y))

        # Make sure we have enough seed points
        while len(seed_points) < n_circles:
            x = random.uniform(0.05, width - 0.05)
            y = random.uniform(0.05, height - 0.05)
            seed_points.append((x, y))

        # Take first n_circles points and assign initial radii
        seed_points = seed_points[:n_circles]
        for i, (x, y) in enumerate(seed_points):
            circles[i] = [x, y, 0.02]

        return circles

    def compute_max_radius_at_position_vectorized(x: float, y: float, existing_circles: np.ndarray,
                                                rect_width: float, rect_height: float) -> float:
        """Vectorized computation of maximum possible radius for a circle at given position"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)

        # Vectorized distance calculation to all existing circles
        if len(existing_circles) > 0:
            positions = existing_circles[:, :2]
            radii = existing_circles[:, 2]

            # Calculate distances to all existing circles
            dx = positions[:, 0] - x
            dy = positions[:, 1] - y
            distances = np.sqrt(dx*dx + dy*dy)

            # Avoid self-distance and compute min distance to other circles
            # Set self-distances to infinity to avoid them
            distances = np.where(distances == 0, float('inf'), distances)

            # Min distance minus sum of radii
            if len(distances) > 0:
                # Calculate min distance to other circles (not self)
                min_dist_to_others = np.min(distances - radii)
                actual_min_dist = min_dist_to_others
            else:
                actual_min_dist = float('inf')
        else:
            actual_min_dist = float('inf')

        # Take minimum of boundary and other-circle distances
        max_radius = min(min_bound, actual_min_dist if actual_min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)

    def is_valid_configuration_vectorized(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Vectorized validation of circle configuration"""
        # Check boundary constraints efficiently
        if np.any(circles[:, 0] - circles[:, 2] < 0) or \
           np.any(circles[:, 0] + circles[:, 2] > rect_width) or \
           np.any(circles[:, 1] - circles[:, 2] < 0) or \
           np.any(circles[:, 1] + circles[:, 2] > rect_height):
            return False

        # Check overlap constraints efficiently using vectorized computation
        if len(circles) < 2:
            return True

        # Use vectorized computation for overlap detection
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Create distance matrix
        dist_matrix = cdist(positions, positions)

        # Set diagonal to infinity (self-distances)
        np.fill_diagonal(dist_matrix, float('inf'))

        # Minimum distances between circles
        min_distances = np.min(dist_matrix, axis=1)

        # Minimum sum of radii for each circle pair
        radii_sums = radii[:, np.newaxis] + radii[np.newaxis, :]

        # Check overlaps - we only consider pairs where distance is less than sum of radii
        overlap_mask = min_distances < np.min(radii_sums, axis=0)

        return not np.any(overlap_mask)

    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def compute_radius_gradient(circles: np.ndarray, idx: int, rect_width: float, rect_height: float) -> Tuple[float, float]:
        """Compute approximate gradient of radius with respect to position"""
        x, y, r = circles[idx]

        # Compute boundary gradients
        d_bound_dx = 1.0 if x < rect_width - x else -1.0
        d_bound_dy = 1.0 if y < rect_height - y else -1.0

        # Compute neighbor gradients
        d_neighbor_dx = 0.0
        d_neighbor_dy = 0.0
        min_dist_to_neighbor = float('inf')

        for i in range(len(circles)):
            if i != idx:
                ex, ey, er = circles[i]
                dx = x - ex
                dy = y - ey
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0.0001:
                    # Compute derivative of distance w.r.t. x and y
                    d_dist_dx = dx / dist if dist > 0 else 0.0
                    d_dist_dy = dy / dist if dist > 0 else 0.0

                    # Derivative of (dist - er) w.r.t. x and y
                    d_neighbor_dx += d_dist_dx
                    d_neighbor_dy += d_dist_dy

                    # Track minimum distance
                    if dist - er < min_dist_to_neighbor:
                        min_dist_to_neighbor = dist - er

        # For simplicity, use the gradient from closest neighbors
        # In practice, we'd want to compute this more precisely
        return d_neighbor_dx, d_neighbor_dy

    def local_refinement_step(circles: np.ndarray, rect_width: float, rect_height: float,
                            iterations: int = 100, relax_overlap: bool = True) -> np.ndarray:
        """Perform local refinement with adaptive parameters and gradient-based moves"""
        current = circles.copy()
        current_sum = calculate_radius_sum(current)

        # Adaptive step sizes and parameters that change during optimization
        initial_step_size = 0.05
        min_step_size = 0.001
        step_reduction_factor = 0.95
        adaptive_threshold = 0.01  # Threshold to decide whether to reduce step size

        for iter_num in range(iterations):
            # Dynamic step size reduction
            step_size = max(min_step_size, initial_step_size * (step_reduction_factor ** (iter_num // 5)))

            # Adaptive iteration control
            if iter_num > 0 and abs(current_sum - calculate_radius_sum(current)) < adaptive_threshold:
                step_size *= 0.7  # Reduce step size if little improvement seen

            # Try to improve each circle
            for i in range(len(current)):
                original_x, original_y, original_r = current[i]

                # Better perturbation strategy with direction awareness
                best_x, best_y, best_r = original_x, original_y, original_r
                best_sum = current_sum

                # Multiple perturbation strategies
                test_moves = []

                # Base Gaussian perturbations
                test_moves.append((step_size * random.gauss(0, 1), step_size * random.gauss(0, 1)))
                test_moves.append((step_size * random.gauss(0, 1), 0))
                test_moves.append((0, step_size * random.gauss(0, 1)))

                # Coordinate-specific perturbations
                test_moves.append((step_size * random.uniform(-1, 1), 0))
                test_moves.append((0, step_size * random.uniform(-1, 1)))

                # Small random moves
                test_moves.append((random.uniform(-step_size/2, step_size/2), random.uniform(-step_size/2, step_size/2)))

                # No move (baseline)
                test_moves.append((0, 0))

                # More focused perturbations in later iterations
                if iter_num > iterations // 3:
                    # Add more precise movement directions
                    test_moves.extend([
                        (step_size * random.choice([-1, 1]), 0),
                        (0, step_size * random.choice([-1, 1])),
                        (step_size, step_size),
                        (-step_size, -step_size),
                    ])

                # Add gradient-based moves in later iterations
                if iter_num > iterations // 2:
                    try:
                        grad_x, grad_y = compute_radius_gradient(current, i, rect_width, rect_height)
                        # Move in direction that maximizes radius increase (opposite of gradient)
                        norm = math.sqrt(grad_x*grad_x + grad_y*grad_y)
                        if norm > 0.001:
                            # Scale gradient to step size
                            scale = step_size / norm
                            test_moves.append((-grad_x * scale, -grad_y * scale))
                    except:
                        pass

                for dx, dy in test_moves:
                    test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
                    test_y = max(0.001, min(rect_height - 0.001, original_y + dy))

                    # Compute maximum possible radius at new position
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, 0.01]  # Temporarily small
                    max_r = compute_max_radius_at_position_vectorized(test_x, test_y, temp_circles, rect_width, rect_height)

                    # Perturb radius with more sophisticated approach
                    test_r = min(max_r, max(0.001, original_r + random.uniform(-0.03, 0.03)))

                    # Apply final adjustment
                    temp_circles[i] = [test_x, test_y, test_r]

                    # Validate and calculate new sum
                    if relax_overlap and iter_num < iterations // 2:
                        # With relaxed overlap checking in early phase
                        valid = True
                        # Just check boundary constraints
                        if (test_x - test_r < 0 or test_x + test_r > rect_width or
                            test_y - test_r < 0 or test_y + test_r > rect_height):
                            valid = False

                        if valid:
                            new_sum = calculate_radius_sum(temp_circles)
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_x, best_y, best_r = test_x, test_y, test_r
                    else:
                        # Strict validation in later phases
                        if is_valid_configuration_vectorized(temp_circles, rect_width, rect_height):
                            new_sum = calculate_radius_sum(temp_circles)
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_x, best_y, best_r = test_x, test_y, test_r

                # Update if improvement found
                if best_sum > current_sum:
                    current[i] = [best_x, best_y, best_r]
                    current_sum = best_sum

        return current

    def multi_start_optimization(n_starts: int = 7) -> np.ndarray:
        """Run multiple optimization starts to find better solutions"""
        best_circles = None
        best_sum = -float('inf')

        # Different initialization strategies for variety
        init_strategies = ['voronoi', 'hexagonal', 'random']

        for start_num in range(n_starts):
            # Select initialization strategy
            strategy = init_strategies[start_num % len(init_strategies)]

            if strategy == 'voronoi':
                circles = create_voronoi_initialization(21, rect_width, rect_height)
            elif strategy == 'hexagonal':
                # Create hexagonal grid initialization
                circles = np.zeros((21, 3))
                rows = int(math.ceil(math.sqrt(21)))
                cols = int(math.ceil(21 / rows))
                cell_width = rect_width / (cols + 1)
                cell_height = rect_height / (rows + 1)
                idx = 0
                for i in range(rows):
                    for j in range(cols):
                        if idx >= 21:
                            break
                        x_offset = 0.0 if i % 2 == 0 else 0.5
                        x = (j + 1 + x_offset) * cell_width
                        y = (i + 1) * cell_height
                        x = max(0.01, min(rect_width - 0.01, x))
                        y = max(0.01, min(rect_height - 0.01, y))
                        circles[idx] = [x, y, 0.02]
                        idx += 1
                        if idx >= 21:
                            break
            else:  # random
                circles = np.zeros((21, 3))
                for i in range(21):
                    x = random.uniform(0.01, rect_width - 0.01)
                    y = random.uniform(0.01, rect_height - 0.01)
                    circles[i] = [x, y, 0.02]

            # Phase 1: Coarse refinement with fewer iterations
            refined_1 = local_refinement_step(circles, rect_width, rect_height, 20, relax_overlap=True)

            # Phase 2: Medium refinement
            refined_2 = local_refinement_step(refined_1, rect_width, rect_height, 50, relax_overlap=True)

            # Phase 3: Fine refinement with more iterations
            refined_3 = local_refinement_step(refined_2, rect_width, rect_height, 50, relax_overlap=False)

            final_sum = calculate_radius_sum(refined_3)

            if final_sum > best_sum:
                best_sum = final_sum
                best_circles = refined_3.copy()

        return best_circles

    # Main optimization workflow
    # Multi-start optimization to avoid local optima
    final_circles = multi_start_optimization(7)

    # Final validation and cleanup
    if final_circles is not None:
        # Double-check validity and ensure all constraints
        while True:
            valid = True

            # Check boundaries and overlaps using vectorized method
            if not is_valid_configuration_vectorized(final_circles, rect_width, rect_height):
                valid = False

            if not valid:
                # Reinitialize if invalid
                final_circles = create_voronoi_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 40)
                continue

            # Final validation check
            if is_valid_configuration_vectorized(final_circles, rect_width, rect_height):
                break
            else:
                # Retry with different initialization
                final_circles = create_voronoi_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 40)

    # Final optimization pass to fine-tune
    if final_circles is not None:
        final_circles = local_refinement_step(final_circles, rect_width, rect_height, 20, relax_overlap=False)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")