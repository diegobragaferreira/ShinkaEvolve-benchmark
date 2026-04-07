# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import math
from typing import Tuple, List

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

    def create_voronoi_based_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using Voronoi-based seeding for better spatial distribution"""
        circles = np.zeros((n_circles, 3))

        # Generate initial seed points using a combination of strategic placement and Voronoi analysis
        seed_points = []

        # Corner positions
        corner_positions = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
        ]

        # Edge center positions
        edge_positions = [
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
        ]

        # Center position
        center_position = (width * 0.5, height * 0.5)

        # Add strategic positions
        seed_points.extend(corner_positions)
        seed_points.extend(edge_positions)
        seed_points.append(center_position)

        # If we need more points, generate additional ones using Voronoi-inspired approach
        # Use a grid pattern to distribute additional points
        additional_points_needed = n_circles - len(seed_points)
        if additional_points_needed > 0:
            # Create a grid pattern that covers the area
            grid_rows = int(math.ceil(math.sqrt(additional_points_needed)))
            grid_cols = int(math.ceil(additional_points_needed / grid_rows))

            cell_width = width / (grid_cols + 1)
            cell_height = height / (grid_rows + 1)

            for i in range(grid_rows):
                for j in range(grid_cols):
                    if len(seed_points) >= n_circles:
                        break
                    x = (j + 1) * cell_width
                    y = (i + 1) * cell_height
                    # Slightly jitter points for better distribution
                    x += random.uniform(-cell_width * 0.1, cell_width * 0.1)
                    y += random.uniform(-cell_height * 0.1, cell_height * 0.1)
                    # Ensure within bounds
                    x = max(0.01, min(width - 0.01, x))
                    y = max(0.01, min(height - 0.01, y))
                    seed_points.append((x, y))
                if len(seed_points) >= n_circles:
                    break

        # Trim to exact number needed
        seed_points = seed_points[:n_circles]

        # Initialize circles with these seed points
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
            min_dist = np.min(distances)
            if len(distances) > 0:
                # Calculate min distance to other circles (not self)
                min_dist_to_others = np.min(distances - radii)
                actual_min_dist = min(min_dist, min_dist_to_others)
            else:
                actual_min_dist = min_dist
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

        # Check overlaps - vectorized operation
        overlap_mask = min_distances < np.min(radii_sums, axis=0)

        return not np.any(overlap_mask)

    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def adaptive_multi_scale_refinement_step(circles: np.ndarray, rect_width: float, rect_height: float,
                                           iterations: int = 100) -> np.ndarray:
        """Perform adaptive multi-scale local refinement with dynamic scale adjustment"""
        current = circles.copy()
        current_sum = calculate_radius_sum(current)

        # Track improvement history for adaptive scaling
        improvement_history = []
        max_history_length = 15

        # Define multiple scales for perturbations
        scales = [0.1, 0.05, 0.02, 0.01, 0.005]  # From coarse to fine
        scale_index = 0  # Start with coarse scale

        for iter_num in range(iterations):
            # Adaptively select scale based on recent performance
            if len(improvement_history) >= 5:
                recent_avg_improvement = np.mean(improvement_history[-5:])
                # If improvement is low, switch to finer scale
                if recent_avg_improvement < 0.0005:
                    scale_index = min(len(scales) - 1, scale_index + 1)
                # If improvement is high, try coarser scale for exploration
                elif recent_avg_improvement > 0.002:
                    scale_index = max(0, scale_index - 1)

            current_scale = scales[scale_index]

            # Try to improve each circle
            for i in range(len(current)):
                original_x, original_y, original_r = current[i]

                # Use adaptive number of perturbation attempts based on scale
                num_attempts = max(5, int(20 / (current_scale * 100)))
                best_x, best_y, best_r = original_x, original_y, original_r
                best_sum = current_sum

                # Generate perturbations with adaptive spread
                test_moves = []

                # Base Gaussian perturbations at current scale
                for _ in range(num_attempts // 2):
                    dx = current_scale * random.gauss(0, 1)
                    dy = current_scale * random.gauss(0, 1)
                    test_moves.append((dx, dy))

                # Coordinate-specific perturbations
                for _ in range(num_attempts // 4):
                    dx = current_scale * random.uniform(-1, 1)
                    dy = 0
                    test_moves.append((dx, dy))
                    dx = 0
                    dy = current_scale * random.uniform(-1, 1)
                    test_moves.append((dx, dy))

                # Small random moves for fine-tuning
                for _ in range(num_attempts // 4):
                    dx = random.uniform(-current_scale/2, current_scale/2)
                    dy = random.uniform(-current_scale/2, current_scale/2)
                    test_moves.append((dx, dy))

                # No move as baseline
                test_moves.append((0, 0))

                # Try different perturbations
                for dx, dy in test_moves:
                    test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
                    test_y = max(0.001, min(rect_height - 0.001, original_y + dy))

                    # Compute maximum possible radius at new position
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, 0.01]  # Temporarily small
                    max_r = compute_max_radius_at_position_vectorized(test_x, test_y, temp_circles, rect_width, rect_height)
                    test_r = min(max_r, max(0.001, original_r + random.uniform(-current_scale*0.5, current_scale*0.5)))

                    # Apply final adjustment
                    temp_circles[i] = [test_x, test_y, test_r]

                    # Validate and calculate new sum (strict validation)
                    if is_valid_configuration_vectorized(temp_circles, rect_width, rect_height):
                        new_sum = calculate_radius_sum(temp_circles)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_x, best_y, best_r = test_x, test_y, test_r

                # Update if improvement found
                if best_sum > current_sum:
                    current[i] = [best_x, best_y, best_r]
                    current_sum = best_sum

            # Track improvement for adaptive scaling
            improvement = current_sum - calculate_radius_sum(current)
            improvement_history.append(improvement)
            if len(improvement_history) > max_history_length:
                improvement_history.pop(0)

        return current

    def multi_start_optimization(n_starts: int = 5) -> np.ndarray:
        """Run multiple optimization starts to find better solutions"""
        best_circles = None
        best_sum = -float('inf')

        for start_num in range(n_starts):
            # Create different initial configurations
            if start_num == 0:
                # Use Voronoi-based initialization
                circles = create_voronoi_based_initialization(21, rect_width, rect_height)
            elif start_num == 1:
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
            else:
                # Create random initial configuration with better placement
                circles = np.zeros((21, 3))
                for i in range(21):
                    x = random.uniform(0.01, rect_width - 0.01)
                    y = random.uniform(0.01, rect_height - 0.01)
                    circles[i] = [x, y, 0.02]

            # Phase 1: Adaptive multi-scale coarse refinement
            refined_1 = adaptive_multi_scale_refinement_step(circles, rect_width, rect_height, 25)

            # Phase 2: Adaptive multi-scale medium refinement
            refined_2 = adaptive_multi_scale_refinement_step(refined_1, rect_width, rect_height, 50)

            # Phase 3: Adaptive multi-scale fine refinement
            refined_3 = adaptive_multi_scale_refinement_step(refined_2, rect_width, rect_height, 50)

            final_sum = calculate_radius_sum(refined_3)

            if final_sum > best_sum:
                best_sum = final_sum
                best_circles = refined_3.copy()

        return best_circles

    # Main optimization workflow
    # Multi-start optimization to avoid local optima
    final_circles = multi_start_optimization(5)

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
                final_circles = create_hybrid_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50)
                continue

            # Final validation check
            if is_valid_configuration_vectorized(final_circles, rect_width, rect_height):
                break
            else:
                # Retry with different initialization
                final_circles = create_hybrid_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50)

    # Final optimization pass to fine-tune
    if final_circles is not None:
        final_circles = adaptive_multi_scale_refinement_step(final_circles, rect_width, rect_height, 25)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")