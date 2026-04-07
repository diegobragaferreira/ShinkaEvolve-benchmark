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

    def create_hybrid_initialization(n_circles: int, width: float, height: float) -> np.ndarray:
        """Create initial configuration using a hybrid approach combining corner placements and hexagonal grid"""
        circles = np.zeros((n_circles, 3))

        # Corner and edge positions to establish boundaries
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

        # Place some circles at strategic positions
        for i in range(min(len(corner_positions), n_circles)):
            x, y = corner_positions[i]
            circles[i] = [x, y, 0.03]

        # Fill remaining positions with hexagonal grid
        remaining = n_circles - len(corner_positions)
        if remaining > 0:
            rows = int(math.ceil(math.sqrt(remaining)))
            cols = int(math.ceil(remaining / rows))

            cell_width = width / (cols + 1)
            cell_height = height / (rows + 1)

            idx = len(corner_positions)
            for i in range(rows):
                for j in range(cols):
                    if idx >= n_circles:
                        break
                    x_offset = 0.0 if i % 2 == 0 else 0.5
                    x = (j + 1 + x_offset) * cell_width
                    y = (i + 1) * cell_height
                    # Ensure within bounds
                    x = max(0.01, min(width - 0.01, x))
                    y = max(0.01, min(height - 0.01, y))
                    circles[idx] = [x, y, 0.02]
                    idx += 1
                    if idx >= n_circles:
                        break

        return circles

    def compute_max_radius_at_position(x: float, y: float, existing_circles: np.ndarray,
                                     rect_width: float, rect_height: float) -> float:
        """Compute maximum possible radius for a circle at given position"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)

        # Distance to other circles (with early termination if already very close)
        min_dist = float('inf')
        for i in range(len(existing_circles)):
            ex, ey, er = existing_circles[i]
            dx = x - ex
            dy = y - ey
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.0001:  # Avoid self-distance
                min_dist = min(min_dist, dist - er)
                # Early termination if already too close to another circle
                if min_dist < 0.001:
                    break

        # Take minimum of boundary and other-circle distances
        max_radius = min(min_bound, min_dist if min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)

    def is_valid_configuration(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Check if configuration is valid (no overlaps, all within bounds)"""
        # Check boundary constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                return False

        # Check overlap constraints with early termination
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_squared = dx*dx + dy*dy
                min_dist_squared = (r1 + r2) * (r1 + r2)
                if dist_squared < min_dist_squared:
                    return False
        return True

    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def local_refinement_step(circles: np.ndarray, rect_width: float, rect_height: float,
                            iterations: int = 100, relax_overlap: bool = True) -> np.ndarray:
        """Perform local refinement to improve circle configuration"""
        current = circles.copy()
        current_sum = calculate_radius_sum(current)

        # Adaptive step sizes
        step_sizes = [0.05, 0.03, 0.01]

        for iter_num in range(iterations):
            # Gradually tighten overlap constraints over time
            overlap_tolerance_factor = 1.0 if not relax_overlap else max(0.1, 1.0 - (iter_num / iterations) * 0.8)

            # Decrease step size over time
            step_size = step_sizes[min(iter_num // 30, len(step_sizes)-1)]

            # Try to improve each circle
            for i in range(len(current)):
                original_x, original_y, original_r = current[i]

                # Try several moves with random directions
                best_x, best_y, best_r = original_x, original_y, original_r
                best_sum = current_sum

                # Try different perturbations
                test_moves = [
                    (step_size * random.gauss(0, 1), step_size * random.gauss(0, 1)),
                    (step_size * random.gauss(0, 1), 0),
                    (0, step_size * random.gauss(0, 1)),
                    (0, 0)
                ]

                for dx, dy in test_moves:
                    test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
                    test_y = max(0.001, min(rect_height - 0.001, original_y + dy))

                    # Compute maximum possible radius at new position
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, 0.01]  # Temporarily small
                    max_r = compute_max_radius_at_position(test_x, test_y, temp_circles, rect_width, rect_height)
                    test_r = min(max_r, max(0.001, original_r + random.uniform(-0.02, 0.02)))

                    # Apply final adjustment
                    temp_circles[i] = [test_x, test_y, test_r]

                    # Validate and calculate new sum
                    # Allow relaxed checking in early iterations
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
                        if is_valid_configuration(temp_circles, rect_width, rect_height):
                            new_sum = calculate_radius_sum(temp_circles)
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_x, best_y, best_r = test_x, test_y, test_r

                # Update if improvement found
                if best_sum > current_sum:
                    current[i] = [best_x, best_y, best_r]
                    current_sum = best_sum

        return current

    def multi_start_optimization(n_starts: int = 5) -> np.ndarray:
        """Run multiple optimization starts to find better solutions"""
        best_circles = None
        best_sum = -float('inf')

        for start_num in range(n_starts):
            # Create different initial configurations
            if start_num == 0:
                # Use hybrid initialization
                circles = create_hybrid_initialization(21, rect_width, rect_height)
            else:
                # Create random initial configuration with better placement
                circles = np.zeros((21, 3))
                for i in range(21):
                    x = random.uniform(0.01, rect_width - 0.01)
                    y = random.uniform(0.01, rect_height - 0.01)
                    circles[i] = [x, y, 0.02]

            # Phase 1: Coarse refinement
            refined_1 = local_refinement_step(circles, rect_width, rect_height, 50)

            # Phase 2: Fine refinement
            refined_2 = local_refinement_step(refined_1, rect_width, rect_height, 100)

            # Phase 3: Additional refinement
            refined_3 = local_refinement_step(refined_2, rect_width, rect_height, 50)

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

            # Check boundaries and overlaps
            for i in range(len(final_circles)):
                x, y, r = final_circles[i]
                if x - r < 0 or x + r > rect_width or y - r < 0 or y + r > rect_height:
                    valid = False
                    break

            if not valid:
                # Reinitialize if invalid
                final_circles = create_hybrid_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50)
                continue

            # Check overlaps
            for i in range(len(final_circles)):
                for j in range(i+1, len(final_circles)):
                    x1, y1, r1 = final_circles[i]
                    x2, y2, r2 = final_circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_squared = dx*dx + dy*dy
                    min_dist_squared = (r1 + r2) * (r1 + r2)
                    if dist_squared < min_dist_squared:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                break
            else:
                # Retry with different initialization
                final_circles = create_hybrid_initialization(21, rect_width, rect_height)
                final_circles = local_refinement_step(final_circles, rect_width, rect_height, 50)

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")