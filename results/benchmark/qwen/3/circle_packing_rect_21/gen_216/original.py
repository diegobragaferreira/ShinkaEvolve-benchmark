# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
import random
import math
from typing import Tuple, List
from scipy.spatial.distance import cdist

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
        """Create initial configuration using Voronoi-based spatial distribution"""
        circles = np.zeros((n_circles, 3))

        # Generate initial points for Voronoi diagram
        # Start with corner and edge positions
        initial_points = [
            (width * 0.1, height * 0.1),    # bottom-left
            (width * 0.9, height * 0.1),    # bottom-right
            (width * 0.1, height * 0.9),    # top-left
            (width * 0.9, height * 0.9),    # top-right
            (width * 0.5, height * 0.1),    # bottom-middle
            (width * 0.5, height * 0.9),    # top-middle
            (width * 0.1, height * 0.5),    # left-middle
            (width * 0.9, height * 0.5),    # right-middle
        ]

        # Add random points to ensure good coverage
        additional_points = []
        for _ in range(n_circles - len(initial_points)):
            x = random.uniform(0.05 * width, 0.95 * width)
            y = random.uniform(0.05 * height, 0.95 * height)
            additional_points.append([x, y])

        # Combine all points
        all_points = initial_points + additional_points[:n_circles - len(initial_points)]

        # Create Voronoi diagram
        try:
            vor = Voronoi(all_points)

            # Get Voronoi vertices that are inside our rectangle
            valid_vertices = []
            for vertex in vor.vertices:
                if (0 <= vertex[0] <= width) and (0 <= vertex[1] <= height):
                    valid_vertices.append(vertex)

            # Use Voronoi vertices as center points
            chosen_centers = valid_vertices[:min(n_circles, len(valid_vertices))]
            if len(chosen_centers) < n_circles:
                # Fill missing circles with random points
                for i in range(len(chosen_centers), n_circles):
                    x = random.uniform(0.05 * width, 0.95 * width)
                    y = random.uniform(0.05 * height, 0.95 * height)
                    chosen_centers.append([x, y])

            # Assign positions with initial small radii
            for i, (x, y) in enumerate(chosen_centers[:n_circles]):
                circles[i] = [x, y, 0.02]

        except Exception:
            # Fallback to simple initialization if Voronoi fails
            for i in range(n_circles):
                x = random.uniform(0.05 * width, 0.95 * width)
                y = random.uniform(0.05 * height, 0.95 * height)
                circles[i] = [x, y, 0.02]

        return circles

    def compute_max_radius_at_position_voronoi(x: float, y: float, existing_circles: np.ndarray,
                                             rect_width: float, rect_height: float) -> float:
        """Compute maximum possible radius using Voronoi-like proximity analysis"""
        # Distance to boundaries
        min_bound = min(x, rect_width - x, y, rect_height - y)

        # Distance to other circles using Voronoi-inspired approach
        min_dist = float('inf')
        for i in range(len(existing_circles)):
            ex, ey, er = existing_circles[i]

            # Calculate distance to circle center
            dx = x - ex
            dy = y - ey
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0.0001:  # Avoid self-distance
                # Minimum distance from point to circle circumference
                dist_to_edge = dist - er
                min_dist = min(min_dist, dist_to_edge)

                # Early termination for very close circles
                if min_dist < 0.001:
                    break

        # Take minimum of boundary and other-circle distances
        max_radius = min(min_bound, min_dist if min_dist < float('inf') else float('inf'))
        return max(0.001, max_radius)

    def validate_configuration_voronoi(circles: np.ndarray, rect_width: float, rect_height: float) -> bool:
        """Validate configuration using vectorized approach with Voronoi insights"""
        # Check boundary constraints
        if np.any(circles[:, 0] - circles[:, 2] < 0) or \
           np.any(circles[:, 0] + circles[:, 2] > rect_width) or \
           np.any(circles[:, 1] - circles[:, 2] < 0) or \
           np.any(circles[:, 1] + circles[:, 2] > rect_height):
            return False

        # Check overlap constraints using more efficient vectorized approach
        if len(circles) < 2:
            return True

        # Use distance matrix for overlap detection
        positions = circles[:, :2]
        radii = circles[:, 2]

        # Create distance matrix
        dist_matrix = cdist(positions, positions)

        # Set diagonal to infinity (self-distances)
        np.fill_diagonal(dist_matrix, float('inf'))

        # Minimum distances between all pairs
        min_distances = np.min(dist_matrix, axis=1)

        # Required minimum distances (sum of radii)
        required_distances = radii[:, np.newaxis] + radii[np.newaxis, :]

        # Check if any pair violates overlap constraint
        overlap_violations = min_distances < np.min(required_distances, axis=0)

        return not np.any(overlap_violations)

    def calculate_radius_sum(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])

    def voronoi_guided_local_search(circles: np.ndarray, rect_width: float, rect_height: float,
                                  iterations: int = 200) -> np.ndarray:
        """Local search guided by Voronoi spatial relationships"""
        current = circles.copy()
        current_sum = calculate_radius_sum(current)

        # Track recent improvements for adaptive search
        recent_improvements = []
        max_recent = 10

        for iter_num in range(iterations):
            # Adaptive step size based on convergence
            base_step_size = 0.05
            if len(recent_improvements) > 5:
                avg_improvement = np.mean(recent_improvements[-5:])
                if avg_improvement < 0.001:
                    current_step_size = base_step_size * 0.8  # Reduce step size
                else:
                    current_step_size = base_step_size * 1.1  # Increase step size
            else:
                current_step_size = base_step_size

            # Randomly select circles to update
            update_order = list(range(len(current)))
            random.shuffle(update_order)

            # Update circles in random order
            for i in update_order:
                original_x, original_y, original_r = current[i]

                # Use Voronoi-inspired move strategies
                # Strategy 1: Move toward nearest neighbors
                best_x, best_y, best_r = original_x, original_y, original_r
                best_sum = current_sum

                # Collect information about neighbors
                neighbor_distances = []
                for j in range(len(current)):
                    if i != j:
                        dx = current[j, 0] - original_x
                        dy = current[j, 1] - original_y
                        dist = math.sqrt(dx*dx + dy*dy)
                        neighbor_distances.append((dist, j))

                # Sort neighbors by distance
                neighbor_distances.sort(key=lambda x: x[0])

                # Try several moves
                moves = []

                # Move towards closest neighbor
                if len(neighbor_distances) > 0:
                    closest_j = neighbor_distances[0][1]
                    dx = current[closest_j, 0] - original_x
                    dy = current[closest_j, 1] - original_y
                    moves.append((dx * 0.05, dy * 0.05))  # Small move toward neighbor

                # Add random moves
                moves.extend([
                    (current_step_size * random.gauss(0, 1), current_step_size * random.gauss(0, 1)),
                    (current_step_size * random.uniform(-1, 1), 0),
                    (0, current_step_size * random.uniform(-1, 1)),
                    (current_step_size * random.choice([-1, 1]), 0),
                    (0, current_step_size * random.choice([-1, 1])),
                    (random.uniform(-current_step_size/2, current_step_size/2),
                     random.uniform(-current_step_size/2, current_step_size/2)),
                    (0, 0)  # No move baseline
                ])

                # Try each move
                for dx, dy in moves:
                    test_x = max(0.001, min(rect_width - 0.001, original_x + dx))
                    test_y = max(0.001, min(rect_height - 0.001, original_y + dy))

                    # Compute maximum possible radius at new location
                    temp_circles = current.copy()
                    temp_circles[i] = [test_x, test_y, 0.01]  # Temporary small radius

                    max_r = compute_max_radius_at_position_voronoi(test_x, test_y, temp_circles, rect_width, rect_height)
                    test_r = min(max_r, max(0.001, original_r + random.uniform(-0.03, 0.03)))

                    # Final adjustment
                    temp_circles[i] = [test_x, test_y, test_r]

                    # Validate and calculate sum
                    if validate_configuration_voronoi(temp_circles, rect_width, rect_height):
                        new_sum = calculate_radius_sum(temp_circles)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_x, best_y, best_r = test_x, test_y, test_r

                # Update if improvement found
                if best_sum > current_sum:
                    current[i] = [best_x, best_y, best_r]
                    current_sum = best_sum

                    # Track improvement for adaptive behavior
                    recent_improvements.append(best_sum - calculate_radius_sum(current))
                    if len(recent_improvements) > max_recent:
                        recent_improvements.pop(0)

        return current

    def multi_stage_voronoi_optimization(n_starts: int = 6) -> np.ndarray:
        """Multi-stage optimization using Voronoi guidance"""
        best_circles = None
        best_sum = -float('inf')

        for start_num in range(n_starts):
            # Different initialization strategies
            if start_num == 0:
                # Voronoi-based initialization
                circles = create_voronoi_initialization(21, rect_width, rect_height)
            elif start_num == 1:
                # Simple random initialization
                circles = np.zeros((21, 3))
                for i in range(21):
                    x = random.uniform(0.01, rect_width - 0.01)
                    y = random.uniform(0.01, rect_height - 0.01)
                    circles[i] = [x, y, 0.02]
            else:
                # Hexagonal grid initialization
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

            # Stage 1: Coarse Voronoi-guided search
            refined_1 = voronoi_guided_local_search(circles, rect_width, rect_height, 50)

            # Stage 2: Medium refinement
            refined_2 = voronoi_guided_local_search(refined_1, rect_width, rect_height, 100)

            # Stage 3: Fine refinement with more iterations
            refined_3 = voronoi_guided_local_search(refined_2, rect_width, rect_height, 150)

            final_sum = calculate_radius_sum(refined_3)

            if final_sum > best_sum:
                best_sum = final_sum
                best_circles = refined_3.copy()

        return best_circles

    # Main optimization workflow
    final_circles = multi_stage_voronoi_optimization(6)

    # Final validation
    if final_circles is not None:
        # Ensure final configuration is valid
        iterations = 0
        max_iterations = 10
        while not validate_configuration_voronoi(final_circles, rect_width, rect_height) and iterations < max_iterations:
            # Regenerate if invalid
            final_circles = create_voronoi_initialization(21, rect_width, rect_height)
            final_circles = voronoi_guided_local_search(final_circles, rect_width, rect_height, 100)
            iterations += 1

    return final_circles

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")